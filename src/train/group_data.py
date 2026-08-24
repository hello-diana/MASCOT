"""Dataset construction for Stage-2 group training.

Moved verbatim from src/train/train_group.py (Phase-2 Part B): group training
dataset construction (fresh rollout or from branching episodes), DPO dataset
construction, deterministic branching-artifact resolution, and dataset-name
inference. The legacy two-run preference conversion lives in
src/data_generation/preference_pairs.py and is re-exported here.
"""

import glob
import json
import logging
import os
import random

from datasets import Dataset

from config.persona_config import AGENTS
from src.const import EMPATHETIC_DATASET, ESCONV_DATASET, QMSUM_DATASET
# Legacy two-run episode conversion (shared preference-pair core).
from src.data_generation.preference_pairs import \
    load_and_convert_to_preference_pairs  # noqa: F401
from src.prompts.persona_templates import MascotPersonaTemplates
from src.train.group_rollout import build_prefix_turns
from src.utils.data_utils import clean_response
from src.utils.paths import artifact_dir

logger = logging.getLogger(__name__)


def create_group_training_dataset(
    tokenizer,
    meeting_data_path: str,
    agents: dict = None,
    num_episode_turns: int = 5,
    directive_generator=None,
    agent_generator=None,
    user_simulator=None,
    user_response_probability: float = 0.3,
    dataset_name: str = QMSUM_DATASET,
) -> Dataset:
    """Create dataset for group directive-generator training.

    For each meeting segment, randomly samples a director turn k
    (0 to num_episode_turns-1). If k > 0, generates prefix turns
    using frozen vLLM models. The resulting conversation state is
    stored as dataset columns for use during reward computation.

    Args:
        tokenizer: Tokenizer instance.
        meeting_data_path: Path to meeting data JSONL.
        agents: Agent dictionary (default: AGENTS).
        num_episode_turns: Max prefix length (k range: 0 to this-1).
        directive_generator: Frozen LLM for prefix directive generation.
        agent_generator: Frozen LLM for prefix agent responses.
        user_simulator: Optional user simulation LLM.
        user_response_probability: Probability of user turn per round.

    Returns:
        Dataset with columns: [prompt, meeting_context, episode_id,
        train_k, prefix_conversation_history, prefix_agents_spoken,
        prefix_turns]
    """
    if agents is None:
        agents = AGENTS

    logger.info(f"Loading meeting data from {meeting_data_path}")

    # Collect all samples with their randomly-sampled k values
    all_samples = []
    with open(meeting_data_path, 'r') as f:
        for line_idx, line in enumerate(f):
            meeting_data = json.loads(line)
            meeting_id = meeting_data.get("meeting_id", f"meeting_{line_idx}")
            transcript = meeting_data.get("original_transcript", [])

            turns_per_segment = 100
            num_segments = min(3, len(transcript) // turns_per_segment)

            for seg_idx in range(num_segments):
                start_turn = seg_idx * turns_per_segment
                end_turn = min(start_turn + turns_per_segment, len(transcript))

                meeting_context = "\n".join([
                    f"{turn.get('speaker', 'Speaker')}: {turn.get('text', '')}"
                    for turn in transcript[start_turn:end_turn]
                ])

                k = random.randint(0, num_episode_turns - 1)
                all_samples.append({
                    "meeting_context": meeting_context,
                    "episode_id": f"{meeting_id}_seg{seg_idx}",
                    "train_k": k,
                })

    logger.info(f"Collected {len(all_samples)} samples, generating prefixes...")

    # Batch-generate prefixes for all samples
    meeting_contexts = [s["meeting_context"] for s in all_samples]
    k_values = [s["train_k"] for s in all_samples]

    if directive_generator is not None and k_values and max(k_values) > 0:
        prefix_states = build_prefix_turns(
            directive_generator=directive_generator,
            agent_generator=agent_generator,
            meeting_contexts=meeting_contexts,
            k_values=k_values,
            agents=agents,
            user_simulator=user_simulator,
            user_response_probability=user_response_probability,
            dataset_name=dataset_name,
        )
    else:
        # No prefix needed (all k=0 or no generator provided)
        prefix_states = [
            {
                "meeting_context": mc,
                "conversation_history": "",
                "agents_spoken": {name: 0 for name in agents},
                "prefix_turns": [],
            }
            for mc in meeting_contexts
        ]

    # Build directive prompts at turn k using prefix state
    dataset_rows = []
    for i, sample in enumerate(all_samples):
        prompt_text = MascotPersonaTemplates.directive_prompt(
            dataset_name,
            agents=agents,
            meeting_context=sample["meeting_context"],
            conversation_history=prefix_states[i]["conversation_history"],
            agents_spoken=prefix_states[i]["agents_spoken"],
        )
        dataset_rows.append({
            "prompt": prompt_text,
            "meeting_context": sample["meeting_context"],
            "episode_id": sample["episode_id"],
            "train_k": sample["train_k"],
            "prefix_conversation_history": prefix_states[i]["conversation_history"],
            "prefix_agents_spoken": json.dumps(prefix_states[i]["agents_spoken"]),
            "prefix_turns": json.dumps(prefix_states[i]["prefix_turns"]),
        })

    k_dist = {}
    for k in k_values:
        k_dist[k] = k_dist.get(k, 0) + 1
    logger.info(
        f"Created {len(dataset_rows)} training prompts "
        f"(k distribution: {dict(sorted(k_dist.items()))})"
    )
    return Dataset.from_list(dataset_rows)


def create_group_dataset_from_episodes(
    episode_dataset_path: str,
    agents: dict,
    dataset_name: str,
) -> Dataset:
    """Create GRPO training dataset from pre-generated branching episode data.

    Reconstructs prefix states from saved episode columns, avoiding frozen
    model inference. Each row's branch_turn_idx becomes train_k.

    Args:
        episode_dataset_path: Path to HF Dataset with branching episode columns.
        agents: Agent dictionary (AGENTS or EMPATHETIC_AGENTS).
        dataset_name: "QMSum" or "Empathetic".

    Returns:
        Dataset with same 7-column schema as create_group_training_dataset():
        [prompt, meeting_context, episode_id, train_k,
         prefix_conversation_history, prefix_agents_spoken, prefix_turns]
    """

    logger.info(f"Loading branching episode dataset from {episode_dataset_path}")
    ds = Dataset.load_from_disk(episode_dataset_path)
    logger.info(f"Loaded {len(ds)} rows, dataset_name={dataset_name}")

    dataset_rows = []
    for row in ds:
        # Parse stored columns
        ctx = json.loads(row["episode_context"])
        turns = json.loads(row["episode_turns"])
        k = int(row["branch_turn_idx"])

        # Slice prefix turns (0..k-1)
        prefix_turns = turns[:k]

        # Recompute prefix-only agents_spoken from prefix turns
        prefix_agents_spoken = {name: 0 for name in agents}
        for t in prefix_turns:
            agent_name = t.get("agent_name", t.get("name", "unknown"))
            if agent_name in prefix_agents_spoken:
                prefix_agents_spoken[agent_name] += 1

        # Build conversation history matching build_prefix_turns format
        conv_lines = []
        for t in prefix_turns:
            agent_name = t.get("agent_name", t.get("name", "unknown"))
            response = t.get("response", "")
            clean = clean_response(response)
            conv_lines.append(f"[{agent_name}]: {clean}")
        conversation_history = "\n".join(conv_lines) + ("\n" if conv_lines else "")

        # Add agent_response key for compatibility with downstream code
        enriched_prefix_turns = []
        for t in prefix_turns:
            et = dict(t)
            if "agent_response" not in et:
                et["agent_response"] = et.get("response", "")
            enriched_prefix_turns.append(et)

        # Build directive prompt at turn k
        if dataset_name == QMSUM_DATASET:
            meeting_context = ctx["meeting_context"]
            prompt_text = MascotPersonaTemplates.directive_prompt_qmsum(
                agents=agents,
                meeting_context=meeting_context,
                conversation_history=conversation_history,
                agents_spoken=prefix_agents_spoken,
            )
        elif dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET):
            meeting_context = json.dumps(ctx)  # Store full context JSON
            prompt_text = MascotPersonaTemplates.directive_prompt_empathetic(
                agents=agents,
                situation=ctx["initial_situation"],
                emotion=ctx["initial_emotion"],
                conversation_history=conversation_history,
                agents_spoken=prefix_agents_spoken,
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        # Episode ID from meeting_id or conv_id
        episode_id = row.get("meeting_id") or row.get("conv_id") or f"ep_{len(dataset_rows)}"

        dataset_rows.append({
            "prompt": prompt_text,
            "meeting_context": meeting_context,
            "episode_id": str(episode_id),
            "train_k": k,
            "prefix_conversation_history": conversation_history,
            "prefix_agents_spoken": json.dumps(prefix_agents_spoken),
            "prefix_turns": json.dumps(enriched_prefix_turns),
        })

    k_dist = {}
    for r in dataset_rows:
        kv = r["train_k"]
        k_dist[kv] = k_dist.get(kv, 0) + 1
    logger.info(
        f"Created {len(dataset_rows)} training prompts from episodes "
        f"(k distribution: {dict(sorted(k_dist.items()))})"
    )
    return Dataset.from_list(dataset_rows)


def resolve_branching_episode_dirs(args, dataset_name: str) -> list:
    """DETERMINISTIC selection of the pre-generated branching episode data:

    1) explicit --episode-dataset-path;
    2) the canonical artifact path for (dataset, speaker model);
    3) exactly one non-canonical candidate (used with a loud warning);
       two or more candidates are an error demanding --episode-dataset-path.

    Returns a list with at most one directory (empty -> no episode data).
    """
    episode_dirs = []
    if getattr(args, "episode_dataset_path", None):
        episode_dirs = [args.episode_dataset_path]
    else:
        canonical = artifact_dir("preference_data", mode="branching",
                                 dataset=dataset_name,
                                 model_name=args.speaker_model_name)
        if canonical.is_dir():
            episode_dirs = [str(canonical)]
        else:
            candidates = sorted(d for d in glob.glob(f"data/rl/branching_{dataset_name}_*")
                                if os.path.isdir(d))
            if len(candidates) >= 2:
                raise ValueError(
                    f"Canonical branching dataset {canonical} not found and "
                    f"{len(candidates)} non-canonical candidates exist:\n  "
                    + "\n  ".join(candidates)
                    + "\nPass --episode-dataset-path explicitly.")
            if len(candidates) == 1:
                logger.warning(
                    f"Canonical branching dataset {canonical} not found; using the "
                    f"single existing candidate {candidates[0]}. Pass "
                    "--episode-dataset-path to silence this warning.")
                episode_dirs = candidates
    return episode_dirs


# =====================================================================
# Offline DPO dataset builder
# =====================================================================

def _extract_directive_text(messages) -> str:
    """Pull the directive text from a chosen/rejected message list."""
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return (last.get("content") or "").strip()
    return ""


def build_director_dpo_dataset(
    episode_dataset_path: str,
    agents: dict,
    dataset_name: str,
    score_margin: float = 0.5,
    min_branches: int = 2,
) -> Dataset:
    """Build a {prompt, chosen, rejected} DPO dataset from a paired branching rollout.

    Inputs are the existing paired branching datasets under data/rl/branching_*.
    Each input row is already a chosen/rejected pair for one branching state:
      - chosen / rejected: list[{role, content}], where the assistant turn
        carries the *directive text* (e.g. "[agent_name]: action Tone: tone").
      - score_chosen / score_rejected: scalar episode-level scores.
      - episode_context / episode_turns / branch_turn_idx: shared prefix state.

    The director prompt at turn k is rebuilt via create_group_dataset_from_episodes,
    which handles JSON parsing, prefix reconstruction, and the dataset-specific
    directive prompt template (QMSum vs Empathetic/ESConv).

    Pairs with empty directives are dropped (counted as malformed).
    Pairs with score_chosen - score_rejected < score_margin are dropped.
    """
    prefix_ds = create_group_dataset_from_episodes(episode_dataset_path, agents, dataset_name)
    raw_ds = Dataset.load_from_disk(episode_dataset_path)
    if len(prefix_ds) != len(raw_ds):
        raise RuntimeError(
            f"Row count mismatch: prefix_ds={len(prefix_ds)} vs raw_ds={len(raw_ds)}. "
            "create_group_dataset_from_episodes is expected to preserve row order."
        )

    logger.info(f"Loaded {len(raw_ds)} raw branching rows")

    out = []
    states = {}
    filtered_by_margin = 0
    malformed = 0
    gaps = []

    for prefix_row, raw_row in zip(prefix_ds, raw_ds):
        prompt_text = prefix_row["prompt"]
        episode_id = prefix_row["episode_id"]
        k = int(prefix_row["train_k"])
        chosen_text = _extract_directive_text(raw_row.get("chosen"))
        rejected_text = _extract_directive_text(raw_row.get("rejected"))
        score_chosen = float(raw_row.get("score_chosen") or 0.0)
        score_rejected = float(raw_row.get("score_rejected") or 0.0)

        if not prompt_text or not chosen_text or not rejected_text:
            malformed += 1
            continue

        state_key = (episode_id, k)
        states[state_key] = states.get(state_key, 0) + 1

        gap = score_chosen - score_rejected
        if gap < score_margin:
            filtered_by_margin += 1
            continue

        out.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "score_chosen": score_chosen,
            "score_rejected": score_rejected,
            "episode_id": str(episode_id),
            "branch_turn_idx": k,
        })
        gaps.append(gap)

    n_states = len(states)
    states_below_min = sum(1 for v in states.values() if v < min_branches)
    logger.info(f"Unique branching states: {n_states} (below min_branches={min_branches}: {states_below_min})")
    logger.info(
        f"Kept {len(out)} DPO pairs after score_margin={score_margin} filter "
        f"(filtered by margin: {filtered_by_margin}; malformed: {malformed})"
    )
    if gaps:
        logger.info(
            f"Score gap stats: mean={sum(gaps) / len(gaps):.3f}, "
            f"min={min(gaps):.3f}, max={max(gaps):.3f}"
        )
        for i in range(min(2, len(out))):
            ex = out[i]
            logger.info(
                f"Sample pair {i}: prompt[:300]={ex['prompt'][:300]!r}; "
                f"chosen[:300]={ex['chosen'][:300]!r}; rejected[:300]={ex['rejected'][:300]!r}; "
                f"gap={ex['score_chosen'] - ex['score_rejected']:.3f}"
            )

    if not out:
        raise RuntimeError(
            "No DPO pairs survived filtering. Lower --score-margin or check the input dataset."
        )

    assert all(r["prompt"] and r["chosen"] and r["rejected"] for r in out), \
        "Empty prompt/chosen/rejected slipped through filtering"

    return Dataset.from_list(out)


def _resolve_dataset_name(args) -> str:
    """Resolve dataset_name from --dataset or by inferring from --episode-dataset-path."""
    explicit = getattr(args, "dataset", None)
    if explicit:
        return explicit
    path = getattr(args, "episode_dataset_path", "") or ""
    if ESCONV_DATASET in path:
        return ESCONV_DATASET
    if EMPATHETIC_DATASET in path:
        return EMPATHETIC_DATASET
    return QMSUM_DATASET
