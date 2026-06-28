#!/usr/bin/env python3
"""Group training script for directive generator using GRPO,
and group reward model training using RewardTrainer.

Usage:
    # Train directive generator (GRPO)
    python src/train/train_group.py --mode train-director \
        --meeting-data data/reward_model/rm_conversations_QMSum_Qwen3-8B.jsonl \
        --reward-model-path outputs/reward_model/group_reward_Qwen3-0.6B \
        --use-vllm

    # Train group reward model
    python src/train/train_group.py --mode train-reward-model \
        --data data/group_reward/annotated_run1.jsonl data/group_reward/annotated_run2.jsonl
"""

import os
import json
import random
import glob
import torch
import logging
import argparse
from dotenv import load_dotenv
from functools import partial
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainerCallback

from trl import GRPOTrainer, GRPOConfig, RewardTrainer, RewardConfig

from config.model_config import ModelConfig
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.utils.model_utils import create_lora_model, save_lora_adapter, load_model_auto
from src.models.llm_models import VLLMModel, AgentResponse
from src.generation.mascot_episode import parse_directive
from src.user.simulated_users import extract_user_json, generate_user_simulation_prompt_qmsum
from src.utils.data_utils import clean_response
from src.train.train_rl import RewardLoggingCallback
from src.prompts.prompt_reward_model import get_prompt_collective_reward_model
from src.prompts.persona_templates import MascotPersonaTemplates
from src.train.train_args import add_group_args, add_vllm_args, add_rl_args, add_dpo_args
from src.const import QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# Batch directive generation with retry
# =====================================================================

def _random_agent(agents: dict) -> dict:
    """Create a random fallback directive."""
    name = random.choice(list(agents.keys()))
    return {
        "agent_name": name,
        "action": "Continue the ongoing discussion",
        "tone": "neutral",
    }


def _generate_directives_batch(
    directive_generator,
    episode_states: list,
    agents: dict,
    indices: list,
    max_retries: int = 3,
    desc: str = "directives",
) -> list:
    """Batch-generate and parse directives with retry on parse failure.

    Args:
        directive_generator: Frozen LLM with generate_batch().
        episode_states: List of episode state dicts (need meeting_context,
            conversation_history, agents_spoken).
        agents: Agent dictionary.
        indices: Which episode_states indices to generate for.
        max_retries: Max retries for failed directives.
        desc: Description for progress bar.

    Returns:
        List of parsed directive dicts, one per index in ``indices``.
    """
    if not indices:
        return []

    prompts = []
    for i in indices:
        ep = episode_states[i]
        prompts.append(MascotPersonaTemplates.directive_prompt_qmsum(
            agents=agents,
            meeting_context=ep["meeting_context"],
            conversation_history=ep["conversation_history"],
            agents_spoken=ep["agents_spoken"],
        ))

    results = [None] * len(indices)
    pending = list(range(len(indices)))

    for attempt in range(1 + max_retries):
        if not pending:
            break

        batch_prompts = [prompts[j] for j in pending]
        label = desc if attempt == 0 else f"{desc} retry {attempt}"
        if attempt > 0:
            logger.debug(f"Retrying {len(pending)} failed directives (attempt {attempt})")
        raw_outputs = directive_generator.generate_batch(
            batch_prompts, max_tokens=1024, temperature=0.9, desc=label,
            system_prompt=None,
        )

        still_pending = []
        for jj, j in enumerate(pending):
            d = parse_directive(raw_outputs[jj], agents)
            if d is not None:
                results[j] = d
            else:
                still_pending.append(j)
        pending = still_pending

    # Last resort fallback after all retries exhausted
    if pending:
        logger.warning(
            f"{len(pending)} directives still unparseable after {max_retries} retries, "
            "using random agent fallback"
        )
        for j in pending:
            results[j] = _random_agent(agents)

    return results


# =====================================================================
# Prefix generation (turns 0..k-1)
# =====================================================================

def build_prefix_turns(
    directive_generator,
    agent_generator,
    meeting_contexts: list,
    k_values: list,
    agents: dict,
    user_simulator=None,
    user_response_probability: float = 0.3,
) -> list:
    """Pre-generate prefix turns 0..k-1 for each sample using frozen vLLM.

    Processes turns sequentially; within each turn, generates directives
    and agent responses in batch (only for samples that need that turn).

    Args:
        directive_generator: Frozen LLM for directive generation.
        agent_generator: Frozen LLM for agent responses.
        meeting_contexts: Meeting context string per sample.
        k_values: Which director turn to train per sample (0 = no prefix).
        agents: Agent dictionary.
        user_simulator: Optional user simulation LLM.
        user_response_probability: Probability of user turn per round.

    Returns:
        List of state dicts with keys: conversation_history, agents_spoken,
        prefix_turns, meeting_context.
    """
    
    states = []
    for i in range(len(meeting_contexts)):
        states.append({
            "meeting_context": meeting_contexts[i],
            "conversation_history": "",
            "agents_spoken": {name: 0 for name in agents},
            "prefix_turns": [],
        })

    max_k = max(k_values) if k_values else 0

    for turn_idx in range(max_k):
        # Only process samples that need this prefix turn
        need_idx = [i for i, k in enumerate(k_values) if k > turn_idx]
        if not need_idx:
            break

        logger.info(f"Prefix turn {turn_idx}: generating for {len(need_idx)} samples")

        # 1. Batch generate directives (frozen director)
        directives = _generate_directives_batch(
            directive_generator, states, agents, need_idx,
            desc=f"Prefix turn {turn_idx} directives",
        )

        # 2. Batch generate agent responses (frozen speaker)
        agent_prompts = []
        for j, i in enumerate(need_idx):
            agent_name = directives[j]["agent_name"]
            persona_spec = agents[agent_name]
            agent_prompts.append(MascotPersonaTemplates.speaker_prompt_qmsum(
                persona=persona_spec,
                context=states[i]["meeting_context"],
                meeting_summary=None,
                conversation_history=states[i]["conversation_history"],
            ))

        agent_responses = agent_generator.generate_batch(
            agent_prompts, max_tokens=512, temperature=0.9,
            system_prompt=None,
            desc=f"Prefix turn {turn_idx} responses",
        )

        # 3. Update states
        for j, i in enumerate(need_idx):
            agent_name = directives[j]["agent_name"]
            clean = clean_response(agent_responses[j])
            states[i]["conversation_history"] += f"[{agent_name}]: {clean}\n"
            states[i]["agents_spoken"][agent_name] += 1
            states[i]["prefix_turns"].append({
                "turn_idx": turn_idx,
                "directive": directives[j],
                "agent_name": agent_name,
                "response": agent_responses[j],
                "agent_response": agent_responses[j],
                "turn_type": "agent",
            })

        # 4. Optional user simulation
        if user_simulator is not None and random.random() < user_response_probability:
            user_prompts = [
                generate_user_simulation_prompt_qmsum(
                    states[i]["meeting_context"],
                    states[i]["conversation_history"],
                )
                for i in need_idx
            ]
            user_responses = user_simulator.generate_batch(
                user_prompts, max_tokens=256, temperature=0.7,
                system_prompt=None,
                desc=f"Prefix turn {turn_idx} user sim",
            )
            for j, i in enumerate(need_idx):
                user_text = user_responses[j]
                parsed = extract_user_json(user_text)
                if parsed is not None:
                    user_text = parsed.get("response", user_text)
                clean_user = clean_response(user_text)
                states[i]["conversation_history"] += f"[User]: {clean_user}\n"
                states[i]["prefix_turns"].append({
                    "turn_idx": turn_idx,
                    "directive": None,
                    "agent_name": "User",
                    "response": user_text,
                    "agent_response": user_text,
                    "turn_type": "user",
                })

    logger.info(
        f"Built prefixes for {len(meeting_contexts)} samples "
        f"(k range: {min(k_values)}-{max(k_values) if k_values else 0})"
    )
    return states


# =====================================================================
# Suffix rollout (turns k..k+num_suffix-1)
# =====================================================================

def rollout_suffix_turns(
    turn_k_directives: list,
    k_values: list,
    prefix_states: list,
    directive_generator,
    agent_generator,
    meeting_contexts: list,
    agents: dict,
    num_episode_turns: int = 5,
    user_simulator=None,
    user_response_probability: float = 0.3,
) -> list:
    """Roll out suffix turns starting from the training turn.

    Generates exactly ``num_episode_turns - k`` suffix rounds so that each
    episode has ``num_episode_turns`` total turns (prefix + suffix).
    Round 0 (suffix_idx=0) uses GRPO-generated directives (training turn);
    subsequent rounds use the frozen directive_generator.

    Args:
        turn_k_directives: Parsed directives from GRPO completions.
        k_values: Which turn each sample trained on.
        prefix_states: States from build_prefix_turns().
        directive_generator: Frozen LLM for suffix directive generation.
        agent_generator: Frozen LLM for agent responses.
        meeting_contexts: Meeting context per sample.
        agents: Agent dictionary.
        num_episode_turns: Total episode length (prefix + suffix).
        user_simulator: Optional user simulation LLM.
        user_response_probability: Probability of user turn per round.

    Returns:
        List of episode dicts with turns, conversation_history, and agents_spoken.
    """
    batch_size = len(turn_k_directives)

    # Initialize from prefix states
    episodes = []
    for i in range(batch_size):
        episodes.append({
            "meeting_context": meeting_contexts[i],
            "turns": list(prefix_states[i]["prefix_turns"]),
            "conversation_history": prefix_states[i]["conversation_history"],
            "agents_spoken": dict(prefix_states[i]["agents_spoken"]),
        })

    all_idx = list(range(batch_size))

    # Compute exact suffix turns needed: all k values should be the same
    # in a sorted, non-shuffled batch
    k = k_values[0]
    num_suffix = num_episode_turns - k
    if num_suffix <= 0:
        logger.warning(f"k={k} >= num_episode_turns={num_episode_turns}, no suffix turns to generate")
        return episodes

    for suffix_idx in range(num_suffix):
        actual_turn_indices = [k_values[i] + suffix_idx for i in range(batch_size)]

        # --- Determine directives ---
        if suffix_idx == 0:
            # Training turn: use GRPO-generated directives
            directives = list(turn_k_directives)
            # Retry any that failed to parse with frozen director
            retry_needed = [i for i in range(batch_size) if directives[i] is None]
            if retry_needed:
                logger.debug(
                    f"Retrying {len(retry_needed)} unparsed GRPO directives "
                    "with frozen director"
                )
                retry_results = _generate_directives_batch(
                    directive_generator, episodes, agents, retry_needed,
                    desc="GRPO directive retry",
                )
                for j, i in enumerate(retry_needed):
                    directives[i] = retry_results[j]
        else:
            # Frozen suffix turns: batch generate directives
            dir_results = _generate_directives_batch(
                directive_generator, episodes, agents, all_idx,
                desc=f"Suffix turn {suffix_idx} directives",
            )
            directives = dir_results

        # --- Batch generate agent responses ---
        agent_prompts = []
        for i in all_idx:
            agent_name = directives[i]["agent_name"]
            persona_spec = agents[agent_name]
            agent_prompts.append(MascotPersonaTemplates.speaker_prompt_qmsum(
                persona=persona_spec,
                context=episodes[i]["meeting_context"],
                meeting_summary=None,
                conversation_history=episodes[i]["conversation_history"],
            ))

        agent_responses = agent_generator.generate_batch(
            agent_prompts, max_tokens=4096, temperature=0.9,
            response_format=AgentResponse,
            system_prompt=None,
            desc=f"Suffix turn {suffix_idx} responses",
        )

        # --- Update episode states ---
        for i in all_idx:
            agent_name = directives[i]["agent_name"]
            raw = agent_responses[i]
            # Parse structured output; fall back to raw text
            try:
                parsed = json.loads(raw)
                response_text = parsed["response"]
            except (json.JSONDecodeError, KeyError, TypeError):
                response_text = raw
            clean = clean_response(response_text)
            episodes[i]["conversation_history"] += f"[{agent_name}]: {clean}\n"
            episodes[i]["agents_spoken"][agent_name] = \
                episodes[i]["agents_spoken"].get(agent_name, 0) + 1
            episodes[i]["turns"].append({
                "turn_idx": actual_turn_indices[i],
                "directive": directives[i],
                "agent_name": agent_name,
                "response": response_text,
                "agent_response": response_text,
                "turn_type": "agent",
            })

        # --- Optional user simulation ---
        if user_simulator is not None and random.random() < user_response_probability:
            user_prompts = [
                generate_user_simulation_prompt_qmsum(
                    episodes[i]["meeting_context"],
                    episodes[i]["conversation_history"],
                )
                for i in all_idx
            ]
            user_responses = user_simulator.generate_batch(
                user_prompts, max_tokens=256, temperature=0.7,
                system_prompt=None,
                desc=f"Suffix turn {suffix_idx} user sim",
            )
            for i in all_idx:
                user_text = user_responses[i]
                parsed_user = extract_user_json(user_text)
                if parsed_user is not None:
                    user_text = parsed_user.get("response", user_text)
                clean_user = clean_response(user_text)
                episodes[i]["conversation_history"] += f"[User]: {clean_user}\n"
                episodes[i]["turns"].append({
                    "turn_idx": actual_turn_indices[i],
                    "directive": None,
                    "agent_name": "User",
                    "response": user_text,
                    "agent_response": user_text,
                    "turn_type": "user",
                })

    logger.info(
        f"Rolled out {batch_size} episodes x {num_suffix} suffix turns "
        f"(k={k}, total={num_episode_turns} turns per episode)"
    )
    return episodes


# =====================================================================
# Dataset creation
# =====================================================================

def create_group_training_dataset(
    tokenizer,
    meeting_data_path: str,
    agents: dict = None,
    num_episode_turns: int = 5,
    directive_generator=None,
    agent_generator=None,
    user_simulator=None,
    user_response_probability: float = 0.3,
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
        prompt_text = MascotPersonaTemplates.directive_prompt_qmsum(
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


# =====================================================================
# Reward computation
# =====================================================================

def compute_group_reward(
    prompts,
    completions,
    meeting_context=None,
    train_k=None,
    prefix_conversation_history=None,
    prefix_agents_spoken=None,
    prefix_turns=None,
    directive_generator=None,
    agents_vllm=None,
    agents_dict=None,
    group_reward_model=None,
    group_reward_tokenizer=None,
    user_simulator=None,
    user_response_probability=0.3,
    num_episode_turns=5,
    **kwargs,
):
    """Compute group-level rewards for directive completions.

    Flow:
        1. Parse turn-k directives from GRPO ``completions``.
        2. Reconstruct prefix states from dataset columns (passed by TRL).
        3. Roll out suffix turns (training turn + frozen continuation).
        4. Truncate to first ``num_episode_turns`` rounds.
        5. Score with the group reward model.

    Extra dataset columns (train_k, prefix_conversation_history,
    prefix_agents_spoken, prefix_turns) are passed by TRL GRPOTrainer
    as keyword arguments.

    Returns:
        list[float]: Rewards, one per completion.
    """
    if agents_dict is None:
        agents_dict = AGENTS

    batch_size = len(completions)
    logger.debug(f"Computing group rewards for batch of {batch_size}")

    # --- Parse k values (TRL may pass as list[int] or tensor) ---
    if train_k is not None:
        k_list = [int(k) for k in train_k]
    else:
        k_list = [0] * batch_size

    # --- Reconstruct prefix states from stored dataset columns ---
    prefix_states_list = []
    for i in range(batch_size):
        conv_hist = (
            prefix_conversation_history[i]
            if prefix_conversation_history
            else ""
        )
        agents_spoken = (
            json.loads(prefix_agents_spoken[i])
            if prefix_agents_spoken
            else {name: 0 for name in agents_dict}
        )
        p_turns = (
            json.loads(prefix_turns[i])
            if prefix_turns
            else []
        )
        mc = meeting_context[i] if meeting_context else ""
        prefix_states_list.append({
            "conversation_history": conv_hist,
            "agents_spoken": agents_spoken,
            "prefix_turns": p_turns,
            "meeting_context": mc,
        })

    # --- Parse GRPO completions as turn-k directives ---
    turn_k_directives = []
    for comp in completions:
        if isinstance(comp, list) and len(comp) > 0:
            directive_text = comp[0].get('content', '')
        else:
            directive_text = str(comp)
        d = parse_directive(directive_text, agents_dict)
        turn_k_directives.append(d)  # may be None; rollout handles retry

    # --- Rollout suffix turns ---
    mc_list = (
        meeting_context if meeting_context else [""] * batch_size
    )
    try:
        episodes = rollout_suffix_turns(
            turn_k_directives=turn_k_directives,
            k_values=k_list,
            prefix_states=prefix_states_list,
            directive_generator=directive_generator,
            agent_generator=agents_vllm,
            meeting_contexts=mc_list,
            agents=agents_dict,
            num_episode_turns=num_episode_turns,
            user_simulator=user_simulator,
            user_response_probability=user_response_probability,
        )
    except Exception as e:
        logger.error(f"Suffix rollout failed: {e}", exc_info=True)
        return [0.0] * batch_size

    # --- Format and score ---
    episode_texts = [get_prompt_collective_reward_model(ep["turns"]) for ep in episodes]

    inputs = group_reward_tokenizer(
        episode_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(group_reward_model.device)

    with torch.no_grad():
        outputs = group_reward_model(**inputs)
        logits = outputs.logits
        if logits.size(-1) == 2:
            rewards = logits[:, 0]
        else:
            rewards = logits.squeeze(-1)

    reward_list = rewards.cpu().tolist()
    logger.debug(f"Group rewards: mean={sum(reward_list)/len(reward_list):.3f}")
    return reward_list


# =====================================================================
# Reward logging callback (mirrors train_rl.py)
# =====================================================================


# =====================================================================
# Director training (GRPO)
# =====================================================================

def train_group_grpo(config: ModelConfig, args):
    """Train directive generator using GRPO with random turn selection.

    The directive generator (small model, e.g. 0.6B) produces a directive at
    a randomly selected turn k. A frozen speaker model (e.g. 8B) executes
    agent turns and generates directives for all other turns.

    Turn structure per episode:
        prefix (0..k-1 frozen turns) + training turn k + suffix (frozen)
        Total = k + num_episode_turns turns.
        Only the first num_episode_turns rounds are used for reward scoring.
    """
    logger.info("=== Group Director Training with GRPO ===")

    # --- Tokenizer for the directive generator (policy) ---
    tokenizer = AutoTokenizer.from_pretrained(args.directive_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # --- Frozen speaker model (also used as directive generator for non-training turns) ---
    # Priority: external vLLM server > HuggingFace fallback (colocate) > local vLLM
    if args.rollout_server_url:
        # Use external vLLM server — single connection shared by all rollout agents
        from src.models.llm_models import VLLMServerModel
        logger.info(
            f"Initializing all frozen rollout agents via vLLM server at "
            f"{args.rollout_server_url} (model: {args.speaker_model_name})"
        )
        _server = VLLMServerModel(
            model_name=args.speaker_model_name,
            base_url=args.rollout_server_url,
        )
        agents_vllm = _server
    elif args.use_vllm:
        # When GRPOTrainer uses colocated vLLM (--use-vllm), we CANNOT run a second
        # vLLM instance in the same process (CUDA context conflict). Fall back to
        # HuggingFace-based Qwen model for the frozen speaker in that case.
        from src.models.llm_models import Qwen
        logger.info(
            f"Initializing frozen speaker model ({args.speaker_model_name}) "
            "via HuggingFace (vLLM colocate mode active, avoiding dual vLLM)"
        )
        agents_vllm = Qwen(model_name=args.speaker_model_name)
    else:
        rollout_gpu_util = args.rollout_gpu_memory_utilization
        logger.info(
            f"Initializing frozen speaker model ({args.speaker_model_name}) "
            f"via vLLM (gpu_memory_utilization={rollout_gpu_util})"
        )
        agents_vllm = VLLMModel(
            model_name=args.speaker_model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=rollout_gpu_util,
        )

    # The frozen speaker model doubles as the directive generator for non-training turns
    directive_generator = agents_vllm

    # Determine training GPU: VLLMModel takes cuda:0, training models go on last GPU
    num_gpus = torch.cuda.device_count()
    train_device = num_gpus - 1 if num_gpus > 1 else 0
    logger.info(f"GPU layout: policy/vLLM on cuda:0, reward model on cuda:{train_device} ({num_gpus} GPUs visible)")

    # --- User simulator (optional) ---
    user_simulator = None
    if args.user_simulator_model:
        if args.rollout_server_url:
            # All rollout agents share the same vLLM server
            user_simulator = agents_vllm
            logger.info("User simulator shares vLLM server instance")
        elif args.user_simulator_model == args.speaker_model_name:
            user_simulator = agents_vllm
            logger.info("User simulator shares speaker model instance")
        else:
            logger.info(f"Loading separate user simulator model: {args.user_simulator_model}")
            if args.use_vllm:
                user_simulator = Qwen(model_name=args.user_simulator_model)
            else:
                user_simulator = VLLMModel(
                    model_name=args.user_simulator_model,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.5,
                )

    # --- Dataset ---
    # Priority: branching episode dataset > auto-cache > generate from scratch
    
    _base = os.path.basename(args.meeting_data or "")
    if ESCONV_DATASET in _base:
        dataset_name = ESCONV_DATASET
    elif EMPATHETIC_DATASET in _base:
        dataset_name = EMPATHETIC_DATASET
    else:
        dataset_name = QMSUM_DATASET
    agents_dict = EMPATHETIC_AGENTS if dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS

    # Check for pre-generated branching episode data
    episode_dirs = [d for d in glob.glob(f"data/rl/branching_{dataset_name}_*") if os.path.isdir(d)]

    meeting_data_short = os.path.splitext(os.path.basename(args.meeting_data))[0]
    speaker_short = os.path.basename(args.speaker_model_name)
    cache_path = f"data/group_prefix/{meeting_data_short}_{speaker_short}_{args.num_episode_turns}turns"

    if episode_dirs:
        logger.info(f"Using branching episode dataset: {episode_dirs[0]}")
        dataset = create_group_dataset_from_episodes(episode_dirs[0], agents_dict, dataset_name)
    elif os.path.exists(cache_path):
        logger.info(f"Loading cached prefix dataset from {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
        k_dist = {}
        for k in dataset["train_k"]:
            k_dist[k] = k_dist.get(k, 0) + 1
        logger.info(f"Loaded {len(dataset)} cached samples (k distribution: {dict(sorted(k_dist.items()))})")
    else:
        dataset = create_group_training_dataset(
            tokenizer=tokenizer,
            meeting_data_path=args.meeting_data,
            agents=agents_dict,
            num_episode_turns=args.num_episode_turns,
            directive_generator=directive_generator,
            agent_generator=agents_vllm,
            user_simulator=user_simulator,
            user_response_probability=args.user_response_probability,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dataset.save_to_disk(cache_path)
        logger.info(f"Saved prefix dataset cache to {cache_path}")

    # Group by train_k, form complete batches, shuffle, append remainders
    batch_size = args.batch_size
    k_unique = sorted(set(dataset["train_k"]))
    complete_chunks = []  # list of Dataset slices, each of size batch_size
    remainders = []       # leftover indices

    for k_val in k_unique:
        indices = [i for i, k in enumerate(dataset["train_k"]) if k == k_val]
        n_complete = (len(indices) // batch_size) * batch_size
        # Split into complete batches
        for start in range(0, n_complete, batch_size):
            complete_chunks.append(dataset.select(indices[start:start + batch_size]))
        # Collect remainders
        if n_complete < len(indices):
            remainders.extend(indices[n_complete:])

    random.shuffle(complete_chunks)

    parts = complete_chunks
    if remainders:
        parts.append(dataset.select(remainders))

    dataset = concatenate_datasets(parts)

    k_dist = {}
    for k in dataset["train_k"]:
        k_dist[k] = k_dist.get(k, 0) + 1
    logger.info(
        f"Grouped {len(dataset)} samples by train_k into batch-aligned chunks "
        f"(batch_size={batch_size}, k distribution: {dict(sorted(k_dist.items()))}, "
        f"{len(remainders)} remainder samples appended at end)"
    )

    # --- Group reward model ---
    reward_model_path = args.reward_model_path
    logger.info(f"Loading group reward model from {reward_model_path}")

    group_reward_model, group_reward_tokenizer = load_model_auto(
        reward_model_path,
        task_type="SEQ_CLS",
        num_labels=1,
        dtype=torch.bfloat16,
        device_map={"": train_device},
    )
    group_reward_model.eval()
    group_reward_model.config.pad_token_id = group_reward_tokenizer.pad_token_id

    # --- Directive generator (policy, LoRA) ---
    logger.info(f"Creating directive generator model from {args.directive_model_name}...")
    directive_config = ModelConfig(
        base_model=args.directive_model_name,
        use_lora=True,
        lora_r=getattr(args, 'lora_r', 16),
        lora_alpha=getattr(args, 'lora_alpha', 32),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
    )
    directive_model = create_lora_model(
        base_model_name=directive_config.base_model,
        task_type="CAUSAL_LM",
        config=directive_config,
        dtype=torch.bfloat16,
        device_map=None,  # GRPOTrainer colocate vLLM handles device placement
    )
    directive_model.config.pad_token_id = tokenizer.pad_token_id

    # --- Output directory ---
    model_short = os.path.basename(args.directive_model_name)
    output_dir = getattr(args, 'output_dir', None) or f"outputs/llm/grpo_Director_{model_short}"
    logger.info(f"Output directory: {output_dir}")

    # --- GRPO config ---
    grpo_config_kwargs = {}
    if hasattr(args, 'max_steps') and args.max_steps > 0:
        grpo_config_kwargs["max_steps"] = args.max_steps

    vllm_kwargs = {
        "vllm_mode": "colocate",
        "vllm_tensor_parallel_size": 1,
        "vllm_gpu_memory_utilization": 0.3,
        "vllm_enable_sleep_mode": True,
    }

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=getattr(args, 'learning_rate', 1e-5),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=getattr(args, 'epochs', 1),
        beta=0.1,
        max_completion_length=1536, 
        temperature=0.9,
        use_vllm=True,
        **vllm_kwargs,
        num_generations=args.num_generations,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        shuffle_dataset=False,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        **grpo_config_kwargs,
    )

    # --- Reward function ---
    reward_fn = partial(
        compute_group_reward,
        directive_generator=directive_generator,
        agents_vllm=agents_vllm,
        agents_dict=agents_dict,
        group_reward_model=group_reward_model,
        group_reward_tokenizer=group_reward_tokenizer,
        user_simulator=user_simulator,
        user_response_probability=args.user_response_probability,
        num_episode_turns=args.num_episode_turns,
    )
    reward_fn.__name__ = "group_reward"

    reward_names = ["group_reward"]
    reward_weights = [1.0]

    # --- Patch TRL 0.28.0 sync_weights for vLLM sleep mode (PR #5147, fixed in TRL 0.29.0) ---
    # vLLM sleep(level=2) frees GPU weight memory. TRL 0.28.0's sync_weights writes into
    # the freed tensors causing "CUDA error: invalid argument". Wake up before sync.
    import trl.generation.vllm_generation as _vg
    _original_sync = _vg.VLLMGeneration.sync_weights

    def _patched_sync_weights(self, *args, **kwargs):
        if self.mode == "colocate" and hasattr(self, "llm") and hasattr(self.llm, "wake_up"):
            torch.cuda.empty_cache()                # Free PyTorch cached memory from backward pass
            self.llm.wake_up(tags=["weights"])       # Wake only weights, not KV cache
        return _original_sync(self, *args, **kwargs)

    _vg.VLLMGeneration.sync_weights = _patched_sync_weights

    # --- Trainer ---
    logger.info("Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=directive_model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            RewardLoggingCallback(reward_names, reward_weights, every_n_steps=50),
        ],
    )

    # --- Train ---
    logger.info("Starting training...")
    trainer.train()

    # --- Save ---
    logger.info(f"Saving model to {output_dir}")
    save_lora_adapter(directive_model, output_dir, tokenizer)
    logger.info("Training completed!")


# =====================================================================
# Group reward model training (absorbed from src/group/train_group.py)
# =====================================================================

def load_and_convert_to_preference_pairs(data_paths: list, score_margin: float = 0.5) -> Dataset:
    """Convert episodes to preference pairs for RewardTrainer.

    Args:
        data_paths: Exactly 2 paths to episode JSONL files.
        score_margin: Minimum score difference to keep a pair.

    Returns:
        HF Dataset with columns: chosen, rejected, score_chosen, score_rejected.
    """
    from tqdm import trange

    if len(data_paths) != 2:
        raise ValueError(f"Expected exactly 2 data files, got {len(data_paths)}")

    episodes_list = []
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        logger.info(f"Loading from {data_path}")
        episodes = []
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    episode = json.loads(line)
                    if "score" not in episode:
                        logger.warning(f"{data_path}:{line_num}: Missing 'score' field")
                        continue
                    episodes.append(episode)
                except json.JSONDecodeError as e:
                    logger.error(f"{data_path}:{line_num}: Invalid JSON - {e}")
        episodes_list.append(episodes)

    episodes_run1, episodes_run2 = episodes_list

    if len(episodes_run1) != len(episodes_run2):
        raise ValueError(
            f"Data files have different lengths: {len(episodes_run1)} vs {len(episodes_run2)}"
        )

    logger.info(f"Loaded {len(episodes_run1)} episodes from each file")

    preference_pairs = []
    for i in trange(len(episodes_run1), desc="Building preference pairs"):
        episode1 = episodes_run1[i]
        episode2 = episodes_run2[i]

        assert episode1['id'] == episode2['id']

        score1 = episode1["score"]
        score2 = episode2["score"]

        if abs(score1 - score2) < score_margin:
            continue

        if score1 > score2:
            chosen_episode, rejected_episode = episode1, episode2
        else:
            chosen_episode, rejected_episode = episode2, episode1

        chosen_text = get_prompt_collective_reward_model(chosen_episode["turns"])
        rejected_text = get_prompt_collective_reward_model(rejected_episode["turns"])
        prompt = (
            "Discuss the following human meeting from multiple perspectives.\n\n"
            f"## Meeting Context\n{episode1['meeting']['meeting_context']}"
        )

        chosen = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_text},
        ]
        rejected = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_text},
        ]

        preference_pairs.append({
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": float(chosen_episode["score"]),
            "score_rejected": float(rejected_episode["score"]),
        })

    logger.info(f"Created {len(preference_pairs)} preference pairs")
    return Dataset.from_list(preference_pairs)


def train_group_reward_model(args):
    """Train group reward model using TRL RewardTrainer.

    Absorbed from the former ``src/group/train_group.py``.

    Supports two data input modes:
        1. ``--data-path``: HF Dataset directory (from ``generate_and_annotate.py --mode group``).
           Already contains preference pairs with columns: chosen, rejected, score_chosen, score_rejected.
        2. ``--data``: Two raw episode JSONL files (legacy 2-JSONL format).
           Converted to preference pairs via ``load_and_convert_to_preference_pairs()``.
    """
    
    model_name = getattr(args, 'model_name', 'Qwen/Qwen3-0.6B')
    data_path = getattr(args, 'data_path', None)
    data_paths = getattr(args, 'data', None)
    batch_size = getattr(args, 'batch_size', 16)
    epochs = getattr(args, 'epochs', 3)
    lr = getattr(args, 'learning_rate', 1e-5)
    score_margin = getattr(args, 'score_margin', 0.5)

    output_model_name = f"outputs/reward_model/group_reward_{os.path.basename(model_name)}"
    os.makedirs(output_model_name, exist_ok=True)

    logger.info(f"=== Group Reward Model Training ===")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Output: {output_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if data_path:
        # New path: load pre-formed preference pairs from generate_and_annotate.py
        dataset = Dataset.load_from_disk(data_path)
        logger.info(f"Loaded HF Dataset from {data_path} ({len(dataset)} rows)")
    elif data_paths:
        # Legacy path: convert 2 JSONL files into preference pairs
        dataset = load_and_convert_to_preference_pairs(data_paths, score_margin)
    else:
        raise ValueError(
            "train-reward-model mode requires --data-path (HF Dataset from "
            "generate_and_annotate.py --mode group) or --data (two episode JSONL files)")

    reward_config = RewardConfig(
        output_dir=output_model_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        max_length=2048,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    )

    logger.info("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model_name,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {output_model_name}")
    trainer.save_model(output_model_name)
    tokenizer.save_pretrained(output_model_name)
    logger.info("Training completed!")


# =====================================================================
# Offline DPO training entry
# =====================================================================

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


def train_group_dpo(args):
    from trl import DPOTrainer, DPOConfig

    if not args.episode_dataset_path:
        raise ValueError("--episode-dataset-path is required for train-director-dpo mode")

    dataset_name = _resolve_dataset_name(args)
    agents_dict = EMPATHETIC_AGENTS if dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    logger.info(f"Offline DPO: dataset_name={dataset_name}, agents={list(agents_dict.keys())}")

    # Build or load the DPO dataset
    if args.dpo_data_path and os.path.exists(args.dpo_data_path):
        logger.info(f"Loading cached DPO dataset from {args.dpo_data_path}")
        dpo_dataset = Dataset.load_from_disk(args.dpo_data_path)
    else:
        dpo_dataset = build_director_dpo_dataset(
            episode_dataset_path=args.episode_dataset_path,
            agents=agents_dict,
            dataset_name=dataset_name,
            score_margin=args.score_margin,
            min_branches=args.min_branches,
        )
        if args.dpo_data_path:
            os.makedirs(os.path.dirname(args.dpo_data_path) or ".", exist_ok=True)
            dpo_dataset.save_to_disk(args.dpo_data_path)
            logger.info(f"Cached processed DPO dataset to {args.dpo_data_path}")

    if args.max_pairs is not None and len(dpo_dataset) > args.max_pairs:
        logger.info(f"Truncating DPO dataset to {args.max_pairs} pairs (--max-pairs)")
        dpo_dataset = dpo_dataset.select(range(args.max_pairs))

    # Train/eval split
    eval_dataset = None
    if not args.no_eval and len(dpo_dataset) >= 10:
        split = dpo_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        logger.info(f"Split: train={len(train_dataset)}, eval={len(eval_dataset)}")
    else:
        train_dataset = dpo_dataset
        logger.info(f"No eval split (size={len(dpo_dataset)}, no_eval={args.no_eval})")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.directive_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # LoRA policy model
    directive_model = create_lora_model(
        base_model_name=args.directive_model_name,
        task_type="CAUSAL_LM",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device_map="auto",
    )

    # Output dir
    model_short = os.path.basename(args.directive_model_name)
    output_dir = args.output_dir or f"outputs/llm/dpo_Director_{dataset_name}_{model_short}"

    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        beta=args.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=directive_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    save_lora_adapter(directive_model, output_dir, tokenizer)
    logger.info(f"Saved LoRA adapter + tokenizer to {output_dir}")


# =====================================================================
# Main entry point
# =====================================================================

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Group training: directive generator (GRPO/DPO) or group reward model")
    add_group_args(parser)
    add_vllm_args(parser)
    add_dpo_args(parser)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=[QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET],
                        help="Canonical dataset name (auto-inferred from --episode-dataset-path if omitted).")

    # Shared training args
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    # RL-specific args (for train-director mode)
    parser.add_argument("--reward-model-path", type=str,
                        default="outputs/reward_model/group_reward_Qwen3-0.6B",
                        help="Path to trained group reward model")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 means use epochs instead)")

    # Reward model-specific args (for train-reward-model mode)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to HF Dataset directory with preference pairs "
                             "(from generate_and_annotate.py --mode group)")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Paths to episode JSONL files (legacy, for reward model training)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B",
                        help="Base model for reward model training")
    parser.add_argument("--score-margin", type=float, default=0.5,
                        help="Minimum score difference for preference pairs")
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.3,
                        help="GPU memory utilization for the frozen rollout vLLM engine")

    args = parser.parse_args()

    # Initialize W&B if configured
    if os.getenv("WANDB_PROJECT"):
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"group-{args.mode}",
        )
        logger.info("W&B tracking enabled")

    if args.mode == "train-director":
        config = ModelConfig(
            base_model=args.directive_model_name,
            use_lora=True,
        )
        train_group_grpo(config, args)

    elif args.mode == "train-reward-model":
        train_group_reward_model(args)

    elif args.mode == "train-director-dpo":
        train_group_dpo(args)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
