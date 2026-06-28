#!/usr/bin/env python3
"""Unified data pipeline: generate, annotate, and format reward-model training data.

Supports both individual-level and group-level data generation for QMSum and
Empathetic datasets.  Outputs an HF Dataset compatible with
``datasets.load_from_disk`` (used by ``create_or_load_rl_dataset``).

Usage examples:
    # Individual-level QMSum (small test)
    python src/dataset/generate_and_annotate.py \
        --mode individual --dataset QMSum --max-samples 5

    # Group-level Empathetic with user simulation and LLM verification
    python src/dataset/generate_and_annotate.py \
        --mode group --dataset Empathetic --verify-pairs
"""

import json
import logging
import os
import random
import re
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.models.llm_models import (GPT, VLLMServerModel)
from src.utils.eval_utils import project_setup
from config.model_config import ModelConfig
from src.prompts.persona_templates import MascotPersonaTemplates
from src.prompts.persona_templates import get_summary_prompt
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.utils.data_utils import (
    load_empathetic_situations,
    load_esconv_situations,
)
from src.eval.rubric_evaluator import IndividualRubricEvaluator, GroupRubricEvaluator
from src.generation.mascot_episode import run_mascot_episode_loop, EpisodeConfig, parse_directive
from src.user.simulated_users import (
    extract_user_json,
    generate_user_simulation_prompt,
    generate_user_simulation_prompt_qmsum,
)
from src.utils.transcript_utils import (
    create_meeting_contexts,
    format_conversation_history,
    format_conversation_history_empathetic,
)


from src.const import (QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET,
                   DATASET2HF, INDIVIDUAL, PAIRED)

logger = logging.getLogger(__name__)


# Alias for backward compatibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_episode_context(episode: dict, dataset_name: str) -> str:
    """Extract context from episode state as JSON string, handling both datasets."""
    if dataset_name == QMSUM_DATASET:
        return json.dumps({"meeting_context": episode.get("meeting", {}).get("meeting_context", "")})
    else:  # Empathetic
        return json.dumps({
            "initial_situation": episode.get("initial_situation", ""),
            "initial_emotion": episode.get("initial_emotion", ""),
        })


def _add_grpo_columns(pair: dict, episode: dict, dataset_name: str) -> dict:
    """Add GRPO training columns to a preference pair in-place."""
    pair["episode_context"] = _extract_episode_context(episode, dataset_name)
    pair["episode_turns"] = json.dumps(episode.get("turns", []), default=str)
    pair["episode_agents_spoken"] = json.dumps(episode.get("agents_spoken", {}))
    pair["dataset_name"] = dataset_name
    return pair

def _scale_scores(records: List[dict]) -> List[dict]:
    """Min-max scale score_chosen / score_rejected to [1, 5] across *records*."""
    all_scores = [r['score_chosen'] for r in records] + [r['score_rejected'] for r in records]
    lo, hi = min(all_scores), max(all_scores)
    rng = hi - lo
    if rng == 0:
        logger.warning("All scores identical — setting all scaled scores to 3.0")
        for r in records:
            r['score_chosen'] = 3.0
            r['score_rejected'] = 3.0
        return records
    for r in records:
        r['score_chosen'] = 1.0 + 4.0 * (r['score_chosen'] - lo) / rng
        r['score_rejected'] = 1.0 + 4.0 * (r['score_rejected'] - lo) / rng
    logger.info(f"Scaled scores: raw range [{lo:.3f}, {hi:.3f}] -> [1, 5]")
    return records


def _make_preference_pair(prompt_text: str, chosen_resp: str, rejected_resp: str,
                          score_chosen: float, score_rejected: float,
                          persona: str, id_key: str, id_val: str) -> dict:
    return {
        "chosen": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_resp},
        ],
        "rejected": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rejected_resp},
        ],
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
        "persona": persona,
        id_key: id_val,
    }


def _clean(text: str) -> str:
    """Strip <think> tags."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _strip_response_prefix(text: str) -> str:
    """Remove echoed prompt prefixes like 'Your response:' from LLM output."""
    if not text:
        return ""
    return re.sub(r'^(?:Your\s+)?(?:JSON\s+)?[Rr]esponse\s*:\s*', '', text.strip()).strip()


def _build_paired_prompt_empathetic(persona, situation: str, rubric_text: str) -> str:
    """Build a rubric-aware prompt that asks for one good + one somewhat-worse response.

    The worse response should target rubric score ≈3 (generic, predictable but on-topic,
    coherent, persona-consistent), NOT a trivially-bad score-1/2 response.
    """
    traits_str = "\n".join(f"- {t}" for t in persona.traits)
    return f"""You are {persona.name}, an emotionally supportive agent.

## Persona Bio
{persona.description_long or persona.name}

## Key traits you must embody
{traits_str}

## Evaluation Rubric (this will be used to score both of your responses)
{rubric_text}

## Your Task: Generate ONE PAIR of responses for the situation above
You must produce exactly TWO responses to the same user situation:

1. "good_response" — TARGET RUBRIC SCORE ≈ 4
   - Vivid persona voice with character-specific vocabulary
   - Names a SPECIFIC unfelt emotion or offers non-obvious insight / reframe
   - Goes beyond surface relevance to deeper unstated implications
   - Introduces a genuinely NEW angle the user hasn't considered

2. "worse_response" — TARGET RUBRIC SCORE ≈ 3 (NOT lower)
   - MUST remain on-topic, coherent, and persona-consistent.
   - MUST be non-trivially bad.
   - MUST NOT be tone-deaf, dismissive, off-topic, or contradictory
   - Uses STANDARD coaching/therapy language (e.g., "What do you think...", "Tell me more...", "I'm here with you...")
   - Functional but generic — recognizable persona but predictable, lacks specific insight
   - The difference vs. good_response should be DEPTH and SPECIFICITY, not character or coherence

## Output Format (STRICT — JSON only, no prose, no markdown fences)
{{
    "good_response": "...",
    "worse_response": "..."
}}

Both responses must be in English, 1-3 sentences, and address the SAME user situation.

## Respond to the following user situation
{situation}

"""


def _parse_paired_response(raw_text: str) -> Optional[Tuple[str, str]]:
    """Extract (good_response, worse_response) from LLM output. Returns None on failure."""
    cleaned = _clean(raw_text)
    if not cleaned:
        return None
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if not m:
        logger.warning(f"Paired parse: no JSON object found in output (first 200 chars): {cleaned[:200]!r}")
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"Paired parse: JSON decode failed ({e}); snippet: {m.group(0)[:200]!r}")
        return None
    good = (parsed.get("good_response") or "").strip()
    worse = (parsed.get("worse_response") or "").strip()
    if not good or not worse:
        logger.warning(f"Paired parse: missing good/worse fields in {list(parsed.keys())}")
        return None
    return good, worse


# ---------------------------------------------------------------------------
# LLM Preference Pair Verification
# ---------------------------------------------------------------------------

def verify_pairs_with_llm(
    pairs: List[dict],
    llm: VLLMServerModel,
) -> List[dict]:
    """Use LLM to verify preference pairs; discard disagreements.

    For each pair, ask the LLM which response is better.  If the LLM says
    the rejected response is better, drop the pair.
    """
    if not pairs:
        return pairs

    prompts = []
    for p in pairs:
        chosen_text = p["chosen"][-1]["content"]
        rejected_text = p["rejected"][-1]["content"]
        context = p["chosen"][0]["content"]
        prompt = (
            f"Given the following context, which response is better?\n\n"
            f"Context:\n{context[:2048]}\n\n"
            f"Response A:\n{_clean(chosen_text)[:1024]}\n\n"
            f"Response B:\n{_clean(rejected_text)[:1024]}\n\n"
            f"Answer with just \"A\" or \"B\"."
        )
        prompts.append(prompt)

    logger.info(f"Verifying {len(prompts)} preference pairs with LLM...")
    verdicts = [(t or "").strip() for t in llm.generate_batch(
        prompts=prompts, temperature=0.0, max_tokens=8,
        system_prompt="You are a helpful judge. Answer with just A or B.",
        desc=f"Pair verification ({len(prompts)})",
    )]

    kept = []
    for pair, verdict in zip(pairs, verdicts):
        v = verdict.strip().upper()
        if "A" in v and "B" not in v:
            kept.append(pair)
        # else: LLM disagrees or ambiguous → drop
    agreement_rate = len(kept) / len(pairs) if pairs else 0
    logger.info(f"LLM verification: kept {len(kept)}/{len(pairs)} pairs "
                f"(agreement rate {agreement_rate:.1%})")
    return kept


# ---------------------------------------------------------------------------
# QMSum user simulation prompt for group-level generation
# ---------------------------------------------------------------------------

# =====================================================================
# INDIVIDUAL PIPELINE
# =====================================================================

def individual_conversation_pipeline(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate individual responses per query (N independent OR one good+worse pair), score, form preference pairs."""
    if args.mode == PAIRED and args.dataset == QMSUM_DATASET:
        raise ValueError("--mode paired only supports Empathetic / ESConv (rubric is empathy-centric)")

    agents = EMPATHETIC_AGENTS if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    config = ModelConfig()
    config.base_model = args.model_name
    N = args.num_agents_per_query  # responses per query
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"

    # ---- Phase 1: prepare queries ----
    queries: List[dict] = []  # each: {id, prompt_text, persona, ...}

    if args.dataset == QMSUM_DATASET:
        dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        # Pre-generate summaries
        meetings = []
        for item in dataset:
            transcript = [{"speaker": u["speaker"], "text": u["content"]}
                          for u in item.get("meeting_transcripts", [])]
            meetings.append({"meeting_id": item["id"], "transcript": transcript})

        logger.info(f"Generating summaries for {len(meetings)} meetings...")
        summary_prompts = [
            get_summary_prompt("".join(f"{t['speaker']}: {t['text']}\n"
                                       for t in m["transcript"][:400]))
            for m in meetings
        ]
        summaries = [_clean((t or "").strip()) for t in llm.generate_batch(
            prompts=summary_prompts, temperature=0.7,
            max_tokens=config.max_new_tokens,
            desc=f"Summaries ({len(summary_prompts)})",
        )]

        # Build per-agent queries (one per meeting × agent)
        TURNS_PER_ROUND = args.turns_per_round
        for mi, m in enumerate(meetings):
            chunks = []
            for i in range(0, len(m["transcript"]), TURNS_PER_ROUND):
                chunks.append(m["transcript"][i:i + TURNS_PER_ROUND])
            accumulated = ""
            for ci, chunk in enumerate(chunks):
                chunk_text = "".join(f"{t['speaker']}: {t['text']}\n" for t in chunk)
                ctx = (accumulated + "\n" + chunk_text).strip() if accumulated else chunk_text
                for aname, pspec in agents.items():
                    prompt_text = MascotPersonaTemplates.speaker_prompt(
                        QMSUM_DATASET, pspec, ctx, summaries[mi])
                    queries.append({
                        "id": m["meeting_id"],
                        "prompt_text": prompt_text,
                        "persona": aname,
                        "context": ctx,
                    })
                accumulated = ctx

    elif args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET):
        if args.dataset == ESCONV_DATASET:
            conv_list = load_esconv_situations(split="train", max_samples=args.max_samples)
        else:
            conv_list = load_empathetic_situations(split="train", max_samples=args.max_samples)
        for c in conv_list:
            c["situation"] = c["situation"].replace("__comma__", ", ")

        for conv in conv_list:
            for aname, pspec in agents.items():
                if args.mode == PAIRED:
                    prompt_text = _build_paired_prompt_empathetic(
                        persona=pspec, situation=conv["situation"],
                        rubric_text=IndividualRubricEvaluator.RUBRIC,
                    )
                else:
                    prompt_text = MascotPersonaTemplates.speaker_prompt_empathetic(
                        persona=pspec, context=conv["situation"],
                        conversation_history="", num_shot=3,
                    )
                queries.append({
                    "id": conv["conv_id"],
                    "prompt_text": prompt_text,
                    "persona": aname,
                    "context": conv["situation"],
                })

    logger.info(f"Total queries: {len(queries)}")

    # ---- Phase 2: Generate responses per query (with replacement) ----
    gen_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(gen_dir, exist_ok=True)
    model_tag = args.model_name.split("/")[-1]

    file_prefix = PAIRED if args.mode == PAIRED else INDIVIDUAL
    suffix = ""
    if args.max_samples:
        suffix += f"_max{args.max_samples}"
    gen_file = os.path.join(gen_dir, f"{file_prefix}_{args.dataset}_{model_tag}{suffix}.jsonl")

    if os.path.exists(gen_file):
        logger.info(f"Found existing generations at {gen_file}, skipping generation")
        queries = [json.loads(line) for line in open(gen_file)]
    elif args.mode == PAIRED:
        K = args.num_pairs_per_query
        all_prompts = []
        prompt_idx_map = []  # (query_idx, pair_idx)
        for qi, q in enumerate(queries):
            for pi in range(K):
                all_prompts.append(q["prompt_text"])
                prompt_idx_map.append((qi, pi))

        logger.info(f"Paired generation: {len(all_prompts)} pair calls ({len(queries)} queries × {K})...")
        raw_outputs = [(t or "").strip() for t in llm.generate_batch(
            prompts=all_prompts, temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            desc=f"Paired candidates ({len(all_prompts)})",
        )]

        # Parse pairs; drop any query whose ANY pair fails to parse so q["responses"]
        # length stays consistent (2*K) for downstream phases.
        per_query_pairs: Dict[int, List[Optional[Tuple[str, str]]]] = {
            qi: [None] * K for qi in range(len(queries))
        }
        for idx, (qi, pi) in enumerate(prompt_idx_map):
            per_query_pairs[qi][pi] = _parse_paired_response(raw_outputs[idx])

        kept_queries = []
        n_dropped = 0
        for qi, q in enumerate(queries):
            pairs = per_query_pairs[qi]
            if any(p is None for p in pairs):
                n_dropped += 1
                continue
            # Flatten [good_0, worse_0, good_1, worse_1, ...]
            flat = []
            for good, worse in pairs:
                flat.extend([good, worse])
            q["responses"] = flat
            kept_queries.append(q)
        logger.info(f"Paired parse: kept {len(kept_queries)}/{len(queries)} queries "
                    f"(dropped {n_dropped} with parse failures)")
        queries = kept_queries

        with open(gen_file, "w") as f:
            for q in queries:
                f.write(json.dumps(q, default=str) + "\n")
        logger.info(f"Saved raw paired generations to {gen_file}")
    else:
        all_prompts = []
        prompt_idx_map = []  # (query_idx, response_idx)
        for qi, q in enumerate(queries):
            for ri in range(N):
                all_prompts.append(q["prompt_text"])
                prompt_idx_map.append((qi, ri))

        logger.info(f"Generating {len(all_prompts)} candidate responses...")
        all_responses = [_strip_response_prefix((t or "").strip()) for t in llm.generate_batch(
            prompts=all_prompts, temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            desc=f"Individual candidates ({len(all_prompts)})",
        )]

        # Attach responses to queries
        for q in queries:
            q["responses"] = [None] * N
        for idx, (qi, ri) in enumerate(prompt_idx_map):
            queries[qi]["responses"][ri] = all_responses[idx]

        # Save raw generation
        with open(gen_file, "w") as f:
            for q in queries:
                f.write(json.dumps(q, default=str) + "\n")
        logger.info(f"Saved raw generations to {gen_file}")

    # ---- Phase 3: Annotate (parallel) ----
    ds = EMPATHETIC_DATASET if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else QMSUM_DATASET
    evaluator = IndividualRubricEvaluator(judge_llm, dataset_name=ds)

    def _annotate_query(qi_q):
        qi, q = qi_q
        scores = []
        for resp in q["responses"]:
            record = {
                "conversation_context": q["context"],
                "agent_response": resp or "",
                "agent_name": q["persona"],
                "id": q["id"],
                "start_idx": None,
                "turn_type": "agent",
            }
            try:
                results = evaluator.evaluate_batch([record], agents, ds)
                scores.append(results[0].total_reward if results else None)
            except Exception as e:
                logger.warning(f"Annotation failed for query {qi}: {e}")
                scores.append(None)
        return qi, scores

    # Resume support: load prior annotations keyed by (id, persona), keep only rows
    # whose responses still match the current queries, rewrite the file with the
    # valid rows, then append each newly annotated query as it completes.
    ann_dir = os.path.join(args.output_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    ann_file = os.path.join(ann_dir, f"{file_prefix}_{args.dataset}_{model_tag}{suffix}.jsonl")

    expected = {(q["id"], q["persona"]): qi for qi, q in enumerate(queries)}
    done: Dict[Tuple[str, str], List] = {}
    if os.path.exists(ann_file):
        with open(ann_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail line from a crash
                key = (row.get("id"), row.get("persona"))
                qi = expected.get(key)
                scores = row.get("scores")
                if (qi is not None and scores
                        and any(s is not None for s in scores)
                        and row.get("responses") == queries[qi]["responses"]):
                    done[key] = scores
        logger.info(f"Resuming annotation: {len(done)}/{len(queries)} queries already done in {ann_file}")

    pending = []
    for qi, q in enumerate(queries):
        key = (q["id"], q["persona"])
        if key in done:
            q["scores"] = done[key]
        else:
            pending.append(qi)

    # Rewrite with validated rows only (dedupes and repairs truncation), then append.
    with open(ann_file, "w") as f:
        for q in queries:
            if "scores" in q:
                f.write(json.dumps(q, default=str) + "\n")

    if pending:
        max_workers = min(args.max_concurrent_requests, len(pending))
        logger.info(f"Annotating {len(pending)} queries with {max_workers} concurrent workers...")
        with open(ann_file, "a") as fout, ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_annotate_query, (qi, queries[qi])): qi
                       for qi in pending}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Annotating individual responses"):
                qi, scores = future.result()
                queries[qi]["scores"] = scores
                fout.write(json.dumps(queries[qi], default=str) + "\n")
                fout.flush()
    logger.info(f"Saved annotated data to {ann_file}")

    # ---- Phase 4: Format preference pairs ----
    pairs = []
    dropped_small = 0
    dropped_reverse = 0
    if args.mode == PAIRED:
        # Paired mode: each query has exactly 2*K responses in [good_0, worse_0, good_1, worse_1, ...] order.
        # Use SIGNED margin filter on (good, worse) — no sort, no swap. Reverse pairs are dropped.
        K = args.num_pairs_per_query
        for q in queries:
            if "scores" not in q:
                continue
            for k in range(K):
                gi, wi = 2 * k, 2 * k + 1
                if gi >= len(q["scores"]) or wi >= len(q["scores"]):
                    continue
                sg, sw = q["scores"][gi], q["scores"][wi]
                if sg is None or sw is None:
                    continue
                if sg - sw < args.score_margin:
                    if sg < sw:
                        dropped_reverse += 1
                    else:
                        dropped_small += 1
                    continue
                resp_g = q["responses"][gi] or ""
                resp_w = q["responses"][wi] or ""
                if not resp_g.strip() or not resp_w.strip():
                    continue
                pairs.append(_make_preference_pair(
                    prompt_text=q["prompt_text"],
                    chosen_resp=resp_g,
                    rejected_resp=resp_w,
                    score_chosen=sg,
                    score_rejected=sw,
                    persona=q["persona"],
                    id_key=id_key,
                    id_val=q["id"],
                ))
        logger.info(f"Paired filter: dropped {dropped_small} small-margin, {dropped_reverse} reverse")
    else:
        for q in queries:
            if "scores" not in q:
                continue
            indexed = [(i, s) for i, s in enumerate(q["scores"]) if s is not None]
            indexed.sort(key=lambda x: x[1], reverse=True)

            for a_idx in range(len(indexed)):
                for b_idx in range(a_idx + 1, len(indexed)):
                    ri_a, score_a = indexed[a_idx]
                    ri_b, score_b = indexed[b_idx]
                    if score_a - score_b < args.score_margin:
                        continue
                    resp_a = q["responses"][ri_a] or ""
                    resp_b = q["responses"][ri_b] or ""
                    if not resp_a.strip() or not resp_b.strip():
                        continue
                    pairs.append(_make_preference_pair(
                        prompt_text=q["prompt_text"],
                        chosen_resp=resp_a,
                        rejected_resp=resp_b,
                        score_chosen=score_a,
                        score_rejected=score_b,
                        persona=q["persona"],
                        id_key=id_key,
                        id_val=q["id"],
                    ))

    logger.info(f"Created {len(pairs)} preference pairs (before verification)")

    # ---- Phase 5: LLM verification ----
    if args.verify_pairs and pairs:
        pairs = verify_pairs_with_llm(pairs, judge_llm)

    # ---- Phase 6: Scale and save ----
    if not pairs:
        logger.warning("No preference pairs to save!")
        return

    pairs = _scale_scores(pairs)

    # Ensure consistent columns (fill missing id_key with empty)
    other_key = "conv_id" if id_key == "meeting_id" else "meeting_id"
    for p in pairs:
        p.setdefault(other_key, "")

    dataset = Dataset.from_list(pairs)
    model_tag = args.model_name.split("/")[-1]
    save_path = os.path.join(args.output_dir, f"{file_prefix}_{args.dataset}_{model_tag}")
    dataset.save_to_disk(save_path)
    logger.info(f"Saved HF Dataset ({len(dataset)} rows) to {save_path}")
    return save_path


# =====================================================================
# Shared helpers for group pipelines (QMSum)
# =====================================================================

def _build_directive_qmsum(state: dict, agents: dict) -> str:
    """Build directive prompt for QMSum group episodes."""
    h = format_conversation_history(state["turns"], name_key='agent_name')
    return MascotPersonaTemplates.directive_prompt_qmsum(
        agents=agents, meeting_context=state["meeting"]["meeting_context"],
        conversation_history=h, agents_spoken=state["agents_spoken"],
    )


def _build_agent_qmsum(state: dict, directive: dict, agents: dict) -> str:
    """Build agent speaking prompt for QMSum group episodes."""
    h = format_conversation_history(state["turns"], name_key='agent_name')
    return MascotPersonaTemplates.speaker_prompt_qmsum(
        persona=agents[directive["agent_name"]],
        context=state["meeting"]["meeting_context"],
        meeting_summary=None, conversation_history=h,
    )


def _on_turn_end_qmsum(
    conv_states: OrderedDict, turn_idx: int,
    llm: VLLMServerModel, user_response_probability: float,
) -> None:
    """Post-turn hook: tag turn_type and optionally run QMSum user simulation."""
    for state in conv_states.values():
        last = state["turns"][-1]
        if "turn_type" not in last:
            last["turn_type"] = "agent"

    if user_response_probability <= 0:
        return

    user_keys = []
    user_prompts = []
    for key, state in conv_states.items():
        if random.random() < user_response_probability:
            user_keys.append(key)
            h = format_conversation_history(state["turns"], name_key='agent_name')
            user_prompts.append(generate_user_simulation_prompt_qmsum(
                meeting_context=state["meeting"]["meeting_context"],
                conversation_history=h,
            ))

    if not user_keys:
        return

    user_responses = [(t or "").strip() for t in llm.generate_batch(
        prompts=user_prompts, max_tokens=512,
        temperature=0.7,
        system_prompt=None,
        desc=f"User sim QMSum turn {turn_idx} ({len(user_prompts)})",
    )]

    for i, key in enumerate(user_keys):
        state = conv_states[key]
        raw = user_responses[i].strip()
        if not raw:
            continue
        parsed = extract_user_json(raw)
        if parsed is not None:
            user_text = parsed.get("response")
            if user_text:
                state["turns"].append({
                    "turn_idx": turn_idx,
                    "turn_type": "user",
                    "response": user_text,
                    "agent_name": "user",
                })


# =====================================================================
# Shared helpers for group pipelines (Empathetic)
# =====================================================================

def _build_directive_emp(state: dict, agents: dict) -> str:
    """Build directive prompt for Empathetic group episodes."""
    h = format_conversation_history_empathetic(state["turns"])
    return MascotPersonaTemplates.directive_prompt_empathetic(
        agents=agents, situation=state['initial_situation'],
        emotion=state.get('current_emotion', ''),
        conversation_history=h, agents_spoken=state["agents_spoken"],
    )


def _build_agent_emp(state: dict, directive: dict, agents: dict) -> str:
    """Build agent speaking prompt for Empathetic group episodes."""
    h = format_conversation_history_empathetic(state["turns"])
    return MascotPersonaTemplates.speaker_prompt_empathetic(
        persona=agents[directive["agent_name"]],
        context=state['initial_situation'],
        conversation_history=h, num_shot=5,
    )


def _on_turn_end_emp(
    conv_states: OrderedDict, turn_idx: int,
    llm: VLLMServerModel, user_response_probability: float,
) -> None:
    """Post-turn hook: tag turn_type and optionally run Empathetic user simulation."""
    for state in conv_states.values():
        last = state["turns"][-1]
        if "turn_type" not in last:
            last["turn_type"] = "agent"

    if user_response_probability <= 0:
        return

    user_keys = []
    user_prompts = []
    for key, state in conv_states.items():
        if random.random() < user_response_probability:
            user_keys.append(key)
            h = format_conversation_history_empathetic(state["turns"])
            user_prompts.append(generate_user_simulation_prompt(
                initial_situation=state['initial_situation'],
                current_emotion=state.get('current_emotion', ''),
                conversation_history=h,
            ))

    if not user_keys:
        return

    user_responses = [(t or "").strip() for t in llm.generate_batch(
        prompts=user_prompts, max_tokens=512,
        temperature=0.7,
        system_prompt=None,
        desc=f"User sim Empathetic turn {turn_idx} ({len(user_prompts)})",
    )]

    for i, key in enumerate(user_keys):
        state = conv_states[key]
        raw = user_responses[i].strip()
        if not raw:
            continue
        parsed = extract_user_json(raw)
        if parsed is not None:
            user_text = parsed.get("response")
            new_emotion = parsed.get("new_emotion")
            if user_text:
                state["turns"].append({
                    "turn_idx": turn_idx,
                    "turn_type": "user",
                    "response": user_text,
                    "agent_name": "user",
                })
                if new_emotion:
                    state["current_emotion"] = new_emotion


# =====================================================================
# GROUP PIPELINE (independent episodes)
# =====================================================================

def group_conversation_pipeline_independent(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate N episodes per query via MASCOT loop, score at episode + turn level."""
    agents = EMPATHETIC_AGENTS if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"
    adapted = llm

    model_tag = args.model_name.split("/")[-1]
    gen_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(gen_dir, exist_ok=True)

    # ---- Phase 1: Generate N episodes per query ----
    gen_file = os.path.join(gen_dir, f"group_{args.dataset}_{model_tag}.jsonl")

    if os.path.exists(gen_file):
        logger.info(f"Found existing group episodes at {gen_file}, skipping generation")
        all_episode_states: Dict[str, List[dict]] = {}
        with open(gen_file) as f:
            for line in f:
                rec = json.loads(line)
                all_episode_states[rec["query_key"]] = rec["episodes"]
    else:
        all_episode_states: Dict[str, List[dict]] = {}  # query_key -> list of episode states

        if args.dataset == QMSUM_DATASET:
            dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            meetings = create_meeting_contexts(dataset, args.turns_per_round)

            for ep_idx in range(args.num_episodes):
                logger.info(f"=== Episode {ep_idx + 1}/{args.num_episodes} ===")
                conversation_states = OrderedDict()
                for m in meetings:
                    conversation_states[(m['id'], m['start_idx'])] = {
                        "meeting": m, "id": m['id'], "start_idx": m['start_idx'],
                        "turns": [],
                        "agents_spoken": {n: 0 for n in agents},
                    }

                ep_config = EpisodeConfig(
                    agents=agents,
                    num_turns=args.num_turns,
                    directive_temperature=0.9,
                    agent_temperature=0.9,
                    directive_max_tokens=4096,
                    agent_max_tokens=4096,
                    enable_thinking=True,
                    disable_directive=False,
                )

                conversation_states = run_mascot_episode_loop(
                    conversation_states=conversation_states,
                    directive_generator=adapted,
                    agent_generator=adapted,
                    config=ep_config,
                    build_directive_prompt=lambda s: _build_directive_qmsum(s, agents),
                    build_agent_prompt=lambda s, d: _build_agent_qmsum(s, d, agents),
                    on_turn_end=lambda cs, ti: _on_turn_end_qmsum(
                        cs, ti, llm, args.user_response_probability),
                )

                for key, state in conversation_states.items():
                    qk = f"{key[0]}_{key[1]}"
                    all_episode_states.setdefault(qk, []).append(dict(state))

        elif args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET):
            if args.dataset == ESCONV_DATASET:
                conv_list = load_esconv_situations(split="train", max_samples=args.max_samples)
            else:
                conv_list = load_empathetic_situations(split="train", max_samples=args.max_samples)

            for ep_idx in range(args.num_episodes):
                logger.info(f"=== Episode {ep_idx + 1}/{args.num_episodes} ===")
                conversation_states = OrderedDict()
                for c in conv_list:
                    conversation_states[(c['conv_id'], None)] = {
                        "conv_id": c['conv_id'],
                        "initial_emotion": c['emotion'],
                        "initial_situation": c['situation'].replace("__comma__", ", "),
                        "current_emotion": c['emotion'],
                        "turns": [],
                        "agents_spoken": {n: 0 for n in agents},
                    }

                ep_config = EpisodeConfig(
                    agents=agents,
                    num_turns=args.num_turns,
                    directive_temperature=0.9,
                    agent_temperature=0.9,
                    directive_max_tokens=4096,
                    agent_max_tokens=4096,
                    enable_thinking=True,
                    disable_directive=False,
                )

                conversation_states = run_mascot_episode_loop(
                    conversation_states=conversation_states,
                    directive_generator=adapted,
                    agent_generator=adapted,
                    config=ep_config,
                    build_directive_prompt=lambda s: _build_directive_emp(s, agents),
                    build_agent_prompt=lambda s, d: _build_agent_emp(s, d, agents),
                    on_turn_end=lambda cs, ti: _on_turn_end_emp(
                        cs, ti, llm, args.user_response_probability),
                )

                for key, state in conversation_states.items():
                    qk = key[0]  # conv_id
                    all_episode_states.setdefault(qk, []).append(dict(state))

        # Save raw episodes
        with open(gen_file, "w") as f:
            for qk, episodes in all_episode_states.items():
                f.write(json.dumps({"query_key": qk, "episodes": episodes}, default=str) + "\n")
        logger.info(f"Saved raw group episodes to {gen_file}")

    # ---- Phase 2: Annotate (parallel) ----
    ds = EMPATHETIC_DATASET if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else QMSUM_DATASET

    # Episode-level scoring
    group_evaluator = GroupRubricEvaluator(judge_llm, dataset_name=ds)
    # Turn-level scoring
    individual_evaluator = IndividualRubricEvaluator(judge_llm, dataset_name=ds)

    # Build flat list of (qk, ep_idx, ep) for parallel processing
    annotation_tasks = []
    for qk, episodes in all_episode_states.items():
        for ep_idx, ep in enumerate(episodes):
            annotation_tasks.append((qk, ep_idx, ep))
            
    if args.max_samples:
        annotation_tasks = annotation_tasks[:args.max_samples]
        
    logger.info(f"Annotating {len(annotation_tasks)} episodes")

    def _annotate_episode(task):
        qk, ep_idx, ep = task
        # Episode-level score
        try:
            results = group_evaluator.evaluate_batch(ep, agents)
            episode_score = results[0].total_reward if results else None
        except Exception as e:
            logger.warning(f"Group eval failed for {qk}: {e}")
            episode_score = None

        # Turn-level scores
        eval_data = []
        for turn in ep.get("turns", []):
            aname = turn.get("agent_name", "unknown")
            if aname not in agents:
                continue
            eval_data.append({
                "conversation_context": ep.get("initial_situation", "") or
                                        ep.get("meeting", {}).get("meeting_context", ""),
                "agent_response": turn.get("response", ""),
                "agent_name": aname,
                "id": qk,
                "start_idx": ep.get("start_idx"),
                "turn_type": turn.get("turn_type", "agent"),
            })

        if eval_data:
            try:
                turn_results = individual_evaluator.evaluate_batch(eval_data, agents, ds)
                turn_scores = [r.total_reward for r in turn_results]
            except Exception as e:
                logger.warning(f"Turn eval failed for {qk}: {e}")
                turn_scores = [None] * len(eval_data)
        else:
            turn_scores = []

        return qk, ep_idx, episode_score, turn_scores

    # Save annotated
    ann_dir = os.path.join(args.output_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    if args.max_samples:
        ann_file = os.path.join(ann_dir, f"group_{args.dataset}_{model_tag}_max{args.max_samples}.jsonl")
    else:
        ann_file = os.path.join(ann_dir, f"group_{args.dataset}_{model_tag}.jsonl")

    max_workers = min(args.max_concurrent_requests, len(annotation_tasks))
    logger.info(f"Annotating {len(annotation_tasks)} episodes with {max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_annotate_episode, t): t for t in annotation_tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Annotating episodes"):
            qk, ep_idx, episode_score, turn_scores = future.result()
            all_episode_states[qk][ep_idx]["episode_score"] = episode_score
            all_episode_states[qk][ep_idx]["turn_scores"] = turn_scores

   
    with open(ann_file, "w") as f:
        for qk, episodes in all_episode_states.items():
            f.write(json.dumps({"query_key": qk, "episodes": episodes}, default=str) + "\n")
    logger.info(f"Saved annotated episodes to {ann_file}")

    # ---- Phase 3: Format preference pairs ----
    episode_pairs = []
    turn_pairs = []

    for qk, episodes in all_episode_states.items():
        if len(episodes) < 2:
            continue

        # Determine the id and prompt for this query
        sample_ep = episodes[0]
        if args.dataset == QMSUM_DATASET:
            ep_id = sample_ep.get("id", qk)
            prompt_context = sample_ep.get("meeting", {}).get("meeting_context", "")
        else:
            ep_id = sample_ep.get("conv_id", qk)
            prompt_context = sample_ep.get("initial_situation", "")

        # -- Episode-level pairs: sort by episode_score, top vs bottom --
        scored_eps = [(i, ep.get("episode_score")) for i, ep in enumerate(episodes)
                      if ep.get("episode_score") is not None]
        scored_eps.sort(key=lambda x: x[1], reverse=True)

        for a_pos in range(len(scored_eps)):
            for b_pos in range(a_pos + 1, len(scored_eps)):
                idx_a, score_a = scored_eps[a_pos]
                idx_b, score_b = scored_eps[b_pos]
                if score_a - score_b < args.score_margin:
                    continue
                # Format full episode as response
                chosen_text = "\n".join(
                    f"[{t.get('agent_name', 'unknown')}]: {t.get('response', '')}"
                    for t in episodes[idx_a].get("turns", [])
                )
                rejected_text = "\n".join(
                    f"[{t.get('agent_name', 'unknown')}]: {t.get('response', '')}"
                    for t in episodes[idx_b].get("turns", [])
                )
                if not chosen_text.strip() or not rejected_text.strip():
                    continue
                pair = _make_preference_pair(
                    prompt_text=prompt_context,
                    chosen_resp=chosen_text,
                    rejected_resp=rejected_text,
                    score_chosen=score_a,
                    score_rejected=score_b,
                    persona="group",
                    id_key=id_key,
                    id_val=ep_id,
                )
                _add_grpo_columns(pair, episodes[idx_a], args.dataset)
                episode_pairs.append(pair)

        # -- Turn-level pairs: compare corresponding turns across episodes --
        # Find agent turns only
        for ep in episodes:
            ep["_agent_turns"] = [
                (ti, t) for ti, t in enumerate(ep.get("turns", []))
                if t.get("turn_type", "agent") == "agent" and t.get("agent_name", "unknown") in agents
            ]

        min_agent_turns = min(len(ep["_agent_turns"]) for ep in episodes) if episodes else 0
        for turn_pos in range(min_agent_turns):
            turn_data = []
            for ep_i, ep in enumerate(episodes):
                ti, turn = ep["_agent_turns"][turn_pos]
                score = ep.get("turn_scores", [None] * (ti + 1))
                turn_score = score[turn_pos] if turn_pos < len(score) else None
                if turn_score is not None:
                    turn_data.append((ep_i, turn, turn_score))

            turn_data.sort(key=lambda x: x[2], reverse=True)
            for a_pos in range(len(turn_data)):
                for b_pos in range(a_pos + 1, len(turn_data)):
                    _, turn_a, score_a = turn_data[a_pos]
                    _, turn_b, score_b = turn_data[b_pos]
                    if score_a - score_b < args.score_margin:
                        continue
                    resp_a = turn_a.get("response", "")
                    resp_b = turn_b.get("response", "")
                    if not resp_a.strip() or not resp_b.strip():
                        continue
                    pair = _make_preference_pair(
                        prompt_text=prompt_context,
                        chosen_resp=resp_a,
                        rejected_resp=resp_b,
                        score_chosen=score_a,
                        score_rejected=score_b,
                        persona=turn_a.get("agent_name", "unknown"),
                        id_key=id_key,
                        id_val=ep_id,
                    )
                    _add_grpo_columns(pair, episodes[0], args.dataset)
                    turn_pairs.append(pair)

        # Clean up temp key
        for ep in episodes:
            ep.pop("_agent_turns", None)

    all_pairs = episode_pairs + turn_pairs
    logger.info(f"Created {len(episode_pairs)} episode-level + {len(turn_pairs)} turn-level "
                f"= {len(all_pairs)} total preference pairs")

    # ---- Phase 4: LLM verification ----
    if args.verify_pairs and all_pairs:
        all_pairs = verify_pairs_with_llm(all_pairs, judge_llm)

    # ---- Phase 5: Scale and save ----
    if not all_pairs:
        logger.warning("No preference pairs to save!")
        return

    all_pairs = _scale_scores(all_pairs)

    other_key = "conv_id" if id_key == "meeting_id" else "meeting_id"
    for p in all_pairs:
        p.setdefault(other_key, "")
        p.setdefault("episode_context", "{}")
        p.setdefault("episode_turns", "[]")
        p.setdefault("episode_agents_spoken", "{}")
        p.setdefault("dataset_name", args.dataset)

    dataset = Dataset.from_list(all_pairs)
    save_path = os.path.join(args.output_dir, f"group_{args.dataset}_{model_tag}")
    dataset.save_to_disk(save_path)
    logger.info(f"Saved HF Dataset ({len(dataset)} rows) to {save_path}")
    return save_path


# =====================================================================
# GROUP PIPELINE (branching episodes)
# =====================================================================

def group_conversation_pipeline_with_branching(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate branching episodes with turn-centric batching.

    Architecture:
    - Pre-assign branch_turn_idx for ALL meetings
    - Maintain conversation_states: OrderedDict[(meeting_idx, candidate_idx)] → state
    - Each turn: single generate_batch for directives, single generate_batch for agents
    - At branch turn: fork states into num_candidates copies with diverse directives
    - Single batched evaluation across ALL episodes at the end

    Supports both QMSum and Empathetic datasets.
    """
    agents = AGENTS if args.dataset == QMSUM_DATASET else EMPATHETIC_AGENTS  # ESConv reuses empathetic personas
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"
    model_tag = args.model_name.split("/")[-1]
    num_candidates = args.num_candidates
    num_turns = args.num_turns

    gen_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(gen_dir, exist_ok=True)
    gen_file = os.path.join(gen_dir, f"branching_{args.dataset}_{model_tag}.jsonl")

    if os.path.exists(gen_file):
        logger.info(f"Found existing branching data at {gen_file}, skipping generation")
        all_query_data: List[dict] = []
        with open(gen_file) as f:
            for line in f:
                all_query_data.append(json.loads(line))
    else:
        # ---- Load dataset ----
        if args.dataset == QMSUM_DATASET:
            dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            items = create_meeting_contexts(dataset, args.turns_per_round)
        else:
            if args.dataset == ESCONV_DATASET:
                items = load_esconv_situations(split="train", max_samples=args.max_samples)
            else:
                items = load_empathetic_situations(split="train", max_samples=args.max_samples)

        # Pre-assign branch positions for ALL items
        branch_turn_indices = {m_idx: random.randint(0, num_turns - 1)
                               for m_idx in range(len(items))}

        # One state per item, keyed by (item_idx, candidate_idx)
        conversation_states = OrderedDict()
        for m_idx, item in enumerate(items):
            if args.dataset == QMSUM_DATASET:
                state = {
                    "meeting": item, "id": item["id"],
                    "start_idx": item["start_idx"],
                    "turns": [], "agents_spoken": {n: 0 for n in agents},
                }
            else:
                state = {
                    "conv_id": item["conv_id"],
                    "initial_emotion": item["emotion"],
                    "initial_situation": item["situation"].replace("__comma__", ", "),
                    "current_emotion": item["emotion"],
                    "turns": [], "agents_spoken": {n: 0 for n in agents},
                }
            conversation_states[(m_idx, 0)] = state

        logger.info(f"Starting turn-centric branching: {len(items)} items, "
                     f"{num_turns} turns, {num_candidates} candidates")

        # ---- Set dataset-specific function references ----
        _build_directive = (lambda s: _build_directive_qmsum(s, agents)) if args.dataset == QMSUM_DATASET \
                           else (lambda s: _build_directive_emp(s, agents))
        _build_agent = (lambda s, d: _build_agent_qmsum(s, d, agents)) if args.dataset == QMSUM_DATASET \
                       else (lambda s, d: _build_agent_emp(s, d, agents))
        _on_turn_end = (lambda cs, ti: _on_turn_end_qmsum(cs, ti, llm, args.user_response_probability)) \
                       if args.dataset == QMSUM_DATASET else \
                       (lambda cs, ti: _on_turn_end_emp(cs, ti, llm, args.user_response_probability))

        branch_directive_prompts = {}  # m_idx -> directive prompt at branch turn
        completed_states = OrderedDict()  # (m_idx, c_idx) → state, for meetings done branching

        # ---- Turn-centric loop ----
        for turn_idx in range(num_turns):
            keys = list(conversation_states.keys())

            # D1. Partition: which meetings branch at this turn?
            branching_keys = [k for k in keys
                              if branch_turn_indices[k[0]] == turn_idx and k[1] == 0]
            branching_set = set(branching_keys)
            normal_keys = [k for k in keys if k not in branching_set]

            logger.info(f"Turn {turn_idx}: {len(keys)} states "
                         f"({len(branching_keys)} branching, {len(normal_keys)} normal)")

            # D2. Build directive prompts
            directive_prompts = []
            prompt_source = []  # (key, candidate_idx_or_None)

            for key in branching_keys:
                prompt = _build_directive(conversation_states[key])
                branch_directive_prompts[key[0]] = prompt
                for c in range(num_candidates):
                    directive_prompts.append(prompt)
                    prompt_source.append((key, c))

            for key in normal_keys:
                directive_prompts.append(
                    _build_directive(conversation_states[key]))
                prompt_source.append((key, None))

            # SINGLE batch call for all directives
            raw_results = llm.generate_batch(
                prompts=directive_prompts, max_tokens=4096, temperature=1.0,
                desc=f"Directives turn {turn_idx} ({len(directive_prompts)})",
            ) if directive_prompts else []

            directive_outputs = [(t or "").strip() for t in raw_results]

            # D3. Parse directives — NO random fallback
            parsed_directives = [parse_directive(output, agents)
                                 for output in directive_outputs]

            # Retry failed non-branching directives up to directive_retries times
            branching_count = len(branching_keys) * num_candidates
            for retry_num in range(args.directive_retries):
                retry_indices = []
                retry_prompts = []
                for i in range(branching_count, len(parsed_directives)):
                    if parsed_directives[i] is None:
                        retry_indices.append(i)
                        retry_prompts.append(directive_prompts[i])

                if not retry_prompts:
                    break

                retry_results = llm.generate_batch(
                    prompts=retry_prompts, max_tokens=4096, temperature=1.0,
                    desc=f"Directive retry {retry_num+1}/{args.directive_retries} ({len(retry_prompts)})",
                )
                for j, idx in enumerate(retry_indices):
                    parsed_directives[idx] = parse_directive(
                        (retry_results[j] or "").strip(), agents)

            # D4. Drop meetings with unrecoverable failures
            dropped_meetings = set()

            # Non-branching: drop if directive still None after retry
            for i in range(branching_count, len(parsed_directives)):
                if parsed_directives[i] is None:
                    key, _ = prompt_source[i]
                    dropped_meetings.add(key[0])

            # Branching: need >=2 valid directives per meeting
            branching_dirs = {}  # key → list of valid directives
            offset = 0
            for key in branching_keys:
                m_idx = key[0]
                dirs_for_meeting = []
                for c in range(num_candidates):
                    d = parsed_directives[offset + c]
                    if d is not None:
                        dirs_for_meeting.append(d)
                offset += num_candidates
                if len(dirs_for_meeting) < 2:
                    logger.warning(f"Meeting {m_idx}: only {len(dirs_for_meeting)} "
                                   f"valid directives at branch turn, dropping")
                    dropped_meetings.add(m_idx)
                else:
                    branching_dirs[key] = dirs_for_meeting

            # Remove dropped meetings from conversation_states
            if dropped_meetings:
                for key in list(conversation_states.keys()):
                    if key[0] in dropped_meetings:
                        del conversation_states[key]
                branching_keys = [k for k in branching_keys
                                  if k[0] not in dropped_meetings]
                normal_keys = [k for k in normal_keys
                               if k[0] not in dropped_meetings]
                logger.info(f"  Dropped {len(dropped_meetings)} meetings "
                             f"due to parse failures")

            # Build normal directive map from surviving states
            normal_directive_map = {}
            for i, (key, candidate) in enumerate(prompt_source):
                if candidate is None and key[0] not in dropped_meetings:
                    normal_directive_map[key] = parsed_directives[i]

            # D5. Fork branching states into candidate copies
            for key in branching_keys:
                m_idx = key[0]
                prefix_state = conversation_states[key]
                actual_num = len(branching_dirs[key])
                for c in range(1, actual_num):
                    fork = {k: v for k, v in prefix_state.items()
                            if k not in ("turns", "agents_spoken")}
                    fork["turns"] = [dict(t) for t in prefix_state["turns"]]
                    fork["agents_spoken"] = dict(prefix_state["agents_spoken"])
                    conversation_states[(m_idx, c)] = fork

            # D6. Build agent prompts for ALL active states
            agent_prompts = []
            agent_keys = []
            agent_directive_list = []

            for key in branching_keys:
                m_idx = key[0]
                dirs = branching_dirs[key]
                for c, directive in enumerate(dirs):
                    state_key = (m_idx, c)
                    agent_prompts.append(
                        _build_agent(
                            conversation_states[state_key], directive))
                    agent_keys.append(state_key)
                    agent_directive_list.append(directive)

            for key in normal_keys:
                directive = normal_directive_map[key]
                agent_prompts.append(
                    _build_agent(conversation_states[key], directive))
                agent_keys.append(key)
                agent_directive_list.append(directive)

            # SINGLE batch call for all agent responses
            agent_results = llm.generate_batch(
                prompts=agent_prompts, max_tokens=4096, temperature=0.9,
                desc=f"Agents turn {turn_idx} ({len(agent_prompts)})",
            ) if agent_prompts else []

            # D7. Update states with agent responses
            for i in range(len(agent_keys)):
                state = conversation_states[agent_keys[i]]
                directive = agent_directive_list[i]
                state["agents_spoken"][directive["agent_name"]] += 1
                state["turns"].append({
                    "turn_idx": turn_idx, "directive": directive,
                    "agent_name": directive["agent_name"],
                    "response": (agent_results[i] or "").strip(), "turn_type": "agent",
                })

            # D8. User simulation (already batches internally)
            _on_turn_end(conversation_states, turn_idx)

            # D9. Early-stop: move completed branching meetings out of active loop
            if args.stop_after_branch:
                for key in branching_keys:
                    m_idx = key[0]
                    if m_idx in dropped_meetings:
                        continue
                    actual_num = len(branching_dirs.get(key, []))
                    for c in range(actual_num):
                        state_key = (m_idx, c)
                        if state_key in conversation_states:
                            completed_states[state_key] = conversation_states.pop(state_key)

        # Merge any remaining states (meetings that never branched in the loop)
        for key, state in conversation_states.items():
            completed_states[key] = state

        # ---- Batched evaluation ----
        group_evaluator = GroupRubricEvaluator(judge_llm, dataset_name=QMSUM_DATASET if args.dataset == QMSUM_DATASET else EMPATHETIC_DATASET)  # ESConv reuses empathetic rubric

        eval_prompts = []
        eval_meta = []  # (key, orig, clean)
        for key, state in completed_states.items():
            prompt, orig, clean = group_evaluator.build_prompt(state, agents)
            if prompt is not None:
                eval_prompts.append(prompt)
                eval_meta.append((key, orig, clean))
            else:
                state["episode_score"] = None

        if eval_prompts:
            eval_results = judge_llm.generate_batch(
                prompts=eval_prompts, temperature=0.0, max_tokens=3000,
                desc=f"Group eval ({len(eval_prompts)} episodes)",
            )
            for i, (key, orig, clean) in enumerate(eval_meta):
                state = completed_states[key]
                try:
                    parsed = group_evaluator.parse_result(
                        eval_results[i], state, orig, clean)
                    state["episode_score"] = (
                        parsed[0].total_reward if parsed else None)
                except Exception as e:
                    logger.warning(f"Eval parse failed for {key}: {e}")
                    state["episode_score"] = None

        # ---- Build query records ----
        meeting_episodes = {}
        for (m_idx, c_idx), state in completed_states.items():
            meeting_episodes.setdefault(m_idx, []).append((c_idx, state))

        all_query_data = []
        for m_idx in sorted(meeting_episodes):
            item = items[m_idx]
            candidates = sorted(meeting_episodes[m_idx], key=lambda x: x[0])
            episodes = [state for _, state in candidates]
            rec = {
                "branch_turn_idx": branch_turn_indices[m_idx],
                "num_candidates": len(episodes),
                "episodes": episodes,
            }
            if args.dataset == QMSUM_DATASET:
                rec["query_key"] = f"{item['id']}_{item['start_idx']}"
                rec["meeting_id"] = item["id"]
                rec["start_idx"] = item["start_idx"]
            else:
                rec["query_key"] = item["conv_id"]
                rec["conv_id"] = item["conv_id"]
            rec["branch_directive_prompt"] = branch_directive_prompts.get(m_idx, "")
            all_query_data.append(rec)

        # Save raw branching data
        with open(gen_file, "w") as f:
            for rec in all_query_data:
                f.write(json.dumps(rec, default=str) + "\n")
        logger.info(f"Saved raw branching data to {gen_file}")

    # ---- Phase 2: Create preference pairs ----
    pairs = []
    for rec in all_query_data:
        episodes = rec["episodes"]
        branch_turn_idx = rec["branch_turn_idx"]
        ep_id = rec.get("meeting_id") or rec.get("conv_id")

        # Use the directive prompt captured at the branch turn
        prompt_context = rec.get("branch_directive_prompt", "")

        # Collect scored episodes
        scored = [(i, ep.get("episode_score")) for i, ep in enumerate(episodes)
                  if ep.get("episode_score") is not None]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Create pairs from all (i, j) where score_diff >= margin
        for a_pos in range(len(scored)):
            for b_pos in range(a_pos + 1, len(scored)):
                idx_a, score_a = scored[a_pos]
                idx_b, score_b = scored[b_pos]
                if score_a - score_b < args.score_margin:
                    continue

                # Format only post-branch turns as response text
                # Skip if branch turn is a user turn (no directive)
                chosen_turn = episodes[idx_a]['turns'][branch_turn_idx]
                rejected_turn = episodes[idx_b]['turns'][branch_turn_idx]
                if 'directive' not in chosen_turn or 'directive' not in rejected_turn:
                    continue
                chosen_directive = chosen_turn['directive']
                rejected_directive = rejected_turn['directive']
                chosen_text = f"[{chosen_directive['agent_name']}]: {chosen_directive['action']} Tone: {chosen_directive['tone']}"
                rejected_text = f"[{rejected_directive['agent_name']}]: {rejected_directive['action']} Tone: {rejected_directive['tone']}"
                if not chosen_text.strip() or not rejected_text.strip():
                    continue

                pair = _make_preference_pair(
                    prompt_text=prompt_context,
                    chosen_resp=chosen_text,
                    rejected_resp=rejected_text,
                    score_chosen=score_a,
                    score_rejected=score_b,
                    persona="group",
                    id_key=id_key,
                    id_val=ep_id,
                )
                pair["branch_turn_idx"] = branch_turn_idx
                _add_grpo_columns(pair, episodes[idx_a], args.dataset)
                pairs.append(pair)

    logger.info(f"Created {len(pairs)} branching preference pairs")

    # ---- Phase 3: LLM verification ----
    if args.verify_pairs and pairs:
        pairs = verify_pairs_with_llm(pairs, judge_llm)

    # ---- Phase 4: Scale and save ----
    if not pairs:
        logger.warning("No branching preference pairs to save!")
        return

    pairs = _scale_scores(pairs)

    for p in pairs:
        p.setdefault("conv_id", "")
        p.setdefault("meeting_id", "")
        p.setdefault("episode_context", "{}")
        p.setdefault("episode_turns", "[]")
        p.setdefault("episode_agents_spoken", "{}")
        p.setdefault("dataset_name", args.dataset)

    dataset = Dataset.from_list(pairs)
    suffix = ""
    if args.max_samples:
        suffix += f"_max{args.max_samples}"
    
    save_path = os.path.join(args.output_dir, f"branching_{args.dataset}_{model_tag}{suffix}")
    dataset.save_to_disk(save_path)
    logger.info(f"Saved HF Dataset ({len(dataset)} rows) to {save_path}")

    # Print K distribution stats
    branch_pos_counts = {}
    for rec in all_query_data:
        branch_pos = rec["branch_turn_idx"]
        branch_pos_counts[branch_pos] = branch_pos_counts.get(branch_pos, 0) + 1
    logger.info(f"Branch position distribution: {dict(sorted(branch_pos_counts.items()))}")

    return save_path


# =====================================================================
# Argument Parsing
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified data pipeline: generate, annotate, format reward-model data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="individual",
                        choices=["individual", "paired", "group"],
                        help="Pipeline mode: individual (N independent responses), "
                             "paired (LLM emits good+worse in one call, Empathetic/ESConv only), "
                             "or group")
    parser.add_argument("--dataset", type=str, default="QMSum",
                        choices=[QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET])
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--generator-type", type=str, default="vllm",
                        choices=["vllm", "gpt"],
                        help="Backend for response generation: vllm server or "
                             "OpenAI/Azure GPT API (judge stays on vLLM).")
    parser.add_argument("--output-dir", type=str, default="data/rl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Individual-specific
    parser.add_argument("--num-agents-per-query", type=int, default=5,
                        help="Number of responses to generate per query (individual mode)")
    parser.add_argument("--turns-per-round", type=int, default=100,
                        help="Transcript turns per meeting segment (QMSum)")
    parser.add_argument("--num-pairs-per-query", type=int, default=1,
                        help="Number of (good, worse) pairs to generate per query "
                             "(paired mode only). Each call yields one pair.")

    # Group-specific
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes per query (group mode)")
    parser.add_argument("--num-turns", type=int, default=10,
                        help="Number of turns per episode (group mode)")
    parser.add_argument("--user-response-probability", type=float, default=0.5,
                        help="Probability of user simulation after each turn (group mode)")
    parser.add_argument("--generation-mode", type=str, default="independent",
                        choices=["independent", "branching"],
                        help="Group generation strategy: independent episodes or branching")
    parser.add_argument("--num-candidates", type=int, default=4,
                        help="Number of directive candidates per branch point (branching mode)")
    parser.add_argument("--directive-retries", type=int, default=3,
                        help="Max retries for failed directive parsing (branching mode)")
    parser.add_argument("--stop-after-branch", action="store_true", default=False,
                        help="Stop generating turns for a meeting after its branch turn "
                             "(saves compute; preference pairs only use the branch turn)")

    # Scoring / filtering
    parser.add_argument("--score-margin", type=float, default=0.5,
                        help="Minimum score difference for preference pairs")
    parser.add_argument("--verify-pairs", action="store_true",
                        help="Use LLM to verify preference pairs")

    # vLLM server
    parser.add_argument("--vllm-server-url", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    
    parser.add_argument("--judge-vllm-server-url", type=str,
                        default="http://localhost:8032/v1")
    
    parser.add_argument("--judge-model-name", type=str, default="Qwen/Qwen3-32B")

    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

def main():
    project_setup()
    args = parse_args()
    random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger.info(f"Pipeline: mode={args.mode}, dataset={args.dataset}, "
                f"model={args.model_name}")

    # Create generation LLM
    if args.generator_type == "gpt":
        llm = GPT(args.model_name, max_workers=args.max_concurrent_requests)
    else:
        llm = VLLMServerModel(
            model_name=args.model_name,
            base_url=args.vllm_server_url,
            max_concurrent_requests=args.max_concurrent_requests,
            enable_thinking=True,
        )
    # Create judge LLM (no thinking for structured JSON)
    judge_llm = VLLMServerModel(
        model_name=args.judge_model_name,
        base_url=args.judge_vllm_server_url,
        max_concurrent_requests=args.max_concurrent_requests,
        enable_thinking=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in (INDIVIDUAL, PAIRED):
        individual_conversation_pipeline(args, llm, judge_llm)

    elif args.generation_mode == "branching":

        # This branch compares two episodes that have the same prefixes and differ starting from a branching point.
        group_conversation_pipeline_with_branching(args, llm, judge_llm)

    else:
        assert args.generation_mode == "independent", f"Unsupported generation mode: {args.generation_mode}"
        # This branch compares completely different episodes.
        group_conversation_pipeline_independent(args, llm, judge_llm)


if __name__ == "__main__":
    main()
