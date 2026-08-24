"""Preference-pair construction for the four distinct pair semantics.

The record constructor (:func:`make_preference_pair`) is shared; the four
builders keep their historical ordering, score handling, margin filtering,
max-pair behavior, provenance metadata, and GRPO columns exactly:

- :func:`build_paired_pairs` — one good+worse generation per call, SIGNED
  margin (reverse pairs dropped, never swapped).
- :func:`build_individual_pairs` — K independent candidates, all unordered
  (K choose 2) pairs sorted by score, margin-filtered.
- :func:`build_independent_group_pairs` — episode-level top-vs-bottom pairs +
  turn-level pairs across sibling episodes.
- :func:`build_branching_pairs` — directive pairs at the branch turn.
- :func:`load_and_convert_to_preference_pairs` — legacy two-run episode JSONL
  conversion (used by group reward-model training).
"""

import logging
import os
from typing import List

from datasets import Dataset

from src.data_generation.common import add_grpo_columns
from src.prompts.prompt_reward_model import get_prompt_collective_reward_model
from src.rubrics import PAPER_V1

logger = logging.getLogger(__name__)


def make_preference_pair(prompt_text: str, chosen_resp: str, rejected_resp: str,
                         score_chosen: float, score_rejected: float,
                         persona: str, id_key: str, id_val: str,
                         args=None) -> dict:
    pair = {
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
    if args is not None:
        # Preference-record provenance (spec: candidate model, judge model,
        # persona, rubric version) — only fields relevant to this artifact.
        pair.update({
            "schema_version": "1.0",
            "dataset": args.dataset,
            "candidate_model": args.model_name,
            "judge_model": args.judge_model_name,
            "rubric_version": PAPER_V1,
        })
    return pair


def build_paired_pairs(queries, args, id_key: str) -> List[dict]:
    """Paired mode: each query has exactly 2*K responses in
    [good_0, worse_0, good_1, worse_1, ...] order. Uses a SIGNED margin
    filter on (good, worse) — no sort, no swap. Reverse pairs are dropped."""
    pairs = []
    dropped_small = 0
    dropped_reverse = 0
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
            pairs.append(make_preference_pair(
                prompt_text=q["prompt_text"],
                chosen_resp=resp_g,
                rejected_resp=resp_w,
                score_chosen=sg,
                score_rejected=sw,
                persona=q["persona"],
                id_key=id_key,
                id_val=q["id"], args=args,
            ))
    logger.info(f"Paired filter: dropped {dropped_small} small-margin, {dropped_reverse} reverse")
    return pairs


def build_individual_pairs(queries, args, id_key: str) -> List[dict]:
    """Individual mode: all unordered (K choose 2) candidate pairs per query,
    sorted by score descending, margin-filtered.
    K=2 -> exactly 1 pair, K=8 -> up to 28 pairs per query."""
    pairs = []
    candidate_pairs_before_margin = 0
    for q in queries:
        if "scores" not in q:
            continue
        indexed = [(i, s) for i, s in enumerate(q["scores"]) if s is not None]
        indexed.sort(key=lambda x: x[1], reverse=True)

        candidate_pairs_before_margin += len(indexed) * (len(indexed) - 1) // 2

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
                pairs.append(make_preference_pair(
                    prompt_text=q["prompt_text"],
                    chosen_resp=resp_a,
                    rejected_resp=resp_b,
                    score_chosen=score_a,
                    score_rejected=score_b,
                    persona=q["persona"],
                    id_key=id_key,
                    id_val=q["id"], args=args,
                ))

    logger.info(f"K={args.num_candidates} candidates/query: "
                f"{candidate_pairs_before_margin} unordered pairs before margin filter, "
                f"{len(pairs)} kept (score_margin={args.score_margin})")
    return pairs


def build_independent_group_pairs(all_episode_states, args, agents,
                                  id_key: str, dataset_name: str):
    """Independent group mode: episode-level pairs (full-episode text, sorted
    by episode_score) plus turn-level pairs comparing corresponding agent
    turns across sibling episodes. Returns (episode_pairs, turn_pairs)."""
    from src.const import QMSUM_DATASET

    episode_pairs = []
    turn_pairs = []

    for qk, episodes in all_episode_states.items():
        if len(episodes) < 2:
            continue

        # Determine the id and prompt for this query
        sample_ep = episodes[0]
        if dataset_name == QMSUM_DATASET:
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
                pair = make_preference_pair(
                    prompt_text=prompt_context,
                    chosen_resp=chosen_text,
                    rejected_resp=rejected_text,
                    score_chosen=score_a,
                    score_rejected=score_b,
                    persona="group",
                    id_key=id_key,
                    id_val=ep_id, args=args,
                )
                add_grpo_columns(pair, episodes[idx_a], dataset_name)
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
                    pair = make_preference_pair(
                        prompt_text=prompt_context,
                        chosen_resp=resp_a,
                        rejected_resp=resp_b,
                        score_chosen=score_a,
                        score_rejected=score_b,
                        persona=turn_a.get("agent_name", "unknown"),
                        id_key=id_key,
                        id_val=ep_id, args=args,
                    )
                    add_grpo_columns(pair, episodes[0], dataset_name)
                    turn_pairs.append(pair)

        # Clean up temp key
        for ep in episodes:
            ep.pop("_agent_turns", None)

    return episode_pairs, turn_pairs


def build_branching_pairs(all_query_data, args, id_key: str) -> List[dict]:
    """Branching mode: directive pairs at the branch turn, scored by
    episode-level scores of the branched trajectories."""
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

                pair = make_preference_pair(
                    prompt_text=prompt_context,
                    chosen_resp=chosen_text,
                    rejected_resp=rejected_text,
                    score_chosen=score_a,
                    score_rejected=score_b,
                    persona="group",
                    id_key=id_key,
                    id_val=ep_id, args=args,
                )
                pair["branch_turn_idx"] = branch_turn_idx
                add_grpo_columns(pair, episodes[idx_a], args.dataset)
                pairs.append(pair)

    logger.info(f"Created {len(pairs)} branching preference pairs")
    return pairs


def load_and_convert_to_preference_pairs(data_paths: list, score_margin: float = 0.5,
                                         rubric_version: str = None) -> Dataset:
    """Convert episodes to preference pairs for RewardTrainer (legacy two-run
    JSONL format used by group reward-model training).

    Args:
        data_paths: Exactly 2 paths to episode JSONL files.
        score_margin: Minimum score difference to keep a pair.

    Returns:
        HF Dataset with columns: chosen, rejected, score_chosen, score_rejected.
    """
    import json

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

        chosen_text = get_prompt_collective_reward_model(chosen_episode["turns"],
                                                         rubric_version=rubric_version or PAPER_V1)
        rejected_text = get_prompt_collective_reward_model(rejected_episode["turns"],
                                                           rubric_version=rubric_version or PAPER_V1)
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


# Historical private name (imported by the compatibility wrapper).
_make_preference_pair = make_preference_pair
