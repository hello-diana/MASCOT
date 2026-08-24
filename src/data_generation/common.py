"""Small shared pure helpers for the data-generation pipelines.

No CLI parsing, no model construction, no pipeline implementations here.
"""

import json
import logging
import os
from typing import List

from src.const import INDIVIDUAL, PAIRED, QMSUM_DATASET
from src.utils.paths import artifact_dir
from src.utils.text import extract_final_response, strip_response_prefix

logger = logging.getLogger(__name__)


def artifact_paths(args) -> dict:
    """Single source of truth for this run's artifact paths, computable from
    args at startup. The pipeline save sites use these same values, so the
    metadata in run_config.json can never drift from what was written.

    Returns {raw_file, annotated_file (None for branching), dataset_dir}.
    """
    model_tag = args.model_name.split("/")[-1]
    max_suffix = f"_max{args.max_samples}" if args.max_samples else ""
    raw_dir = os.path.join(args.output_dir, "raw")
    ann_dir = os.path.join(args.output_dir, "annotated")

    if args.mode in (INDIVIDUAL, PAIRED):
        prefix = PAIRED if args.mode == PAIRED else INDIVIDUAL
        return {
            "raw_file": os.path.join(raw_dir, f"{prefix}_{args.dataset}_{model_tag}{max_suffix}.jsonl"),
            "annotated_file": os.path.join(ann_dir, f"{prefix}_{args.dataset}_{model_tag}{max_suffix}.jsonl"),
            "dataset_dir": str(artifact_dir("preference_data", dataset=args.dataset,
                                            model_name=args.model_name, mode=prefix,
                                            root=args.output_dir)),
        }
    if args.generation_mode == "branching":
        return {
            "raw_file": os.path.join(raw_dir, f"branching_{args.dataset}_{model_tag}.jsonl"),
            "annotated_file": None,  # branching scores are parsed inline, no annotated jsonl
            "dataset_dir": str(artifact_dir("preference_data", dataset=args.dataset,
                                            model_name=args.model_name, mode="branching",
                                            root=args.output_dir)) + max_suffix,
        }
    return {  # group independent
        "raw_file": os.path.join(raw_dir, f"group_{args.dataset}_{model_tag}.jsonl"),
        "annotated_file": os.path.join(ann_dir, f"group_{args.dataset}_{model_tag}{max_suffix}.jsonl"),
        "dataset_dir": str(artifact_dir("preference_data", dataset=args.dataset,
                                        model_name=args.model_name, mode="group",
                                        root=args.output_dir)),
    }


def extract_episode_context(episode: dict, dataset_name: str) -> str:
    """Extract context from episode state as JSON string, handling both datasets."""
    if dataset_name == QMSUM_DATASET:
        return json.dumps({"meeting_context": episode.get("meeting", {}).get("meeting_context", "")})
    else:  # Empathetic
        return json.dumps({
            "initial_situation": episode.get("initial_situation", ""),
            "initial_emotion": episode.get("initial_emotion", ""),
        })


def add_grpo_columns(pair: dict, episode: dict, dataset_name: str) -> dict:
    """Add GRPO training columns to a preference pair in-place."""
    pair["episode_context"] = extract_episode_context(episode, dataset_name)
    pair["episode_turns"] = json.dumps(episode.get("turns", []), default=str)
    pair["episode_agents_spoken"] = json.dumps(episode.get("agents_spoken", {}))
    pair["dataset_name"] = dataset_name
    return pair


def scale_scores(records: List[dict]) -> List[dict]:
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


def clean(text: str) -> str:
    """Strip <think> tags (see src.utils.text.extract_final_response)."""
    return extract_final_response(text)


def needs_min_candidates(mode: str, generation_mode: str) -> bool:
    """True when the run forms preference pairs from K candidates (K >= 2)."""
    from src.const import GROUP
    return mode == INDIVIDUAL or (mode == GROUP and generation_mode == "branching")


# Historical private names (imported by the compatibility wrapper).
_extract_episode_context = extract_episode_context
_add_grpo_columns = add_grpo_columns
_scale_scores = scale_scores
_clean = clean
_strip_response_prefix = strip_response_prefix
