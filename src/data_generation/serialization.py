"""Artifact writing and run-lifecycle policy for the data-generation pipelines.

Owns raw/annotated JSONL writing, HF-Dataset saving with the historical column
defaults, the default-refuse/--resume/--overwrite policy, and per-run sidecar
metadata placement (`<dataset_dir>.runs/<run_id>/`). Paths, filenames, schemas,
and the resume/overwrite behavior are unchanged from Phase 1.5.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from datasets import Dataset

from src.const import GROUP
from src.rubrics import PAPER_V1
from src.utils.run_metadata import (new_counters, new_run_id,
                                    write_run_config, write_run_summary)

logger = logging.getLogger(__name__)


def write_jsonl(path: str, records, make_record=None) -> None:
    """Write records to a JSONL file (default=str for non-JSON types)."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(make_record(rec) if make_record else rec,
                               default=str) + "\n")


def read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_pairs_dataset(pairs: List[dict], save_path: str, *,
                       id_key: Optional[str] = None,
                       both_id_keys: bool = False,
                       episode_defaults: bool = False,
                       dataset_name: Optional[str] = None) -> str:
    """Fill the historical per-pipeline column defaults, then save an HF Dataset.

    - individual: the counterpart id key gets ''.
    - group independent: counterpart id key + episode GRPO defaults + dataset.
    - branching: BOTH id keys + episode GRPO defaults + dataset.
    """
    for p in pairs:
        if both_id_keys:
            p.setdefault("conv_id", "")
            p.setdefault("meeting_id", "")
        elif id_key:
            other_key = "conv_id" if id_key == "meeting_id" else "meeting_id"
            p.setdefault(other_key, "")
        if episode_defaults:
            p.setdefault("episode_context", "{}")
            p.setdefault("episode_turns", "[]")
            p.setdefault("episode_agents_spoken", "{}")
            p.setdefault("dataset_name", dataset_name)

    dataset = Dataset.from_list(pairs)
    dataset.save_to_disk(save_path)
    logger.info(f"Saved HF Dataset ({len(dataset)} rows) to {save_path}")
    return save_path


def apply_output_policy(args, paths: dict) -> None:
    """Default-refuse / --resume / --overwrite policy for existing outputs.

    Exits with SystemExit(2) when outputs exist and neither flag was passed;
    with --overwrite the raw/annotated files and dataset dir are removed.
    """
    dataset_dir = Path(paths["dataset_dir"])
    existing = [p for p in (paths["raw_file"], paths["annotated_file"]) if p and os.path.exists(p)]
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        existing.append(str(dataset_dir))
    if existing and not args.resume and not args.overwrite:
        logger.error(
            "Outputs for this configuration already exist:\n  "
            + "\n  ".join(existing)
            + "\nPass --resume to continue from them or --overwrite to regenerate.")
        raise SystemExit(2)
    if args.overwrite and existing:
        import shutil
        for p in (paths["raw_file"], paths["annotated_file"]):
            if p and os.path.exists(p):
                os.unlink(p)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        logger.info("--overwrite: removed existing raw/annotated files and dataset dir")


def start_run_metadata(args, paths: dict):
    """Create the per-run sidecar (`<dataset_dir>.runs/<run_id>/`) with
    run_config.json, attach fresh failure counters to args, and return
    (run_id, run_meta_dir, started_at) for the closing run_summary."""
    run_id = new_run_id(f"{args.mode if args.mode != GROUP else args.generation_mode}"
                        f"_{args.dataset}_{os.path.basename(args.model_name)}")
    run_meta_dir = Path(str(paths["dataset_dir"]) + ".runs") / run_id
    started_at = time.time()
    args._counters = new_counters()
    write_run_config(
        run_meta_dir, run_id=run_id, dataset=args.dataset, system="mascot",
        persona_set="default",
        model_roles={"candidate": args.model_name, "judge": args.judge_model_name},
        rubric_version=PAPER_V1, args=args, seed=args.seed,
        extra={"raw_file": paths["raw_file"],
               "annotated_file": paths["annotated_file"],
               "artifact_dir": str(paths["dataset_dir"])},
    )
    return run_id, run_meta_dir, started_at


def finish_run_metadata(run_meta_dir, run_id: str, status: str, args,
                        started_at: float) -> None:
    """Write the closing run_summary.json for a pipeline run."""
    write_run_summary(run_meta_dir, run_id=run_id, status=status,
                      counters=args._counters, started_at=started_at)
