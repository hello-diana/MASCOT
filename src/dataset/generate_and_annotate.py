#!/usr/bin/env python3
"""Unified data pipeline: generate, annotate, and format reward-model training data.

Supports both individual-level and group-level data generation for QMSum and
Empathetic datasets.  Outputs an HF Dataset compatible with
``datasets.load_from_disk`` (used by ``create_or_load_rl_dataset``).

This module is the documented CLI (``python -m src.dataset.generate_and_annotate``)
and the backward-compatible import surface; the implementation lives in
``src/data_generation/`` (Phase-2 split).

Usage examples:
    # Individual-level QMSum (small test)
    python src/dataset/generate_and_annotate.py \
        --mode individual --dataset QMSum --max-samples 5

    # Group-level Empathetic with user simulation and LLM verification
    python src/dataset/generate_and_annotate.py \
        --mode group --dataset Empathetic --verify-pairs
"""

import argparse
import logging
import random

from src.models.llm_models import GPT, VLLMServerModel
from src.utils.eval_utils import project_setup
from src.utils.run_metadata import StrictModeError

from src.const import (QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET,
                       INDIVIDUAL, PAIRED, GROUP,
                       DEFAULT_SPEAKER_MODEL, DEFAULT_LOCAL_JUDGE_MODEL,
                       DEFAULT_NUM_CANDIDATES, DEFAULT_SEED)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports: everything previously defined in this
# module keeps its historical (including private) name.
# ---------------------------------------------------------------------------
from src.data_generation.common import (  # noqa: F401
    artifact_paths,
    needs_min_candidates,
    _add_grpo_columns,
    _clean,
    _extract_episode_context,
    _scale_scores,
    _strip_response_prefix,
)
from src.data_generation.annotation import verify_pairs_with_llm  # noqa: F401
from src.data_generation.preference_pairs import _make_preference_pair  # noqa: F401
from src.data_generation.individual import (  # noqa: F401
    individual_conversation_pipeline,
    _build_paired_prompt_empathetic,
    _parse_paired_response,
)
from src.data_generation.group_independent import (  # noqa: F401
    group_conversation_pipeline_independent,
    _build_agent_emp,
    _build_agent_qmsum,
    _build_directive_emp,
    _build_directive_qmsum,
    _on_turn_end_emp,
    _on_turn_end_qmsum,
)
from src.data_generation.group_branching import (  # noqa: F401
    group_conversation_pipeline_with_branching,
)
from src.data_generation.serialization import (  # noqa: F401
    apply_output_policy,
    finish_run_metadata,
    start_run_metadata,
)

logger = logging.getLogger(__name__)


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
    parser.add_argument("--model-name", type=str, default=DEFAULT_SPEAKER_MODEL,
                        help="Model that generates candidate responses/episodes")
    parser.add_argument("--generator-type", type=str, default="vllm",
                        choices=["vllm", "openai", "azure-openai", "gpt"],
                        help="Backend for response generation: vllm server, or an "
                             "explicit API provider (judge stays on vLLM). "
                             "'gpt' is ambiguous and rejected — name the provider.")
    parser.add_argument("--output-dir", type=str, default="data/rl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Individual-specific
    parser.add_argument("--num-agents-per-query", type=int, default=None,
                        help="(deprecated) alias for --num-candidates")
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
    parser.add_argument("--num-candidates", type=int, default=None,
                        help="Number of candidate responses per query/persona (individual "
                             "mode) and directive candidates per branch point (branching "
                             f"mode). Default: {DEFAULT_NUM_CANDIDATES} (paper K=8)")
    parser.add_argument("--directive-retries", type=int, default=3,
                        help="Max retries for failed directive parsing (branching mode)")
    parser.add_argument("--stop-after-branch", action="store_true", default=False,
                        help="Stop generating turns for a meeting after its branch turn "
                             "(saves compute; preference pairs only use the branch turn)")

    # Scoring / filtering
    parser.add_argument("--score-margin", type=float, default=0.5,
                        help="Minimum score difference for preference pairs")
    parser.add_argument("--strict", action="store_true",
                        help="Terminate on the first generation/annotation failure. "
                             "Default (tolerant) mode records failure counters in "
                             "run_summary.json and continues.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true",
                              help="Reuse existing raw generations and annotations for "
                                   "this exact configuration and continue where the "
                                   "previous run stopped. Default: refuse to run when "
                                   "outputs already exist.")
    resume_group.add_argument("--overwrite", action="store_true",
                              help="Delete this configuration's existing raw/annotated "
                                   "files and dataset directory, then regenerate from "
                                   "scratch. Default: refuse to run when outputs exist.")
    parser.add_argument("--verify-pairs", action="store_true",
                        help="Use LLM to verify preference pairs")

    # vLLM server endpoints (no localhost defaults baked in — pass explicitly)
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="vLLM server URL for candidate/episode generation "
                             "(required with --generator-type vllm)")
    parser.add_argument("--max-concurrent-requests", type=int, default=32)

    parser.add_argument("--judge-server-url", type=str, default=None,
                        help="vLLM server URL for the annotation judge (required)")
    parser.add_argument("--judge-vllm-server-url", type=str, default=None,
                        help="(deprecated) alias for --judge-server-url")

    parser.add_argument("--judge-model-name", type=str, default=DEFAULT_LOCAL_JUDGE_MODEL,
                        help="Local judge model for annotation (paper's external judge is configured separately)")

    args = parser.parse_args()

    from src.arguments import resolve_deprecated
    resolve_deprecated(args, parser, {"judge_vllm_server_url": "judge_server_url",
                                      "num_agents_per_query": "num_candidates"})

    # Canonical K: candidate responses per query (individual) / branch factor
    # (branching). K=8 matches the paper; preference pairs need K >= 2.
    if args.num_candidates is None:
        args.num_candidates = DEFAULT_NUM_CANDIDATES
    if needs_min_candidates(args.mode, args.generation_mode):
        if args.num_candidates < 2:
            parser.error("--num-candidates must be >= 2 to form preference pairs "
                         f"(got {args.num_candidates})")

    if args.generator_type == "gpt":
        parser.error("--generator-type gpt is ambiguous; pass 'openai' or "
                     "'azure-openai' explicitly (no provider is ever inferred)")
    if args.generator_type == "vllm" and not args.vllm_server_url:
        parser.error("--generator-type vllm requires --vllm-server-url "
                     "(e.g. http://localhost:8008/v1)")
    if not args.judge_server_url:
        parser.error("annotation requires --judge-server-url "
                     "(e.g. http://localhost:8008/v1)")
    return args


# =====================================================================
# Main
# =====================================================================

def main():
    import os

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
    if args.generator_type in ("openai", "azure-openai"):
        llm = GPT(args.model_name, max_workers=args.max_concurrent_requests,
                  provider=args.generator_type)
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
        base_url=args.judge_server_url,
        max_concurrent_requests=args.max_concurrent_requests,
        enable_thinking=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Existing-output policy: default refuse / --resume / --overwrite ---
    paths = artifact_paths(args)
    apply_output_policy(args, paths)

    # Run-level config + failure counters (see src/utils/run_metadata.py).
    # Metadata lives in a per-run sidecar next to the artifact
    # (<dataset_dir>.runs/<run_id>/), so repeated runs never overwrite each
    # other's metadata and the HF dataset dir itself stays clean.
    run_id, run_meta_dir, started_at = start_run_metadata(args, paths)

    try:
        if args.mode in (INDIVIDUAL, PAIRED):
            individual_conversation_pipeline(args, llm, judge_llm)

        elif args.generation_mode == "branching":

            # This branch compares two episodes that have the same prefixes and differ starting from a branching point.
            group_conversation_pipeline_with_branching(args, llm, judge_llm)

        else:
            if args.generation_mode != "independent":
                raise ValueError(f"Unsupported generation mode: {args.generation_mode}")
            # This branch compares completely different episodes.
            group_conversation_pipeline_independent(args, llm, judge_llm)
    except StrictModeError as e:
        logger.error(str(e))
        finish_run_metadata(run_meta_dir, run_id, "failed_strict", args, started_at)
        raise SystemExit(1)

    finish_run_metadata(run_meta_dir, run_id, "completed", args, started_at)


if __name__ == "__main__":
    main()
