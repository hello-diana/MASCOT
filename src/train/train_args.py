"""Shared CLI argument definitions for training scripts."""

import argparse
import os

from src.const import (
    DATASETS,
    DEFAULT_DIRECTOR_MODEL,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_PERSONA_REWARD_MODEL,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_USER_MODEL,
    QMSUM_DATASET,
)


def add_training_args(parser: argparse.ArgumentParser):
    """Add shared training hyperparameter arguments."""
    parser.add_argument("--model-name", type=str, default=DEFAULT_PERSONA_REWARD_MODEL,
                        help="Base model to train (reward-model trainer default)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for dataset splits and the trainer")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 means use epochs instead)")


def add_data_args(parser: argparse.ArgumentParser):
    """Add dataset loading arguments."""
    parser.add_argument("--dataset", type=str, default=QMSUM_DATASET,
                        choices=list(DATASETS),
                        help="Canonical dataset name (QMSum, Empathetic, ESConv)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Explicit data path (overrides --dataset). Comma-separated for multiple.")
    parser.add_argument("--data-source", type=str, default=None,
                        help="Path to source JSONL for building dataset (if --data-path doesn't exist)")


def resolve_data_path(args: argparse.Namespace, model_name: str = DEFAULT_SPEAKER_MODEL,
                      mode: str = "individual") -> str:
    """Build the data path from --data-path or --dataset + model_name.

    If --data-path is explicitly set, return it as-is. Otherwise return the
    canonical preference-data dir produced by ``generate_and_annotate``
    (``data/rl/{mode}_{dataset}_{model_short}``), falling back to the legacy
    training-side name ``data/rl/{dataset}_{model_short}`` if only that exists.
    """
    from src.utils.paths import artifact_dir, model_short

    if args.data_path is not None:
        return args.data_path
    canonical = artifact_dir("preference_data", dataset=args.dataset,
                             model_name=model_name, mode=mode)
    legacy = f"data/rl/{args.dataset}_{model_short(model_name)}"
    if not canonical.exists() and os.path.isdir(legacy):
        import logging
        logging.getLogger(__name__).warning(
            "Canonical dataset path %s not found; using legacy path %s",
            canonical, legacy)
        return legacy
    return str(canonical)


def add_lora_args(parser: argparse.ArgumentParser):
    """Add LoRA configuration arguments."""
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT,
                        help="LoRA dropout")


def add_vllm_args(parser: argparse.ArgumentParser):
    """Add vLLM inference arguments."""
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for inference (only for HuggingFace models)")
    parser.add_argument("--vllm-mode", type=str, choices=["server", "colocate"],
                        default="colocate",
                        help="vLLM mode: 'server' (connect to external vLLM server) or 'colocate' (spawn vLLM in-process)")
    parser.add_argument("--vllm-server-host", type=str, default="localhost",
                        help="vLLM server host (only used with --vllm-mode server)")
    parser.add_argument("--vllm-server-port", type=int, default=None,
                        help="vLLM server port (required with --vllm-mode server; "
                             "no default is baked in)")


def add_reward_model_args(parser: argparse.ArgumentParser):
    """Add reward-model-specific arguments."""
    parser.add_argument("--center-rewards-coefficient", type=float, default=0.0,
                        help="Coefficient for centering rewards toward 0 (0 to disable)")


def add_output_policy_args(parser: argparse.ArgumentParser):
    """Existing-output policy shared by all trainers.

    Default: refuse to write into a non-empty output directory.
    --overwrite replaces it; --resume continues from the last HF trainer
    checkpoint inside it (resume_from_checkpoint).
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true",
                       help="Write into a non-empty output directory "
                            "(default: refuse)")
    group.add_argument("--resume", action="store_true",
                       help="Resume from the last trainer checkpoint in the "
                            "output directory")


def add_rubric_args(parser: argparse.ArgumentParser):
    """Rubric-schema version selection (see src/rubrics.py).

    Training: --rubric-version selects the criteria text baked into
    reward-model prompts (default paper_v1). Scoring with an existing
    checkpoint: the version is resolved from the checkpoint's
    mascot_metadata.json; checkpoints without metadata REQUIRE an explicit
    --rubric-version or --assume-legacy-rubric (never silently assumed).
    """
    from src.rubrics import RUBRIC_VERSIONS
    parser.add_argument("--rubric-version", type=str, default=None,
                        choices=list(RUBRIC_VERSIONS),
                        help="Rubric schema version (default: paper_v1 for training; "
                             "resolved from checkpoint metadata for scoring)")
    parser.add_argument("--assume-legacy-rubric", action="store_true",
                        help="Score a metadata-less reward checkpoint with the "
                             "legacy_v0 rubric (explicit opt-in)")


def add_group_args(parser: argparse.ArgumentParser):
    """Group-training-specific arguments."""
    parser.add_argument("--mode", type=str,
                        choices=["train-director-grpo", "train-reward-model",
                                 "train-director-dpo", "train-director"],
                        default="train-director-grpo",
                        help="'train-director-grpo': GRPO on the Director (paper-default "
                             "Stage-2 method); 'train-reward-model': RewardTrainer on the "
                             "group reward model; 'train-director-dpo': offline DPO on the "
                             "Director (optional alternative, not the paper default); "
                             "'train-director' is a deprecated alias for train-director-grpo")
    parser.add_argument("--meeting-data", type=str, default=None,
                        help="Path to meeting data JSONL")
    parser.add_argument("--director-model-name", type=str, default=DEFAULT_DIRECTOR_MODEL,
                        help="Director (directive generator) policy model")
    parser.add_argument("--directive-model-name", type=str, default=None,
                        help="(deprecated) alias for --director-model-name")
    parser.add_argument("--speaker-model-name", type=str, default=DEFAULT_SPEAKER_MODEL,
                        help="Frozen Speaker model used during rollouts")
    parser.add_argument("--num-episode-turns", type=int, default=5)
    parser.add_argument("--user-model-name", type=str, default=DEFAULT_USER_MODEL,
                        help="Model for the simulated User during rollouts")
    parser.add_argument("--disable-user-simulation", action="store_true",
                        help="Disable the simulated User during Director-GRPO "
                             "rollouts (user simulation is ON by default)")
    parser.add_argument("--user-simulator-model", type=str, default=None,
                        help="(deprecated) alias for --user-model-name")
    parser.add_argument("--user-response-probability", type=float, default=0.3)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--rollout-server-url", type=str, default=None,
                        help="vLLM server URL for frozen rollout models "
                             "(e.g. http://localhost:8008/v1). "
                             "When set, uses VLLMServerModel instead of loading models in-process.")


def add_dpo_args(parser: argparse.ArgumentParser):
    """Offline-DPO-specific arguments (mode: train-director-dpo)."""
    parser.add_argument("--episode-dataset-path", type=str, default=None,
                        help="Path to pre-generated branching rollout HF Dataset "
                             "(e.g. data/rl/branching_QMSum_Qwen3-8B).")
    parser.add_argument("--dpo-data-path", type=str, default=None,
                        help="Optional cache path for the processed DPO dataset.")
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO beta (regularization strength).")
    parser.add_argument("--min-branches", type=int, default=2,
                        help="Minimum candidates per branching state. "
                             "Paired inputs always have 2; raised to gate forward-compat.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for train/eval split and trainer.")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Trainer checkpoint frequency.")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Trainer evaluation frequency.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable train/eval split.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap on DPO pairs (smoke-test convenience).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")


def build_group_parser() -> argparse.ArgumentParser:
    """Full CLI parser for group training (dispatcher + direct module CLIs).

    Argument names, defaults, choices, and help text are the Phase-1.5
    ``python -m src.train.train_group`` surface, unchanged.
    """
    from src.const import (DEFAULT_GROUP_REWARD_MODEL, EMPATHETIC_DATASET,
                           ESCONV_DATASET)

    parser = argparse.ArgumentParser(description="Group training: directive generator (GRPO/DPO) or group reward model")
    add_group_args(parser)
    add_vllm_args(parser)
    add_dpo_args(parser)
    add_rubric_args(parser)
    add_output_policy_args(parser)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=[QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET],
                        help="Canonical dataset name (auto-inferred from --episode-dataset-path if omitted).")

    # Shared training args
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length for reward-model tokenization "
                             "(longer episodes are filtered out)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    # RL-specific args (for train-director mode)
    parser.add_argument("--reward-model-path", type=str, default=None,
                        help="Path to trained group reward model. Defaults to the "
                             "canonical outputs/reward_model/group_reward_{dataset}_{model} "
                             "(falling back to the legacy group_reward_{model} if present)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 means use epochs instead)")

    # Reward model-specific args (for train-reward-model mode)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to HF Dataset directory with preference pairs "
                             "(from generate_and_annotate.py --mode group)")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Paths to episode JSONL files (legacy, for reward model training)")
    parser.add_argument("--group-reward-model-name", type=str, default=None,
                        help=f"Base model for group reward model training "
                             f"(default: {DEFAULT_GROUP_REWARD_MODEL})")
    parser.add_argument("--model-name", type=str, default=None,
                        help="(deprecated, ambiguous) alias for --group-reward-model-name")
    parser.add_argument("--score-margin", type=float, default=0.5,
                        help="Minimum score difference for preference pairs")
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.3,
                        help="GPU memory utilization for the frozen rollout vLLM engine")
    return parser


def finalize_group_args(args: argparse.Namespace,
                        parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Resolve deprecated aliases and map canonical names onto the historical
    attribute names the trainers read. Behavior identical to Phase 1.5."""
    from src.arguments import resolve_deprecated
    from src.const import DEFAULT_GROUP_REWARD_MODEL

    # Resolve deprecated aliases (conflicting values are a hard CLI error).
    # Role args carry canonical non-None defaults; a value equal to the
    # default is treated as "not explicitly set" for conflict detection.
    resolve_deprecated(args, parser, {
        "directive_model_name": "director_model_name",
        "user_simulator_model": "user_model_name",
        "model_name": "group_reward_model_name",
    }, new_defaults={
        "director_model_name": DEFAULT_DIRECTOR_MODEL,
        "user_model_name": DEFAULT_USER_MODEL,
    })
    if args.group_reward_model_name is None:
        args.group_reward_model_name = DEFAULT_GROUP_REWARD_MODEL
    # Internal code paths still read the historical attribute names.
    # User simulation is ON by default; --disable-user-simulation turns it off
    # (user_simulator_model=None is the internal "no user sim" signal).
    args.directive_model_name = args.director_model_name
    args.user_simulator_model = None if args.disable_user_simulation else args.user_model_name
    args.model_name = args.group_reward_model_name

    if args.mode == "train-director":
        import warnings
        warnings.warn("--mode train-director is deprecated; use train-director-grpo",
                      DeprecationWarning)
        args.mode = "train-director-grpo"
    return args


def add_rl_args(parser: argparse.ArgumentParser):
    """Add RL-specific arguments."""
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--algorithm", type=str, choices=["grpo"],
                        required=True,
                        help="RL algorithm to use: grpo (Group Relative Policy Optimization)")
    parser.add_argument("--reward-model-path", type=str,
                        required=True,
                        help="Path to trained reward model, such as `outputs/reward_model/Individual_Qwen3-0.6B`")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 means use epochs instead)")
    parser.add_argument("--max-completion-length", type=int, default=1024,
                        help="Maximum completion length for GRPO generation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for dataset split and the trainer")
