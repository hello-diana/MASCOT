"""Shared CLI argument definitions for training scripts."""

import argparse
import os

from src.const import DATASETS, QMSUM_DATASET


def add_training_args(parser: argparse.ArgumentParser):
    """Add shared training hyperparameter arguments."""
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
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


def add_data_args(parser: argparse.ArgumentParser):
    """Add dataset loading arguments."""
    parser.add_argument("--dataset", type=str, default=QMSUM_DATASET,
                        choices=list(DATASETS),
                        help="Canonical dataset name (QMSum, Empathetic, ESConv)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Explicit data path (overrides --dataset). Comma-separated for multiple.")
    parser.add_argument("--data-source", type=str, default=None,
                        help="Path to source JSONL for building dataset (if --data-path doesn't exist)")


def resolve_data_path(args: argparse.Namespace, model_name: str = "Qwen3-8B") -> str:
    """Build the data path from --data-path or --dataset + model_name.

    If --data-path is explicitly set, return it as-is.
    Otherwise, construct ``data/rl/{dataset}_{model_short_name}``.
    """
    if args.data_path is not None:
        return args.data_path
    model_short = os.path.basename(model_name)
    return f"data/rl/{args.dataset}_{model_short}"


def add_lora_args(parser: argparse.ArgumentParser):
    """Add LoRA configuration arguments."""
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
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
    parser.add_argument("--vllm-server-port", type=int, default=8609,
                        help="vLLM server port (only used with --vllm-mode server)")


def add_reward_model_args(parser: argparse.ArgumentParser):
    """Add reward-model-specific arguments."""
    parser.add_argument("--center-rewards-coefficient", type=float, default=0.0,
                        help="Coefficient for centering rewards toward 0 (0 to disable)")


def add_group_args(parser: argparse.ArgumentParser):
    """Group-training-specific arguments."""
    parser.add_argument("--mode", type=str,
                        choices=["train-director", "train-reward-model", "train-director-dpo"],
                        default="train-director",
                        help="'train-director': GRPO on directive generator; "
                             "'train-reward-model': RewardTrainer on group reward model; "
                             "'train-director-dpo': offline DPO on directive generator")
    parser.add_argument("--meeting-data", type=str, default=None,
                        help="Path to meeting data JSONL")
    parser.add_argument("--directive-model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--speaker-model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num-episode-turns", type=int, default=5)
    parser.add_argument("--user-simulator-model", type=str, default=None,
                        help="Model for simulated user (if None, no user sim)")
    parser.add_argument("--user-response-probability", type=float, default=0.3)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--rollout-server-url", type=str, default=None,
                        help="vLLM server URL for frozen rollout models "
                             "(e.g. http://localhost:8032/v1). "
                             "When set, uses VLLMServerModel instead of loading models in-process.")


def add_dpo_args(parser: argparse.ArgumentParser):
    """Offline-DPO-specific arguments (mode: train-director-dpo)."""
    parser.add_argument("--episode-dataset-path", type=str, default=None,
                        help="Path to pre-generated branching rollout HF Dataset "
                             "(e.g. data/rl/branching_QMSum_Qwen3-32B).")
    parser.add_argument("--dpo-data-path", type=str, default=None,
                        help="Optional cache path for the processed DPO dataset.")
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO beta (regularization strength).")
    parser.add_argument("--min-branches", type=int, default=2,
                        help="Minimum candidates per branching state. "
                             "Paired inputs always have 2; raised to gate forward-compat.")
    parser.add_argument("--seed", type=int, default=42,
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
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")


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
