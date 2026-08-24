#!/usr/bin/env python3
"""Group training dispatcher: Director (GRPO/DPO) and group reward model.

This module is the documented CLI (``python -m src.train.train_group``) and
the backward-compatible import surface; the implementations live in
src/train/{group_rollout,group_data,group_reward,director_grpo,director_dpo}.py
(Phase-2 split). Direct module CLIs are also available:

    python -m src.train.director_grpo
    python -m src.train.director_dpo
    python -m src.train.group_reward

Usage:
    # Train directive generator (GRPO, paper default)
    python -m src.train.train_group --mode train-director-grpo \
        --dataset QMSum \
        --episode-dataset-path data/rl/branching_QMSum_Qwen3-8B \
        --rollout-server-url http://localhost:8008/v1 --use-vllm

    # Train group reward model
    python -m src.train.train_group --mode train-reward-model \
        --dataset QMSum --data-path data/rl/group_QMSum_Qwen3-8B
"""

import logging

from dotenv import load_dotenv

from config.model_config import ModelConfig

# ---------------------------------------------------------------------------
# Backward-compatible re-exports: everything previously defined in this
# module keeps its historical (including private) name.
# ---------------------------------------------------------------------------
from src.train.group_rollout import (  # noqa: F401
    build_prefix_turns,
    compute_group_reward,
    rollout_suffix_turns,
    _generate_directives_batch,
    _random_agent,
    _user_sim_prompt,
)
from src.train.group_data import (  # noqa: F401
    build_director_dpo_dataset,
    create_group_dataset_from_episodes,
    create_group_training_dataset,
    load_and_convert_to_preference_pairs,
    resolve_branching_episode_dirs,
    _extract_directive_text,
    _resolve_dataset_name,
)
from src.train.group_reward import (  # noqa: F401
    train_group_reward_model,
    resolve_group_reward_output_dir,
    preflight_reward_length_filter,
)
from src.train.director_grpo import train_group_grpo  # noqa: F401
from src.train.director_dpo import train_group_dpo  # noqa: F401
from src.train.callbacks import (  # noqa: F401
    RewardLoggingCallback,
    maybe_init_wandb_group,
)
from src.train.train_args import build_group_parser, finalize_group_args

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    parser = build_group_parser()
    args = finalize_group_args(parser.parse_args(), parser)

    # Initialize W&B if configured
    maybe_init_wandb_group(args.mode)

    if args.mode == "train-director-grpo":
        config = ModelConfig(
            base_model=args.director_model_name,
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
