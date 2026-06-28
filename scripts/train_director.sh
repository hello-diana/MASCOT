#!/usr/bin/env bash
# Train the Director (directive generator) on branching group episodes.
#   train-director      -> GRPO with group-level rewards
#   train-reward-model  -> group reward model on annotated episodes
#   train-director-dpo  -> offline DPO on branching preference data
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-train-director-dpo}"
EPISODE_DATASET_PATH="${EPISODE_DATASET_PATH:-data/rl/branching_QMSum_Qwen3-8B}"

python -m src.train.train_group \
    --mode "$MODE" \
    --episode-dataset-path "$EPISODE_DATASET_PATH"
