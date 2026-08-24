#!/usr/bin/env bash
# Train the Speaker policy with GRPO against the trained reward model.
# Run from the repository root.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-23456}"

DATASET="${DATASET:-QMSum}"
GEN_MODEL_SHORT="${GEN_MODEL_SHORT:-Qwen3-8B}"
DATA_PATH="${DATA_PATH:-data/rl/individual_${DATASET}_${GEN_MODEL_SHORT}}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-outputs/reward_model/Individual_${DATASET}_Qwen3-0.6B}"

python -m src.train.train_rl --algorithm grpo \
    --dataset "$DATASET" \
    --data-path "$DATA_PATH" \
    --reward-model-path "$REWARD_MODEL_PATH" \
    --max-completion-length 1024 --batch-size 8 --num-epochs 1
