#!/usr/bin/env bash
# Train the persona-aware speaker reward model on preference pairs
# produced by scripts/generate_preference_data.sh (individual mode).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.train.train_reward_model \
    --dataset "${DATASET:-QMSum}" \
    --model-name "${MODEL:-Qwen/Qwen3-0.6B}"
