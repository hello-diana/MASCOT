#!/usr/bin/env bash
# Train the persona-aware speaker reward model on the preference pairs
# produced by scripts/generate_preference_data.sh (individual mode).
# Consumes the EXACT artifact that script writes:
#   data/rl/individual_${DATASET}_${GEN_MODEL_SHORT}
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${DATASET:-QMSum}"
GEN_MODEL_SHORT="${GEN_MODEL_SHORT:-Qwen3-8B}"
DATA_PATH="${DATA_PATH:-data/rl/individual_${DATASET}_${GEN_MODEL_SHORT}}"

python -m src.train.train_reward_model \
    --dataset "$DATASET" \
    --model-name "${MODEL:-Qwen/Qwen3-0.6B}" \
    --data-path "$DATA_PATH"
