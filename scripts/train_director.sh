#!/usr/bin/env bash
# Train the Director (directive generator).
#   train-director-grpo -> GRPO with group-level rewards (paper-default method)
#   train-director-dpo  -> offline DPO on branching preference data (optional alternative)
#   train-reward-model  -> group reward model on annotated episodes
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-train-director-grpo}"
DATASET="${DATASET:-QMSum}"
DIRECTOR_MODEL="${DIRECTOR_MODEL:-Qwen/Qwen3-8B}"
EPISODE_DATASET_PATH="${EPISODE_DATASET_PATH:-data/rl/branching_${DATASET}_Qwen3-8B}"

if [ "$MODE" = "train-director-dpo" ]; then
    python -m src.train.train_group \
        --mode "$MODE" \
        --dataset "$DATASET" \
        --director-model-name "$DIRECTOR_MODEL" \
        --episode-dataset-path "$EPISODE_DATASET_PATH"
else
    # GRPO (paper default): consumes the branching episode dataset from
    # scripts/generate_preference_data.sh; the group reward model is resolved
    # automatically from outputs/reward_model/group_reward_${DATASET}_Qwen3-0.6B.
    ROLLOUT_URL="${ROLLOUT_URL:-http://localhost:8008/v1}"
    python -m src.train.train_group \
        --mode "$MODE" \
        --dataset "$DATASET" \
        --director-model-name "$DIRECTOR_MODEL" \
        --episode-dataset-path "$EPISODE_DATASET_PATH" \
        --rollout-server-url "$ROLLOUT_URL" \
        --use-vllm
fi
