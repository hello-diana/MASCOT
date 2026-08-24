#!/usr/bin/env bash
# Generate preference data for MASCOT training (canonical QMSum chain).
#   individual mode          -> per-speaker candidates + rubric annotation -> preference pairs
#   group mode (independent) -> episodes + group-level annotation -> group RM preference pairs
#   group mode (branching)   -> branching episodes -> Director GRPO prefix data / DPO pairs
#
# Requires a running vLLM OpenAI-compatible server (see scripts/serve_vllm.py).
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8008/v1}"
JUDGE_URL="${JUDGE_URL:-$VLLM_URL}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-QMSum}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl}"
# K candidate responses per query (paper setup: K=8; use 4 to halve generation cost)
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"

# Stage 1 data: individual candidates for the persona reward model / Speaker GRPO
# -> data/rl/individual_${DATASET}_Qwen3-8B (consumed by scripts/train_reward_model.sh)
python -m src.dataset.generate_and_annotate \
    --mode individual --dataset "$DATASET" \
    --model-name "$MODEL" \
    --num-candidates "$NUM_CANDIDATES" --score-margin 0.5 \
    --vllm-server-url "$VLLM_URL" \
    --judge-server-url "$JUDGE_URL" \
    --output-dir "$OUTPUT_DIR"

# Stage 2a data: independent group episodes for the group reward model
# -> data/rl/group_${DATASET}_Qwen3-8B
python -m src.dataset.generate_and_annotate \
    --mode group --dataset "$DATASET" \
    --model-name "$MODEL" --generation-mode independent \
    --num-episodes 2 --num-turns 5 \
    --user-response-probability 0.5 --score-margin 0.5 --seed 42 \
    --vllm-server-url "$VLLM_URL" \
    --judge-server-url "$JUDGE_URL" \
    --output-dir "$OUTPUT_DIR"

# Stage 2b data: branching group episodes for Director GRPO / DPO
# -> data/rl/branching_${DATASET}_Qwen3-8B (consumed by scripts/train_director.sh)
python -m src.dataset.generate_and_annotate \
    --mode group --dataset "$DATASET" \
    --model-name "$MODEL" --generation-mode branching \
    --max-samples 1000 --num-candidates "$NUM_CANDIDATES" --num-turns 5 \
    --user-response-probability 0.5 --score-margin 0.5 --seed 42 \
    --vllm-server-url "$VLLM_URL" \
    --judge-server-url "$JUDGE_URL" \
    --output-dir "$OUTPUT_DIR"
