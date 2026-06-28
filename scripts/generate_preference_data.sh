#!/usr/bin/env bash
# Generate preference data for MASCOT training.
#   individual mode -> per-speaker candidate responses + rubric annotation -> preference pairs
#   group mode      -> branching multi-turn episodes + group-level rubric annotation
#
# Requires a running vLLM OpenAI-compatible server (see scripts/serve_vllm.py).
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8008/v1}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl}"

# Stage 1 data: individual candidates for the speaker reward model / GRPO
python -m src.dataset.generate_and_annotate \
    --mode individual --dataset ESConv \
    --model-name "$MODEL" \
    --num-agents-per-query 1 --score-margin 0.5 \
    --vllm-server-url "$VLLM_URL" \
    --output-dir "$OUTPUT_DIR"

# Stage 2 data: branching group episodes for director / group-reward training
python -m src.dataset.generate_and_annotate \
    --mode group --dataset QMSum \
    --model-name "$MODEL" --generation-mode branching \
    --max-samples 1000 --num-candidates 4 --num-turns 5 \
    --user-response-probability 0.5 --score-margin 0.5 --seed 42 \
    --vllm-server-url "$VLLM_URL" \
    --output-dir "$OUTPUT_DIR"
