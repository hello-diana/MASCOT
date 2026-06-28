#!/usr/bin/env bash
# Generate MASCOT episodes (Director -> Speaker multi-turn conversations).
#
# Requires a running vLLM OpenAI-compatible server, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 python scripts/serve_vllm.py --model Qwen/Qwen3-8B --gpus 2 --port 8008
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8008/v1}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

# QMSum meeting-style episodes (Minutes Scribe / Decision Logger / Action Item Captain / Critic)
python -m src.eval.eval_mascot_qmsum \
    --dataset QMSum --split test \
    --directive-model "$MODEL" --agent-model "$MODEL" --share-model \
    --vllm-server-url "$VLLM_URL" \
    --output-dir "$OUTPUT_DIR"

# Emotional-support episodes on ESConv (Anchor / Catalyst / Beacon), with user simulation
python -m src.eval.eval_mascot_empathetic \
    --dataset ESConv --split test \
    --directive-model "$MODEL" --agent-model "$MODEL" --user-model "$MODEL" --share-model \
    --enable-user-simulation \
    --vllm-server-url "$VLLM_URL" \
    --output-dir "$OUTPUT_DIR"
