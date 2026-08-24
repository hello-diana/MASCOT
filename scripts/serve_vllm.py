#!/usr/bin/env python3
"""Parameterized vLLM OpenAI-server launcher.

Examples:
    # Explicit device ids (sets CUDA_VISIBLE_DEVICES, TP inferred from count):
    python scripts/serve_vllm.py --model Qwen/Qwen3-8B --gpus 0,1 --port 8008

    # GPU count over already-visible devices (CUDA_VISIBLE_DEVICES set outside):
    CUDA_VISIBLE_DEVICES=0,1 python scripts/serve_vllm.py \
        --model Qwen/Qwen3-8B --gpus 2 --port 8008
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch a vLLM OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--gpus", required=True,
                        help="Comma-separated CUDA device ids (e.g. '0,1'), or — when "
                             "CUDA_VISIBLE_DEVICES is already set — a bare integer GPU "
                             "count; tensor-parallel size is inferred either way")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--reasoning-parser", default=None,
                        help="Optional reasoning parser (e.g. 'qwen3')")
    parser.add_argument("--lora-modules", nargs="+", default=None, metavar="NAME=PATH",
                        help="Serve LoRA adapters as additional model ids alongside "
                             "the base model, e.g. "
                             "--lora-modules mascot-director=outputs/llm/grpo_Director_QMSum_Qwen3-8B "
                             "mascot-speaker=outputs/llm/grpo_Speaker_individual_QMSum_Qwen3-8B. "
                             "Implies --enable-lora; --max-loras and --max-lora-rank "
                             "are inferred (override via passthrough flags).")
    args, extra = parser.parse_known_args()

    lora_flags = []
    if args.lora_modules:
        import json
        max_rank = 16
        for pair in args.lora_modules:
            if "=" not in pair:
                parser.error(f"--lora-modules entries must be NAME=PATH, got {pair!r}")
            name, path = pair.split("=", 1)
            cfg_path = os.path.join(path, "adapter_config.json")
            if not os.path.isfile(cfg_path):
                parser.error(f"LoRA module {name!r}: {cfg_path} not found — "
                             "point at a saved adapter directory")
            with open(cfg_path) as f:
                max_rank = max(max_rank, int(json.load(f).get("r", 16)))
        lora_flags = [
            "--enable-lora",
            "--lora-modules", *args.lora_modules,
            "--max-loras", str(len(args.lora_modules)),
            "--max-lora-rank", str(max_rank),
        ]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and "," not in args.gpus and int(args.gpus) <= len(visible.split(",")):
        # Bare count over externally selected devices
        tp_size = int(args.gpus)
        args.gpus = visible
    else:
        tp_size = len(args.gpus.split(","))
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(tp_size),
        "--port", str(args.port),
    ]
    if args.reasoning_parser:
        cmd += ["--reasoning-parser", args.reasoning_parser]
    cmd += lora_flags
    cmd += extra  # pass any additional vLLM flags through (explicit flags win)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    print("Launching:", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={args.gpus})")
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
