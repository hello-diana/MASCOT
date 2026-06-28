#!/usr/bin/env python3
"""CLI wrapper for merging LoRA adapters into a full model.

The actual merge logic lives in src/utils/model_utils.merge_lora_to_full_model().
This script auto-detects the base model from adapter_config.json when --base-model
is not provided.
"""

import argparse
from pathlib import Path

from src.utils.model_utils import merge_lora_to_full_model, is_lora_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model for vLLM deployment"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name or path (auto-detected from adapter_config.json if omitted)"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., outputs/sft_models/final_model)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for merged model (default: <lora_checkpoint>_merged)"
    )
    args = parser.parse_args()

    # Auto-detect base model from adapter_config.json
    if args.base_model is None:
        if is_lora_checkpoint(args.lora_checkpoint):
            import json
            adapter_cfg = json.loads(
                (Path(args.lora_checkpoint) / "adapter_config.json").read_text()
            )
            args.base_model = adapter_cfg["base_model_name_or_path"]
            print(f"Auto-detected base model: {args.base_model}")
        else:
            parser.error(
                "No adapter_config.json found; please provide --base-model explicitly."
            )

    # Set default output path if not provided
    if args.output_path is None:
        lora_path = Path(args.lora_checkpoint)
        args.output_path = str(lora_path.parent / f"{lora_path.name}_merged")

    merge_lora_to_full_model(
        base_model_name=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
