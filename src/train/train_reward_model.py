#!/usr/bin/env python3
"""Revised Reward Model Training Script using TRL."""

import logging
import os
import argparse
import torch

from transformers import AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_from_disk, concatenate_datasets
from src.const import DEFAULT_SEED
from src.utils.data_utils import create_or_load_rl_dataset, clean_response
from src.utils.paths import artifact_dir, ensure_writable_dir
from src.utils.model_utils import create_lora_model, save_lora_adapter
from src.prompts.prompt_reward_model import get_prompt_persona_reward_model
from src.rubrics import (PAPER_V1, validate_dataset_rubric_version,
                         write_checkpoint_metadata)
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from dotenv import load_dotenv


def main(args: argparse.Namespace):
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Weights & Biases logging is fully optional: enabled only when
    # WANDB_PROJECT is set (entity comes from WANDB_ENTITY, if provided).
    if os.getenv("WANDB_PROJECT"):
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"rm-{args.model_name.split('/')[-1]}"
        )

    output_dir = str(artifact_dir("persona_reward", dataset=args.dataset,
                                  model_name=args.model_name,
                                  root=os.path.join(args.output_dir, "reward_model")))
    ensure_writable_dir(output_dir, overwrite=args.overwrite or args.resume)

    # Load base model with LoRA for reward modeling (scalar output)
    logger.info(f"Loading model and tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = create_lora_model(
        base_model_name=args.model_name,
        task_type="SEQ_CLS",
        num_labels=1,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Verify model outputs scalar reward
    assert model.config.num_labels == 1, (
        f"Expected num_labels=1 for scalar reward, got {model.config.num_labels}"
    )

    # Load dataset(s) — supports comma-separated paths for multi-dataset loading
    data_paths = [p.strip() for p in args.data_path.split(",") if p.strip()]
    datasets_list = []
    for dp in data_paths:
        if not os.path.exists(dp) and args.dataset:
            logger.info(f"Dataset path {dp} not found; building from --data-source {args.data_source}")
            ds = create_or_load_rl_dataset(dp, tokenizer, source_jsonl=args.data_source, dataset_name=args.dataset)
        else:
            ds = load_from_disk(dp)
        logger.info(f"Loaded {len(ds)} rows from {dp}")
        datasets_list.append(ds)

    if len(datasets_list) == 1:
        full_dataset = datasets_list[0]
    else:
        # Align columns before concatenation
        all_cols = set()
        for ds in datasets_list:
            all_cols.update(ds.column_names)
        # Only fill missing metadata columns; core reward columns must exist in all datasets
        core_cols = {"chosen", "rejected", "score_chosen", "score_rejected"}
        for i, ds in enumerate(datasets_list):
            for col in all_cols:
                if col not in ds.column_names and col not in core_cols:
                    datasets_list[i] = ds.add_column(col, [""] * len(ds))
        full_dataset = concatenate_datasets(datasets_list)
        logger.info(f"Combined {len(full_dataset)} total rows from {len(datasets_list)} datasets")

    # Rubric schema baked into the reward prompts: validated against the
    # dataset's recorded rubric_version (mixed/mismatched → hard error;
    # metadata-less data requires explicit --rubric-version/--assume-legacy-rubric).
    rubric_version = validate_dataset_rubric_version(
        full_dataset, requested=getattr(args, "rubric_version", None),
        assume_legacy=getattr(args, "assume_legacy_rubric", False),
        source=f"training data ({args.data_path})")
    logger.info(f"Training rubric version: {rubric_version}")

    if isinstance(full_dataset, dict) and "test" in full_dataset:
        train_dataset = full_dataset["train"]
        eval_dataset = full_dataset["test"]
    else:
        split_ds = full_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split_ds["train"]
        eval_dataset = split_ds["test"]

    # Wrap responses in evaluation prompt so the reward model learns to score
    # based on explicit persona/role/criteria (matching compute_reward at inference)
    def wrap_with_eval_prompt(example):
        persona_key = example.get("persona", "")
        if persona_key in EMPATHETIC_AGENTS:
            persona_spec = EMPATHETIC_AGENTS[persona_key]
        elif persona_key in AGENTS:
            persona_spec = AGENTS[persona_key]
        else:
            return example  # unknown persona, leave unchanged

        for col in ("chosen", "rejected"):
            messages = example[col]
            # Extract assistant response (last message)
            response_text = messages[-1]["content"]
            # Strip <think> tags for clean evaluation
            response_text = clean_response(response_text)
            # Wrap in evaluation prompt
            eval_prompt = get_prompt_persona_reward_model(persona_key, persona_spec, response_text,
                                                          rubric_version=rubric_version)
            example[col] = [{"role": "user", "content": eval_prompt}]
        return example

    train_dataset = train_dataset.map(wrap_with_eval_prompt)
    eval_dataset = eval_dataset.map(wrap_with_eval_prompt)
    logger.info("Wrapped responses in evaluation prompt (persona/role/criteria)")

    logger.info(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        max_length=args.max_length,
        gradient_checkpointing=True,
        center_rewards_coefficient=args.center_rewards_coefficient,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting reward model training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    save_lora_adapter(model, output_dir, tokenizer)
    write_checkpoint_metadata(output_dir, base_model=args.model_name,
                              reward_type="persona", rubric_version=rubric_version)

    logger.info(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    from src.train.train_args import (add_training_args, add_data_args, add_lora_args,
                                      add_output_policy_args, add_reward_model_args,
                                      add_rubric_args, resolve_data_path)

    parser = argparse.ArgumentParser(description="Train reward model")
    add_training_args(parser)
    add_data_args(parser)
    add_lora_args(parser)
    add_reward_model_args(parser)
    add_rubric_args(parser)
    add_output_policy_args(parser)
    args = parser.parse_args()
    args.data_path = resolve_data_path(args)
    main(args)
