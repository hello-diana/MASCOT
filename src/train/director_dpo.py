"""Director DPO training (optional alternative to the paper-default GRPO).

Moved verbatim from src/train/train_group.py (Phase-2 Part B). Dataset
semantics, training configuration, and the LoRA adapter format are unchanged.

Direct CLI: ``python -m src.train.director_dpo`` (same arguments as the
``python -m src.train.train_group`` dispatcher, mode fixed to
train-director-dpo).
"""

import logging
import os

from datasets import Dataset
from transformers import AutoTokenizer

from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.const import EMPATHETIC_DATASET, ESCONV_DATASET
from src.train.callbacks import maybe_init_wandb_group
from src.train.group_data import _resolve_dataset_name, build_director_dpo_dataset
from src.utils.model_utils import create_lora_model, save_lora_adapter
from src.utils.paths import artifact_dir, ensure_writable_dir

logger = logging.getLogger(__name__)


def train_group_dpo(args):
    from trl import DPOTrainer, DPOConfig

    if not args.episode_dataset_path:
        raise ValueError("--episode-dataset-path is required for train-director-dpo mode")

    dataset_name = _resolve_dataset_name(args)
    agents_dict = EMPATHETIC_AGENTS if dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    logger.info(f"Offline DPO: dataset_name={dataset_name}, agents={list(agents_dict.keys())}")

    # Build or load the DPO dataset
    if args.dpo_data_path and os.path.exists(args.dpo_data_path):
        logger.info(f"Loading cached DPO dataset from {args.dpo_data_path}")
        dpo_dataset = Dataset.load_from_disk(args.dpo_data_path)
    else:
        dpo_dataset = build_director_dpo_dataset(
            episode_dataset_path=args.episode_dataset_path,
            agents=agents_dict,
            dataset_name=dataset_name,
            score_margin=args.score_margin,
            min_branches=args.min_branches,
        )
        if args.dpo_data_path:
            os.makedirs(os.path.dirname(args.dpo_data_path) or ".", exist_ok=True)
            dpo_dataset.save_to_disk(args.dpo_data_path)
            logger.info(f"Cached processed DPO dataset to {args.dpo_data_path}")

    if args.max_pairs is not None and len(dpo_dataset) > args.max_pairs:
        logger.info(f"Truncating DPO dataset to {args.max_pairs} pairs (--max-pairs)")
        dpo_dataset = dpo_dataset.select(range(args.max_pairs))

    # Train/eval split
    eval_dataset = None
    if not args.no_eval and len(dpo_dataset) >= 10:
        split = dpo_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        logger.info(f"Split: train={len(train_dataset)}, eval={len(eval_dataset)}")
    else:
        train_dataset = dpo_dataset
        logger.info(f"No eval split (size={len(dpo_dataset)}, no_eval={args.no_eval})")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.directive_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # LoRA policy model
    directive_model = create_lora_model(
        base_model_name=args.directive_model_name,
        task_type="CAUSAL_LM",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device_map="auto",
    )

    # Output dir
    output_dir = args.output_dir or str(
        artifact_dir("director_dpo", dataset=dataset_name,
                     model_name=args.directive_model_name))
    ensure_writable_dir(output_dir, overwrite=args.overwrite or args.resume)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=getattr(args, 'max_steps', -1) or -1,
        beta=args.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=directive_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    save_lora_adapter(directive_model, output_dir, tokenizer)
    logger.info(f"Saved LoRA adapter + tokenizer to {output_dir}")


def main():
    """Direct CLI for Director DPO (mode fixed to train-director-dpo)."""
    from dotenv import load_dotenv

    from src.train.train_args import build_group_parser, finalize_group_args

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = build_group_parser()
    parser.set_defaults(mode="train-director-dpo")
    args = finalize_group_args(parser.parse_args(), parser)
    if args.mode != "train-director-dpo":
        parser.error(f"python -m src.train.director_dpo trains with DPO only "
                     f"(got --mode {args.mode}); use python -m src.train.train_group "
                     "for other modes")
    maybe_init_wandb_group(args.mode)
    train_group_dpo(args)


if __name__ == "__main__":
    main()
