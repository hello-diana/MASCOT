"""Group reward-model training (full-weight, behavior unchanged).

Moved verbatim from src/train/train_group.py (Phase-2 Part B). Supports the
HF-Dataset path (with dataset rubric-version validation) and the legacy
two-run JSONL conversion (src/data_generation/preference_pairs.py).

Output directory precedence: an explicit ``--output-dir`` is used exactly;
otherwise the canonical ``artifact_dir("group_reward", dataset, model)`` when a
``--dataset`` is given; otherwise the legacy dataset-less name.

Direct CLI: ``python -m src.train.group_reward`` (same arguments as the
``python -m src.train.train_group`` dispatcher, mode fixed to
train-reward-model).
"""

import logging
import math
import os
import statistics

from datasets import Dataset
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.const import DEFAULT_GROUP_REWARD_MODEL, DEFAULT_SEED
from src.data_generation.preference_pairs import \
    load_and_convert_to_preference_pairs
from src.rubrics import (PAPER_V1, validate_dataset_rubric_version,
                         write_checkpoint_metadata)
from src.train.callbacks import maybe_init_wandb_group
from src.utils.paths import artifact_dir, ensure_writable_dir, legacy_dir

logger = logging.getLogger(__name__)


def resolve_group_reward_output_dir(args, model_name):
    """Resolve the group reward-model checkpoint directory.

    Precedence:
      1. an explicit ``--output-dir`` (used verbatim);
      2. the canonical ``artifact_dir("group_reward", dataset, model)`` when a
         ``--dataset`` is provided;
      3. the legacy dataset-less name (``group_reward_{model}``).

    Returns ``(path, source)`` with source in {"explicit","canonical","legacy"}
    so the caller can log appropriately. Pure/testable: no side effects.
    """
    explicit = getattr(args, "output_dir", None)
    if explicit:
        return str(explicit), "explicit"
    dataset_name = getattr(args, "dataset", None)
    if dataset_name:
        return str(artifact_dir("group_reward", dataset=dataset_name,
                                model_name=model_name)), "canonical"
    return str(legacy_dir("group_reward", dataset="", model_name=model_name)), "legacy"


def _reward_pair_length(example, tokenizer):
    """Length RewardTrainer compares against ``max_length`` for one pair.

    Mirrors TRL's conversational tokenization: each of ``chosen`` / ``rejected``
    is rendered with ``apply_chat_template(..., return_dict=True)`` and a pair is
    kept iff BOTH token lengths are <= max_length. The retention-deciding length
    is therefore ``max(len(chosen_ids), len(rejected_ids))``.

    Raises ValueError on malformed required fields (non-list/empty chosen or
    rejected) so the preflight fails clearly instead of deep inside the trainer.
    """
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if not isinstance(chosen, list) or not chosen \
            or not isinstance(rejected, list) or not rejected:
        raise ValueError(
            "Malformed preference pair: 'chosen' and 'rejected' must be "
            f"non-empty message lists, got chosen={type(chosen).__name__} "
            f"rejected={type(rejected).__name__}.")
    chosen_ids = tokenizer.apply_chat_template(chosen, return_dict=True)["input_ids"]
    rejected_ids = tokenizer.apply_chat_template(rejected, return_dict=True)["input_ids"]
    return max(len(chosen_ids), len(rejected_ids))


def preflight_reward_length_filter(dataset, tokenizer, max_length, source="dataset"):
    """Report the effect of the ``max_length`` filter and fail early if it would
    leave zero training pairs.

    Uses the same tokenizer, prompt representation, and length convention as
    ``RewardTrainer`` (see ``_reward_pair_length``); it does NOT change which
    examples are retained. Logs original/retained counts, the configured
    ``max_length``, and min/median/p95/max observed pair lengths.

    Raises:
        ValueError: distinct messages for an already-empty input dataset and for
            the case where the filter would drop every pair.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError(
            f"Group reward-model training dataset is empty ({source}): 0 "
            "preference pairs to train on. Check the group preference-pair "
            "generation step (generate_and_annotate.py --mode group).")

    lengths = [_reward_pair_length(ex, tokenizer) for ex in dataset]
    if max_length is None:
        logger.info("Reward-length preflight: %d pairs, no --max-length filter "
                    "(min=%d, median=%d, max=%d tokens)", n, min(lengths),
                    int(statistics.median(lengths)), max(lengths))
        return n

    retained = sum(1 for L in lengths if L <= max_length)
    ordered = sorted(lengths)
    p95 = ordered[max(0, min(n - 1, math.ceil(0.95 * n) - 1))]
    logger.info(
        "Reward-length preflight: %d pairs, %d retained at --max-length=%d "
        "(observed pair token lengths: min=%d, median=%d, p95=%d, max=%d)",
        n, retained, max_length, min(lengths), int(statistics.median(lengths)),
        p95, max(lengths))

    if retained == 0:
        raise ValueError(
            f"All {n} group reward-model preference pairs exceed "
            f"--max-length={max_length} tokens, so RewardTrainer would filter "
            f"every pair and train on 0 samples (observed pair token lengths: "
            f"min={min(lengths)}, median={int(statistics.median(lengths))}, "
            f"p95={p95}, max={max(lengths)}). Raise --max-length "
            f"(e.g. to >= {min(lengths)} to keep at least one pair) or inspect "
            f"the formatted preference examples in {source}.")
    return retained


def train_group_reward_model(args):
    """Train group reward model using TRL RewardTrainer.

    Absorbed from the former ``src/group/train_group.py``.

    Supports two data input modes:
        1. ``--data-path``: HF Dataset directory (from ``generate_and_annotate.py --mode group``).
           Already contains preference pairs with columns: chosen, rejected, score_chosen, score_rejected.
        2. ``--data``: Two raw episode JSONL files (legacy 2-JSONL format).
           Converted to preference pairs via ``load_and_convert_to_preference_pairs()``.
    """

    model_name = getattr(args, 'model_name', DEFAULT_GROUP_REWARD_MODEL)
    # Rubric schema baked into training prompts (recorded in checkpoint metadata).
    rubric_version = getattr(args, 'rubric_version', None) or PAPER_V1
    data_path = getattr(args, 'data_path', None)
    data_paths = getattr(args, 'data', None)
    batch_size = getattr(args, 'batch_size', 16)
    epochs = getattr(args, 'epochs', 3)
    lr = getattr(args, 'learning_rate', 1e-5)
    score_margin = getattr(args, 'score_margin', 0.5)

    # Output dir precedence: explicit --output-dir > canonical > legacy.
    output_model_name, _dir_source = resolve_group_reward_output_dir(args, model_name)
    if _dir_source == "explicit":
        logger.info("Using explicit --output-dir for the group reward model: %s",
                    output_model_name)
    elif _dir_source == "legacy":
        logger.warning("No --dataset or --output-dir provided; using legacy "
                       "output name %s", output_model_name)
    ensure_writable_dir(output_model_name, overwrite=args.overwrite or args.resume)

    logger.info(f"=== Group Reward Model Training ===")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Output: {output_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if data_path:
        # New path: load pre-formed preference pairs from generate_and_annotate.py.
        # Validate the dataset's recorded rubric_version before training
        # (mixed/mismatched → hard error; metadata-less requires explicit opt-in).
        dataset = Dataset.load_from_disk(data_path)
        logger.info(f"Loaded HF Dataset from {data_path} ({len(dataset)} rows)")
        rubric_version = validate_dataset_rubric_version(
            dataset, requested=getattr(args, "rubric_version", None),
            assume_legacy=getattr(args, "assume_legacy_rubric", False),
            source=f"training data ({data_path})")
        logger.info(f"Training rubric version: {rubric_version}")
    elif data_paths:
        # Legacy path: convert 2 JSONL files into preference pairs. The prompts
        # are built in-process with the resolved rubric version, so no
        # per-row validation applies (the rows are created here).
        dataset = load_and_convert_to_preference_pairs(data_paths, score_margin,
                                                       rubric_version=rubric_version)
    else:
        raise ValueError(
            "train-reward-model mode requires --data-path (HF Dataset from "
            "generate_and_annotate.py --mode group) or --data (two episode JSONL files)")

    # Preflight the max_length filter with RewardTrainer's own tokenization so
    # a fully-filtered (or empty) dataset fails with an actionable message
    # instead of TRL's opaque "num_samples=0".
    max_length = getattr(args, 'max_length', 2048)
    preflight_reward_length_filter(
        dataset, tokenizer, max_length,
        source=data_path or (", ".join(data_paths) if data_paths else "dataset"))

    reward_config = RewardConfig(
        output_dir=output_model_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=getattr(args, 'max_steps', -1) or -1,
        seed=getattr(args, 'seed', DEFAULT_SEED),
        learning_rate=lr,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        max_length=max_length,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    )

    logger.info("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model_name,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    logger.info(f"Saving model to {output_model_name}")
    trainer.save_model(output_model_name)
    tokenizer.save_pretrained(output_model_name)
    write_checkpoint_metadata(output_model_name, base_model=model_name,
                              reward_type="group", rubric_version=rubric_version)
    logger.info("Training completed!")


def main():
    """Direct CLI for group reward-model training (mode fixed to train-reward-model)."""
    from dotenv import load_dotenv

    from src.train.train_args import build_group_parser, finalize_group_args

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = build_group_parser()
    parser.set_defaults(mode="train-reward-model")
    args = finalize_group_args(parser.parse_args(), parser)
    if args.mode != "train-reward-model":
        parser.error(f"python -m src.train.group_reward trains the group reward "
                     f"model only (got --mode {args.mode}); use "
                     "python -m src.train.train_group for other modes")
    maybe_init_wandb_group(args.mode)
    train_group_reward_model(args)


if __name__ == "__main__":
    main()
