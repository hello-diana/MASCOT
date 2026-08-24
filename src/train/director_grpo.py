"""Director GRPO training (paper-default Stage-2 method).

Moved verbatim from src/train/train_group.py (Phase-2 Part B). The GRPO
objective, rollout order, reward composition, same-k batching, guarded TRL
compatibility patch, and LoRA adapter saving are unchanged.

Direct CLI: ``python -m src.train.director_grpo`` (same arguments as the
``python -m src.train.train_group`` dispatcher, mode fixed to
train-director-grpo).
"""

import glob
import json
import logging
import os
import random
from functools import partial

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config.model_config import ModelConfig
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.const import (DEFAULT_GROUP_REWARD_MODEL, DEFAULT_SEED,
                       EMPATHETIC_DATASET, ESCONV_DATASET, QMSUM_DATASET)
from src.models.llm_models import VLLMModel
from src.train.callbacks import RewardLoggingCallback, maybe_init_wandb_group
from src.train.group_data import (create_group_dataset_from_episodes,
                                  create_group_training_dataset,
                                  resolve_branching_episode_dirs)
from src.train.group_rollout import compute_group_reward
from src.rubrics import resolve_checkpoint_rubric_version
from src.utils.model_utils import (create_lora_model, load_model_auto,
                                   save_lora_adapter)
from src.utils.paths import (artifact_dir, ensure_writable_dir,
                             file_fingerprint, group_prefix_cache_path,
                             resolve_artifact_dir, rollout_fingerprint)

logger = logging.getLogger(__name__)


def train_group_grpo(config: ModelConfig, args):
    """Train directive generator using GRPO with random turn selection.

    The Director (directive generator; canonically Qwen3-8B, trained as a
    LoRA policy) produces a directive at
    a randomly selected turn k. A frozen Speaker model (canonically Qwen3-8B) executes
    agent turns and generates directives for all other turns.

    Turn structure per episode:
        prefix (0..k-1 frozen turns) + training turn k + suffix (frozen)
        Total = k + num_episode_turns turns.
        Only the first num_episode_turns rounds are used for reward scoring.
    """
    logger.info("=== Group Director Training with GRPO ===")

    # --- Tokenizer for the directive generator (policy) ---
    tokenizer = AutoTokenizer.from_pretrained(args.directive_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # --- Frozen speaker model (also used as directive generator for non-training turns) ---
    # Priority: external vLLM server > HuggingFace fallback (colocate) > local vLLM
    if args.rollout_server_url:
        # Use external vLLM server — single connection shared by all rollout agents
        from src.models.llm_models import VLLMServerModel
        logger.info(
            f"Initializing all frozen rollout agents via vLLM server at "
            f"{args.rollout_server_url} (model: {args.speaker_model_name})"
        )
        _server = VLLMServerModel(
            model_name=args.speaker_model_name,
            base_url=args.rollout_server_url,
        )
        agents_vllm = _server
    elif args.use_vllm:
        # When GRPOTrainer uses colocated vLLM (--use-vllm), we CANNOT run a second
        # vLLM instance in the same process (CUDA context conflict). Fall back to
        # HuggingFace-based Qwen model for the frozen speaker in that case.
        from src.models.llm_models import Qwen
        logger.info(
            f"Initializing frozen speaker model ({args.speaker_model_name}) "
            "via HuggingFace (vLLM colocate mode active, avoiding dual vLLM)"
        )
        agents_vllm = Qwen(model_name=args.speaker_model_name)
    else:
        rollout_gpu_util = args.rollout_gpu_memory_utilization
        logger.info(
            f"Initializing frozen speaker model ({args.speaker_model_name}) "
            f"via vLLM (gpu_memory_utilization={rollout_gpu_util})"
        )
        agents_vllm = VLLMModel(
            model_name=args.speaker_model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=rollout_gpu_util,
        )

    # The frozen speaker model doubles as the directive generator for non-training turns
    directive_generator = agents_vllm

    # Determine training GPU: VLLMModel takes cuda:0, training models go on last GPU
    num_gpus = torch.cuda.device_count()
    train_device = num_gpus - 1 if num_gpus > 1 else 0
    logger.info(f"GPU layout: policy/vLLM on cuda:0, reward model on cuda:{train_device} ({num_gpus} GPUs visible)")

    # --- User simulator (optional) ---
    user_simulator = None
    if args.user_simulator_model:
        if args.rollout_server_url:
            # All rollout agents share the same vLLM server
            user_simulator = agents_vllm
            logger.info("User simulator shares vLLM server instance")
        elif args.user_simulator_model == args.speaker_model_name:
            user_simulator = agents_vllm
            logger.info("User simulator shares speaker model instance")
        else:
            logger.info(f"Loading separate user simulator model: {args.user_simulator_model}")
            if args.use_vllm:
                from src.models.llm_models import Qwen
                user_simulator = Qwen(model_name=args.user_simulator_model)
            else:
                user_simulator = VLLMModel(
                    model_name=args.user_simulator_model,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.5,
                )

    # --- Dataset ---
    # Priority: explicit --episode-dataset-path > auto-discovered branching
    # episode dataset > auto-cache > generate from scratch

    _base = os.path.basename(args.meeting_data or "")
    if args.dataset:
        dataset_name = args.dataset
    elif ESCONV_DATASET in _base:
        dataset_name = ESCONV_DATASET
    elif EMPATHETIC_DATASET in _base:
        dataset_name = EMPATHETIC_DATASET
    else:
        dataset_name = QMSUM_DATASET
    agents_dict = EMPATHETIC_AGENTS if dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS

    # Pre-generated branching episode data — DETERMINISTIC selection (see
    # resolve_branching_episode_dirs).
    episode_dirs = resolve_branching_episode_dirs(args, dataset_name)

    # Fingerprinted prefix cache: the key covers the complete rollout
    # configuration + source-data identity. Pre-fingerprint cache dirs are
    # never loaded (they are ignored and eventually regenerated).
    fp_payload = {
        "speaker_model": args.speaker_model_name,
        "director_model": args.directive_model_name,
        "num_episode_turns": args.num_episode_turns,
        "user_model": args.user_simulator_model,
        "user_response_probability": args.user_response_probability,
        "seed": getattr(args, "seed", DEFAULT_SEED),
        "dataset": dataset_name,
        "source": file_fingerprint(args.meeting_data) if args.meeting_data else None,
    }
    fp12 = rollout_fingerprint(fp_payload)
    cache_path = str(group_prefix_cache_path(dataset_name, args.speaker_model_name,
                                             args.num_episode_turns, fp12))
    legacy_caches = [d for d in glob.glob("data/group_prefix/*turns")
                     if os.path.isdir(d)]
    if legacy_caches:
        logger.info(f"Ignoring {len(legacy_caches)} legacy pre-fingerprint prefix "
                    "cache dir(s) (delete manually to reclaim space): "
                    f"{legacy_caches[:3]}")

    if episode_dirs:
        logger.info(f"Using branching episode dataset: {episode_dirs[0]}")
        dataset = create_group_dataset_from_episodes(episode_dirs[0], agents_dict, dataset_name)
    elif os.path.exists(cache_path):
        logger.info(f"Loading cached prefix dataset from {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
        k_dist = {}
        for k in dataset["train_k"]:
            k_dist[k] = k_dist.get(k, 0) + 1
        logger.info(f"Loaded {len(dataset)} cached samples (k distribution: {dict(sorted(k_dist.items()))})")
    else:
        dataset = create_group_training_dataset(
            tokenizer=tokenizer,
            meeting_data_path=args.meeting_data,
            agents=agents_dict,
            num_episode_turns=args.num_episode_turns,
            directive_generator=directive_generator,
            agent_generator=agents_vllm,
            user_simulator=user_simulator,
            user_response_probability=args.user_response_probability,
            dataset_name=dataset_name,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dataset.save_to_disk(cache_path)
        # Auditability: record what this cache was built from.
        with open(os.path.join(cache_path, "fingerprint.json"), "w") as f:
            json.dump({"fingerprint": fp12, "payload": fp_payload}, f, indent=2, default=str)
        logger.info(f"Saved prefix dataset cache to {cache_path} (fingerprint {fp12})")

    # Group by train_k, form complete batches, shuffle, append remainders
    batch_size = args.batch_size
    k_unique = sorted(set(dataset["train_k"]))
    complete_chunks = []  # list of Dataset slices, each of size batch_size
    remainders = []       # leftover indices

    for k_val in k_unique:
        indices = [i for i, k in enumerate(dataset["train_k"]) if k == k_val]
        n_complete = (len(indices) // batch_size) * batch_size
        # Split into complete batches
        for start in range(0, n_complete, batch_size):
            complete_chunks.append(dataset.select(indices[start:start + batch_size]))
        # Collect remainders
        if n_complete < len(indices):
            remainders.extend(indices[n_complete:])

    # Seeded shuffle so chunk order (and thus training batches) is reproducible.
    random.Random(getattr(args, "seed", DEFAULT_SEED)).shuffle(complete_chunks)

    # Drop per-k remainders instead of appending them as one mixed-k chunk:
    # rollout_suffix_turns computes the suffix length from a single k per
    # batch, so a mixed-k batch would roll out wrong-length episodes and
    # produce mislabeled rewards.
    if remainders:
        logger.warning(
            f"Dropping {len(remainders)} remainder samples that do not fill a "
            f"complete same-k batch of {batch_size} (k-homogeneous batches are "
            "required for correct suffix rollout)")

    if not complete_chunks:
        raise ValueError(
            f"No complete same-k batches of size {batch_size} could be formed "
            f"from {len(dataset)} samples; reduce --batch-size")

    dataset = concatenate_datasets(complete_chunks)

    k_dist = {}
    for k in dataset["train_k"]:
        k_dist[k] = k_dist.get(k, 0) + 1
    logger.info(
        f"Grouped {len(dataset)} samples by train_k into batch-aligned chunks "
        f"(batch_size={batch_size}, k distribution: {dict(sorted(k_dist.items()))}, "
        f"{len(remainders)} mixed-k remainder samples dropped)"
    )

    # --- Group reward model ---
    reward_model_path = args.reward_model_path or str(
        resolve_artifact_dir("group_reward", dataset=dataset_name,
                             model_name=getattr(args, "model_name", DEFAULT_GROUP_REWARD_MODEL)))
    logger.info(f"Loading group reward model from {reward_model_path}")

    # Prompt schema must match what the reward checkpoint was trained with.
    rm_rubric_version = resolve_checkpoint_rubric_version(
        reward_model_path,
        cli_version=getattr(args, "rubric_version", None),
        assume_legacy=getattr(args, "assume_legacy_rubric", False),
    )
    logger.info(f"Group reward-model rubric version: {rm_rubric_version}")

    group_reward_model, group_reward_tokenizer = load_model_auto(
        reward_model_path,
        task_type="SEQ_CLS",
        num_labels=1,
        dtype=torch.bfloat16,
        device_map={"": train_device},
    )
    group_reward_model.eval()
    group_reward_model.config.pad_token_id = group_reward_tokenizer.pad_token_id

    # --- Directive generator (policy, LoRA) ---
    logger.info(f"Creating directive generator model from {args.directive_model_name}...")
    directive_config = ModelConfig(
        base_model=args.directive_model_name,
        use_lora=True,
        lora_r=getattr(args, 'lora_r', 16),
        lora_alpha=getattr(args, 'lora_alpha', 32),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
    )
    directive_model = create_lora_model(
        base_model_name=directive_config.base_model,
        task_type="CAUSAL_LM",
        config=directive_config,
        dtype=torch.bfloat16,
        device_map=None,  # GRPOTrainer colocate vLLM handles device placement
    )
    directive_model.config.pad_token_id = tokenizer.pad_token_id

    # --- Output directory ---
    output_dir = getattr(args, 'output_dir', None) or str(
        artifact_dir("director_grpo", dataset=dataset_name,
                     model_name=args.directive_model_name))
    ensure_writable_dir(output_dir, overwrite=args.overwrite or args.resume)
    logger.info(f"Output directory: {output_dir}")

    # --- GRPO config ---
    grpo_config_kwargs = {}
    if hasattr(args, 'max_steps') and args.max_steps > 0:
        grpo_config_kwargs["max_steps"] = args.max_steps

    vllm_kwargs = {
        "vllm_mode": "colocate",
        "vllm_tensor_parallel_size": 1,
        "vllm_gpu_memory_utilization": 0.3,
        "vllm_enable_sleep_mode": True,
    }

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=getattr(args, 'learning_rate', 1e-5),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=getattr(args, 'epochs', 1),
        beta=0.1,
        max_completion_length=1536,
        temperature=0.9,
        use_vllm=True,
        **vllm_kwargs,
        num_generations=args.num_generations,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        shuffle_dataset=False,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        **grpo_config_kwargs,
    )

    # --- Reward function ---
    reward_fn = partial(
        compute_group_reward,
        directive_generator=directive_generator,
        agents_vllm=agents_vllm,
        agents_dict=agents_dict,
        group_reward_model=group_reward_model,
        group_reward_tokenizer=group_reward_tokenizer,
        user_simulator=user_simulator,
        user_response_probability=args.user_response_probability,
        num_episode_turns=args.num_episode_turns,
        rubric_version=rm_rubric_version,
        dataset_name=dataset_name,
    )
    reward_fn.__name__ = "group_reward"

    reward_names = ["group_reward"]
    reward_weights = [1.0]

    # --- Patch TRL 0.28.x sync_weights for vLLM sleep mode (PR #5147, fixed in TRL 0.29.0) ---
    # vLLM sleep(level=2) frees GPU weight memory. TRL 0.28.0's sync_weights writes into
    # the freed tensors causing "CUDA error: invalid argument". Wake up before sync.
    # Version-guarded: applied only on the affected 0.28.x series.
    import trl as _trl
    if _trl.__version__.startswith("0.28."):
        import trl.generation.vllm_generation as _vg
        _original_sync = _vg.VLLMGeneration.sync_weights

        def _patched_sync_weights(self, *args, **kwargs):
            if self.mode == "colocate" and hasattr(self, "llm") and hasattr(self.llm, "wake_up"):
                torch.cuda.empty_cache()                # Free PyTorch cached memory from backward pass
                self.llm.wake_up(tags=["weights"])       # Wake only weights, not KV cache
            return _original_sync(self, *args, **kwargs)

        _vg.VLLMGeneration.sync_weights = _patched_sync_weights
        logger.info(f"Applied TRL {_trl.__version__} sync_weights sleep-mode patch")
    else:
        logger.info(f"TRL {_trl.__version__}: sync_weights sleep-mode patch not needed/applied")

    # --- Trainer ---
    logger.info("Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=directive_model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            RewardLoggingCallback(reward_names, reward_weights, every_n_steps=50),
        ],
    )

    # --- Train ---
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # --- Save ---
    logger.info(f"Saving model to {output_dir}")
    save_lora_adapter(directive_model, output_dir, tokenizer)
    logger.info("Training completed!")


def main():
    """Direct CLI for Director GRPO (mode fixed to train-director-grpo)."""
    from dotenv import load_dotenv

    from src.train.train_args import build_group_parser, finalize_group_args

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = build_group_parser()
    parser.set_defaults(mode="train-director-grpo")
    args = finalize_group_args(parser.parse_args(), parser)
    if args.mode != "train-director-grpo":
        parser.error(f"python -m src.train.director_grpo trains with GRPO only "
                     f"(got --mode {args.mode}); use python -m src.train.train_group "
                     "for other modes")
    maybe_init_wandb_group(args.mode)
    config = ModelConfig(base_model=args.director_model_name, use_lora=True)
    train_group_grpo(config, args)


if __name__ == "__main__":
    main()
