#!/usr/bin/env python3
"""Unified RL training script supporting PPO, GRPO, and SimPO algorithms."""

import os
import re
import argparse
import logging
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback
try:
    import wandb
except ImportError:  # wandb is an optional dependency
    wandb = None
from trl import GRPOTrainer, GRPOConfig, extract_prompt

from config.model_config import ModelConfig
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.utils.model_utils import create_lora_model, load_lora_model, save_lora_adapter, load_model_auto
from src.prompts.persona_templates import CRITERIA, MascotPersonaTemplates
from src.utils.data_utils import create_or_load_rl_dataset, clean_response
from src.prompts.prompt_reward_model import get_prompt_persona_reward_model
from functools import partial


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def compute_reward(prompts, responses, personas, reward_model, reward_tokenizer, agents_dict=None, max_length=2048):
    """Compute rewards using trained reward model."""
    if agents_dict is None:
        agents_dict = AGENTS
    rewards = []
    input_texts = []
    for response, prompt, persona in zip(responses, prompts, personas):
        # Extract text from completion dict (GRPOTrainer passes [{"role": "assistant", "content": "..."}])
        if isinstance(response, list):
            response_text = response[0]['content']
        elif isinstance(response, dict):
            response_text = response['content']
        else:
            response_text = str(response)

        # Strip <think>...</think> tags — reward model was trained on clean responses
        response_text = clean_response(response_text)

        persona_spec = agents_dict[persona]
        
        input_text = get_prompt_persona_reward_model(persona, persona_spec, response_text)

        input_texts.append(input_text)

    inputs = reward_tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(reward_model.device)

    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits

        if logits.size(-1) == 2:
            logits = logits[:, 0]
            # logits = logits[:, 1] - logits[:, 0]
        else:
            logits = logits.squeeze(-1)
    return logits


def format_reward(prompts, completions, **kwargs):
    """Check if completions contain <think>...</think> tags.

    Returns:
        list[float]: 0.1 if tags present and properly formatted, 0.0 otherwise.
        Scaled down to prevent auxiliary reward dominance.
    """
    pattern = r"<think>.*?</think>"
    matches = [re.search(pattern, comp[0]['content'], re.DOTALL) for comp in completions]
    return [0.1 if match else 0.0 for match in matches]


def length_reward(prompts, completions, tokenizer, 
                  expected_thinking_length=420, expected_answer_length=64, **kwargs):
    """
    Penalize completions that violate length constraints.
    Returns 0.0 if valid, -0.2 per violation.
    Scaled down to prevent auxiliary reward dominance.
    If format is missing, returns 0.0 (neutral) for length checks
    (relying on format_reward to apply the penalty).
    """
    
    # Regex to cleanly extract parts
    # <think> (group 1) </think> (group 2)
    pattern = r"<think>(.*?)</think>(.*)"
    
    thinking_texts = []
    answer_texts = []
    
    for c in completions:
        content = c[0]['content']
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # Successfully extracted parts
            thinking_texts.append(match.group(1).strip())
            answer_texts.append(match.group(2).strip())
        else:
            # Malformed: treat lengths as 0 (pass) 
            # so we don't double-penalize. 
            # The format_reward function will handle the punishment for this.
            thinking_texts.append("")
            answer_texts.append("")

    # Batch Tokenize
    tokenized_thinkings = tokenizer(thinking_texts, add_special_tokens=False)
    tokenized_answers = tokenizer(answer_texts, add_special_tokens=False)
    
    rewards = []
    for t_ids, a_ids in zip(tokenized_thinkings['input_ids'], tokenized_answers['input_ids']):
        # Logic: Start at 0.0. Subtract 0.2 for every violation.
        score = 0.0
        if len(t_ids) < expected_thinking_length:
            score -= 0.2
        if len(a_ids) > expected_answer_length:
            score -= 0.2
        rewards.append(score)
        
    return rewards


class RewardLoggingCallback(TrainerCallback):
    """Log per-component reward breakdown every N steps (rank 0 only)."""

    def __init__(self, reward_names, reward_weights, every_n_steps=50):
        self.reward_names = reward_names
        self.reward_weights = reward_weights
        self.every_n_steps = every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return

        lines = [f"[Step {step}] Reward breakdown:"]
        weighted_mean = 0.0
        wb_metrics = {}

        for name, w in zip(self.reward_names, self.reward_weights):
            mean_key = f"rewards/{name}/mean"
            std_key = f"rewards/{name}/std"
            m = logs.get(mean_key)
            s = logs.get(std_key, 0.0)
            if m is not None:
                lines.append(f"  {name:25s}  w={w:.2f}  mean={m:+.4f}  std={s:.4f}")
                weighted_mean += w * m
                wb_metrics[f"reward_breakdown/{name}_weighted"] = w * m

        trl_combined = logs.get("reward", None)
        lines.append(f"  {'weighted_sum':25s}         mean={weighted_mean:+.4f}")
        if trl_combined is not None:
            diff = weighted_mean - trl_combined
            lines.append(f"  {'trl_combined':25s}         mean={trl_combined:+.4f}  diff={diff:+.4f}")
            wb_metrics["reward_breakdown/reconstructed"] = weighted_mean
            wb_metrics["reward_breakdown/trl_combined"] = trl_combined
            wb_metrics["reward_breakdown/recon_minus_trl"] = diff

        logger.info("\n".join(lines))

        if wandb is not None and wandb.run is not None and wb_metrics:
            wandb.log(wb_metrics, step=step)


def train_grpo(config: ModelConfig, args):
    """Train using GRPO algorithm."""
    from trl import GRPOConfig, GRPOTrainer
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset_path = args.data_path
    dataset = create_or_load_rl_dataset(dataset_path, tokenizer, dataset_name=args.dataset)

    # Detect which agents dict to use based on dataset personas
    sample_persona = dataset[0]["persona"]
    if sample_persona in EMPATHETIC_AGENTS:
        agents_dict = EMPATHETIC_AGENTS
        logger.info(f"Detected empathetic persona '{sample_persona}', using EMPATHETIC_AGENTS")
    else:
        agents_dict = AGENTS
        logger.info(f"Detected QMSum persona '{sample_persona}', using AGENTS")

    # Auto-generate output dir: outputs/llm/{algorithm}_Speaker_{dataset}_{model}
    model_name = os.path.basename(config.base_model)  # e.g. "Qwen3-8B"
    # Extract dataset name by matching known model pattern at end of basename
    # e.g. "individual_Empathetic_Qwen3-32B" → "individual_Empathetic"
    data_basename = os.path.basename(dataset_path)
    model_suffix_match = re.search(r'_([A-Za-z]+\d+-[\w.]+)$', data_basename)
    if model_suffix_match:
        dataset_name = data_basename[:model_suffix_match.start()]
    else:
        dataset_name = data_basename
    output_dir = f"outputs/llm/{args.algorithm}_Speaker_{dataset_name}_{model_name}"
    logger.info(f"Output directory: {output_dir}")

    logger.info("=== Training with GRPO ===")

    # Load reward model on cuda:0 (physical CUDA:1)
    reward_model_path = args.reward_model_path
    logger.info(f"Loading reward model from {reward_model_path}...")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    reward_model, reward_tokenizer = load_model_auto(
        reward_model_path,
        task_type="SEQ_CLS",
        dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    reward_model.eval()
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # Create policy model on cuda:1 (physical CUDA:2, no value head needed)
    logger.info("Creating policy model...")
    policy_model = create_lora_model(
        base_model_name=config.base_model,
        task_type="CAUSAL_LM",
        config=config,
        dtype=torch.bfloat16,
        device_map=None if args.use_vllm else "auto"
    )
    
    

    
    # GRPO expects left padding for causal LM batching.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # Sync pad_token_id to model config (required for batch generation)
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    
    # GRPO config
    grpo_config_kwargs = {}
    if args.max_steps > 0:
        grpo_config_kwargs["max_steps"] = args.max_steps

    # vLLM mode-specific config
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            vllm_kwargs["vllm_tensor_parallel_size"] = 1
            vllm_kwargs["vllm_gpu_memory_utilization"] = 0.3
            vllm_kwargs["vllm_enable_sleep_mode"] = True
        elif args.vllm_mode == "server":
            vllm_kwargs["vllm_server_host"] = args.vllm_server_host
            vllm_kwargs["vllm_server_port"] = args.vllm_server_port
        logger.info(f"vLLM mode: {args.vllm_mode}")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        beta=0.0,
        max_completion_length=args.max_completion_length,
        temperature=config.temperature,
        # top_p=config.top_p,
        use_vllm=args.use_vllm,
        **vllm_kwargs,
        gradient_accumulation_steps=4,

        gradient_checkpointing=True,  # Explicitly enable it
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Fixes the "marked as ready twice"
        num_generations=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        metric_for_best_model="reward",
        greater_is_better=True,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none",
        reward_weights=[1.0, 0.1, 0.1],  # Down-weight aux rewards: [reward_model, length, format]
        **grpo_config_kwargs,
        
    )
    
    
    # Create dataset and split into train/eval
    dataset = dataset.map(extract_prompt)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Reward function for GRPO - persona passed automatically by GRPOTrainer
    def reward_model_fn(prompts, completions, persona=None, **kwargs):
        """Wrapper for reward model to match GRPOTrainer signature."""
        # persona is already a list, repeated for each generation by GRPOTrainer
        return compute_reward(prompts, completions, persona, reward_model, reward_tokenizer, agents_dict=agents_dict)




    # Create length reward partial with __name__ attribute
    length_reward_partial = partial(length_reward, tokenizer=tokenizer,
                expected_thinking_length=420, expected_answer_length=64)
    length_reward_partial.__name__ = "length_reward"

    # Reward function names/weights must match reward_funcs order
    reward_names = ["reward_model_fn", "length_reward", "format_reward"]
    reward_weights = [1.0, 0.1, 0.1]

    # Create trainer
    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[
            reward_model_fn,
            length_reward_partial,
            format_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            RewardLoggingCallback(reward_names, reward_weights, every_n_steps=50),
        ],
    )

    # Train
    trainer.train()

    # Save
    save_lora_adapter(policy_model, output_dir, tokenizer)
    logger.info(f"Model saved to {output_dir}")


def main():
    from src.train.train_args import add_data_args, add_vllm_args, add_rl_args, resolve_data_path

    parser = argparse.ArgumentParser(description="Train with RL algorithms")
    add_data_args(parser)
    add_vllm_args(parser)
    add_rl_args(parser)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    args = parser.parse_args()

    load_dotenv()

    # Weights & Biases logging is fully optional: enabled only when
    # WANDB_PROJECT is set (entity comes from WANDB_ENTITY, if provided).
    if os.getenv("WANDB_PROJECT"):
        if wandb is None:
            raise ImportError("WANDB_PROJECT is set but the 'wandb' package is not installed")
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
        )
    config = ModelConfig()
    args.data_path = resolve_data_path(args, config.base_model)

    logger.info(f"Selected algorithm: {args.algorithm.upper()}")

    if args.algorithm == "grpo":
        train_grpo(config, args)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")


if __name__ == "__main__":
    main()
