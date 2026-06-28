"""LoRA Model Utilities for Multi-Persona Training

This module provides centralized utilities for managing LoRA (Low-Rank Adaptation)
models across both reward modeling (SEQ_CLS) and persona generation (CAUSAL_LM) tasks.

Storage efficiency:
- Full Qwen3-8B model: ~16GB
- LoRA adapter only: ~10MB
- Savings: 99.9%
"""

from typing import Optional, Literal, Union
from pathlib import Path
import json
import torch
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from config.model_config import ModelConfig

logger = logging.getLogger(__name__)

TaskTypeStr = Literal["SEQ_CLS", "CAUSAL_LM"]


def get_lora_config_from_model_config(
    config: ModelConfig,
    task_type: TaskTypeStr,
    target_modules: Optional[list[str]] = None
) -> LoraConfig:
    """Convert ModelConfig to PEFT LoraConfig.

    Args:
        config: ModelConfig instance with LoRA parameters
        task_type: Either "SEQ_CLS" or "CAUSAL_LM"
        target_modules: Optional override for target modules

    Returns:
        LoraConfig instance
    """
    if task_type not in ["SEQ_CLS", "CAUSAL_LM"]:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be 'SEQ_CLS' or 'CAUSAL_LM'")

    task_type_enum = TaskType.SEQ_CLS if task_type == "SEQ_CLS" else TaskType.CAUSAL_LM

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=task_type_enum
    )


def create_lora_model(
    base_model_name: str,
    task_type: TaskTypeStr,
    config: Optional[ModelConfig] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list[str]] = None,
    num_labels: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, dict, None] = "auto",
) -> PeftModel:
    """Create a base model with LoRA adapters attached.

    Args:
        base_model_name: HuggingFace model name or path
        task_type: Either "SEQ_CLS" or "CAUSAL_LM"
        config: Optional ModelConfig instance (uses its LoRA settings if provided)
        lora_r: LoRA rank (defaults to config or 16)
        lora_alpha: LoRA alpha (defaults to config or 32)
        lora_dropout: LoRA dropout (defaults to config or 0.1)
        target_modules: Modules to apply LoRA to
        num_labels: Number of labels for SEQ_CLS tasks (defaults to 1)
        dtype: Model dtype (defaults to bfloat16)

    Returns:
        tuple of (peft_model, tokenizer)
    """
    if task_type not in ["SEQ_CLS", "CAUSAL_LM"]:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be 'SEQ_CLS' or 'CAUSAL_LM'")

    logger.info(f"Loading base model {base_model_name} for {task_type}...")
    
    # ✅ IMPORTANT: model must stay on CPU for vLLM
    model_kwargs = dict(
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # Load base model
    if task_type == "SEQ_CLS":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels or 1,
            **model_kwargs
        )
    else:  # CAUSAL_LM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        
        
    # We must always disable checkpointing for GRPO/RLHF
    # base_model.gradient_checkpointing_disable()
    base_model.config.use_cache = True

    # Create LoRA config
    if config is not None:
        lora_config = get_lora_config_from_model_config(config, task_type, target_modules)
    else:
        task_type_enum = TaskType.SEQ_CLS if task_type == "SEQ_CLS" else TaskType.CAUSAL_LM
        lora_config = LoraConfig(
            r=lora_r or 16,
            lora_alpha=lora_alpha or 32,
            lora_dropout=lora_dropout or 0.1,
            target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=task_type_enum
        )

    logger.info(f"Applying LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"dropout={lora_config.lora_dropout}, targets={lora_config.target_modules}")

    # Apply LoRA
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def load_lora_model(
    base_model_name: str,
    adapter_path: str,
    task_type: TaskTypeStr,
    num_labels: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, dict, None] = "auto"
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a base model and attach pre-trained LoRA adapters.

    Args:
        base_model_name: HuggingFace model name or path for base model
        adapter_path: Path to the saved LoRA adapter directory
        task_type: Either "SEQ_CLS" or "CAUSAL_LM"
        num_labels: Number of labels for SEQ_CLS (defaults to 1)
        dtype: Model dtype (defaults to bfloat16)
        device_map: Device mapping strategy (defaults to "auto")

    Returns:
        tuple of (peft_model, tokenizer)
    """
    if task_type not in ["SEQ_CLS", "CAUSAL_LM"]:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be 'SEQ_CLS' or 'CAUSAL_LM'")

    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    logger.info(f"Loading base model {base_model_name} for {task_type}...")

    # Load base model
    if task_type == "SEQ_CLS":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels or 1,
            torch_dtype=dtype,
            device_map=device_map
        )
    else:  # CAUSAL_LM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device_map
        )

    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Load tokenizer (try adapter path first, fallback to base model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        logger.info("Loaded tokenizer from adapter path")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        logger.info("Loaded tokenizer from base model")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    logger.info("LoRA model loaded successfully")
    return peft_model, tokenizer


def save_lora_adapter(
    peft_model: PeftModel,
    save_path: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    safe_serialization: bool = True
) -> None:
    """Save only the LoRA adapter weights (not the full model).

    Args:
        peft_model: The PEFT model with LoRA adapters
        save_path: Directory path to save adapter and config
        tokenizer: Optional tokenizer to save alongside adapter
        safe_serialization: Whether to use safetensors format
    """
    save_path_obj = Path(save_path)
    save_path_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to {save_path}...")
    peft_model.save_pretrained(save_path, safe_serialization=safe_serialization)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved tokenizer to {save_path}")

    logger.info(f"LoRA adapter saved successfully to {save_path}")


def is_lora_checkpoint(path: str) -> bool:
    """Check whether a checkpoint directory contains LoRA adapters.

    Returns True if ``adapter_config.json`` exists inside *path*.
    """
    return (Path(path) / "adapter_config.json").exists()


def load_model_auto(
    model_path: str,
    task_type: TaskTypeStr,
    num_labels: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, dict, None] = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a model checkpoint, auto-detecting LoRA vs full format.

    * If *model_path* contains ``adapter_config.json`` → reads
      ``base_model_name_or_path`` from that file and loads via
      :func:`load_lora_model`.
    * Otherwise → loads directly with ``AutoModelFor*.from_pretrained()``.

    Returns:
        (model, tokenizer)
    """
    if is_lora_checkpoint(model_path):
        adapter_cfg = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base_model_name = adapter_cfg["base_model_name_or_path"]
        logger.info(f"Detected LoRA checkpoint at {model_path}, base model: {base_model_name}")
        return load_lora_model(
            base_model_name=base_model_name,
            adapter_path=model_path,
            task_type=task_type,
            num_labels=num_labels,
            dtype=dtype,
            device_map=device_map,
        )

    # Full model checkpoint
    logger.info(f"Loading full model checkpoint from {model_path}")
    if task_type == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels or 1, torch_dtype=dtype, device_map=device_map
        )
    elif task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map
        )
    else:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be 'SEQ_CLS' or 'CAUSAL_LM'")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def merge_lora_to_full_model(
    base_model_name: str,
    lora_checkpoint_path: str,
    output_path: str,
) -> None:
    """Merge LoRA adapters into base model and save as a full model.

    Args:
        base_model_name: HuggingFace model name or path for base model
        lora_checkpoint_path: Path to LoRA checkpoint directory
        output_path: Path to save the merged full model
    """
    print(f"\n{'='*60}")
    print("Merging LoRA Adapters into Base Model")
    print(f"{'='*60}")
    print(f"Base model: {base_model_name}")
    print(f"LoRA checkpoint: {lora_checkpoint_path}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}\n")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("Tokenizer loaded")

    print(f"Loading LoRA adapters from {lora_checkpoint_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    print("LoRA adapters loaded")

    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    print("Adapters merged")

    print(f"Saving merged model to {output_path}...")
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path_obj, safe_serialization=True)
    tokenizer.save_pretrained(output_path_obj)
    print("Merged model saved")

    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"Merged model saved to: {output_path}")
    print(f"{'='*60}\n")
