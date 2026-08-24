from dataclasses import dataclass

from src.const import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_NUM_CANDIDATES,
)


@dataclass
class ModelConfig:
    """Configuration for base model and reward model training."""

    # Base model
    model_type: str = "qwen"  # Options: 'qwen', 'gpt', 'claude', 'gemini'
    base_model: str = DEFAULT_BASE_MODEL
    reward_model_output: str = "outputs/models/reward_model"
    max_length: int = 2048
    device: str = "cuda"

    # LoRA configuration (policy training; see src.const for the canonical values)
    use_lora: bool = True
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT

    # Generation configuration
    num_candidates: int = DEFAULT_NUM_CANDIDATES
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 3072
