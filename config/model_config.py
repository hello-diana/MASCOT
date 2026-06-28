from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for base model and reward model training."""

    # Base model
    model_type: str = "qwen"  # Options: 'qwen', 'gpt', 'claude', 'gemini'
    base_model: str = "Qwen/Qwen3-8B"  # Specific model name
    reward_model_output: str = "outputs/models/reward_model"
    max_length: int = 2048
    device: str = "cuda"

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Generation configuration
    num_candidates: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 3072
