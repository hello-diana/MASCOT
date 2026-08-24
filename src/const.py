EMPATHETIC_DATASET = "Empathetic"
QMSUM_DATASET = "QMSum"
ESCONV_DATASET = "ESConv"

INDIVIDUAL = "individual"
PAIRED = "paired"
GROUP = "group"

DATASET2HF = {
    EMPATHETIC_DATASET: "DianaW/empathetic_dialogues",
    QMSUM_DATASET: "Ahren09/QMSum",
    ESCONV_DATASET: "Ahren09/ESConv",
}

HF2DATASET = {v: k for k, v in DATASET2HF.items()}

DATASETS = (EMPATHETIC_DATASET, QMSUM_DATASET, ESCONV_DATASET)
# ---------------------------------------------------------------------------
# Single source of truth for defaults duplicated across CLIs.
#
# Policy/generative roles (paper setup): Director, Speaker, and the simulated
# User all use the same 8B backbone. Reward models use the small 0.6B backbone.
# ---------------------------------------------------------------------------
DEFAULT_DIRECTOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_SPEAKER_MODEL = "Qwen/Qwen3-8B"
DEFAULT_USER_MODEL = "Qwen/Qwen3-8B"

DEFAULT_PERSONA_REWARD_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GROUP_REWARD_MODEL = "Qwen/Qwen3-0.6B"

# Backward-compatible alias; prefer the role-specific names above.
DEFAULT_BASE_MODEL = DEFAULT_SPEAKER_MODEL

# Judge used in the paper's teacher/evaluation experiments (external API).
DEFAULT_JUDGE_MODEL = "gpt-4o"
# Judge for fully local open-source runs and smoke tests only. The paper's
# evaluation judge is DEFAULT_JUDGE_MODEL; do not conflate the two.
DEFAULT_LOCAL_JUDGE_MODEL = "Qwen/Qwen3-8B"

# Number of candidate responses (K) per context/persona for individual
# preference generation and the group branching factor. K=8 matches the paper.
DEFAULT_NUM_CANDIDATES = 8

# LoRA hyperparameters for policy training (Speaker/Director).
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1

DEFAULT_SEED = 42
N_BOOTSTRAP = 10000
