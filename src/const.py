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
# Single source of truth for defaults duplicated across CLIs.
DEFAULT_JUDGE_MODEL = "gpt-4o"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_SEED = 42
N_BOOTSTRAP = 10000
