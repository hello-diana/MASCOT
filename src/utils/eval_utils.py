import logging
from pathlib import Path
from typing import Dict
import random
import numpy as np

from datasets import load_dataset

from src.const import QMSUM_DATASET, DATASET2HF

def set_seed(seed: int = 42, use_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
        except ImportError:
            print("Fail to import torch. Skipping torch seed setting")



def project_setup():
    import logging
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def load_qmsum_topic_mapping(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load meeting ID to topic mapping from QMSum dataset."""
    topics = ["Academic", "Committee", "Product"]

    topic_map = {split: {} for split in ["train", "validation", "test"]}

        
    dataset_dict = load_dataset(DATASET2HF[QMSUM_DATASET])

    for split in ["train", "validation", "test"]:
        dataset = dataset_dict[split]
        topic_map[split].update(
            dataset
            .select_columns(['id', 'topic'])
            .to_pandas()
            .set_index('id')['topic']
            .to_dict()
        )

    return topic_map


def print_colored(text, color):
    # Define ANSI color codes
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"  # Reset to default color
    }

    # Get the color code from the dictionary (default to reset if not found)
    color_code = colors.get(color.lower(), colors["reset"])

    # Print the colored text
    print(f"{color_code}{text}{colors['reset']}")
