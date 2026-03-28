"""Utility functions: seeds, logging, paths."""

import os
import random
import logging
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(ROOT_DIR, "..", "Data")          # original data
DATA_AUDIO_DIR  = os.path.join(DATA_RAW, "Audio")
LABEL_CSV       = os.path.join(DATA_RAW, "label.csv")
METADATA_XLSX   = os.path.join(DATA_RAW, "Metadata.xlsx")

DATA_PROCESSED  = os.path.join(ROOT_DIR, "data", "processed")
DATA_SPLITS     = os.path.join(ROOT_DIR, "data", "splits")
MODELS_DIR      = os.path.join(ROOT_DIR, "models", "checkpoints")
RESULTS_DIR     = os.path.join(ROOT_DIR, "results")

for d in (DATA_PROCESSED, DATA_SPLITS, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str = "back2speak", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
