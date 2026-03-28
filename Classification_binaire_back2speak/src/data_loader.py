"""Load and prepare label CSV + metadata, produce a clean DataFrame."""

import os
import pandas as pd
from src.utils import LABEL_CSV, DATA_AUDIO_DIR, get_logger

logger = get_logger(__name__)

# ── Label mapping ─────────────────────────────────────────────────────────────
CORRECT_LABEL   = "correct"
BINARY_MAP = {
    "correct":               1,
    "substitution_ch_vers_s": 0,
    "distorsion":             0,
    "indéterminé":            0,
    "indetermin\u00e9":       0,   # encoding variant
}


def load_labels(label_csv: str = LABEL_CSV, audio_dir: str = DATA_AUDIO_DIR) -> pd.DataFrame:
    """
    Load label.csv and return a cleaned DataFrame with binary target.

    Returns
    -------
    pd.DataFrame with columns:
        audio_id, collecteur_id, speaker, age, sexe,
        position, type_item, decision, label, audio_path
    """
    df = pd.read_csv(label_csv)
    logger.info(f"Loaded {len(df)} rows from {label_csv}")

    # ── Binary target ─────────────────────────────────────────────────────────
    # Strip whitespace and normalise encoding issues in decision values
    df["decision"] = df["decision"].str.strip()
    df["label"] = df["decision"].map(BINARY_MAP)
    unmapped = df["label"].isna()
    if unmapped.any():
        logger.warning(f"Unmapped decisions: {df.loc[unmapped, 'decision'].unique()}")
        df = df[~unmapped].copy()

    df["label"] = df["label"].astype(int)

    # ── Audio path ────────────────────────────────────────────────────────────
    df["audio_path"] = df["audio_id"].apply(
        lambda aid: os.path.join(audio_dir, aid)
    )
    exists_mask = df["audio_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing:
        logger.warning(f"{n_missing} audio files not found on disk — dropping them")
    df = df[exists_mask].copy().reset_index(drop=True)

    # ── Fill / encode categoricals ────────────────────────────────────────────
    df["sexe"]     = df["sexe"].fillna("U")
    df["position"] = df["position"].fillna("unknown")

    logger.info(
        f"Final dataset: {len(df)} samples | "
        f"correct={df['label'].sum()} | incorrect={(df['label']==0).sum()}"
    )
    return df


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts of the raw 'decision' column."""
    return df["decision"].value_counts()
