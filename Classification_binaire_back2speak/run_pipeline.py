"""
Back2speaK — Pipeline complet d'entraînement et d'évaluation.

Usage:
    python run_pipeline.py                    # pipeline complet
    python run_pipeline.py --skip-features   # si features déjà extraites
    python run_pipeline.py --model svm_rbf   # entraîner un seul modèle
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, get_logger, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, SEED
from src.data_loader import load_labels
from src.preprocessing import load_audio
from src.features import build_feature_matrix
from src.train import build_pipelines, cross_validate_models, train_final_model
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve, plot_cv_results,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

logger = get_logger("run_pipeline")


def step_extract_features(df: pd.DataFrame, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load audio, extract features, cache to disk."""
    out_path = os.path.join(DATA_PROCESSED, "features.npz")
    if os.path.exists(out_path) and not force:
        logger.info(f"Loading cached features from {out_path}")
        data = np.load(out_path, allow_pickle=True)
        return data["X"], data["y"]

    logger.info(f"Loading {len(df)} audio files …")
    waveforms = []
    for _, row in df.iterrows():
        waveforms.append(load_audio(row["audio_path"]))
    logger.info("Extracting features …")
    X = build_feature_matrix(df, waveforms, use_metadata=True)
    y = df["label"].values.astype(int)

    meta_dict = {col: df[col].tolist() for col in
                 ["audio_id", "speaker", "age", "sexe", "position", "type_item", "decision"]}

    np.savez(out_path, X=X, y=y, meta=meta_dict)
    logger.info(f"Features cached → {out_path}  shape={X.shape}")
    return X, y


def step_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.15) -> dict:
    """Stratified train/test split (CV will be used for val inside training)."""
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size,
        stratify=y,
        random_state=SEED,
    )
    split_path = os.path.join(DATA_PROCESSED, "splits.npz")
    np.savez(split_path, train_idx=idx_tr, test_idx=idx_te)
    logger.info(
        f"Train: {len(y_tr)} (pos={y_tr.sum()}) | "
        f"Test:  {len(y_te)} (pos={y_te.sum()})"
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def step_train_cv(X_train, y_train, model_filter: str = "all") -> pd.DataFrame:
    """Cross-validate all models on training data."""
    pipelines = build_pipelines()
    if model_filter != "all":
        pipelines = {model_filter: pipelines[model_filter]}

    results = cross_validate_models(X_train, y_train, pipelines)

    results_path = os.path.join(RESULTS_DIR, "cv_results.csv")
    results.to_csv(results_path, index=False)
    logger.info(f"CV results → {results_path}")
    print("\n=== Cross-validation results ===")
    print(results.to_string(index=False))
    return results


def step_final_train_eval(X_train, y_train, X_test, y_test, best_model_name: str):
    """Re-train best model on full training set, evaluate on test."""
    pipelines = build_pipelines()
    pipe = pipelines[best_model_name]
    pipe = train_final_model(X_train, y_train, pipe, best_model_name, save=True)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print("\n=== Test set performance ===")
    print(classification_report(y_test, y_pred,
          target_names=["Incorrecte", "Correcte"], zero_division=0))
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # Save figures
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(RESULTS_DIR, f"confusion_matrix_{best_model_name}.png"),
    )
    plot_roc_curve(
        y_test, y_prob, best_model_name,
        save_path=os.path.join(RESULTS_DIR, f"roc_curve_{best_model_name}.png"),
    )
    plt.close("all")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Back2speaK binary classification pipeline")
    parser.add_argument("--skip-features", action="store_true",
                        help="Load cached features instead of re-extracting")
    parser.add_argument("--model", default="all",
                        choices=["all", "logistic_regression", "svm_rbf",
                                 "random_forest", "gradient_boosting"])
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load labels ────────────────────────────────────────────────────────
    logger.info("=== Step 1: Loading data ===")
    df = load_labels()

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    logger.info("=== Step 2: Feature extraction ===")
    X, y = step_extract_features(df, force=not args.skip_features)

    # ── 3. Train/test split ───────────────────────────────────────────────────
    logger.info("=== Step 3: Train/test split ===")
    splits = step_split(X, y, test_size=args.test_size)

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    logger.info("=== Step 4: Cross-validation ===")
    cv_results = step_train_cv(splits["X_train"], splits["y_train"], args.model)

    # Plot CV comparison
    plot_cv_results(
        cv_results,
        save_path=os.path.join(RESULTS_DIR, "cv_model_comparison.png"),
    )
    plt.close("all")

    # ── 5. Final model training + test evaluation ─────────────────────────────
    logger.info("=== Step 5: Final training & test evaluation ===")
    best_name = cv_results.iloc[0]["model"]
    logger.info(f"Best model (by AUC): {best_name}")

    metrics = step_final_train_eval(
        splits["X_train"], splits["y_train"],
        splits["X_test"],  splits["y_test"],
        best_name,
    )

    logger.info("=== Pipeline complete ===")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Model saved to:   {MODELS_DIR}")


if __name__ == "__main__":
    main()
