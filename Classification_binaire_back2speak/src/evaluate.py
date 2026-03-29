"""
Evaluate a trained model on a held-out test set and generate all figures.

Usage:
    python -m src.evaluate --model models/checkpoints/svm_rbf.joblib
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)

from src.utils import get_logger, MODELS_DIR, RESULTS_DIR

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Incorrecte", "Correcte"],
        yticklabels=["Incorrecte", "Correcte"],
        ax=ax,
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved → {save_path}")
    return fig


def plot_roc_curve(y_true, y_prob, model_name: str, save_path: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"Courbe ROC — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"ROC curve saved → {save_path}")
    return fig


def plot_cv_results(cv_results: pd.DataFrame, save_path: str | None = None):
    """Bar chart comparing models across metrics."""
    metrics = ["accuracy_mean", "f1_mean", "roc_auc_mean"]
    melted  = cv_results.melt(id_vars="model", value_vars=metrics,
                               var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].str.replace("_mean", "")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="score", y="model", hue="metric", ax=ax)
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Comparaison des modèles (CV)")
    ax.set_xlabel("Score moyen")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ── Main evaluation routine ───────────────────────────────────────────────────

def evaluate_on_test(
    model_path: str,
    features_path: str,
    split_path: str | None = None,
) -> dict:
    """
    Load a saved model + test features and print/save metrics + figures.
    """
    pipeline = joblib.load(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    logger.info(f"Loaded model: {model_name}")

    data = np.load(features_path, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Use saved test indices if available
    if split_path and os.path.exists(split_path):
        splits = np.load(split_path)
        test_idx = splits["test_idx"]
        X_test, y_test = X[test_idx], y[test_idx]
        logger.info(f"Using pre-defined test split: {len(test_idx)} samples")
    else:
        # Fallback: evaluate on full set (for quick checks)
        logger.warning("No test split found — evaluating on full dataset (optimistic estimate)")
        X_test, y_test = X, y

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info("\n" + classification_report(y_test, y_pred,
                target_names=["Incorrecte", "Correcte"], zero_division=0))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png"),
    )
    plot_roc_curve(
        y_test, y_prob, model_name,
        save_path=os.path.join(RESULTS_DIR, f"roc_curve_{model_name}.png"),
    )
    plt.show()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Path to .joblib model")
    parser.add_argument("--features", default=None)
    parser.add_argument("--split",    default=None,  help="Path to splits .npz")
    args = parser.parse_args()

    from src.utils import DATA_PROCESSED
    features = args.features or os.path.join(DATA_PROCESSED, "features.npz")
    split    = args.split    or os.path.join(DATA_PROCESSED, "splits.npz")

    metrics = evaluate_on_test(args.model, features, split)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
