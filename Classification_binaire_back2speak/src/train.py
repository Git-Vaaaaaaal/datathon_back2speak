"""
Train classical ML models (SVM, Random Forest, XGBoost-style GradientBoosting)
and optionally a CNN on mel-spectrograms.

Usage (from project root):
    python -m src.train --model all --save
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    make_scorer,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils import set_seed, get_logger, DATA_PROCESSED, DATA_SPLITS, MODELS_DIR, SEED

logger = get_logger(__name__)

# ── Scorer dict ──────────────────────────────────────────────────────────────
SCORERS = {
    "accuracy":  make_scorer(accuracy_score),
    "f1":        make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score, zero_division=0),
    "roc_auc":   make_scorer(roc_auc_score, needs_proba=True),
}


def load_features(features_path: str | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load pre-extracted feature matrix, labels, and metadata."""
    if features_path is None:
        features_path = os.path.join(DATA_PROCESSED, "features.npz")

    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = pd.DataFrame(data["meta"].item())
    logger.info(f"Loaded features: X={X.shape}, y distribution: {np.bincount(y)}")
    return X, y, meta


def build_pipelines(seed: int = SEED) -> dict:
    """Return a dict of name → imbalanced-learn Pipeline."""
    smote = SMOTE(random_state=seed, k_neighbors=3)   # k=3 because minority is tiny

    pipelines = {
        "logistic_regression": ImbPipeline([
            ("smote",  smote),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced", max_iter=2000,
                C=1.0, solver="lbfgs", random_state=seed,
            )),
        ]),
        "svm_rbf": ImbPipeline([
            ("smote",  smote),
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", class_weight="balanced",
                probability=True, C=10, gamma="scale",
                random_state=seed,
            )),
        ]),
        "random_forest": ImbPipeline([
            ("smote", smote),
            ("clf",   RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                max_depth=None, min_samples_leaf=2,
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": ImbPipeline([
            ("smote", smote),
            ("scaler", StandardScaler()),
            ("clf",   GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=seed,
            )),
        ]),
    }
    return pipelines


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    pipelines: dict,
    cv: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Run stratified k-fold CV for each pipeline and return a results DataFrame.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rows = []

    for name, pipe in pipelines.items():
        logger.info(f"Cross-validating: {name}")
        scores = cross_validate(
            pipe, X, y,
            cv=skf,
            scoring=SCORERS,
            return_train_score=False,
            error_score="raise",
        )
        row = {"model": name}
        for metric, vals in scores.items():
            if metric.startswith("test_"):
                m = metric[5:]
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"]  = vals.std()
        rows.append(row)
        logger.info(
            f"  F1={row['f1_mean']:.3f}±{row['f1_std']:.3f} | "
            f"AUC={row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}"
        )

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    model_name: str,
    save: bool = True,
) -> Pipeline:
    """Fit a pipeline on the full dataset and optionally save it."""
    logger.info(f"Training final model: {model_name} on {len(X)} samples")
    pipeline.fit(X, y)
    if save:
        path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        joblib.dump(pipeline, path)
        logger.info(f"Model saved → {path}")
    return pipeline


def main(args):
    set_seed()
    X, y, meta = load_features(args.features)
    pipelines   = build_pipelines()

    if args.model != "all":
        pipelines = {args.model: pipelines[args.model]}

    results = cross_validate_models(X, y, pipelines, cv=args.cv)

    results_path = os.path.join(os.path.dirname(MODELS_DIR), "..", "results", "cv_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)
    logger.info(f"CV results saved → {results_path}")
    print("\n" + results.to_string(index=False))

    if args.save:
        best_name = results.iloc[0]["model"]
        train_final_model(X, y, pipelines[best_name], best_name, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Back2speaK binary classifier")
    parser.add_argument("--features", default=None,
                        help="Path to features.npz (default: data/processed/features.npz)")
    parser.add_argument("--model",    default="all",
                        choices=["all", "logistic_regression", "svm_rbf",
                                 "random_forest", "gradient_boosting"])
    parser.add_argument("--cv",       default=5, type=int, help="Number of CV folds")
    parser.add_argument("--save",     action="store_true",  help="Save best model")
    args = parser.parse_args()
    main(args)
