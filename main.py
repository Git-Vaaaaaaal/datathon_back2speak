"""
Back2speaK — Pipeline principal

Usage:
    python main.py --dataset donnees          # pipeline complet
    python main.py --dataset bad_data         # sans nettoyage
    python main.py --skip-mfa                 # sans extraction MFA
    python main.py --skip-wav2vec             # sans analyse wav2vec
    python main.py                            # prompt interactif

Modules utilisés :
    Cleaning/cleaning_main.py                  → nettoyage audio
    pre_processing/mfa_phoneme_extractor.py    → extraction phonème ʃ (MFA)
    pre_processing/audio_extractor.py          → résolution des chemins audio
    pre_processing/data_augmentation.py        → augmentation spectrale
    Classification_binaire_back2speak/         → classification ML (SVM, RF…)
    wav2vec/main.py                            → analyse phonémique wav2vec2
"""

from __future__ import annotations

import importlib.util
import io
import sys
import argparse

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError with special chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).parent

# ── Ajout des modules au path ───────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "Cleaning"))
sys.path.insert(0, str(ROOT / "pre_processing"))
sys.path.insert(0, str(ROOT / "Classification_binaire_back2speak"))

# ── Chargement du module wav2vec (peut échouer si eSpeak absent) ────────────
_w2v = None
WAV2VEC_AVAILABLE = False

def _charger_wav2vec() -> bool:
    global _w2v, WAV2VEC_AVAILABLE
    try:
        spec = importlib.util.spec_from_file_location(
            "wav2vec_main", ROOT / "wav2vec" / "main.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["wav2vec_main"] = mod  # nécessaire pour @dataclass
        spec.loader.exec_module(mod)       # exécute configure_espeak()
        _w2v = mod
        WAV2VEC_AVAILABLE = True
        return True
    except Exception as e:
        print(f"      [SKIP] wav2vec non disponible : {e}")
        return False


# ── Mapping nom de fichier → transcription (csv_database_extractor.py) ──────
DICT_SYL: dict[str, str] = {
    "ISO01": "ch",           "SYL01": "cha",          "SYL02": "chi",
    "SYL03": "cho",          "I01":   "chat",          "L02":   "chien",
    "L03":   "chaise",       "I04":   "chaussure",     "I05":   "chapeau",
    "I06":   "cheval",       "M01":   "machine",       "M02":   "bouchon",
    "M03":   "échelle",      "M04":   "t-shirt",       "M05":   "fourchette",
    "M06":   "rocher",       "F01":   "bouche",        "F02":   "flèche",
    "F03":   "niche",        "F04":   "manche",        "F05":   "vache",
    "F06":   "poche",        "P01":   "chouquette a la crème",
    "P02":   "chocolat au lait",      "P03":   "charriot de courses",
    "P04":   "la fourchette tombe",   "P05":   "le bouchon est bleu",
    "P06":   "il lave le tee-shirt",  "P07":   "il dort dans la niche",
    "P08":   "elle colorie une vache","P09":   "le garcon pêche",
}


def transcription_from_filename(name: str) -> str | None:
    for code, text in DICT_SYL.items():
        if code in name:
            return text
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 1 — Nettoyage audio
# ═══════════════════════════════════════════════════════════════════════════════

def etape_nettoyage(input_dir: Path, output_dir: Path) -> None:
    print(f"\n[1/7] Nettoyage audio : {input_dir.name} → {output_dir.name}")
    from cleaning_main import run_batch, FILTRE_DEFAULTS
    stats = run_batch(input_dir, output_dir, FILTRE_DEFAULTS)
    print(f"      Traités : {stats['processed']} | Ignorés : {stats['skipped']} | Erreurs : {len(stats['errors'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 2 — Extraction du phonème ʃ via MFA
# ═══════════════════════════════════════════════════════════════════════════════

def etape_extraction_phonemes(audio_dir: Path) -> Path:
    import tempfile
    print(f"\n[2/7] Extraction phonème ʃ (MFA) depuis {audio_dir.name}")
    try:
        from mfa_phoneme_extractor import mfa_pipeline
    except ImportError as e:
        print(f"      [SKIP] MFA non disponible : {e}")
        return audio_dir

    seg_dir = ROOT / "mfa_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    extracted, skipped = 0, 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for wav in sorted(audio_dir.glob("*.wav")):
            transcription = transcription_from_filename(wav.stem)
            if transcription is None:
                skipped += 1
                continue
            txt = tmp_path / (wav.stem + ".txt")
            txt.write_text(transcription, encoding="utf-8")
            try:
                segs = mfa_pipeline(
                    audio_path=str(wav),
                    transcript_path=str(txt),
                    target_phoneme="ʃ",
                    extract_all=True,
                    output_dir=str(seg_dir),
                )
                extracted += len(segs)
            except Exception as e:
                print(f"      [WARN] {wav.name} : {e}")
                skipped += 1

    print(f"      Segments extraits : {extracted} | Ignorés : {skipped}")
    return seg_dir if extracted > 0 else audio_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 3 — Chargement des labels
# ═══════════════════════════════════════════════════════════════════════════════

def etape_labels(label_csv: Path, audio_dir: Path) -> pd.DataFrame:
    print(f"\n[3/7] Chargement des labels depuis {label_csv.name}")
    from src.data_loader import load_labels

    df = load_labels(str(label_csv), str(audio_dir))
    df = df.dropna(subset=["audio_path"])

    print(f"      {len(df)} fichiers | {df['label'].sum()} corrects / {(df['label'] == 0).sum()} incorrects")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 4 — Augmentation de données
# ═══════════════════════════════════════════════════════════════════════════════

def etape_augmentation(df: pd.DataFrame, aug_dir: Path, n_variants: int = 3) -> pd.DataFrame:
    print(f"\n[4/7] Augmentation de données ({n_variants} variantes par fichier)")
    import librosa
    import soundfile as sf
    from data_augmentation import generate_augmented_variants

    aug_dir.mkdir(parents=True, exist_ok=True)
    aug_rows = []

    for _, row in df.iterrows():
        try:
            y, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
            for i, v in enumerate(generate_augmented_variants(y, sr, n_variants=n_variants)):
                out = aug_dir / f"{Path(row['audio_path']).stem}_aug{i}.wav"
                sf.write(str(out), v["audio"], sr)
                new_row = row.copy()
                new_row["audio_path"] = str(out)
                aug_rows.append(new_row)
        except Exception as e:
            print(f"      [WARN] {Path(row['audio_path']).name} : {e}")

    if aug_rows:
        df = pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)

    print(f"      Dataset total après augmentation : {len(df)} fichiers")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 5 — Classification ML
# ═══════════════════════════════════════════════════════════════════════════════

def etape_classification(df: pd.DataFrame) -> dict:
    print("\n[5/7] Classification ML (features + entraînement + évaluation)")
    import joblib
    from run_pipeline import step_extract_features, step_split, step_train_cv, step_final_train_eval

    X, y = step_extract_features(df)
    splits = step_split(X, y)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    cv_results = step_train_cv(X_train, y_train)
    best_model = cv_results.iloc[0]["model"]
    print(f"\n      Meilleur modèle (CV) : {best_model} (AUC={cv_results.iloc[0]['roc_auc_mean']:.3f})")

    metrics = step_final_train_eval(X_train, y_train, X_test, y_test, best_model)

    # Charger le modèle sauvegardé pour obtenir y_pred (utilisé pour les comparaisons)
    model_path = ROOT / "Classification_binaire_back2speak" / "models" / "checkpoints" / f"{best_model}.joblib"
    y_pred_ml, y_prob_ml = None, None
    if model_path.exists():
        pipe = joblib.load(str(model_path))
        y_pred_ml = pipe.predict(X_test)
        y_prob_ml = pipe.predict_proba(X_test)[:, 1]

    return {
        "best_model": best_model,
        "metrics": metrics,
        "cv_results": cv_results,
        "y_test": y_test,
        "y_pred_ml": y_pred_ml,
        "y_prob_ml": y_prob_ml,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 6 — Analyse wav2vec
# ═══════════════════════════════════════════════════════════════════════════════

def _analyser_fichier_wav2vec(audio: np.ndarray, transcription: str) -> dict:
    """Appelle le pipeline wav2vec pour un signal audio et sa transcription."""
    text = _w2v.normalize_text(transcription)
    expected_tokens = _w2v.tokenize_phonemes(_w2v.text_to_phonemes_raw(text))
    predicted_items = _w2v.audio_to_phoneme_items(audio)
    predicted_tokens = [x["token"] for x in predicted_items]
    ops    = _w2v.align_sequences(expected_tokens, predicted_tokens)
    spans  = _w2v.build_word_spans(text)
    pa     = _w2v.build_phoneme_analysis(ops, spans, predicted_items)
    ac     = _w2v.build_acoustic_checks(audio, pa)
    pa     = _w2v.apply_acoustic_overrides(pa, ac)
    errors = _w2v.rebuild_errors_from_phoneme_analysis(pa)
    words  = _w2v.build_word_results(spans, pa)
    summary = _w2v.build_summary(expected_tokens, predicted_tokens, errors, pa, words)
    decision = _w2v.compute_final_decision(words)
    return {
        "is_correct":        decision.status == "valid",
        "pronunciation_score": summary["pronunciation_score"],   # 0–100
        "phoneme_error_rate":  summary["phoneme_error_rate"],
    }


def etape_wav2vec(df_original: pd.DataFrame) -> dict | None:
    """
    Analyse wav2vec sur les fichiers originaux dont la transcription
    est connue via DICT_SYL.  Retourne les métriques ou None.
    """
    print("\n[6/7] Analyse wav2vec (Cnam-LMSSC/wav2vec2-french-phonemizer)")
    if not WAV2VEC_AVAILABLE:
        print("      [SKIP] wav2vec non disponible")
        return None

    import soundfile as sf
    y_true, y_pred, y_score = [], [], []

    for _, row in df_original.iterrows():
        stem = Path(row["audio_path"]).stem
        if "_aug" in stem:          # ignorer les fichiers augmentés
            continue
        transcription = transcription_from_filename(stem)
        if transcription is None:
            continue
        try:
            audio_bytes = Path(row["audio_path"]).read_bytes()
            audio = _w2v.load_audio_bytes(audio_bytes)
            result = _analyser_fichier_wav2vec(audio, transcription)
            y_true.append(int(row["label"]))
            y_pred.append(1 if result["is_correct"] else 0)
            y_score.append(result["pronunciation_score"] / 100.0)
        except Exception as e:
            print(f"      [WARN] {stem} : {e}")

    if not y_true:
        print("      Aucun fichier analysable.")
        return None

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_score = np.array(y_score)

    auc = 0.0
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
    except Exception:
        pass

    result = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "roc_auc":   auc,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "n_files":   len(y_true),
    }
    print(f"      Fichiers analysés : {result['n_files']} | "
          f"Acc={result['accuracy']:.3f} | F1={result['f1']:.3f} | AUC={result['roc_auc']:.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Génération des PNGs de comparaison
# ═══════════════════════════════════════════════════════════════════════════════

def generer_comparaison_png(
    cv_results: pd.DataFrame,
    wav2vec_metrics: dict | None,
    ml_best_metrics: dict,
    y_test_ml: np.ndarray | None,
    y_pred_ml: np.ndarray | None,
    results_dir: Path,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Métriques comparées : tous les modèles ML (CV) + wav2vec ────────
    models   = list(cv_results["model"])
    acc_vals = list(cv_results["accuracy_mean"])
    f1_vals  = list(cv_results["f1_mean"])
    auc_vals = list(cv_results["roc_auc_mean"])

    if wav2vec_metrics:
        models.append("wav2vec")
        acc_vals.append(wav2vec_metrics["accuracy"])
        f1_vals.append(wav2vec_metrics["f1"])
        auc_vals.append(wav2vec_metrics["roc_auc"])

    x     = np.arange(len(models))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 6))
    ax.bar(x - width, acc_vals, width, label="Accuracy", color=colors[0])
    ax.bar(x,         f1_vals,  width, label="F1",       color=colors[1])
    ax.bar(x + width, auc_vals, width, label="AUC",      color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles — Accuracy / F1 / AUC\n(ML : moyenne CV  |  wav2vec : sur fichiers avec transcription)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path1 = results_dir / "comparison_metrics.png"
    fig.savefig(str(path1), dpi=150)
    plt.close(fig)
    print(f"      PNG → {path1}")

    # ── 2. Matrices de confusion côte à côte (meilleur ML vs wav2vec) ───────
    has_ml_cm  = y_test_ml is not None and y_pred_ml is not None
    has_w2v_cm = wav2vec_metrics is not None

    if has_ml_cm or has_w2v_cm:
        ncols = int(has_ml_cm) + int(has_w2v_cm)
        fig2, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        idx = 0
        if has_ml_cm:
            cm = confusion_matrix(y_test_ml, y_pred_ml)
            _plot_cm(axes[idx], cm, title=f"ML — {cv_results.iloc[0]['model']}")
            idx += 1
        if has_w2v_cm:
            cm = confusion_matrix(wav2vec_metrics["y_true"], wav2vec_metrics["y_pred"])
            _plot_cm(axes[idx], cm, title="wav2vec")

        fig2.suptitle("Matrices de confusion", fontsize=13)
        fig2.tight_layout()
        path2 = results_dir / "comparison_confusion_matrices.png"
        fig2.savefig(str(path2), dpi=150)
        plt.close(fig2)
        print(f"      PNG → {path2}")


def _plot_cm(ax, cm: np.ndarray, title: str) -> None:
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    labels = ["Incorrecte", "Correcte"]
    ax.set_xticks([0, 1]);  ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks([0, 1]);  ax.set_yticklabels(labels)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 7 — Résumé
# ═══════════════════════════════════════════════════════════════════════════════

def etape_resume(dataset: str, n_fichiers: int, resultats: dict, wav2vec_metrics: dict | None) -> None:
    results_dir = ROOT / "Classification_binaire_back2speak" / "results"
    print("\n[7/7] Résumé")
    print(f"      Dataset          : {dataset}")
    print(f"      Fichiers traités : {n_fichiers}")
    m = resultats["metrics"]
    print(f"      Meilleur modèle  : {resultats['best_model']}  "
          f"(AUC={m.get('roc_auc', 0):.3f}  F1={m.get('f1', 0):.3f})")
    if wav2vec_metrics:
        print(f"      wav2vec          : {wav2vec_metrics['n_files']} fichiers  "
              f"(AUC={wav2vec_metrics['roc_auc']:.3f}  F1={wav2vec_metrics['f1']:.3f})")
    print(f"      Résultats PNG    : {results_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def choose_dataset(args_dataset: str | None) -> str:
    if args_dataset:
        return args_dataset
    print("\nChoisissez le dataset :")
    print("  1 - Donnees/ch/Fichiers audio (données complètes)")
    print("  2 - bad_data (données problématiques)")
    while True:
        choix = input("Votre choix (1/2) : ").strip()
        if choix == "1":  return "donnees"
        if choix == "2":  return "bad_data"
        print("Choix invalide, entrez 1 ou 2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back2speaK — pipeline principal")
    parser.add_argument("--dataset", choices=["bad_data", "donnees"])
    parser.add_argument("--skip-mfa",     action="store_true", help="Ignorer l'extraction MFA")
    parser.add_argument("--skip-wav2vec", action="store_true", help="Ignorer l'analyse wav2vec")
    args = parser.parse_args()

    dataset = choose_dataset(args.dataset)
    print(f"\n=== Back2speaK | Dataset : {dataset} ===")

    label_csv    = ROOT / "audio_db_id.csv"
    results_dir  = ROOT / "Classification_binaire_back2speak" / "results"

    # ── Chemins selon le dataset ────────────────────────────────────────────
    if dataset == "donnees":
        input_dir  = ROOT / "Donnees" / "ch" / "Fichiers audio"
        output_dir = ROOT / "Donnees" / "ch" / "Fichiers audio cleaned"
        aug_dir    = ROOT / "Donnees" / "ch" / "Fichiers audio augmented"
        etape_nettoyage(input_dir, output_dir)
        audio_dir = output_dir
    else:
        audio_dir = ROOT / "bad_data"
        aug_dir   = ROOT / "bad_data" / "augmented"
        print("\n[1/7] Nettoyage ignoré (bad_data déjà pré-traité)")

    # ── MFA ─────────────────────────────────────────────────────────────────
    if not args.skip_mfa:
        audio_dir = etape_extraction_phonemes(audio_dir)
    else:
        print("\n[2/7] Extraction MFA ignorée (--skip-mfa)")

    # ── Labels & augmentation ───────────────────────────────────────────────
    df_original = etape_labels(label_csv, audio_dir)
    df_augmente  = etape_augmentation(df_original.copy(), aug_dir)

    # ── Classification ML ───────────────────────────────────────────────────
    resultats = etape_classification(df_augmente)

    # ── Wav2Vec ─────────────────────────────────────────────────────────────
    wav2vec_metrics = None
    if not args.skip_wav2vec:
        print("\n      Chargement du modèle wav2vec…")
        _charger_wav2vec()
        wav2vec_metrics = etape_wav2vec(df_original)

    # ── PNGs de comparaison ─────────────────────────────────────────────────
    print("\n      Génération des PNGs de comparaison…")
    generer_comparaison_png(
        cv_results      = resultats["cv_results"],
        wav2vec_metrics = wav2vec_metrics,
        ml_best_metrics = resultats["metrics"],
        y_test_ml       = resultats.get("y_test"),
        y_pred_ml       = resultats.get("y_pred_ml"),
        results_dir     = results_dir,
    )

    etape_resume(dataset, len(df_augmente), resultats, wav2vec_metrics)


if __name__ == "__main__":
    main()
