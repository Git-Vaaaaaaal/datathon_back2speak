"""
Back2speaK — Pipeline principal

Usage:
    python main.py --dataset donnees     # Donnees/ch/Fichiers audio
    python main.py --dataset bad_data    # bad_data/
    python main.py                       # prompt interactif
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

# Ajout des modules au path
sys.path.insert(0, str(ROOT / "Cleaning"))
sys.path.insert(0, str(ROOT / "pre_processing"))
sys.path.insert(0, str(ROOT / "Classification_binaire_back2speak"))


# ─────────────────────────────────────────────
# Choix du dataset
# ─────────────────────────────────────────────

def choose_dataset(args_dataset: str | None) -> str:
    if args_dataset:
        return args_dataset
    print("\nChoisissez le dataset :")
    print("  1 - Donnees/ch/Fichiers audio (données complètes)")
    print("  2 - bad_data (données problématiques)")
    while True:
        choix = input("Votre choix (1/2) : ").strip()
        if choix == "1":
            return "donnees"
        if choix == "2":
            return "bad_data"
        print("Choix invalide, entrez 1 ou 2.")


# ─────────────────────────────────────────────
# Étape 1 — Nettoyage audio
# ─────────────────────────────────────────────

def etape_nettoyage(input_dir: Path, output_dir: Path) -> None:
    print(f"\n[1/5] Nettoyage audio : {input_dir.name} → {output_dir.name}")
    from cleaning_main import run_batch, FILTRE_DEFAULTS
    stats = run_batch(input_dir, output_dir, FILTRE_DEFAULTS)
    print(f"      Traités : {stats['processed']} | Ignorés : {stats['skipped']} | Erreurs : {len(stats['errors'])}")


# ─────────────────────────────────────────────
# Étape 2 — Chargement des labels
# ─────────────────────────────────────────────

def etape_labels(label_csv: Path, audio_dir: Path) -> pd.DataFrame:
    print(f"\n[2/5] Chargement des labels depuis {label_csv.name}")
    from src.data_loader import load_labels
    df = load_labels(str(label_csv), str(audio_dir))
    print(f"      {len(df)} fichiers audio chargés ({df['label'].sum()} corrects / {(df['label'] == 0).sum()} incorrects)")
    return df


# ─────────────────────────────────────────────
# Étape 3 — Augmentation de données
# ─────────────────────────────────────────────

def etape_augmentation(df: pd.DataFrame, aug_dir: Path, n_variants: int = 3) -> pd.DataFrame:
    print(f"\n[3/5] Augmentation de données ({n_variants} variantes par fichier)")
    import librosa
    import soundfile as sf
    from data_augmentation import generate_augmented_variants

    aug_dir.mkdir(parents=True, exist_ok=True)
    aug_rows = []

    for _, row in df.iterrows():
        try:
            y, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
            variants = generate_augmented_variants(y, sr, n_variants=n_variants)
            for i, v in enumerate(variants):
                stem = Path(row["audio_path"]).stem
                out_path = aug_dir / f"{stem}_aug{i}.wav"
                sf.write(str(out_path), v["audio"], sr)
                new_row = row.copy()
                new_row["audio_path"] = str(out_path)
                aug_rows.append(new_row)
        except Exception as e:
            print(f"      [WARN] Augmentation ignorée pour {Path(row['audio_path']).name} : {e}")

    if aug_rows:
        df_aug = pd.DataFrame(aug_rows)
        df = pd.concat([df, df_aug], ignore_index=True)

    print(f"      Dataset total après augmentation : {len(df)} fichiers")
    return df


# ─────────────────────────────────────────────
# Étape 4 — Extraction de features + entraînement
# ─────────────────────────────────────────────

def etape_classification(df: pd.DataFrame) -> dict:
    print("\n[4/5] Classification binaire (extraction features + entraînement)")
    from run_pipeline import step_extract_features, step_split, step_train_cv, step_final_train_eval

    X, y = step_extract_features(df)
    splits = step_split(X, y)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    cv_results = step_train_cv(X_train, y_train)
    best_model = cv_results.iloc[0]["model"]
    print(f"\n      Meilleur modèle (CV) : {best_model} "
          f"(AUC={cv_results.iloc[0]['roc_auc_mean']:.3f})")

    metrics = step_final_train_eval(X_train, y_train, X_test, y_test, best_model)
    return {"best_model": best_model, "metrics": metrics, "cv_results": cv_results}


# ─────────────────────────────────────────────
# Étape 5 — Résumé
# ─────────────────────────────────────────────

def etape_resume(dataset: str, n_fichiers: int, resultats: dict) -> None:
    results_dir = ROOT / "Classification_binaire_back2speak" / "results"
    print("\n[5/5] Résumé")
    print(f"      Dataset         : {dataset}")
    print(f"      Fichiers traités: {n_fichiers}")
    print(f"      Meilleur modèle : {resultats['best_model']}")
    m = resultats["metrics"]
    print(f"      AUC test        : {m.get('roc_auc', 0):.3f}")
    print(f"      F1 test         : {m.get('f1', 0):.3f}")
    print(f"      Résultats       : {results_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Back2speaK — pipeline principal")
    parser.add_argument(
        "--dataset",
        choices=["bad_data", "donnees"],
        help="Dataset à utiliser (bad_data ou donnees). Prompt interactif si absent.",
    )
    args = parser.parse_args()

    dataset = choose_dataset(args.dataset)
    print(f"\n=== Back2speaK | Dataset : {dataset} ===")

    label_csv = ROOT / "audio_db_id.csv"

    if dataset == "donnees":
        input_dir  = ROOT / "Donnees" / "ch" / "Fichiers audio"
        output_dir = ROOT / "Donnees" / "ch" / "Fichiers audio cleaned"
        aug_dir    = ROOT / "Donnees" / "ch" / "Fichiers audio augmented"
        etape_nettoyage(input_dir, output_dir)
        audio_dir = output_dir
    else:
        audio_dir = ROOT / "bad_data"
        aug_dir   = ROOT / "bad_data" / "augmented"
        print("\n[1/5] Nettoyage ignoré (bad_data déjà pré-traité)")

    df = etape_labels(label_csv, audio_dir)
    df = etape_augmentation(df, aug_dir)
    resultats = etape_classification(df)
    etape_resume(dataset, len(df), resultats)


if __name__ == "__main__":
    main()
