"""
Back2speaK — Pipeline principal

Usage:
    python main.py --dataset donnees     # Donnees/ch/Fichiers audio
    python main.py --dataset bad_data    # bad_data/
    python main.py                       # prompt interactif

Modules utilisés :
    Cleaning/cleaning_main.py            → nettoyage audio
    pre_processing/mfa_phoneme_extractor.py → extraction phonème ʃ (MFA)
    pre_processing/audio_extractor.py    → résolution des chemins audio
    pre_processing/data_augmentation.py  → augmentation spectrale
    Classification_binaire_back2speak/   → classification ML
"""

import sys
import argparse
import tempfile
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

# Ajout des modules au path
sys.path.insert(0, str(ROOT / "Cleaning"))
sys.path.insert(0, str(ROOT / "pre_processing"))
sys.path.insert(0, str(ROOT / "Classification_binaire_back2speak"))

# Mapping nom de fichier → transcription (extrait de csv_database_extractor.py)
DICT_SYL = {
    "ISO01": "ch",
    "SYL01": "cha",
    "SYL02": "chi",
    "SYL03": "cho",
    "I01":   "chat",
    "L02":   "chien",
    "L03":   "chaise",
    "I04":   "chaussure",
    "I05":   "chapeau",
    "I06":   "cheval",
    "M01":   "machine",
    "M02":   "bouchon",
    "M03":   "échelle",
    "M04":   "t-shirt",
    "M05":   "fourchette",
    "M06":   "rocher",
    "F01":   "bouche",
    "F02":   "flèche",
    "F03":   "niche",
    "F04":   "manche",
    "F05":   "vache",
    "F06":   "poche",
    "P01":   "chouquette a la crème",
    "P02":   "chocolat au lait",
    "P03":   "charriot de courses",
    "P04":   "la fourchette tombe",
    "P05":   "le bouchon est bleu",
    "P06":   "il lave le tee-shirt",
    "P07":   "il dort dans la niche",
    "P08":   "elle colorie une vache",
    "P09":   "le garcon pêche",
}


def transcription_from_filename(name: str) -> str | None:
    """Retrouve la transcription d'un fichier depuis son nom (DICT_SYL)."""
    for code, text in DICT_SYL.items():
        if code in name:
            return text
    return None


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
    print(f"\n[1/6] Nettoyage audio : {input_dir.name} → {output_dir.name}")
    from cleaning_main import run_batch, FILTRE_DEFAULTS
    stats = run_batch(input_dir, output_dir, FILTRE_DEFAULTS)
    print(f"      Traités : {stats['processed']} | Ignorés : {stats['skipped']} | Erreurs : {len(stats['errors'])}")


# ─────────────────────────────────────────────
# Étape 2 — Extraction du phonème ʃ via MFA
# ─────────────────────────────────────────────

def etape_extraction_phonemes(audio_dir: Path, output_dir: Path) -> Path:
    """
    Pour chaque .wav du dossier :
      1. Génère une transcription depuis le nom de fichier (DICT_SYL)
      2. Appelle mfa_pipeline pour extraire le segment ʃ
    Retourne le dossier contenant les segments extraits.
    """
    print(f"\n[2/6] Extraction phonème ʃ (MFA) depuis {audio_dir.name}")
    try:
        from mfa_phoneme_extractor import mfa_pipeline
    except ImportError as e:
        print(f"      [SKIP] MFA non disponible : {e}")
        return audio_dir

    seg_dir = output_dir.parent / (output_dir.name + " segments")
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_dir.glob("*.wav"))
    extracted = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for wav in wav_files:
            transcription = transcription_from_filename(wav.stem)
            if transcription is None:
                skipped += 1
                continue
            transcript_file = tmp_path / (wav.stem + ".txt")
            transcript_file.write_text(transcription, encoding="utf-8")
            try:
                segments = mfa_pipeline(
                    audio_path=str(wav),
                    transcript_path=str(transcript_file),
                    target_phoneme="ʃ",
                    extract_all=True,
                    output_dir=str(seg_dir),
                )
                extracted += len(segments)
            except Exception as e:
                print(f"      [WARN] {wav.name} : {e}")
                skipped += 1

    print(f"      Segments extraits : {extracted} | Ignorés : {skipped}")
    return seg_dir if extracted > 0 else audio_dir


# ─────────────────────────────────────────────
# Étape 3 — Chargement des labels
# ─────────────────────────────────────────────

def etape_labels(label_csv: Path, audio_dir: Path) -> pd.DataFrame:
    print(f"\n[3/6] Chargement des labels depuis {label_csv.name}")
    from src.data_loader import load_labels
    from audio_extractor import add_audio_path_column

    df = load_labels(str(label_csv), str(audio_dir))

    # Résolution robuste des chemins manquants via audio_extractor
    missing = df["audio_path"].isna().sum()
    if missing > 0:
        print(f"      Résolution de {missing} chemins manquants (audio_extractor)…")
        df = add_audio_path_column(df, id_col="audio_id", audio_dir=str(audio_dir))
        df = df.dropna(subset=["audio_path"])

    print(f"      {len(df)} fichiers audio | {df['label'].sum()} corrects / {(df['label'] == 0).sum()} incorrects")
    return df


# ─────────────────────────────────────────────
# Étape 4 — Augmentation de données
# ─────────────────────────────────────────────

def etape_augmentation(df: pd.DataFrame, aug_dir: Path, n_variants: int = 3) -> pd.DataFrame:
    print(f"\n[4/6] Augmentation de données ({n_variants} variantes par fichier)")
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
            print(f"      [WARN] {Path(row['audio_path']).name} : {e}")

    if aug_rows:
        df = pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)

    print(f"      Dataset total après augmentation : {len(df)} fichiers")
    return df


# ─────────────────────────────────────────────
# Étape 5 — Extraction de features + entraînement
# ─────────────────────────────────────────────

def etape_classification(df: pd.DataFrame) -> dict:
    print("\n[5/6] Classification binaire (extraction features + entraînement)")
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
# Étape 6 — Résumé
# ─────────────────────────────────────────────

def etape_resume(dataset: str, n_fichiers: int, resultats: dict) -> None:
    results_dir = ROOT / "Classification_binaire_back2speak" / "results"
    print("\n[6/6] Résumé")
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
    parser.add_argument(
        "--skip-mfa",
        action="store_true",
        help="Ignorer l'extraction MFA (si MFA non installé).",
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
        print("\n[1/6] Nettoyage ignoré (bad_data déjà pré-traité)")

    if not args.skip_mfa:
        audio_dir = etape_extraction_phonemes(audio_dir, audio_dir)
    else:
        print("\n[2/6] Extraction MFA ignorée (--skip-mfa)")

    df = etape_labels(label_csv, audio_dir)
    df = etape_augmentation(df, aug_dir)
    resultats = etape_classification(df)
    etape_resume(dataset, len(df), resultats)


if __name__ == "__main__":
    main()
