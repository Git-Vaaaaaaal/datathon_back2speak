import os
import subprocess
import pandas as pd
from pathlib import Path
from pathlib import Path
import tgt
from pydub import AudioSegment
import shutil

from pathlib import Path

def add_audio_path_column(
    df: pd.DataFrame,
    id_col: str,
    audio_dir: str,
    new_col: str = "audio_path",
    recursive: bool = True
) -> pd.DataFrame:

    audio_dir = Path(audio_dir)

    if recursive:
        file_map = {p.name: str(p) for p in audio_dir.rglob("*")}
    else:
        file_map = {p.name: str(p) for p in audio_dir.glob("*")}

    df[new_col] = df[id_col].astype(str).map(file_map)

    return df

def align_audio_dataframe(
    df: pd.DataFrame,
    audio_col: str,
    transcript_col: str,
    audio_dir: str,
    output_dir: str = "output_textgrid",
    acoustic_model: str = "french_mfa",
    dictionary: str = "french_mfa",
    jobs: int = 4,
) -> None:

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    mfa_input = Path("mfa_input_tmp")

    if mfa_input.exists():
        shutil.rmtree(mfa_input)

    mfa_input.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        audio_name = str(row[audio_col]).strip()
        text = str(row[transcript_col]).strip().lower()

        src = audio_dir / audio_name
        if not src.exists():
            continue

        shutil.copy(src, mfa_input / audio_name)

        lab = mfa_input / f"{Path(audio_name).stem}.lab"
        lab.write_text(text, encoding="utf-8")

    subprocess.run(
        [
            "mfa", "align",
            str(mfa_input),
            dictionary,
            acoustic_model,
            str(output_dir),
            "--jobs", str(jobs),
            "--clean",
            "--overwrite"
        ],
        check=True
    )

    shutil.rmtree(mfa_input)

df = pd.read_csv("ton_fichier_avec_value.csv")

df["id"] = df["audio_id"].str.replace(r"\.wav$", "", regex=True)

df = add_audio_path_column(
    df,
    id_col="id",
    audio_dir=r"F:\back2speak\audio_db_2",
    new_col="path"
)

align_audio_dataframe(
    df,
    "audio_id",
    "traduction",
    r"F:\back2speak\audio_db_2"
)


""" #Extract phoneme 
def extract_phoneme(audio_path: str, textgrid_path: str, phoneme: str) -> None:
    tg = tgt.io.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name("phones")

    intervals = [i for i in tier.intervals if i.text == phoneme]
    if not intervals:
        raise ValueError(f"Phonème '{phoneme}' introuvable dans {textgrid_path}")

    audio = AudioSegment.from_file(audio_path)
    stem = Path(audio_path).stem

    for idx, interval in enumerate(intervals):
        start_ms = interval.start_time * 1000
        end_ms = interval.end_time * 1000
        segment = audio[start_ms:end_ms]

        suffix = f"_{idx}" if len(intervals) > 1 else ""
        out_path = Path(audio_path).parent / f"{stem}_{phoneme}{suffix}.wav"
        segment.export(out_path, format="wav")
        print(f"Saved: {out_path}")

 """


def extract_phoneme(audio_path: str, textgrid_path: str, phoneme: str, verbose=False) -> int:
    try:
        tg = tgt.io.read_textgrid(textgrid_path)
        tier = tg.get_tier_by_name("phones")

        intervals = [i for i in tier.intervals if i.text == phoneme]

        if not intervals:
            if verbose:
                print(f"[SKIP] Phonème '{phoneme}' introuvable dans {textgrid_path}")
            return 0  # rien extrait

        audio = AudioSegment.from_file(audio_path)
        stem = Path(audio_path).stem
        count = 0

        for idx, interval in enumerate(intervals):
            start_ms = interval.start_time * 1000
            end_ms = interval.end_time * 1000
            segment = audio[start_ms:end_ms]

            suffix = f"_{idx}" if len(intervals) > 1 else ""
            out_path = Path(audio_path).parent / f"{stem}_{phoneme}{suffix}.wav"

            segment.export(out_path, format="wav")
            count += 1

            if verbose:
                print(f"[OK] Saved: {out_path}")

        return count

    except Exception as e:
        if verbose:
            print(f"[ERROR] {audio_path} -> {e}")
        return 0


# Dossiers
AUDIO_DIR = r"F:\back2speak\audio_db_2"
TEXTGRID_DIR = "output_textgrid"

# Phonème ciblé
PHONEME = "ʃ"

# Nom des colonnes
ID_COLUMN = "id"


def process_dataframe(df):
    total_extracted = 0

    for _, row in df.iterrows():
        patient_id = row[ID_COLUMN]

        audio_path = AUDIO_DIR / f"{patient_id}.wav"
        textgrid_path = TEXTGRID_DIR / f"{patient_id}.TextGrid"

        count = extract_phoneme(audio_path, textgrid_path, PHONEME)

        total_extracted += count

    print(f"\nTotal segments extraits: {total_extracted}")



#Extract audio
def extract_word(audio_path: str, textgrid_path: str, word: str) -> None:
    tg = tgt.io.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name("words")

    intervals = [i for i in tier.intervals if i.text == word]
    if not intervals:
        raise ValueError(f"Mot '{word}' introuvable dans {textgrid_path}")

    audio = AudioSegment.from_file(audio_path)
    stem = Path(audio_path).stem

    for idx, interval in enumerate(intervals):
        segment = audio[interval.start_time * 1000 : interval.end_time * 1000]
        suffix = f"_{idx}" if len(intervals) > 1 else ""
        out_path = Path(audio_path).parent / f"{stem}_{word}{suffix}.wav"
        segment.export(out_path, format="wav")
        print(f"Saved: {out_path}")