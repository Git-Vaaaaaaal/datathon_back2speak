"""Extract hand-crafted features from raw audio waveforms.

Feature vector (per clip) — 193 dimensions:
    - MFCC (40 coeff): mean, std, Δ-mean, Δ-std        → 160
    - Spectral centroid                                   →   2 (mean, std)
    - Spectral bandwidth                                  →   2
    - Spectral rolloff (85 %)                             →   2
    - Zero-crossing rate                                  →   2
    - RMS energy                                          →   2
    - Chroma STFT (12 bins): mean, std                    →  24 (12×2) — captures harmonic content
    - Spectral contrast (7 bands): mean                   →   7
    ─────────────────────────────────────────────────────────────
    Total                                                  ≈ 201 + metadata (see below)
"""

import numpy as np
import librosa
import pandas as pd
from typing import Optional
from src.utils import get_logger

logger = get_logger(__name__)

TARGET_SR  = 16_000
N_MFCC     = 40
HOP_LENGTH = 512
N_FFT      = 2048


def extract_audio_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Compute the full hand-crafted feature vector for a single audio clip.

    Parameters
    ----------
    y  : mono float32 waveform (fixed length)
    sr : sample rate

    Returns
    -------
    features : 1-D float32 array
    """
    feats = []

    # ── MFCCs ────────────────────────────────────────────────────────────────
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    dmfcc = librosa.feature.delta(mfcc)
    for coeff_matrix in (mfcc, dmfcc):
        feats.extend(coeff_matrix.mean(axis=1))
        feats.extend(coeff_matrix.std(axis=1))

    # ── Spectral centroid ─────────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH)[0]
    feats += [centroid.mean(), centroid.std()]

    # ── Spectral bandwidth ────────────────────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT,
                                                    hop_length=HOP_LENGTH)[0]
    feats += [bandwidth.mean(), bandwidth.std()]

    # ── Spectral rolloff ──────────────────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH, roll_percent=0.85)[0]
    feats += [rolloff.mean(), rolloff.std()]

    # ── Zero-crossing rate ────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    feats += [zcr.mean(), zcr.std()]

    # ── RMS energy ────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    feats += [rms.mean(), rms.std()]

    # ── Chroma STFT ───────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    feats.extend(chroma.mean(axis=1))
    feats.extend(chroma.std(axis=1))

    # ── Spectral contrast ─────────────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT,
                                                   hop_length=HOP_LENGTH)
    feats.extend(contrast.mean(axis=1))

    return np.array(feats, dtype=np.float32)


def add_metadata_features(
    audio_feats: np.ndarray,
    age: float,
    sexe: str,
    position: str,
    type_item: str,
) -> np.ndarray:
    """
    Concatenate metadata as extra numeric features.

    Encodings
    ---------
    sexe     : M=0, F=1, U=0.5
    position : initial=0, median=1, final=2, unknown=3
    type_item: mot=0, phrase=1, syllabe=2, isole=3
    """
    sexe_enc = {"M": 0.0, "F": 1.0, "U": 0.5}.get(str(sexe).strip(), 0.5)

    pos_enc = {"initial": 0, "median": 1, "final": 2, "unknown": 3}.get(
        str(position).strip().lower(), 3
    )
    type_enc = {"mot": 0, "phrase": 1, "syllabe": 2, "isole": 3}.get(
        str(type_item).strip().lower(), 0
    )
    meta = np.array([age / 100.0, sexe_enc, pos_enc / 3.0, type_enc / 3.0],
                    dtype=np.float32)
    return np.concatenate([audio_feats, meta])


def build_feature_matrix(
    df: pd.DataFrame,
    waveforms: list[np.ndarray],
    sr: int = TARGET_SR,
    use_metadata: bool = True,
) -> np.ndarray:
    """
    Build the feature matrix X from a list of pre-loaded waveforms.

    Parameters
    ----------
    df        : DataFrame with columns age, sexe, position, type_item
    waveforms : list of waveforms aligned with df rows
    use_metadata : whether to append metadata features

    Returns
    -------
    X : shape (n_samples, n_features)
    """
    rows = []
    for i, (y, (_, row)) in enumerate(zip(waveforms, df.iterrows())):
        if i % 50 == 0:
            logger.info(f"  Feature extraction: {i}/{len(df)}")
        af = extract_audio_features(y, sr=sr)
        if use_metadata:
            af = add_metadata_features(
                af,
                age=float(row["age"]),
                sexe=str(row["sexe"]),
                position=str(row["position"]),
                type_item=str(row["type_item"]),
            )
        rows.append(af)
    return np.vstack(rows)
