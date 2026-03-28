"""Audio loading, normalisation and resampling."""

import numpy as np
import librosa
from src.utils import get_logger

logger = get_logger(__name__)

TARGET_SR = 16_000   # resample everything to 16 kHz (standard for speech)
MAX_DURATION = 5.0   # seconds — clips longer than this are trimmed


def load_audio(path: str, target_sr: int = TARGET_SR, max_duration: float = MAX_DURATION) -> np.ndarray:
    """
    Load a WAV file, convert to mono, resample, normalise amplitude.

    Parameters
    ----------
    path        : path to .wav file
    target_sr   : output sample rate (default 16 000 Hz)
    max_duration: maximum clip length in seconds (clips are trimmed/padded)

    Returns
    -------
    y : float32 numpy array, shape (target_sr * max_duration,)
    """
    try:
        y, _ = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return np.zeros(int(target_sr * max_duration), dtype=np.float32)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalise to [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Fixed-length output (zero-pad or truncate)
    target_len = int(target_sr * max_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32)


def augment_audio(y: np.ndarray, sr: int = TARGET_SR) -> list[np.ndarray]:
    """
    Return a list of augmented variants of y (used for minority class oversampling).

    Augmentations applied:
        1. Time stretch  ×0.9
        2. Time stretch  ×1.1
        3. Pitch shift   +1 semitone
        4. Pitch shift   -1 semitone
        5. Add white noise (SNR ≈ 25 dB)
    """
    augmented = []

    target_len = len(y)

    def _fix_len(arr):
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        return arr[:target_len]

    # Time stretching
    for rate in (0.9, 1.1):
        try:
            aug = librosa.effects.time_stretch(y, rate=rate)
            augmented.append(_fix_len(aug))
        except Exception:
            pass

    # Pitch shifting
    for steps in (+1, -1):
        try:
            aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            augmented.append(_fix_len(aug))
        except Exception:
            pass

    # White noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, size=y.shape).astype(np.float32)
    augmented.append(np.clip(y + noise, -1, 1))

    return augmented
