#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filtre.py

Fonctions de traitement du signal pour le pipeline de nettoyage audio
du dataset 'ch' (phonème ʃ).

Toutes les fonctions :
- Acceptent un tableau numpy float32 (1D mono ou 2D samples×channels)
- Renvoient un tableau de même forme
- Sont sans effets de bord (pas d'I/O)

Dépendances : numpy, scipy
"""

import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# Filtre passe-bande
# ============================================================

def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Filtre passe-bande Butterworth.

    Conserve uniquement la plage de la voix humaine (80–8000 Hz).
    La limite haute à 8 kHz est cohérente avec un rééchantillonnage
    cible à 16 kHz (Nyquist = 8 kHz).

    Parameters
    ----------
    audio   : tableau float32 (N,) ou (N, C)
    sr      : fréquence d'échantillonnage en Hz
    low_hz  : fréquence de coupure basse (Hz)
    high_hz : fréquence de coupure haute (Hz)
    order   : ordre du filtre Butterworth (4 par défaut)
    """
    nyq = sr / 2.0
    low = min(max(low_hz / nyq, 1e-6), 0.999)
    high = min(max(high_hz / nyq, 1e-6), 0.999)

    if low >= high:
        raise ValueError(
            f"low_hz ({low_hz}) doit être inférieur à high_hz ({high_hz})"
        )

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


# ============================================================
# Suppresseur de transitoires (coups sur table, chocs micro)
# ============================================================

def transient_suppressor(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
    energy_delta_db: float = 20.0,
    flatness_threshold: float = 0.7,
    attack_hold_ms: float = 20.0,
) -> np.ndarray:
    """
    Détecte et atténue les transitoires broadband (coups sur table, chocs).

    Deux conditions combinées pour ne pas supprimer les consonnes plosives :
      1. Delta d'énergie entre trames consécutives > energy_delta_db dB
      2. Platitude spectrale de la trame > flatness_threshold
         (voix ≈ 0.1–0.3, impact broadband ≈ 0.7–1.0)

    Le masque d'atténuation utilise un fondu cosinus pour éviter les clics.

    Parameters
    ----------
    audio               : tableau float32
    sr                  : fréquence d'échantillonnage
    frame_ms            : durée d'une trame en ms
    energy_delta_db     : seuil de montée d'énergie (dB) pour détecter un impact
    flatness_threshold  : seuil de platitude spectrale (0–1)
    attack_hold_ms      : durée (ms) maintenue après le début d'un transitoire
    """
    frame_size = max(1, int(frame_ms * sr / 1000))
    hold_frames = max(1, int(attack_hold_ms / frame_ms))
    n_samples = audio.shape[0]

    # Signal de référence mono (moyenne des canaux) pour la détection
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    n_frames = max(1, math.ceil(n_samples / frame_size))

    energy_db = np.zeros(n_frames)
    flatness = np.zeros(n_frames)

    for k in range(n_frames):
        start = k * frame_size
        end = min(start + frame_size, n_samples)
        frame = mono[start:end].astype(np.float64)

        # Énergie RMS en dB
        rms = np.sqrt(np.mean(frame ** 2))
        energy_db[k] = 20.0 * np.log10(max(rms, 1e-12))

        # Platitude spectrale = moyenne géométrique / moyenne arithmétique du spectre
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        mag = np.maximum(mag, 1e-12)
        geo_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        flatness[k] = float(geo_mean / arith_mean) if arith_mean > 1e-12 else 0.0

    # Delta d'énergie entre trames consécutives
    delta = np.diff(energy_db, prepend=energy_db[0])

    # Détection : grande montée d'énergie ET spectre plat
    transient_flag = (delta > energy_delta_db) & (flatness > flatness_threshold)

    # Propagation du masque avec hold
    suppress_mask = np.zeros(n_frames, dtype=bool)
    hold_count = 0
    for k in range(n_frames):
        if transient_flag[k]:
            hold_count = hold_frames
        if hold_count > 0:
            suppress_mask[k] = True
            hold_count -= 1

    # Construction du masque au niveau sample avec fondu cosinus
    sample_mask = np.ones(n_samples, dtype=np.float32)
    fade_len = frame_size  # fondu d'un frame

    for k in range(n_frames):
        if suppress_mask[k]:
            start = k * frame_size
            end = min(start + frame_size, n_samples)
            sample_mask[start:end] = 0.0

    # Adoucissement : fondu cosinus aux transitions 1→0 et 0→1
    sample_mask = _smooth_mask(sample_mask, fade_len)

    if audio.ndim == 2:
        return (audio * sample_mask[:, np.newaxis]).astype(np.float32)
    return (audio * sample_mask).astype(np.float32)


def _smooth_mask(mask: np.ndarray, fade_len: int) -> np.ndarray:
    """Applique un fondu cosinus aux bords du masque binaire."""
    mask = mask.copy()
    n = len(mask)
    half = max(1, fade_len // 2)

    i = 0
    while i < n - 1:
        if mask[i] > 0.5 and mask[i + 1] < 0.5:
            # Transition descendante
            for j in range(min(half, n - i)):
                t = j / half
                mask[i + j] = 0.5 * (1.0 + np.cos(np.pi * t))
            i += half
        elif mask[i] < 0.5 and mask[i + 1] > 0.5:
            # Transition montante
            for j in range(min(half, n - i)):
                t = j / half
                mask[i + j] = 0.5 * (1.0 - np.cos(np.pi * t))
            i += half
        else:
            i += 1

    return mask


# ============================================================
# Soustraction spectrale
# ============================================================

def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_frames: int = 10,
    n_fft: int = 512,
    hop_length: int = 128,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> np.ndarray:
    """
    Réduction du bruit de fond par soustraction spectrale.

    Estime le bruit de fond sur les `noise_frames` premières trames
    (supposées silencieuses, avant que le locuteur commence à parler),
    puis le soustrait du spectre de magnitude de chaque trame.

    Formule : mag_clean[k] = max(|X[k]| - alpha*noise_mag, beta*|X[k]|)
    Le plancher beta*|X[k]| évite la sur-soustraction (bruit musical).

    Parameters
    ----------
    audio        : tableau float32
    sr           : fréquence d'échantillonnage
    noise_frames : nombre de trames initiales pour estimer le bruit
    n_fft        : taille de la FFT
    hop_length   : décalage entre trames (overlap-add)
    alpha        : facteur de sur-soustraction (2.0 = agressif)
    beta         : plancher spectral (fraction du spectre original)
    """
    if audio.ndim == 1:
        return _spectral_subtraction_mono(
            audio, noise_frames, n_fft, hop_length, alpha, beta
        )

    # Traitement canal par canal
    result = np.zeros_like(audio)
    for c in range(audio.shape[1]):
        result[:, c] = _spectral_subtraction_mono(
            audio[:, c], noise_frames, n_fft, hop_length, alpha, beta
        )
    return result


def _spectral_subtraction_mono(
    signal: np.ndarray,
    noise_frames: int,
    n_fft: int,
    hop_length: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Soustraction spectrale sur un signal mono 1D."""
    n = len(signal)
    window = np.hanning(n_fft).astype(np.float32)

    # Découpage en trames avec overlap
    frames = []
    positions = []
    pos = 0
    while pos + n_fft <= n:
        frames.append(signal[pos:pos + n_fft] * window)
        positions.append(pos)
        pos += hop_length

    # Dernière trame partielle si nécessaire
    if pos < n:
        frame = np.zeros(n_fft, dtype=np.float32)
        remaining = n - pos
        frame[:remaining] = signal[pos:] * window[:remaining]
        frames.append(frame)
        positions.append(pos)

    if not frames:
        return signal.copy()

    # STFT
    spectra = [np.fft.rfft(f) for f in frames]
    magnitudes = np.array([np.abs(s) for s in spectra])  # (n_frames, n_fft//2+1)
    phases = [np.angle(s) for s in spectra]

    # Estimation du bruit sur les premières trames
    n_est = min(noise_frames, len(frames))
    noise_mag = np.mean(magnitudes[:n_est], axis=0)

    # Alpha adaptatif selon le SNR estimé :
    # Pour les signaux très faibles (SNR bas), alpha=2.0 crée du bruit musical
    # car on soustrait plus que ce qui existe. On réduit alpha vers 1.0 quand
    # le SNR descend en dessous de 20 dB.
    # SNR ≤ 6 dB  → alpha_eff = 1.0  (soustraction minimale)
    # SNR ≥ 20 dB → alpha_eff = alpha (comportement nominal)
    signal_power = float(np.mean(magnitudes ** 2))
    noise_power = float(np.mean(noise_mag ** 2))
    snr_db = 10.0 * np.log10(max(signal_power / max(noise_power, 1e-12), 1e-6))
    snr_norm = float(np.clip((snr_db - 6.0) / 14.0, 0.0, 1.0))
    effective_alpha = 1.0 + (alpha - 1.0) * snr_norm

    # Soustraction spectrale avec plancher et alpha adaptatif
    clean_magnitudes = np.maximum(
        magnitudes - effective_alpha * noise_mag,
        beta * magnitudes,
    )

    # Reconstruction avec la phase originale
    output = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)

    for i, (pos, mag, phase) in enumerate(zip(positions, clean_magnitudes, phases)):
        spectrum_clean = mag * np.exp(1j * phase)
        frame_clean = np.fft.irfft(spectrum_clean).astype(np.float32)[:n_fft]
        end = min(pos + n_fft, n)
        length = end - pos
        output[pos:end] += frame_clean[:length] * window[:length]
        weight[pos:end] += window[:length] ** 2

    # Normalisation overlap-add
    nonzero = weight > 1e-8
    output[nonzero] /= weight[nonzero]

    return output.astype(np.float32)


# ============================================================
# Noise gate
# ============================================================

def noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    frame_ms: float = 20.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    """
    Noise gate par énergie de trame.

    Supprime les passages dont le niveau RMS est inférieur au seuil.
    L'enveloppe de gain utilise attack/release exponentiels pour éviter
    les transitions brutales (même approche que compressor() dans egaliseur.py).

    Parameters
    ----------
    audio         : tableau float32
    sr            : fréquence d'échantillonnage
    threshold_db  : seuil en dBFS (-40 dBFS par défaut)
    frame_ms      : durée d'une trame pour le calcul RMS
    attack_ms     : temps d'ouverture du gate (ms)
    release_ms    : temps de fermeture du gate (ms)
    """
    frame_size = max(1, int(frame_ms * sr / 1000))
    n_samples = audio.shape[0]

    # Signal mono de référence pour la détection
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    # Seuil adaptatif : pour les fichiers très faibles, le seuil fixe à -40 dBFS
    # couperait l'intégralité du signal. On abaisse le seuil pour qu'il reste
    # au maximum 10 dB sous le RMS global du signal.
    # Exemples :
    #   RMS global = -20 dBFS  → effective_threshold = min(-40, -30) = -40  (inchangé)
    #   RMS global = -65 dBFS  → effective_threshold = min(-40, -75) = -75  (seuil abaissé)
    overall_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    overall_rms_db = 20.0 * np.log10(max(overall_rms, 1e-12))
    effective_threshold_db = min(threshold_db, overall_rms_db - 10.0)

    # Calcul du RMS par trame
    n_frames = max(1, math.ceil(n_samples / frame_size))
    frame_rms_db = np.zeros(n_frames)

    for k in range(n_frames):
        start = k * frame_size
        end = min(start + frame_size, n_samples)
        rms = np.sqrt(np.mean(mono[start:end].astype(np.float64) ** 2))
        frame_rms_db[k] = 20.0 * np.log10(max(rms, 1e-12))

    # Masque binaire au niveau trame
    gate_open = (frame_rms_db >= effective_threshold_db).astype(np.float32)

    # Lissage avec enveloppe exponentielle attack/release
    attack_coeff = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr / frame_size))
    release_coeff = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr / frame_size))

    smoothed = np.zeros(n_frames, dtype=np.float32)
    env = 0.0
    for k in range(n_frames):
        target = gate_open[k]
        if target > env:
            env = attack_coeff * env + (1.0 - attack_coeff) * target
        else:
            env = release_coeff * env + (1.0 - release_coeff) * target
        smoothed[k] = env

    # Rééchantillonnage du masque au niveau sample
    sample_mask = np.repeat(smoothed, frame_size)[:n_samples].astype(np.float32)

    if audio.ndim == 2:
        return (audio * sample_mask[:, np.newaxis]).astype(np.float32)
    return (audio * sample_mask).astype(np.float32)


# ============================================================
# Rééchantillonnage
# ============================================================

def resample_to_target(
    audio: np.ndarray,
    sr_in: int,
    sr_out: int = 48000,
) -> np.ndarray:
    """
    Rééchantillonne l'audio vers sr_out Hz.

    Utilise scipy.signal.resample_poly avec ratio réduit par pgcd pour
    une qualité optimale (filtre anti-repliement intégré).

    Exemples de ratios :
      44100 → 16000 : up=160, down=441
      48000 → 16000 : up=1,   down=3
      22050 → 16000 : up=320, down=441

    Parameters
    ----------
    audio  : tableau float32 (N,) ou (N, C)
    sr_in  : fréquence d'échantillonnage source (Hz)
    sr_out : fréquence d'échantillonnage cible (Hz, défaut 16000)
    """
    if sr_in == sr_out:
        return audio.copy()

    gcd = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // gcd
    down = int(sr_in) // gcd

    resampled = resample_poly(audio, up, down, axis=0)
    return resampled.astype(np.float32)
