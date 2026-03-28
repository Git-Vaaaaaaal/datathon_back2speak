#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_spectra.py
Outil de diagnostic visuel : comparaison avant/après nettoyage pour 6 fichiers audio.
Lancement depuis la racine du projet : python Cleaning/visualize_spectra.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import librosa
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
ROOT = _HERE.parent
INPUT_DIR = ROOT / "Donnees" / "ch" / "Fichiers audio"
OUTPUT_DIR = ROOT / "Donnees" / "ch" / "Fichiers audio cleaned"

# ---------------------------------------------------------------------------
# Fichiers à visualiser
# ---------------------------------------------------------------------------
FILE_REGISTRY = [
    ("P03_S01_W_M04.wav", "substitution_ch→s | moyen | volume faible"),
    ("P03_S01_W_I04.wav", "substitution_ch→s | moyen | bruit"),
    ("P03_S01_W_F05.wav", "indeterminé | inaudible"),
    ("P03_S01_W_M05.wav", "indeterminé | inaudible"),
    ("P03_S01_W_I02.wav", "indeterminé | inaudible"),
    ("P03_S01_P05.wav",   "phrase | indeterminé | inaudible"),
]

# ---------------------------------------------------------------------------
# Paramètres de visualisation
# ---------------------------------------------------------------------------
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
WELCH_NPERSEG = 1024
WELCH_NOVERLAP = 512
PSD_XLIM_HZ = 8000  # range commun raw/cleaned (bandpass cutoff)


def load_audio_pair(filename, input_dir, output_dir):
    """Charge la paire (avant, après) d'un fichier audio.

    Returns:
        (audio_in, audio_out, sr_in, sr_out, warning)
        audio_out / sr_out peuvent être None si le fichier nettoyé est absent.
        audio_in / sr_in sont None si le fichier source est absent.
    """
    path_in = input_dir / filename
    path_out = output_dir / filename

    if not path_in.exists():
        return None, None, None, None, f"ABSENT (source) : {path_in}"

    audio_in, sr_in = sf.read(str(path_in), always_2d=False)
    audio_in = audio_in.astype(np.float32)
    if audio_in.ndim == 2:
        audio_in = audio_in.mean(axis=1)

    audio_out, sr_out = None, None
    warning = ""
    if path_out.exists():
        audio_out, sr_out = sf.read(str(path_out), always_2d=False)
        audio_out = audio_out.astype(np.float32)
        if audio_out.ndim == 2:
            audio_out = audio_out.mean(axis=1)
    else:
        warning = f"INFO : fichier nettoyé absent : {path_out}"

    return audio_in, audio_out, sr_in, sr_out, warning


def compute_psd(audio, sr):
    """Densité spectrale de puissance via Welch. Retourne (freqs_hz, psd_db)."""
    freqs, psd = welch(audio, fs=sr, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    return freqs, psd_db


def compute_spectrogram(audio, sr):
    """Mel spectrogramme en dB. Retourne (times, mel_freqs, S_db)."""
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(mel_S, ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=HOP_LENGTH)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr / 2)
    return times, mel_freqs, S_db


def plot_file_figure(filename, label, audio_in, audio_out, sr_in, sr_out):
    """Construit la figure de diagnostic (3 lignes, GridSpec 3×2)."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)

    ax_wave_in  = fig.add_subplot(gs[0, 0])
    ax_wave_out = fig.add_subplot(gs[0, 1])
    ax_spec_in  = fig.add_subplot(gs[1, 0])
    ax_spec_out = fig.add_subplot(gs[1, 1])
    ax_psd      = fig.add_subplot(gs[2, :])  # pleine largeur

    # --- Formes d'onde ---
    t_in = np.arange(len(audio_in)) / sr_in
    ax_wave_in.plot(t_in, audio_in, color="steelblue", linewidth=0.6)
    ax_wave_in.set_title("Avant (raw)", fontsize=10)
    ax_wave_in.set_xlabel("Temps (s)")
    ax_wave_in.set_ylabel("Amplitude")

    ymax = float(np.abs(audio_in).max())
    if audio_out is not None:
        t_out = np.arange(len(audio_out)) / sr_out
        ax_wave_out.plot(t_out, audio_out, color="tomato", linewidth=0.6)
        ymax = max(ymax, float(np.abs(audio_out).max()))
    else:
        ax_wave_out.text(0.5, 0.5, "Fichier nettoyé absent",
                         ha="center", va="center", transform=ax_wave_out.transAxes,
                         color="gray", fontsize=11)

    ymax = ymax * 1.1 or 1.0
    ax_wave_in.set_ylim(-ymax, ymax)
    ax_wave_out.set_ylim(-ymax, ymax)
    ax_wave_out.set_title("Après (nettoyé)", fontsize=10)
    ax_wave_out.set_xlabel("Temps (s)")

    # --- Mel spectrogrammes ---
    times_in, mf_in, S_db_in = compute_spectrogram(audio_in, sr_in)
    vmin, vmax = S_db_in.min(), S_db_in.max()

    if audio_out is not None:
        times_out, mf_out, S_db_out = compute_spectrogram(audio_out, sr_out)
        vmin = min(vmin, S_db_out.min())
        vmax = max(vmax, S_db_out.max())

    _plot_spectrogram(ax_spec_in, times_in, mf_in, S_db_in, vmin, vmax, "Avant (raw)")
    if audio_out is not None:
        _plot_spectrogram(ax_spec_out, times_out, mf_out, S_db_out, vmin, vmax, "Après (nettoyé)")
    else:
        ax_spec_out.text(0.5, 0.5, "Fichier nettoyé absent",
                         ha="center", va="center", transform=ax_spec_out.transAxes,
                         color="gray", fontsize=11)
        ax_spec_out.set_title("Après (nettoyé)", fontsize=10)

    # --- PSD overlay ---
    freqs_in, psd_in = compute_psd(audio_in, sr_in)
    ax_psd.plot(freqs_in, psd_in, color="steelblue", linewidth=1.2, label="Avant (raw)")

    if audio_out is not None:
        freqs_out, psd_out = compute_psd(audio_out, sr_out)
        ax_psd.plot(freqs_out, psd_out, color="tomato", linewidth=1.2, label="Après (nettoyé)")

    ax_psd.axvline(PSD_XLIM_HZ, color="gray", linestyle="--", linewidth=0.8,
                   label="Limite passe-bande (8 kHz)")
    ax_psd.set_xlim(0, PSD_XLIM_HZ)
    ax_psd.set_xlabel("Fréquence (Hz)")
    ax_psd.set_ylabel("DSP (dB/Hz)")
    ax_psd.set_title("Densité Spectrale de Puissance", fontsize=10)
    ax_psd.legend(fontsize=9)
    ax_psd.grid(True, alpha=0.3)

    fig.suptitle(f"{filename}\n{label}", fontsize=12, fontweight="bold")
    return fig


def _plot_spectrogram(ax, times, freqs, S_db, vmin, vmax, title):
    img = ax.pcolormesh(times, freqs, S_db, shading="auto",
                        cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)


def main():
    out_dir = _HERE  # Cleaning/
    for filename, label in FILE_REGISTRY:
        audio_in, audio_out, sr_in, sr_out, warning = load_audio_pair(
            filename, INPUT_DIR, OUTPUT_DIR
        )
        if warning:
            print(warning)
        if audio_in is None:
            print(f"  -> Fichier source introuvable, passage au suivant.")
            continue
        print(f"Chargé : {filename}  (sr_in={sr_in}"
              + (f", sr_out={sr_out}" if sr_out else ", pas de version nettoyée")
              + ")")
        fig = plot_file_figure(filename, label, audio_in, audio_out, sr_in, sr_out)

        stem = Path(filename).stem
        out_path = out_dir / f"diag_{stem}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
