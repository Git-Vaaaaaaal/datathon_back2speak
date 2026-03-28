#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_process.py
Post-traitement ciblé des fichiers audio nettoyés :
  1. Suppression de bruits ponctuels à des timestamps précis (par fichier)
  2. Crop du silence en début/fin (avec 0.2s de marge) pour tous les fichiers

Modifie les fichiers en place dans le dossier 'Fichiers audio cleaned'.
Lancement depuis la racine : python Cleaning/post_process.py
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
ROOT = _HERE.parent
OUTPUT_DIR = ROOT / "Donnees" / "ch" / "Fichiers audio cleaned"

# ---------------------------------------------------------------------------
# Registre des corrections ciblées
# Clé   : nom du fichier (dans le dossier cleaned, à 16 kHz)
# Valeur : liste de ("zero", t_start_s, t_end_s)
#
# Justification par fichier :
#   W_M04 : burst non-vocal à 1.2-1.55s visible en spectro avant la parole
#   W_I04 : pic de saturation micro (ligne verticale bright ~1.75s en spectro)
#   W_F05 : bursts parasites 1.48-1.82s + soufflement nasal ~2.75s
#   W_M05 : spike à 1.75s = fricative 'f' naturelle (broadband OK) → crop seul
#   W_I02 : bruit diffus résiduel mais parole identifiable → crop seul
#   P05   : saturation micro visible à ±1.0 dans la forme d'onde vers 2.25s
# ---------------------------------------------------------------------------
PATCH_REGISTRY = {
    "P03_S01_W_M04.wav": [
        ("zero", 1.20, 1.55),    # bruit non-vocal avant la parole
    ],
    "P03_S01_W_I04.wav": [
        ("zero", 1.73, 1.83),    # pic de saturation micro
    ],
    "P03_S01_W_F05.wav": [
        ("zero", 0.0,  1.82),    # tout le début est du bruit (parole réelle commence à ~2.5s)
        ("zero", 2.68, 2.92),    # soufflement nasal après la parole
    ],
    "P03_S01_P05.wav": [
        ("zero", 2.20, 2.35),    # saturation micro
    ],
}

# Paramètres de crop spécifiques par fichier.
# Utilisé quand le seuil global -48 dBFS est trop agressif et coupe
# des phonèmes finaux faibles (schwa de "fourchette", voyelle "bleu").
CROP_OVERRIDES = {
    "P03_S01_W_M05.wav": {"pad_s": 0.45, "threshold_db": -58.0},
    "P03_S01_P05.wav":   {"pad_s": 0.45, "threshold_db": -58.0},
}

# ---------------------------------------------------------------------------
# Paramètres globaux
# ---------------------------------------------------------------------------
CROP_PAD_S = 0.20        # secondes de silence conservées avant/après la parole
CROP_THRESHOLD_DB = -48.0  # seuil RMS (dBFS) pour considérer une trame comme active
CROP_FRAME_MS = 20.0     # durée d'une trame RMS (ms)
FADE_MS = 20.0           # durée des fondus cosinus aux bords des zones zérotées (ms)


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def zero_out_range(
    audio: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
    fade_ms: float = FADE_MS,
) -> np.ndarray:
    """
    Remet à zéro la région [t_start, t_end] avec un fondu cosinus aux bords.

    Le fondu évite les clics (discontinuité brutale en amplitude) aux
    transitions entre la région silenciée et le signal adjacent.

    Parameters
    ----------
    audio   : tableau float32 1D (mono)
    sr      : fréquence d'échantillonnage (Hz)
    t_start : début de la zone à supprimer (s)
    t_end   : fin de la zone à supprimer (s)
    fade_ms : durée du fondu cosinus (ms) de chaque côté
    """
    audio = audio.copy()
    n = len(audio)
    i_start = max(0, int(t_start * sr))
    i_end = min(n, int(t_end * sr))
    fade_samples = max(1, int(fade_ms * sr / 1000))

    if i_start >= i_end:
        return audio

    # Fondu descendant : de 1.0 → 0.0 sur [i_start, i_start + fade]
    fade_out_end = min(i_start + fade_samples, i_end)
    fade_out_len = fade_out_end - i_start
    if fade_out_len > 0:
        t = np.linspace(0.0, 1.0, fade_out_len, endpoint=False)
        audio[i_start:fade_out_end] *= 0.5 * (1.0 + np.cos(np.pi * t))

    # Zone centrale à zéro
    center_start = min(i_start + fade_samples, i_end)
    center_end = max(i_end - fade_samples, center_start)
    audio[center_start:center_end] = 0.0

    # Fondu montant : de 0.0 → 1.0 sur [i_end - fade, i_end]
    fade_in_start = max(i_end - fade_samples, center_end)
    fade_in_len = i_end - fade_in_start
    if fade_in_len > 0:
        t = np.linspace(0.0, 1.0, fade_in_len, endpoint=False)
        audio[fade_in_start:i_end] *= 0.5 * (1.0 - np.cos(np.pi * t))

    return audio.astype(np.float32)


def crop_silence(
    audio: np.ndarray,
    sr: int,
    pad_s: float = CROP_PAD_S,
    threshold_db: float = CROP_THRESHOLD_DB,
    frame_ms: float = CROP_FRAME_MS,
) -> np.ndarray:
    """
    Supprime le silence en début et fin de fichier, en conservant `pad_s`
    secondes de marge de chaque côté de la parole active.

    Parameters
    ----------
    audio        : tableau float32 1D (mono)
    sr           : fréquence d'échantillonnage (Hz)
    pad_s        : secondes de silence conservées avant/après la parole
    threshold_db : seuil RMS (dBFS) en dessous duquel une trame = silence
    frame_ms     : durée d'une trame pour le calcul RMS (ms)
    """
    n = len(audio)
    frame_size = max(1, int(frame_ms * sr / 1000))
    n_frames = max(1, math.ceil(n / frame_size))

    # RMS par trame
    frame_rms_db = np.full(n_frames, -120.0)
    for k in range(n_frames):
        start = k * frame_size
        end = min(start + frame_size, n)
        rms = float(np.sqrt(np.mean(audio[start:end].astype(np.float64) ** 2)))
        frame_rms_db[k] = 20.0 * np.log10(max(rms, 1e-12))

    active = np.where(frame_rms_db >= threshold_db)[0]

    if len(active) == 0:
        # Signal entièrement silencieux : on retourne tel quel
        return audio

    first_frame = active[0]
    last_frame = active[-1]

    pad_frames = int(pad_s * sr / frame_size)
    i_start = max(0, (first_frame - pad_frames) * frame_size)
    i_end = min(n, (last_frame + 1 + pad_frames) * frame_size)

    return audio[i_start:i_end].astype(np.float32)


def process_file(path: Path) -> tuple:
    """
    Applique les patches ciblés (si le fichier est dans PATCH_REGISTRY)
    puis le crop silence. Sauvegarde en place.

    Returns
    -------
    (duration_before_s, duration_after_s)
    """
    audio, sr = sf.read(str(path), always_2d=False)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    duration_before = len(audio) / sr

    # Patches ciblés
    patches = PATCH_REGISTRY.get(path.name, [])
    for op, t_start, t_end in patches:
        if op == "zero":
            audio = zero_out_range(audio, sr, t_start, t_end)

    # Crop silence (paramètres globaux ou override par fichier)
    crop_kwargs = {}
    if path.name in CROP_OVERRIDES:
        crop_kwargs.update(CROP_OVERRIDES[path.name])
    audio = crop_silence(audio, sr, **crop_kwargs)

    duration_after = len(audio) / sr

    sf.write(str(path), audio, sr, subtype="PCM_16")

    return duration_before, duration_after


def main():
    verify = "--verify" in sys.argv

    if not OUTPUT_DIR.exists():
        print(f"Dossier introuvable : {OUTPUT_DIR}")
        return

    wav_files = sorted(OUTPUT_DIR.glob("*.wav"))
    if not wav_files:
        print("Aucun fichier WAV trouvé.")
        return

    total_before = 0.0
    total_after = 0.0
    patched_files = set(PATCH_REGISTRY.keys())

    print(f"Post-traitement de {len(wav_files)} fichiers dans :")
    print(f"  {OUTPUT_DIR}\n")

    for path in wav_files:
        dur_before, dur_after = process_file(path)
        total_before += dur_before
        total_after += dur_after
        saved = dur_before - dur_after
        tags = []
        if path.name in patched_files:
            tags.append("patche")
        if path.name in CROP_OVERRIDES:
            tags.append("crop-override")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {path.name}{tag_str}  {dur_before:.2f}s -> {dur_after:.2f}s  (-{saved:.2f}s)")

    print(f"\nResume :")
    print(f"  Fichiers traites : {len(wav_files)}")
    print(f"  Duree totale avant : {total_before:.1f}s")
    print(f"  Duree totale apres : {total_after:.1f}s")
    print(f"  Silence supprime  : {total_before - total_after:.1f}s")

    # Mode verification : regenere les PNG diagnostics apres traitement
    if verify:
        print("\n--- Verification : regeneration des diagnostics ---")
        viz_script = _HERE / "visualize_spectra.py"
        subprocess.run([sys.executable, str(viz_script)], check=False)


if __name__ == "__main__":
    main()
