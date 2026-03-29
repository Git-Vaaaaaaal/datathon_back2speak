#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalisateur.py

But :
- Normalisation intelligente de la voix des patients.
- Remplace l'ancienne normalisation en crête (peak_normalize) par une
  normalisation RMS avec plafond de gain adaptatif et limiteur soft-knee,
  pour éviter la saturation des fichiers forts et l'effet "haché" sur les
  fichiers faibles.

Exemples :
    python normalisateur.py entree.wav sortie.wav
    python normalisateur.py dossier_entree dossier_sortie --preset far
    python normalisateur.py entree.wav sortie.wav --preset custom --hp 100 --presence 4.0 --mud -4.0

Dépendances :
    pip install numpy scipy soundfile
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, iirpeak


# ============================================================
# Outils audio
# ============================================================

def db_to_linear(db: float) -> float:
    """Convertit des dB en gain linéaire."""
    return 10 ** (db / 20.0)


def linear_to_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Convertit un signal linéaire en dB."""
    return 20 * np.log10(np.maximum(np.abs(x), floor))


def ensure_2d(audio: np.ndarray) -> np.ndarray:
    """
    Garantit un tableau 2D :
    - mono : (samples,) -> (samples, 1)
    - stéréo : inchangé
    """
    if audio.ndim == 1:
        return audio[:, np.newaxis]
    return audio


def restore_shape(audio: np.ndarray) -> np.ndarray:
    """Reconvertit en mono si le signal n'a qu'un canal."""
    if audio.shape[1] == 1:
        return audio[:, 0]
    return audio


def rms_db(audio: np.ndarray, floor: float = 1e-12) -> float:
    """Calcule le niveau RMS global en dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(max(rms, floor))


def active_speech_rms_db(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    floor_db: float = -70.0,
) -> float:
    """
    Calcule le RMS des trames actives (voix) uniquement.

    Utilise une fenêtre de 20 ms et ne retient que les trames dont le RMS
    dépasse `floor_db` pour éviter que le silence ne tire la moyenne vers
    le bas. Sur un signal entièrement silencieux, retourne floor_db.

    Parameters
    ----------
    audio    : tableau float32 (N,) ou (N, C)
    sr       : fréquence d'échantillonnage
    frame_ms : durée d'une trame (ms)
    floor_db : seuil minimal pour qu'une trame soit considérée active (dBFS)
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    frame_size = max(1, int(frame_ms * sr / 1000))
    n = len(mono)

    active_rms_values = []
    for i in range(0, n, frame_size):
        frame = mono[i:i + frame_size]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        rms_val_db = 20.0 * np.log10(max(rms, 1e-12))
        if rms_val_db > floor_db:
            active_rms_values.append(rms)

    if not active_rms_values:
        return floor_db
    return 20.0 * np.log10(float(np.mean(active_rms_values)))


def rms_normalize(
    audio: np.ndarray,
    sr: int,
    target_rms_db: float = -20.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """
    Normalisation RMS avec plafond de gain adaptatif.

    Contrairement à peak_normalize, cette fonction cible un niveau RMS
    de parole actif. Le gain est plafonné à `max_gain_db` pour éviter
    l'amplification excessive des fichiers très faibles ou inaudibles,
    ce qui provoquerait saturation et effet "haché".

    Parameters
    ----------
    audio         : tableau float32
    sr            : fréquence d'échantillonnage
    target_rms_db : niveau RMS cible (dBFS). Défaut : -20 dBFS.
    max_gain_db   : gain maximum applicable (dB). Défaut : 20 dB.
                    Limiter à 20 dB évite de transformer du bruit de fond
                    en signal audible pour les fichiers inaudibles.
    """
    current_rms = active_speech_rms_db(audio, sr)
    gain_db = target_rms_db - current_rms
    # Plafonnement : on ne descend pas en dessous de -40 dB (atténuation max)
    gain_db = float(np.clip(gain_db, -40.0, max_gain_db))
    return (audio * db_to_linear(gain_db)).astype(np.float32)


def soft_limiter(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """
    Limiteur soft-knee via tanh.

    Remplace le hard-clip (np.clip) par une saturation progressive :
    - En dessous du plafond : comportement linéaire (tanh(x) ≈ x pour |x| << ceiling)
    - Au-dessus du plafond : compression douce vers ±ceiling sans transition brutale

    Cela évite les artefacts de distorsion harmonique du hard-clip.

    Parameters
    ----------
    audio       : tableau float32
    ceiling_db  : plafond de sortie (dBFS). Défaut : -1 dBFS.
    """
    ceiling = db_to_linear(ceiling_db)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-12 or peak <= ceiling:
        return audio  # rien à faire
    return (ceiling * np.tanh(audio / ceiling)).astype(np.float32)


# ============================================================
# Filtres
# ============================================================

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    """Filtre coupe-bas Butterworth."""
    nyq = sr / 2.0
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.999999)
    sos = butter(order, cutoff, btype="highpass", output="sos")
    return sosfiltfilt(sos, audio, axis=0)


def lowpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 12000.0, order: int = 4) -> np.ndarray:
    """Filtre coupe-haut léger pour retirer un peu de souffle si besoin."""
    nyq = sr / 2.0
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.999999)
    sos = butter(order, cutoff, btype="lowpass", output="sos")
    return sosfiltfilt(sos, audio, axis=0)


def peaking_eq(audio: np.ndarray, sr: int, center_hz: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """
    EQ paramétrique simplifié :
    - si gain_db > 0 : on renforce la bande
    - si gain_db < 0 : on atténue la bande
    Approche simple basée sur un filtre résonant mélangé au signal original.
    """
    if abs(gain_db) < 1e-9:
        return audio

    nyq = sr / 2.0
    freq = min(max(center_hz / nyq, 1e-6), 0.999999)

    # Filtre centré autour de la fréquence ciblée
    b, a = iirpeak(freq, Q=max(q, 0.1))
    filtered = np.zeros_like(audio)

    # Application canal par canal
    from scipy.signal import filtfilt
    for ch in range(audio.shape[1]):
        filtered[:, ch] = filtfilt(b, a, audio[:, ch])

    # Mélange simple du signal filtré
    amount = np.tanh(abs(gain_db) / 12.0)
    if gain_db > 0:
        return audio + amount * filtered
    else:
        return audio - amount * filtered


# ============================================================
# Compression
# ============================================================

def compressor(
    audio: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    makeup_gain_db: float = 0.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    sr: int = 44100
) -> np.ndarray:
    """
    Compresseur simple par enveloppe.
    Fonctionne canal par canal.
    """
    audio_out = np.zeros_like(audio)

    attack_coeff = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr))
    release_coeff = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr))
    makeup_gain = db_to_linear(makeup_gain_db)

    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        env = 0.0
        gain_reduction_db = np.zeros_like(x)

        for i, sample in enumerate(x):
            rectified = abs(sample)

            if rectified > env:
                env = attack_coeff * env + (1.0 - attack_coeff) * rectified
            else:
                env = release_coeff * env + (1.0 - release_coeff) * rectified

            env_db = 20 * np.log10(max(env, 1e-12))

            if env_db > threshold_db:
                over_db = env_db - threshold_db
                compressed_over_db = over_db / ratio
                reduction_db = over_db - compressed_over_db
            else:
                reduction_db = 0.0

            gain_reduction_db[i] = reduction_db

        gain = 10 ** ((makeup_gain_db - gain_reduction_db) / 20.0)
        audio_out[:, ch] = x * gain

    return audio_out * makeup_gain


# ============================================================
# Chaîne voix principale
# ============================================================

def apply_voice_chain(
    audio: np.ndarray,
    sr: int,
    hp_cutoff: float = 80.0,
    mud_gain_db: float = -3.0,
    presence_gain_db: float = 3.0,
    air_gain_db: float = 1.5,
    use_lowpass: bool = False,
    lowpass_cutoff: float = 12000.0,
    comp_threshold_db: float = -20.0,
    comp_ratio: float = 3.0,
    comp_makeup_db: float = 2.0,
    rms_target_db: float = -20.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """
    Chaîne de traitement orientée voix parlée avec normalisation intelligente.

    Remplace la normalisation en crête par une normalisation RMS avec plafond
    de gain adaptatif + limiteur soft-knee, pour éviter :
    - La saturation des fichiers déjà forts
    - L'effet "haché" des fichiers très faibles (où peak_normalize amplifiait
      les seuls pics survivants au noise gate jusqu'au clipping)

    Parameters
    ----------
    rms_target_db : niveau RMS cible de la parole active (dBFS). Défaut : -20 dBFS.
    max_gain_db   : gain maximal applicable (dB). Défaut : 20 dB.
                    Empêche l'amplification excessive des fichiers inaudibles.
    """

    # 1) Coupe-bas : retire les grondements / vibrations / graves inutiles
    processed = highpass_filter(audio, sr, cutoff_hz=hp_cutoff, order=4)

    # 2) Retrait du "mud" vers 200-350 Hz
    processed = peaking_eq(processed, sr, center_hz=250.0, gain_db=mud_gain_db, q=1.0)

    # 3) Ajout de présence vers 3 kHz pour l'intelligibilité
    processed = peaking_eq(processed, sr, center_hz=3000.0, gain_db=presence_gain_db, q=1.0)

    # 4) Un peu d'air vers 8 kHz
    processed = peaking_eq(processed, sr, center_hz=8000.0, gain_db=air_gain_db, q=0.8)

    # 5) Coupe-haut facultatif
    if use_lowpass:
        processed = lowpass_filter(processed, sr, cutoff_hz=lowpass_cutoff, order=4)

    # 6) Compression pour homogénéiser les niveaux dynamiques
    processed = compressor(
        processed,
        threshold_db=comp_threshold_db,
        ratio=comp_ratio,
        makeup_gain_db=comp_makeup_db,
        attack_ms=10.0,
        release_ms=120.0,
        sr=sr
    )

    # 7) Normalisation RMS intelligente avec plafond de gain
    #    Contrairement à peak_normalize, cette étape :
    #    - Cible le niveau RMS de la parole active (pas le pic absolu)
    #    - Plafonne le gain à max_gain_db pour les fichiers très faibles
    #    - Évite l'amplification excessive qui crée l'effet "haché"
    processed = rms_normalize(processed, sr, target_rms_db=rms_target_db, max_gain_db=max_gain_db)

    # 8) Limiteur soft-knee (remplace np.clip)
    #    Évite la distorsion harmonique du hard-clip en atténuant
    #    progressivement les débordements via tanh
    processed = soft_limiter(processed, ceiling_db=-1.0)

    return processed


def preset_parameters(preset_name: str) -> dict:
    """
    Renvoie des réglages simples :
    - near : voix déjà assez proche du micro
    - far : voix enregistrée loin du micro
    - custom : valeurs données en argument
    """
    presets = {
        "near": {
            "hp_cutoff": 75.0,
            "mud_gain_db": -2.0,
            "presence_gain_db": 2.0,
            "air_gain_db": 1.0,
            "comp_threshold_db": -18.0,
            "comp_ratio": 2.5,
            "comp_makeup_db": 1.5,
            "rms_target_db": -20.0,
            "max_gain_db": 20.0,
        },
        "far": {
            "hp_cutoff": 90.0,
            "mud_gain_db": -4.0,
            "presence_gain_db": 4.0,
            "air_gain_db": 2.0,
            "comp_threshold_db": -24.0,
            "comp_ratio": 4.0,
            "comp_makeup_db": 4.0,
            "rms_target_db": -20.0,
            "max_gain_db": 20.0,
        },
        "custom": {}
    }

    if preset_name not in presets:
        raise ValueError(f"Preset inconnu : {preset_name}")

    return presets[preset_name]


# ============================================================
# Lecture / écriture WAV
# ============================================================

def read_wav(path: Path):
    """Lit un fichier WAV en float32."""
    audio, sr = sf.read(str(path), always_2d=False)
    audio = audio.astype(np.float32)

    if audio.dtype.kind in ("i", "u"):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val

    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int):
    """Écrit un WAV PCM 16 bits."""
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ============================================================
# Traitement principal
# ============================================================

def process_file(input_path: Path, output_path: Path, params: dict):
    """Traite un fichier WAV."""
    audio, sr = read_wav(input_path)
    audio = ensure_2d(audio)

    before_rms = rms_db(audio)

    processed = apply_voice_chain(audio, sr, **params)

    after_rms = rms_db(processed)
    processed = restore_shape(processed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, processed, sr)

    print(f"[OK] {input_path.name} -> {output_path}")
    print(f"     RMS avant : {before_rms:.2f} dBFS")
    print(f"     RMS après : {after_rms:.2f} dBFS")


def process_path(input_path: Path, output_path: Path, params: dict):
    """
    Si input_path est :
    - un fichier WAV : traite ce fichier
    - un dossier : traite tous les WAV du dossier
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError("Le fichier d'entrée doit être un .wav")
        process_file(input_path, output_path, params)

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(input_path.glob("*.wav"))

        if not wav_files:
            print("Aucun fichier .wav trouvé dans le dossier.")
            return

        for wav_file in wav_files:
            out_file = output_path / wav_file.name
            process_file(wav_file, out_file, params)

    else:
        raise FileNotFoundError(f"Chemin introuvable : {input_path}")


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Normalisateur intelligent de voix pour enregistrements WAV."
    )

    parser.add_argument("input", help="Fichier WAV d'entrée ou dossier contenant des WAV")
    parser.add_argument("output", help="Fichier WAV de sortie ou dossier de sortie")

    parser.add_argument(
        "--preset",
        choices=["near", "far", "custom"],
        default="far",
        help="Preset de traitement. 'far' est conseillé pour les voix éloignées du micro."
    )

    parser.add_argument("--hp", type=float, default=80.0, help="Fréquence du coupe-bas (Hz)")
    parser.add_argument("--mud", type=float, default=-3.0, help="Gain en dB vers 250 Hz")
    parser.add_argument("--presence", type=float, default=3.0, help="Gain en dB vers 3 kHz")
    parser.add_argument("--air", type=float, default=1.5, help="Gain en dB vers 8 kHz")
    parser.add_argument("--threshold", type=float, default=-20.0, help="Seuil compresseur (dB)")
    parser.add_argument("--ratio", type=float, default=3.0, help="Ratio de compression")
    parser.add_argument("--makeup", type=float, default=2.0, help="Make-up gain (dB)")
    parser.add_argument("--rms-target", type=float, default=-20.0, help="Niveau RMS cible (dBFS)")
    parser.add_argument("--max-gain", type=float, default=20.0, help="Gain maximum applicable (dB)")
    parser.add_argument("--lowpass", action="store_true", help="Active le coupe-haut")
    parser.add_argument("--lowpass-cutoff", type=float, default=12000.0, help="Fréquence coupe-haut (Hz)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.preset == "custom":
        params = {
            "hp_cutoff": args.hp,
            "mud_gain_db": args.mud,
            "presence_gain_db": args.presence,
            "air_gain_db": args.air,
            "use_lowpass": args.lowpass,
            "lowpass_cutoff": args.lowpass_cutoff,
            "comp_threshold_db": args.threshold,
            "comp_ratio": args.ratio,
            "comp_makeup_db": args.makeup,
            "rms_target_db": args.rms_target,
            "max_gain_db": args.max_gain,
        }
    else:
        params = preset_parameters(args.preset)
        params["use_lowpass"] = args.lowpass
        params["lowpass_cutoff"] = args.lowpass_cutoff

    try:
        process_path(input_path, output_path, params)
        print("\nTraitement terminé.")
    except Exception as e:
        print(f"\nErreur : {e}")


if __name__ == "__main__":
    main()
