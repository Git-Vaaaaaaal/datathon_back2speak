#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cleaning_main.py

Pipeline de nettoyage batch pour le dataset audio 'ch' (phonème ʃ).

Étapes pour chaque fichier :
  1. Conversion M4A → WAV si nécessaire (participant P13)
  2. Filtre passe-bande 80–8000 Hz
  3. Suppresseur de transitoires (coups sur table, chocs)
  4. Soustraction spectrale (bruit de fond stationnaire)
  5. Noise gate
  6. Chaîne voix : EQ + compression + normalisation RMS (normalisateur.py)
  7. Rééchantillonnage → 16 kHz (format wav2vec2)

Usage :
    python cleaning_main.py
    python cleaning_main.py --input-dir PATH --output-dir PATH --preset near
    python cleaning_main.py --input-dir PATH --output-dir PATH \\
        --preset custom --hp 90 --presence 4 --gate-threshold -35
    python cleaning_main.py --dry-run

Dépendances :
    pip install numpy scipy soundfile
    pip install pydub   # uniquement pour les fichiers M4A (P13)
    # ffmpeg doit être dans le PATH pour la conversion M4A

Compatibilité :
    Les fichiers de sortie sont en WAV PCM 16 bits, 16 kHz mono,
    compatibles avec model/dataset_wav2vec.py.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Import des modules locaux (même dossier)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from normalisateur import (
    apply_voice_chain,
    preset_parameters,
    ensure_2d,
    restore_shape,
    rms_db,
    write_wav,
)
from filtre import (
    bandpass_filter,
    transient_suppressor,
    spectral_subtraction,
    noise_gate,
    resample_to_target,
)

logger = logging.getLogger(__name__)


# ============================================================
# Valeurs par défaut du pipeline filtre
# ============================================================

FILTRE_DEFAULTS = {
    # Passe-bande
    "bp_low_hz": 80.0,
    "bp_high_hz": 8000.0,
    # Suppresseur de transitoires
    "transient_delta_db": 20.0,
    "transient_flatness": 0.7,
    "transient_hold_ms": 20.0,
    # Soustraction spectrale
    "spec_sub_noise_frames": 10,
    "spec_sub_alpha": 2.0,
    # Noise gate
    "gate_threshold_db": -40.0,
    # Rééchantillonnage
    "target_sr": 16000,
    # Feature flags
    "skip_transient": False,
    "skip_spec_sub": False,
    "skip_gate": False,
}


# ============================================================
# Conversion M4A → WAV
# ============================================================

def convert_m4a_to_wav(m4a_path: Path, out_dir: Path) -> Path:
    """
    Convertit un fichier M4A en WAV via pydub/ffmpeg.

    Le WAV de sortie est placé dans out_dir (pas dans la source).
    Si le WAV existe déjà, la conversion est sautée.

    Returns
    -------
    Path vers le fichier WAV converti.
    """
    out_path = out_dir / (m4a_path.stem + ".wav")

    if out_path.exists():
        logger.debug("Conversion sautée (déjà existant) : %s", out_path.name)
        return out_path

    try:
        from pydub import AudioSegment  # import lazy
    except ImportError:
        raise ImportError(
            "pydub est requis pour les fichiers M4A.\n"
            "  pip install pydub\n"
            "ffmpeg doit également être dans le PATH."
        )

    logger.info("Conversion M4A → WAV : %s", m4a_path.name)
    try:
        seg = AudioSegment.from_file(str(m4a_path), format="m4a")
        seg = seg.set_channels(1)  # force mono
        seg.export(str(out_path), format="wav")
    except Exception as e:
        raise RuntimeError(
            f"Échec de la conversion {m4a_path.name} : {e}\n"
            "Vérifiez que ffmpeg est installé et dans le PATH."
        ) from e

    return out_path


# ============================================================
# Lecture audio
# ============================================================

def read_audio(path: Path):
    """
    Lit un fichier audio en float32.

    Returns
    -------
    (audio: np.ndarray float32, sr: int)
    """
    audio, sr = sf.read(str(path), always_2d=False)
    audio = audio.astype(np.float32)
    return audio, sr


# ============================================================
# Pipeline de traitement d'un fichier
# ============================================================

def process_single_file(
    audio: np.ndarray,
    sr: int,
    params: dict,
) -> tuple:
    """
    Applique la chaîne de nettoyage complète sur un tableau audio.

    Pipeline :
      ensure_2d → bandpass → transient_suppressor → spectral_subtraction
      → noise_gate → apply_voice_chain → resample → restore_shape → clip

    Parameters
    ----------
    audio  : tableau float32
    sr     : fréquence d'échantillonnage source
    params : dictionnaire de paramètres (voir build_pipeline_params)

    Returns
    -------
    (cleaned: np.ndarray float32, sr_out: int)
    """
    # 1. Normalisation de forme
    audio = ensure_2d(audio)

    # 2. Filtre passe-bande vocal
    audio = bandpass_filter(
        audio, sr,
        low_hz=params["bp_low_hz"],
        high_hz=params["bp_high_hz"],
    )

    # 3. Suppresseur de transitoires (coups, chocs)
    if not params["skip_transient"]:
        audio = transient_suppressor(
            audio, sr,
            energy_delta_db=params["transient_delta_db"],
            flatness_threshold=params["transient_flatness"],
            attack_hold_ms=params["transient_hold_ms"],
        )

    # 4. Soustraction spectrale (bruit stationnaire)
    if not params["skip_spec_sub"]:
        audio = spectral_subtraction(
            audio, sr,
            noise_frames=params["spec_sub_noise_frames"],
            alpha=params["spec_sub_alpha"],
        )

    # 5. Noise gate
    if not params["skip_gate"]:
        audio = noise_gate(
            audio, sr,
            threshold_db=params["gate_threshold_db"],
        )

    # 6. Chaîne voix (EQ + compression + normalisation)
    voice_keys = {
        "hp_cutoff", "mud_gain_db", "presence_gain_db", "air_gain_db",
        "use_lowpass", "lowpass_cutoff",
        "comp_threshold_db", "comp_ratio", "comp_makeup_db",
        "rms_target_db", "max_gain_db",
    }
    voice_params = {k: v for k, v in params.items() if k in voice_keys}
    audio = apply_voice_chain(audio, sr, **voice_params)

    # 7. Rééchantillonnage vers le SR cible
    sr_out = params["target_sr"]
    if sr != sr_out:
        audio = resample_to_target(audio, sr_in=sr, sr_out=sr_out)

    # 8. Restauration de la forme + sécurité anti-clipping
    audio = restore_shape(audio)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    return audio, sr_out


# ============================================================
# Pipeline batch
# ============================================================

def run_batch(
    input_dir: Path,
    output_dir: Path,
    params: dict,
    dry_run: bool = False,
) -> dict:
    """
    Traite tous les fichiers audio d'un dossier.

    1. Convertit les M4A → WAV dans output_dir
    2. Traite tous les WAV (originaux + convertis)

    Returns
    -------
    dict avec clés : processed, skipped, errors (liste de (nom, message))
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    m4a_files = sorted(input_dir.glob("*.m4a"))
    wav_files = sorted(input_dir.glob("*.wav"))

    # Conversion M4A d'abord
    converted_wavs = []
    for m4a in m4a_files:
        if dry_run:
            logger.info("[DRY-RUN] Convertirait : %s", m4a.name)
            converted_wavs.append(output_dir / (m4a.stem + ".wav"))
            continue
        try:
            wav_path = convert_m4a_to_wav(m4a, output_dir)
            converted_wavs.append(wav_path)
        except Exception as e:
            logger.error("Erreur conversion %s : %s", m4a.name, e)

    # Liste complète à traiter : WAV originaux + WAV convertis depuis M4A
    # Évite les doublons si un WAV et un M4A ont le même stem
    wav_stems_already_in_source = {f.stem for f in wav_files}
    extra_wavs = [
        p for p in converted_wavs
        if p.stem not in wav_stems_already_in_source and p.exists()
    ]
    to_process = wav_files + extra_wavs

    processed = 0
    skipped = 0
    errors = []

    total = len(to_process)
    logger.info(
        "Fichiers à traiter : %d WAV source + %d M4A convertis = %d total",
        len(wav_files), len(extra_wavs), total,
    )

    for i, wav_path in enumerate(to_process, 1):
        out_path = output_dir / wav_path.name

        if dry_run:
            logger.info("[DRY-RUN] %d/%d %s → %s", i, total, wav_path.name, out_path)
            skipped += 1
            continue

        logger.info("[%d/%d] %s", i, total, wav_path.name)
        try:
            audio, sr = read_audio(wav_path)
            rms_before = rms_db(audio)

            cleaned, sr_out = process_single_file(audio, sr, params)

            rms_after = rms_db(cleaned)
            write_wav(out_path, cleaned, sr_out)

            logger.info(
                "  SR : %d Hz → %d Hz | RMS : %.1f → %.1f dBFS",
                sr, sr_out, rms_before, rms_after,
            )
            processed += 1

        except Exception as e:
            logger.error("  ERREUR : %s", e)
            errors.append((wav_path.name, str(e)))

    return {"processed": processed, "skipped": skipped, "errors": errors}


# ============================================================
# Construction des paramètres
# ============================================================

def build_pipeline_params(args: argparse.Namespace) -> dict:
    """
    Construit le dictionnaire de paramètres du pipeline en fusionnant :
      - Les défauts filtre (FILTRE_DEFAULTS)
      - Le preset egaliseur (near/far/custom)
      - Les overrides CLI

    Force use_lowpass=True, lowpass_cutoff=bp_high_hz lorsque
    target_sr=16000 pour éviter l'aliasing au rééchantillonnage.
    """
    params = dict(FILTRE_DEFAULTS)

    # Preset chaîne voix
    if args.preset == "custom":
        voice_params = {
            "hp_cutoff": args.hp,
            "mud_gain_db": args.mud,
            "presence_gain_db": args.presence,
            "air_gain_db": args.air,
            "comp_threshold_db": args.threshold,
            "comp_ratio": args.ratio,
            "comp_makeup_db": args.makeup,
            "rms_target_db": args.normalize,
            "max_gain_db": 20.0,
        }
    else:
        voice_params = preset_parameters(args.preset)

    params.update(voice_params)

    # Overrides CLI
    if args.gate_threshold is not None:
        params["gate_threshold_db"] = args.gate_threshold
    if args.spec_sub_frames is not None:
        params["spec_sub_noise_frames"] = args.spec_sub_frames
    if args.spec_sub_alpha is not None:
        params["spec_sub_alpha"] = args.spec_sub_alpha
    if args.transient_delta is not None:
        params["transient_delta_db"] = args.transient_delta
    if args.transient_flatness is not None:
        params["transient_flatness"] = args.transient_flatness

    params["target_sr"] = args.target_sr
    params["skip_transient"] = args.no_transient
    params["skip_spec_sub"] = args.no_spectral_sub
    params["skip_gate"] = args.no_gate

    # Anti-aliasing obligatoire quand on rééchantillonne à 16 kHz
    if params["target_sr"] == 16000:
        params["use_lowpass"] = True
        params["lowpass_cutoff"] = params["bp_high_hz"]
    else:
        params.setdefault("use_lowpass", False)
        params.setdefault("lowpass_cutoff", 12000.0)

    return params


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    # Chemin par défaut relatif au script
    default_input = str(_HERE.parent / "Donnees" / "ch" / "Fichiers audio")
    default_output = str(_HERE.parent / "Donnees" / "ch" / "Fichiers audio cleaned")

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de nettoyage audio pour le dataset 'ch' (ʃ).\n"
            "Produit des WAV 16 kHz prêts pour wav2vec2."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Chemins
    parser.add_argument(
        "--input-dir", default=default_input,
        help="Dossier source (WAV + M4A). Défaut : Donnees/ch/Fichiers audio",
    )
    parser.add_argument(
        "--output-dir", default=default_output,
        help="Dossier de sortie. Défaut : Donnees/ch/Fichiers audio cleaned",
    )

    # Preset voix
    parser.add_argument(
        "--preset", choices=["near", "far", "custom"], default="near",
        help="Preset de la chaîne voix (near/far/custom). Défaut : near",
    )

    # Rééchantillonnage
    parser.add_argument(
        "--target-sr", type=int, default=16000,
        help="Fréquence d'échantillonnage de sortie (Hz). Défaut : 16000",
    )

    # Noise gate
    parser.add_argument(
        "--gate-threshold", type=float, default=None,
        help="Seuil du noise gate (dBFS). Défaut : -40.0",
    )

    # Soustraction spectrale
    parser.add_argument(
        "--spec-sub-frames", type=int, default=None,
        help="Trames initiales pour estimer le bruit. Défaut : 10",
    )
    parser.add_argument(
        "--spec-sub-alpha", type=float, default=None,
        help="Facteur de sur-soustraction. Défaut : 2.0",
    )

    # Suppresseur de transitoires
    parser.add_argument(
        "--transient-delta", type=float, default=None,
        help="Seuil delta énergie pour détection impact (dB). Défaut : 20.0",
    )
    parser.add_argument(
        "--transient-flatness", type=float, default=None,
        help="Seuil platitude spectrale (0–1). Défaut : 0.7",
    )

    # Feature flags
    parser.add_argument(
        "--no-spectral-sub", action="store_true",
        help="Désactive la soustraction spectrale",
    )
    parser.add_argument(
        "--no-transient", action="store_true",
        help="Désactive le suppresseur de transitoires",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Désactive le noise gate",
    )

    # Custom preset — paramètres voix
    parser.add_argument("--hp", type=float, default=80.0, help="Coupe-bas (Hz)")
    parser.add_argument("--mud", type=float, default=-3.0, help="Gain mud 250 Hz (dB)")
    parser.add_argument("--presence", type=float, default=3.0, help="Gain présence 3 kHz (dB)")
    parser.add_argument("--air", type=float, default=1.5, help="Gain air 8 kHz (dB)")
    parser.add_argument("--threshold", type=float, default=-20.0, help="Seuil compresseur (dB)")
    parser.add_argument("--ratio", type=float, default=3.0, help="Ratio compression")
    parser.add_argument("--makeup", type=float, default=2.0, help="Make-up gain (dB)")
    parser.add_argument("--normalize", type=float, default=-1.0, help="Normalisation finale (dBFS)")

    # Divers
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Liste les fichiers sans les traiter",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de verbosité des logs. Défaut : INFO",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("Dossier source introuvable : %s", input_dir)
        sys.exit(1)

    params = build_pipeline_params(args)

    logger.info("=== Pipeline de nettoyage audio ===")
    logger.info("Source  : %s", input_dir)
    logger.info("Sortie  : %s", output_dir)
    logger.info("Preset  : %s", args.preset)
    logger.info("SR cible: %d Hz", params["target_sr"])
    if args.dry_run:
        logger.info("Mode    : DRY-RUN (aucun fichier écrit)")

    summary = run_batch(input_dir, output_dir, params, dry_run=args.dry_run)

    logger.info("")
    logger.info("=== Résumé ===")
    logger.info("Traités  : %d", summary["processed"])
    logger.info("Sautés   : %d", summary["skipped"])
    logger.info("Erreurs  : %d", len(summary["errors"]))

    if summary["errors"]:
        logger.warning("Fichiers en erreur :")
        for name, msg in summary["errors"]:
            logger.warning("  %s : %s", name, msg)

    if not args.dry_run and summary["processed"] > 0:
        logger.info("Fichiers nettoyés dans : %s", output_dir)


if __name__ == "__main__":
    main()
