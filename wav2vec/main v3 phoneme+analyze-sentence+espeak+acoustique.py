from __future__ import annotations

import io
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import AutoModelForCTC, AutoProcessor

# =========================
# CONFIG
# =========================

DEFAULT_MODEL_NAME = os.getenv(
    "PHONEME_MODEL_NAME",
    "Cnam-LMSSC/wav2vec2-french-phonemizer",
)
TARGET_SAMPLING_RATE = 16_000

MULTI_CHAR_PHONEMES = {
    "ɑ̃", "ɛ̃", "ɔ̃", "œ̃",
    "dʒ", "tʃ",
}

CONFIDENCE_THRESHOLD = 0.65
LOW_CONFIDENCE_THRESHOLD = 0.50

MIN_WORD_DURATION_PER_PHONEME = 0.055
LOW_ENERGY_THRESHOLD = 2e-5
VERY_LOW_ENERGY_THRESHOLD = 8e-6

SENSITIVE_PHONEMES = {"ʃ", "s", "ʒ", "z"}
SH_S_CENTROID_THRESHOLD = 3600.0
ZH_Z_CENTROID_THRESHOLD = 3300.0

ACOUSTIC_DISAGREE_THRESHOLD = 0.55

# =========================
# ESPEAK CONFIG (WINDOWS)
# =========================

def configure_espeak() -> None:
    if os.name != "nt":
        return

    base_dirs = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
        r"C:\espeak-ng",
    ]

    exe = None
    dll = None

    for d in base_dirs:
        if not exe:
            for name in ["espeak-ng.exe", "espeak.exe"]:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    exe = p

        if not dll:
            for name in [
                "libespeak-ng.dll",
                "espeak-ng.dll",
                "libespeak.dll",
                "espeak.dll",
            ]:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    dll = p

    if not exe:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")

    if exe:
        os.environ["PHONEMIZER_ESPEAK_PATH"] = exe
        print(f"[OK] eSpeak exe : {exe}")

    if dll:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = dll
        print(f"[OK] eSpeak dll : {dll}")

    if not exe or not dll:
        raise RuntimeError(
            "eSpeak NG mal configuré.\n"
            "Vérifie installation dans : C:\\Program Files\\eSpeak NG\n"
            "Il doit contenir au moins :\n"
            " - espeak-ng.exe\n"
            " - libespeak-ng.dll"
        )


configure_espeak()

# =========================
# FASTAPI
# =========================

app = FastAPI(title="French Pronunciation Analyzer V8", version="8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DATA MODELS
# =========================

class ErrorItem(BaseModel):
    expected: str | None
    predicted: str | None
    type: str
    expected_index: int | None = None
    predicted_index: int | None = None
    word: str | None = None
    word_index: int | None = None
    phoneme_index_in_word: int | None = None


class PhonemeAlternative(BaseModel):
    phoneme: str
    score: float


class ArticulatoryFeedback(BaseModel):
    tongue_position: str | None = None
    airflow: str | None = None
    nasalization: str | None = None
    strength: str | None = None
    message: str


class PhonemeAnalysisItem(BaseModel):
    expected: str | None
    predicted: str | None
    status: str
    confidence: float | None = None
    expected_index: int | None = None
    predicted_index: int | None = None
    word: str | None = None
    word_index: int | None = None
    phoneme_index_in_word: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    energy: float | None = None
    alternatives: list[PhonemeAlternative] = []
    articulatory_feedback: ArticulatoryFeedback | None = None


class AcousticCheckItem(BaseModel):
    word: str
    word_index: int
    expected_phoneme: str
    phoneme_index_in_word: int
    acoustic_guess: str
    confidence: float
    spectral_centroid_hz: float
    energy: float
    start_sec: float
    end_sec: float
    agrees_with_main_model: bool
    note: str


class ProblemPosition(BaseModel):
    phoneme: str | None
    index: int | None
    issue: str
    confidence: float | None = None
    energy: float | None = None
    articulatory_feedback: ArticulatoryFeedback | None = None


class WordResult(BaseModel):
    word: str
    word_index: int
    expected_phonemes: list[str]
    errors: list[ErrorItem]
    clarity_score: float
    completeness_score: float
    duration_sec: float
    mean_energy: float
    is_valid: bool
    is_crushed: bool
    score: float
    problem_positions: list[ProblemPosition]
    message: str


class ReportDetail(BaseModel):
    phoneme: str | None
    phoneme_index_in_word: int | None
    issue: str
    severity: str
    confidence: float | None = None
    energy: float | None = None
    articulatory_feedback: ArticulatoryFeedback | None = None
    message: str


class WordReport(BaseModel):
    word: str
    word_index: int
    is_valid: bool
    main_reason: str
    message: str
    details: list[ReportDetail]


class FinalDecision(BaseModel):
    status: str  # valid / invalid
    message: str


class AnalyzeResponse(BaseModel):
    input_text: str
    mode: str
    expected_phonemes_raw: str
    predicted_phonemes_raw: str
    expected_phonemes: list[str]
    predicted_phonemes: list[str]
    errors: list[ErrorItem]
    phoneme_analysis: list[PhonemeAnalysisItem]
    acoustic_checks: list[AcousticCheckItem]
    words: list[WordResult]
    word_reports: list[WordReport]
    final_decision: FinalDecision
    is_correct: bool
    pronunciation_score: float
    message: str
    summary: dict[str, Any]


@dataclass
class AlignmentOp:
    op_type: str
    expected: str | None
    predicted: str | None
    expected_index: int | None
    predicted_index: int | None


# =========================
# MODEL LOADING
# =========================

@lru_cache(maxsize=1)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_processor():
    return AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)


@lru_cache(maxsize=1)
def get_model():
    model = AutoModelForCTC.from_pretrained(DEFAULT_MODEL_NAME)
    model.to(get_device())
    model.eval()
    return model


# =========================
# TEXT -> PHONEMES
# =========================

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def split_ipa_compact(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []

    result = []
    i = 0
    while i < len(token):
        if i + 1 < len(token):
            pair = token[i:i + 2]
            if pair in MULTI_CHAR_PHONEMES:
                result.append(pair)
                i += 2
                continue

        char = token[i]
        if char in {"ː", "ˈ", "ˌ"}:
            i += 1
            continue

        result.append(char)
        i += 1

    return result


def tokenize_phonemes(phoneme_string: str) -> list[str]:
    phoneme_string = phoneme_string.replace("|", " | ")
    phoneme_string = re.sub(r"\s+", " ", phoneme_string).strip()

    if not phoneme_string:
        return []

    chunks = [c for c in phoneme_string.split(" ") if c and c != "|"]

    out = []
    for chunk in chunks:
        out.extend(split_ipa_compact(chunk))
    return out


@lru_cache(maxsize=512)
def text_to_phonemes_raw(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""

    return phonemize(
        cleaned,
        language="fr-fr",
        backend="espeak",
        separator=Separator(phone=" ", word=" | "),
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        njobs=1,
    )


def text_to_phonemes(text: str) -> list[str]:
    return tokenize_phonemes(text_to_phonemes_raw(text))


# =========================
# AUDIO
# =========================

def load_audio_bytes(contents: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(contents), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}") from exc

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SAMPLING_RATE:
        audio = resample_audio(audio, sr, TARGET_SAMPLING_RATE)

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio provided.")

    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio

    duration = len(audio) / float(source_sr)
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_length = int(round(duration * target_sr))
    new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def get_id_to_token_map(processor) -> dict[int, str]:
    vocab = processor.tokenizer.get_vocab()
    return {idx: tok for tok, idx in vocab.items()}


def clean_token(token: str) -> str:
    if token is None:
        return ""
    token = token.replace("|", "").strip()
    token = token.replace("ː", "")
    return token


def decode_ctc_with_confidence(audio: np.ndarray) -> list[dict[str, Any]]:
    processor = get_processor()
    model = get_model()

    inputs = processor(audio, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt")
    device = get_device()

    input_values = inputs.input_values.to(device)
    attention_mask = None
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values=input_values, attention_mask=attention_mask).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_ids = np.argmax(probs, axis=-1).tolist()

    id_to_token = get_id_to_token_map(processor)
    blank_id = processor.tokenizer.pad_token_id

    collapsed: list[tuple[int, list[int]]] = []
    prev_id = None
    current_frames: list[int] = []

    for t, token_id in enumerate(pred_ids):
        if token_id == prev_id:
            current_frames.append(t)
            continue

        if prev_id is not None and prev_id != blank_id:
            collapsed.append((prev_id, current_frames))

        prev_id = token_id
        current_frames = [t]

    if prev_id is not None and prev_id != blank_id:
        collapsed.append((prev_id, current_frames))

    total_frames = probs.shape[0]
    duration_sec = len(audio) / float(TARGET_SAMPLING_RATE)

    items = []
    for token_id, frames in collapsed:
        token = clean_token(id_to_token.get(token_id, ""))
        if not token:
            continue

        frame_probs = probs[frames]
        confidence = float(frame_probs[:, token_id].mean())

        mean_probs = frame_probs.mean(axis=0)
        top_ids = np.argsort(mean_probs)[-5:][::-1]

        alternatives = []
        for alt_id in top_ids:
            if alt_id == blank_id:
                continue
            alt_token = clean_token(id_to_token.get(int(alt_id), ""))
            if not alt_token:
                continue
            alternatives.append(
                {
                    "phoneme": alt_token,
                    "score": float(mean_probs[int(alt_id)]),
                }
            )

        start_frame = min(frames)
        end_frame = max(frames) + 1

        start_sec = (start_frame / max(total_frames, 1)) * duration_sec
        end_sec = (end_frame / max(total_frames, 1)) * duration_sec

        items.append(
            {
                "token": token,
                "confidence": round(confidence, 4),
                "alternatives": alternatives,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
            }
        )

    return items


def extract_audio_segment(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * TARGET_SAMPLING_RATE))
    end = min(len(audio), int(end_sec * TARGET_SAMPLING_RATE))
    if end <= start:
        return np.array([], dtype=np.float32)
    return audio[start:end]


def compute_energy(audio_segment: np.ndarray) -> float:
    if len(audio_segment) == 0:
        return 0.0
    return float(np.mean(audio_segment ** 2))


def compute_spectral_centroid(audio_segment: np.ndarray, sr: int = TARGET_SAMPLING_RATE) -> float:
    if len(audio_segment) < 64:
        return 0.0
    window = np.hanning(len(audio_segment))
    spec = np.abs(np.fft.rfft(audio_segment * window))
    freqs = np.fft.rfftfreq(len(audio_segment), d=1.0 / sr)
    if np.sum(spec) <= 1e-12:
        return 0.0
    return float(np.sum(freqs * spec) / np.sum(spec))


def merge_nasalized_tokens(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    i = 0

    while i < len(items):
        current = items[i]

        if i + 1 < len(items) and items[i + 1]["token"] == "̃":
            base = current["token"]
            combined = base + "̃"

            if combined in {"ɑ̃", "ɛ̃", "ɔ̃", "œ̃"}:
                merged.append(
                    {
                        "token": combined,
                        "confidence": min(
                            float(current.get("confidence", 1.0)),
                            float(items[i + 1].get("confidence", 1.0)),
                        ),
                        "alternatives": current.get("alternatives", []),
                        "start_sec": current.get("start_sec"),
                        "end_sec": items[i + 1].get("end_sec", current.get("end_sec")),
                        "energy": float(
                            (
                                float(current.get("energy", 0.0))
                                + float(items[i + 1].get("energy", 0.0))
                            ) / 2.0
                        ),
                    }
                )
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def audio_to_phoneme_items(audio: np.ndarray) -> list[dict[str, Any]]:
    raw_items = decode_ctc_with_confidence(audio)

    expanded_items = []
    for item in raw_items:
        split_tokens = split_ipa_compact(item["token"])
        if not split_tokens:
            continue

        dur = max(0.0, item["end_sec"] - item["start_sec"])
        unit = dur / max(len(split_tokens), 1)

        for idx, tok in enumerate(split_tokens):
            start_sec = item["start_sec"] + idx * unit
            end_sec = item["start_sec"] + (idx + 1) * unit
            segment = extract_audio_segment(audio, start_sec, end_sec)
            energy = compute_energy(segment)

            expanded_items.append(
                {
                    "token": tok,
                    "confidence": item["confidence"],
                    "alternatives": item["alternatives"],
                    "start_sec": round(start_sec, 4),
                    "end_sec": round(end_sec, 4),
                    "energy": round(energy, 8),
                }
            )

    expanded_items = merge_nasalized_tokens(expanded_items)
    return expanded_items


# =========================
# ALIGNMENT
# =========================

def align_sequences(expected: list[str], predicted: list[str]) -> list[AlignmentOp]:
    m, n = len(expected), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = ("delete", i - 1, 0)

    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = ("insert", 0, j - 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = ("equal", i - 1, j - 1)
            else:
                substitute = dp[i - 1][j - 1] + 1
                delete = dp[i - 1][j] + 1
                insert = dp[i][j - 1] + 1
                best = min(substitute, delete, insert)
                dp[i][j] = best

                if best == substitute:
                    back[i][j] = ("replace", i - 1, j - 1)
                elif best == delete:
                    back[i][j] = ("delete", i - 1, j)
                else:
                    back[i][j] = ("insert", i, j - 1)

    ops: list[AlignmentOp] = []
    i, j = m, n

    while i > 0 or j > 0:
        move = back[i][j]
        if move is None:
            break

        op_type, _, _ = move

        if op_type == "equal":
            ops.append(AlignmentOp("equal", expected[i - 1], predicted[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op_type == "replace":
            ops.append(AlignmentOp("replace", expected[i - 1], predicted[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op_type == "delete":
            ops.append(AlignmentOp("delete", expected[i - 1], None, i - 1, None))
            i -= 1
        elif op_type == "insert":
            ops.append(AlignmentOp("insert", None, predicted[j - 1], None, j - 1))
            j -= 1
        else:
            raise RuntimeError(f"Unknown op type: {op_type}")

    ops.reverse()
    return ops


# =========================
# ARTICULATORY FEEDBACK
# =========================

def build_articulatory_feedback(
    expected: str | None,
    predicted: str | None,
    status: str,
    energy: float | None,
    acoustic_guess: str | None = None,
) -> ArticulatoryFeedback | None:
    if expected is None:
        return None

    if expected == "ʃ":
        if status in {"uncertain", "mismatch"} and acoustic_guess == "s":
            return ArticulatoryFeedback(
                tongue_position="trop_avant",
                airflow="frication_trop_aigue",
                strength="correcte" if (energy or 0.0) >= LOW_ENERGY_THRESHOLD else "faible",
                message="La langue semble trop avancée. Le son se rapproche de « s » au lieu de « ch »."
            )
        if status == "uncertain":
            return ArticulatoryFeedback(
                tongue_position="a_verifier",
                airflow="frottement_peu_net",
                strength="faible" if (energy or 0.0) < LOW_ENERGY_THRESHOLD else "moyenne",
                message="Le son « ch » n'est pas assez net. La fin du mot semble peu précise."
            )

    if expected == "s":
        if status in {"uncertain", "mismatch"} and acoustic_guess == "ʃ":
            return ArticulatoryFeedback(
                tongue_position="trop_reculée",
                airflow="frication_trop_diffuse",
                strength="correcte" if (energy or 0.0) >= LOW_ENERGY_THRESHOLD else "faible",
                message="La langue semble trop reculée. Le son se rapproche de « ch » au lieu de « s »."
            )

    if expected in {"ɔ̃", "ɑ̃", "ɛ̃", "œ̃"}:
        if status in {"uncertain", "mismatch"}:
            return ArticulatoryFeedback(
                tongue_position=None,
                airflow=None,
                nasalization="insuffisante",
                strength="faible" if (energy or 0.0) < LOW_ENERGY_THRESHOLD else "moyenne",
                message="La nasalisation semble insuffisante ou incomplète."
            )

    if status == "deleted":
        return ArticulatoryFeedback(
            tongue_position=None,
            airflow=None,
            nasalization=None,
            strength="trop_faible",
            message="Le son semble manquant ou avalé."
        )

    if status == "uncertain" and (energy or 0.0) < LOW_ENERGY_THRESHOLD:
        return ArticulatoryFeedback(
            tongue_position=None,
            airflow=None,
            nasalization=None,
            strength="faible",
            message="L'articulation semble trop faible ou peu nette."
        )

    if status == "mismatch":
        return ArticulatoryFeedback(
            tongue_position="a_verifier",
            airflow="a_verifier",
            nasalization=None,
            strength="a_verifier",
            message="Le son produit ne correspond pas assez au phonème attendu."
        )

    return None


# =========================
# WORD / PHONEME MAPPING
# =========================

def build_word_spans(text: str) -> list[dict[str, Any]]:
    words = normalize_text(text).split()
    spans: list[dict[str, Any]] = []
    cursor = 0

    for idx, word in enumerate(words):
        phs = text_to_phonemes(word)
        start = cursor
        end = cursor + len(phs) - 1 if phs else cursor - 1

        spans.append(
            {
                "word": word,
                "word_index": idx,
                "phonemes": phs,
                "start": start,
                "end": end,
            }
        )
        cursor += len(phs)

    return spans


def find_word_for_expected_index(expected_index: int | None, spans: list[dict[str, Any]]) -> tuple[str | None, int | None, int | None]:
    if expected_index is None:
        return None, None, None

    for span in spans:
        if span["start"] <= expected_index <= span["end"]:
            return (
                span["word"],
                span["word_index"],
                expected_index - span["start"],
            )

    return None, None, None


def build_phoneme_analysis(
    ops: list[AlignmentOp],
    spans: list[dict[str, Any]],
    predicted_items: list[dict[str, Any]],
) -> list[PhonemeAnalysisItem]:
    result: list[PhonemeAnalysisItem] = []

    for op in ops:
        word, word_index, phoneme_index_in_word = find_word_for_expected_index(op.expected_index, spans)

        confidence = None
        alternatives: list[PhonemeAlternative] = []
        start_sec = None
        end_sec = None
        energy = None

        if op.predicted_index is not None and 0 <= op.predicted_index < len(predicted_items):
            pred_item = predicted_items[op.predicted_index]
            confidence = pred_item["confidence"]
            start_sec = pred_item.get("start_sec")
            end_sec = pred_item.get("end_sec")
            energy = pred_item.get("energy")
            alternatives = [
                PhonemeAlternative(
                    phoneme=a["phoneme"],
                    score=round(float(a["score"]), 4),
                )
                for a in pred_item["alternatives"][:3]
            ]

        if op.op_type == "equal":
            status = "correct"
            if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
                status = "uncertain"
            if energy is not None and energy < LOW_ENERGY_THRESHOLD:
                status = "uncertain"
        elif op.op_type == "replace":
            status = "mismatch"
        elif op.op_type == "delete":
            status = "deleted"
        else:
            status = "inserted"

        result.append(
            PhonemeAnalysisItem(
                expected=op.expected,
                predicted=op.predicted,
                status=status,
                confidence=round(confidence, 4) if confidence is not None else None,
                expected_index=op.expected_index,
                predicted_index=op.predicted_index,
                word=word,
                word_index=word_index,
                phoneme_index_in_word=phoneme_index_in_word,
                start_sec=round(start_sec, 4) if start_sec is not None else None,
                end_sec=round(end_sec, 4) if end_sec is not None else None,
                energy=round(energy, 8) if energy is not None else None,
                alternatives=alternatives,
                articulatory_feedback=None,
            )
        )

    return result


def rebuild_errors_from_phoneme_analysis(analysis: list[PhonemeAnalysisItem]) -> list[ErrorItem]:
    errors: list[ErrorItem] = []

    for p in analysis:
        if p.status not in {"mismatch", "deleted", "inserted"}:
            continue

        errors.append(
            ErrorItem(
                expected=p.expected,
                predicted=p.predicted,
                type=p.status,
                expected_index=p.expected_index,
                predicted_index=p.predicted_index,
                word=p.word,
                word_index=p.word_index,
                phoneme_index_in_word=p.phoneme_index_in_word,
            )
        )

    return errors


# =========================
# SECONDARY ACOUSTIC CHECK
# =========================

def acoustic_guess_for_sensitive_phoneme(expected_phoneme: str, segment: np.ndarray) -> tuple[str, float, float, float, str]:
    centroid = compute_spectral_centroid(segment)
    energy = compute_energy(segment)

    if energy < 1e-6:
        return "unknown", 0.0, centroid, energy, "segment trop faible"

    if expected_phoneme in {"ʃ", "s"}:
        if centroid >= SH_S_CENTROID_THRESHOLD:
            guess = "s"
            dist = min(1.0, abs(centroid - SH_S_CENTROID_THRESHOLD) / 2000.0)
            conf = 0.55 + 0.4 * dist
        else:
            guess = "ʃ"
            dist = min(1.0, abs(centroid - SH_S_CENTROID_THRESHOLD) / 2000.0)
            conf = 0.55 + 0.4 * dist
        return guess, round(conf, 4), centroid, energy, "heuristique centroid /ʃ/ vs /s/"

    if expected_phoneme in {"ʒ", "z"}:
        if centroid >= ZH_Z_CENTROID_THRESHOLD:
            guess = "z"
            dist = min(1.0, abs(centroid - ZH_Z_CENTROID_THRESHOLD) / 2000.0)
            conf = 0.55 + 0.4 * dist
        else:
            guess = "ʒ"
            dist = min(1.0, abs(centroid - ZH_Z_CENTROID_THRESHOLD) / 2000.0)
            conf = 0.55 + 0.4 * dist
        return guess, round(conf, 4), centroid, energy, "heuristique centroid /ʒ/ vs /z/"

    return expected_phoneme, 0.0, centroid, energy, "pas de contrôle secondaire défini"


def build_acoustic_checks(
    audio: np.ndarray,
    phoneme_analysis: list[PhonemeAnalysisItem],
) -> list[AcousticCheckItem]:
    checks: list[AcousticCheckItem] = []

    for item in phoneme_analysis:
        if item.expected not in SENSITIVE_PHONEMES:
            continue
        if item.start_sec is None or item.end_sec is None:
            continue

        segment = extract_audio_segment(audio, item.start_sec, item.end_sec)
        acoustic_guess, conf, centroid, energy, note = acoustic_guess_for_sensitive_phoneme(
            item.expected,
            segment,
        )

        agrees = acoustic_guess == item.predicted or acoustic_guess == item.expected

        checks.append(
            AcousticCheckItem(
                word=item.word or "",
                word_index=item.word_index or -1,
                expected_phoneme=item.expected or "",
                phoneme_index_in_word=item.phoneme_index_in_word or -1,
                acoustic_guess=acoustic_guess,
                confidence=conf,
                spectral_centroid_hz=round(centroid, 2),
                energy=round(energy, 8),
                start_sec=round(item.start_sec, 3),
                end_sec=round(item.end_sec, 3),
                agrees_with_main_model=agrees,
                note=note,
            )
        )

    return checks


def apply_acoustic_overrides(
    phoneme_analysis: list[PhonemeAnalysisItem],
    acoustic_checks: list[AcousticCheckItem],
) -> list[PhonemeAnalysisItem]:
    updated = list(phoneme_analysis)

    for check in acoustic_checks:
        if check.word_index < 0:
            continue

        for item in updated:
            if (
                item.word_index == check.word_index
                and item.phoneme_index_in_word == check.phoneme_index_in_word
                and item.expected == check.expected_phoneme
            ):
                if item.expected == "ʃ" and check.acoustic_guess == "s":
                    if check.confidence >= ACOUSTIC_DISAGREE_THRESHOLD and item.status == "correct":
                        item.status = "uncertain"

                if item.expected == "s" and check.acoustic_guess == "ʃ":
                    if check.confidence >= ACOUSTIC_DISAGREE_THRESHOLD and item.status == "correct":
                        item.status = "uncertain"

                if item.expected == "ʒ" and check.acoustic_guess == "z":
                    if check.confidence >= ACOUSTIC_DISAGREE_THRESHOLD and item.status == "correct":
                        item.status = "uncertain"

                if item.expected == "z" and check.acoustic_guess == "ʒ":
                    if check.confidence >= ACOUSTIC_DISAGREE_THRESHOLD and item.status == "correct":
                        item.status = "uncertain"

                if item.energy is not None and item.energy < VERY_LOW_ENERGY_THRESHOLD:
                    if item.status == "correct":
                        item.status = "uncertain"

                item.articulatory_feedback = build_articulatory_feedback(
                    expected=item.expected,
                    predicted=item.predicted,
                    status=item.status,
                    energy=item.energy,
                    acoustic_guess=check.acoustic_guess,
                )

    # pour les phonèmes sans acoustic check
    for item in updated:
        if item.articulatory_feedback is None and item.status != "correct":
            item.articulatory_feedback = build_articulatory_feedback(
                expected=item.expected,
                predicted=item.predicted,
                status=item.status,
                energy=item.energy,
                acoustic_guess=None,
            )

    return updated


# =========================
# WORD DIAGNOSTIC
# =========================

def build_word_results(
    spans: list[dict[str, Any]],
    phoneme_analysis: list[PhonemeAnalysisItem],
) -> list[WordResult]:
    results: list[WordResult] = []

    for span in spans:
        word_items = [p for p in phoneme_analysis if p.word_index == span["word_index"]]
        word_errors = [p for p in word_items if p.status in {"mismatch", "deleted", "inserted"}]
        word_uncertain = [p for p in word_items if p.status == "uncertain"]

        phoneme_count = max(len(span["phonemes"]), 1)

        starts = [p.start_sec for p in word_items if p.start_sec is not None]
        ends = [p.end_sec for p in word_items if p.end_sec is not None]
        duration_sec = round(max(ends) - min(starts), 4) if starts and ends else 0.0

        energies = [p.energy for p in word_items if p.energy is not None]
        mean_energy = round(float(np.mean(energies)) if energies else 0.0, 8)

        completeness_score = round(
            max(0.0, 1.0 - (len([e for e in word_errors if e.status == "deleted"]) / phoneme_count)) * 100,
            2,
        )

        weak_count = sum(1 for p in word_items if (p.energy is not None and p.energy < LOW_ENERGY_THRESHOLD))
        low_conf_count = sum(1 for p in word_items if (p.confidence is not None and p.confidence < LOW_CONFIDENCE_THRESHOLD))

        clarity_penalty = (
            len(word_errors)
            + 0.8 * len(word_uncertain)
            + 0.35 * weak_count
            + 0.35 * low_conf_count
        ) / phoneme_count

        clarity_score = round(max(0.0, 1.0 - clarity_penalty) * 100, 2)
        overall_score = round((clarity_score * 0.6) + (completeness_score * 0.4), 2)

        crushed_reasons = []
        if (
            duration_sec > 0
            and duration_sec / phoneme_count < MIN_WORD_DURATION_PER_PHONEME
            and mean_energy < LOW_ENERGY_THRESHOLD
        ):
            crushed_reasons.append("too_short_and_low_energy")
        if clarity_score < 65:
            crushed_reasons.append("low_clarity")
        if completeness_score < 100:
            crushed_reasons.append("incomplete")

        is_crushed = len(crushed_reasons) > 0

        problem_positions: list[ProblemPosition] = []
        for p in word_items:
            if p.status == "correct":
                continue

            if p.status == "mismatch":
                issue = "phoneme_remplace"
            elif p.status == "deleted":
                issue = "phoneme_manquant"
            elif p.status == "inserted":
                issue = "son_ajoute"
            else:
                issue = "phoneme_peu_net"

            # ajustement métier
            if p.phoneme_index_in_word is not None and p.phoneme_index_in_word == len(span["phonemes"]) - 1 and p.status == "uncertain":
                issue = "fin_de_mot_peu_nette"
            if is_crushed and p.status == "uncertain":
                issue = "mot_ecrase"

            problem_positions.append(
                ProblemPosition(
                    phoneme=p.expected,
                    index=p.phoneme_index_in_word,
                    issue=issue,
                    confidence=p.confidence,
                    energy=p.energy,
                    articulatory_feedback=p.articulatory_feedback,
                )
            )

        if not problem_positions and is_crushed:
            last = word_items[-1] if word_items else None
            problem_positions.append(
                ProblemPosition(
                    phoneme=last.expected if last else None,
                    index=last.phoneme_index_in_word if last else None,
                    issue="mot_ecrase",
                    confidence=last.confidence if last else None,
                    energy=last.energy if last else None,
                    articulatory_feedback=last.articulatory_feedback if last else None,
                )
            )

        is_valid = len(word_errors) == 0 and len(word_uncertain) == 0 and not is_crushed

        if is_valid:
            message = f'Le mot "{span["word"]}" semble correct.'
        else:
            reasons = []
            if any(p.issue == "phoneme_remplace" for p in problem_positions):
                reasons.append("phonème remplacé")
            if any(p.issue == "phoneme_manquant" for p in problem_positions):
                reasons.append("phonème manquant")
            if any(p.issue == "son_ajoute" for p in problem_positions):
                reasons.append("son ajouté")
            if any(p.issue == "fin_de_mot_peu_nette" for p in problem_positions):
                reasons.append("fin de mot peu nette")
            if any(p.issue == "mot_ecrase" for p in problem_positions):
                reasons.append("mot écrasé")
            if any(p.issue == "phoneme_peu_net" for p in problem_positions):
                reasons.append("phonème peu net")

            reasons = list(dict.fromkeys(reasons))
            message = f'Le mot "{span["word"]}" n\'est pas validé : ' + ", ".join(reasons)

        results.append(
            WordResult(
                word=span["word"],
                word_index=span["word_index"],
                expected_phonemes=span["phonemes"],
                errors=[
                    ErrorItem(
                        expected=p.expected,
                        predicted=p.predicted,
                        type=p.status,
                        expected_index=p.expected_index,
                        predicted_index=p.predicted_index,
                        word=p.word,
                        word_index=p.word_index,
                        phoneme_index_in_word=p.phoneme_index_in_word,
                    )
                    for p in word_items
                    if p.status in {"mismatch", "deleted", "inserted", "uncertain"}
                ],
                clarity_score=clarity_score,
                completeness_score=completeness_score,
                duration_sec=duration_sec,
                mean_energy=mean_energy,
                is_valid=is_valid,
                is_crushed=is_crushed,
                score=overall_score,
                problem_positions=problem_positions,
                message=message,
            )
        )

    return results


def build_word_reports(words: list[WordResult]) -> list[WordReport]:
    reports: list[WordReport] = []

    for word in words:
        if word.is_valid:
            continue

        details: list[ReportDetail] = []

        for problem in word.problem_positions:
            if problem.issue == "phoneme_remplace":
                severity = "high"
                message = f'Le phonème "{problem.phoneme}" semble remplacé par un autre son.'
                main_reason = "phoneme_remplace"
            elif problem.issue == "phoneme_manquant":
                severity = "high"
                message = f'Le phonème "{problem.phoneme}" semble manquant ou avalé.'
                main_reason = "phoneme_manquant"
            elif problem.issue == "son_ajoute":
                severity = "medium"
                message = f'Un son supplémentaire semble apparaître autour du phonème "{problem.phoneme}".'
                main_reason = "son_ajoute"
            elif problem.issue == "fin_de_mot_peu_nette":
                severity = "medium"
                message = f'La fin du mot est peu nette sur le phonème "{problem.phoneme}".'
                main_reason = "fin_de_mot_peu_nette"
            elif problem.issue == "mot_ecrase":
                severity = "medium"
                message = f'Le mot semble écrasé ou trop peu articulé.'
                main_reason = "mot_ecrase"
            else:
                severity = "low"
                message = f'Le phonème "{problem.phoneme}" reste peu net.'
                main_reason = "phoneme_peu_net"

            details.append(
                ReportDetail(
                    phoneme=problem.phoneme,
                    phoneme_index_in_word=problem.index,
                    issue=problem.issue,
                    severity=severity,
                    confidence=problem.confidence,
                    energy=problem.energy,
                    articulatory_feedback=problem.articulatory_feedback,
                    message=message,
                )
            )

        if details:
            priority = [
                "phoneme_remplace",
                "phoneme_manquant",
                "mot_ecrase",
                "fin_de_mot_peu_nette",
                "phoneme_peu_net",
                "son_ajoute",
            ]
            found = [d.issue for d in details]
            main_reason = next((p for p in priority if p in found), found[0])
        else:
            main_reason = "mot_non_valide"

        if main_reason == "phoneme_remplace":
            msg = f'Le mot "{word.word}" n\'est pas validé : un phonème semble remplacé.'
        elif main_reason == "phoneme_manquant":
            msg = f'Le mot "{word.word}" n\'est pas validé : un phonème semble manquant.'
        elif main_reason == "mot_ecrase":
            msg = f'Le mot "{word.word}" n\'est pas validé : il semble écrasé ou trop peu articulé.'
        elif main_reason == "fin_de_mot_peu_nette":
            msg = f'Le mot "{word.word}" n\'est pas validé : la fin du mot est peu nette.'
        elif main_reason == "phoneme_peu_net":
            msg = f'Le mot "{word.word}" n\'est pas validé : un phonème reste peu net.'
        elif main_reason == "son_ajoute":
            msg = f'Le mot "{word.word}" n\'est pas validé : un son supplémentaire semble présent.'
        else:
            msg = f'Le mot "{word.word}" n\'est pas validé.'

        reports.append(
            WordReport(
                word=word.word,
                word_index=word.word_index,
                is_valid=word.is_valid,
                main_reason=main_reason,
                message=msg,
                details=details,
            )
        )

    return reports


# =========================
# GLOBAL SCORING / DECISION
# =========================

def build_summary(
    expected: list[str],
    predicted: list[str],
    errors: list[ErrorItem],
    phoneme_analysis: list[PhonemeAnalysisItem],
    words: list[WordResult],
) -> dict[str, Any]:
    substitutions = sum(1 for e in errors if e.type == "mismatch")
    deletions = sum(1 for e in errors if e.type == "deleted")
    insertions = sum(1 for e in errors if e.type == "inserted")
    uncertain = sum(1 for p in phoneme_analysis if p.status == "uncertain")
    crushed_words = sum(1 for w in words if w.is_crushed)

    total_expected = max(len(expected), 1)
    error_rate = (
        substitutions
        + deletions
        + insertions
        + 0.35 * uncertain
        + 0.5 * crushed_words
    ) / total_expected

    pronunciation_score = round(max(0.0, 1.0 - error_rate) * 100, 2)

    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "uncertain": uncertain,
        "crushed_words": crushed_words,
        "phoneme_error_rate": round(error_rate, 4),
        "pronunciation_score": pronunciation_score,
    }


def compute_final_decision(words: list[WordResult]) -> FinalDecision:
    invalid_words = [w.word for w in words if not w.is_valid]
    if invalid_words:
        return FinalDecision(
            status="invalid",
            message=f'Le ou les mots suivants ne sont pas validés : {", ".join(invalid_words)}.'
        )
    return FinalDecision(
        status="valid",
        message="Tous les mots sont validés."
    )


# =========================
# ROUTE
# =========================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    input_text: str = Form(...),
    audio_file: UploadFile = File(...),
) -> AnalyzeResponse:
    input_text = normalize_text(input_text)
    if not input_text:
        raise HTTPException(status_code=400, detail="input_text is required.")

    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="audio_file is required.")

    contents = await audio_file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    audio = load_audio_bytes(contents)
    mode = "sentence" if " " in input_text else "word"

    expected_raw = text_to_phonemes_raw(input_text)
    expected_tokens = tokenize_phonemes(expected_raw)

    predicted_items = audio_to_phoneme_items(audio)
    predicted_tokens = [x["token"] for x in predicted_items]
    predicted_raw = " ".join(predicted_tokens)

    ops = align_sequences(expected_tokens, predicted_tokens)
    spans = build_word_spans(input_text)

    phoneme_analysis = build_phoneme_analysis(ops, spans, predicted_items)
    acoustic_checks = build_acoustic_checks(audio, phoneme_analysis)
    phoneme_analysis = apply_acoustic_overrides(phoneme_analysis, acoustic_checks)

    errors = rebuild_errors_from_phoneme_analysis(phoneme_analysis)
    words = build_word_results(spans, phoneme_analysis)
    word_reports = build_word_reports(words)
    summary = build_summary(expected_tokens, predicted_tokens, errors, phoneme_analysis, words)

    pronunciation_score = summary["pronunciation_score"]
    final_decision = compute_final_decision(words)
    is_correct = final_decision.status == "valid"
    message = final_decision.message

    return AnalyzeResponse(
        input_text=input_text,
        mode=mode,
        expected_phonemes_raw=expected_raw,
        predicted_phonemes_raw=predicted_raw,
        expected_phonemes=expected_tokens,
        predicted_phonemes=predicted_tokens,
        errors=errors,
        phoneme_analysis=phoneme_analysis,
        acoustic_checks=acoustic_checks,
        words=words,
        word_reports=word_reports,
        final_decision=final_decision,
        is_correct=is_correct,
        pronunciation_score=pronunciation_score,
        message=message,
        summary=summary,
    )


if __name__ == "__main__":
    print("PHONEMIZER_ESPEAK_PATH =", os.getenv("PHONEMIZER_ESPEAK_PATH"))
    print("PHONEMIZER_ESPEAK_LIBRARY =", os.getenv("PHONEMIZER_ESPEAK_LIBRARY"))
    print("Test mot:", text_to_phonemes("vache"))
    print("Test phrase:", text_to_phonemes("la fourchette tombe"))