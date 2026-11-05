"""Whisper-based WER as an intelligibility-regression metric.

The "pseudo-GT" pattern: transcribe a clean reference clip with Whisper to get a
reference transcript, then transcribe the degraded / enhanced version with the
same model and compute WER between the two. This is the standard way to use ASR
as a speech-enhancement quality signal when ground-truth transcripts aren't
available (cf. Erdogan et al. 2023, Reddy et al. DNS Challenge 2022).

Engine: **faster-whisper** (CTranslate2 backend). The transformers ASR pipeline
ran the large-v3 decode loop on the CPU even with the weights on the GPU (~45 s
for a 12 s clip, ~5 % GPU util — verified 2026-05-30), which dominated every
training run's eval. CTranslate2 actually exercises the GPU (int8_float16) and
does the same clip in ~1-2 s. Same large-v3 weights, converted once. Lazy-loaded
singleton.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

WHISPER_SR = 16000

# faster-whisper model name (size alias like "large-v3" → Systran/faster-whisper-large-v3,
# or a CT2-converted repo id / local path). Override via env var for faster iteration.
# NB: this is NOT the openai/whisper-* HF id — faster-whisper uses its own CT2 repos.
WHISPER_MODEL_ID = os.environ.get("VOICE_CLEANUP_WHISPER_MODEL", "large-v3")

_MODEL: Any = None  # faster_whisper.WhisperModel; typed Any to avoid hard import at module scope


def _fw_repo_id() -> str:
    """The HF repo faster-whisper resolves a size alias to (for cache inspection)."""
    if "/" in WHISPER_MODEL_ID:  # already a full repo id / local path
        return WHISPER_MODEL_ID
    return f"Systran/faster-whisper-{WHISPER_MODEL_ID}"


def _is_whisper_cached() -> tuple[bool, Path | None, int]:
    """Return (cached, path, size_bytes). Walks the HF hub cache for the CT2 model."""
    from huggingface_hub import constants as hf_constants

    cache_root = Path(hf_constants.HF_HUB_CACHE)
    model_dir = cache_root / f"models--{_fw_repo_id().replace('/', '--')}"
    if not model_dir.is_dir():
        return False, None, 0
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False, model_dir, 0
    total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
    has_weights = any(model_dir.rglob("model.bin"))  # CT2 weights file
    return has_weights, model_dir, total


def _get_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # int8_float16 on GPU = the sweet spot (full large-v3 accuracy, ~3 GB, fast); int8 on CPU.
    compute_type = "int8_float16" if device == "cuda" else "int8"

    cached, cache_path, cache_bytes = _is_whisper_cached()
    if cached:
        logger.info(
            "Whisper(faster) · model %s already cached at %s (%.2f GB) — loading from disk",
            _fw_repo_id(), cache_path, cache_bytes / 1024**3,
        )
    else:
        logger.info(
            "Whisper(faster) · model %s NOT cached — first run downloads ~1.5 GB to %s",
            _fw_repo_id(), cache_path or "<HF default cache>",
        )

    logger.info(
        "Whisper(faster) · loading WhisperModel · device=%s · compute_type=%s",
        device, compute_type,
    )
    t0 = time.monotonic()
    _MODEL = WhisperModel(WHISPER_MODEL_ID, device=device, compute_type=compute_type)
    elapsed = time.monotonic() - t0
    cached_after, _, bytes_after = _is_whisper_cached()
    logger.info(
        "Whisper(faster) · model ready in %.1fs · on-disk size %.2f GB%s",
        elapsed, bytes_after / 1024**3,
        " (download completed during load)" if not cached and cached_after else "",
    )
    return _MODEL


def _to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


def _resample_if_needed(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sr, target_sr)
    return resampled.squeeze(0).numpy()


def transcribe(signal: Any, sr: int = WHISPER_SR, *, language: str = "en") -> str:
    """Transcribe a 1D waveform with Whisper. Returns the text string."""
    audio = _to_numpy_1d(signal)
    if sr != WHISPER_SR:
        logger.info("Whisper · resampling %d → %d Hz", sr, WHISPER_SR)
        audio = _resample_if_needed(audio, sr, WHISPER_SR)
    model = _get_model()
    duration_s = len(audio) / WHISPER_SR
    logger.info("Whisper · transcribing %.2fs of audio", duration_s)
    t0 = time.monotonic()
    # faster-whisper takes a 16 kHz float32 mono array directly and returns a lazy
    # generator of segments + an info struct; materialise the segments to get the text.
    segments, _info = model.transcribe(
        audio, language=language, task="transcribe", beam_size=5, vad_filter=False,
    )
    text = " ".join(seg.text for seg in segments).strip()
    logger.info(
        "Whisper · transcribed %.2fs in %.1fs · %d chars",
        duration_s, time.monotonic() - t0, len(text),
    )
    return text


def whisper_wer(
    reference: Any,
    hypothesis: Any,
    sr: int = WHISPER_SR,
    *,
    language: str = "en",
    ref_text: str | None = None,
) -> dict[str, Any]:
    """Compute WER between Whisper transcripts of `reference` and `hypothesis`.

    Args:
        reference: 1D clean reference waveform. Transcribed with Whisper to form
            the pseudo-ground-truth. Pass `ref_text` instead to skip this if you
            already have the transcript cached from an earlier call.
        hypothesis: 1D degraded / enhanced waveform to score.
        sr: Sample rate for both inputs. Resampled to 16 kHz internally.
        language: Whisper language hint (default English).
        ref_text: Optional precomputed reference transcript — saves one Whisper
            pass when scoring many hypotheses against the same reference.

    Returns:
        Dict with keys: wer (float in [0, ∞), 0 = perfect), substitutions,
        deletions, insertions, hits, ref_text, hyp_text, n_ref_words.
    """
    import jiwer

    if ref_text is None:
        ref_text = transcribe(reference, sr=sr, language=language)
    hyp_text = transcribe(hypothesis, sr=sr, language=language)

    if not ref_text.strip():
        logger.warning("Whisper produced empty reference transcript — WER undefined")
        return {
            "wer": float("nan"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "hits": 0,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "n_ref_words": 0,
        }

    transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    measures = jiwer.process_words(
        ref_text,
        hyp_text,
        reference_transform=transforms,
        hypothesis_transform=transforms,
    )
    wer = float(measures.wer)
    logger.info(
        "WER = %.3f · S=%d D=%d I=%d H=%d · ref=%d words",
        wer, measures.substitutions, measures.deletions, measures.insertions,
        measures.hits, len(ref_text.split()),
    )
    return {
        "wer": wer,
        "substitutions": int(measures.substitutions),
        "deletions": int(measures.deletions),
        "insertions": int(measures.insertions),
        "hits": int(measures.hits),
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "n_ref_words": len(ref_text.split()),
    }


__all__ = ["transcribe", "whisper_wer", "WHISPER_MODEL_ID", "WHISPER_SR"]
