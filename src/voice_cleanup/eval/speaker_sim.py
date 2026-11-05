"""Speaker-similarity metric — does the restored speech still sound like the speaker?

A **verdict-tier** metric for the restoration eval (CLAUDE.md panel, added
2026-05-28). Generative restorers regenerate speech rather than filter it, so the
real failure mode is no longer "did noise survive?" but "did the model invent a
*different voice*?" — timbre drift, over-smoothing, speaker leakage. Intrusive
metrics (PESQ/SI-SDR) can't see this (they penalise any resynthesis); a speaker
embedding can.

We embed both clips with a speaker-verification model and take the **cosine
similarity** of the x-vectors (∈ [-1, 1]; 1 = same speaker). Two uses:
- studio-reference track: similarity(clean studio reference, restored output) —
  how much speaker identity the restoration preserved vs the *true* voice.
- real-podcast track: similarity(noisy input, restored output) — did enhancement
  drift the voice away from the source.

Model: **microsoft/wavlm-base-plus-sv** (WavLM-Base+ fine-tuned for SV with an
x-vector head; the model CLAUDE.md flagged WavLM-vs-ECAPA for — WavLM picked as it
needs no extra speechbrain dependency). Expects 16 kHz mono; we resample in.
Lazy-loaded singleton; model-load verbosity contract (cache hit/miss + size +
timing) per feedback_model_load_verbosity.
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

SPEAKER_SIM_SR = 16000

# WavLM-Base+ SV head. Override via env var for experiments (e.g. an ECAPA export).
SPEAKER_SIM_MODEL_ID = os.environ.get(
    "VOICE_CLEANUP_SPEAKER_SIM_MODEL", "microsoft/wavlm-base-plus-sv"
)

_MODEL: Any = None
_FEATURE_EXTRACTOR: Any = None


def _is_model_cached() -> tuple[bool, Path | None, int]:
    """Return (cached, path, size_bytes). Walks the HF hub cache (mirrors wer.py)."""
    from huggingface_hub import constants as hf_constants

    cache_root = Path(hf_constants.HF_HUB_CACHE)
    model_dir = cache_root / f"models--{SPEAKER_SIM_MODEL_ID.replace('/', '--')}"
    if not model_dir.is_dir():
        return False, None, 0
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False, model_dir, 0
    total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
    has_weights = any(model_dir.rglob("model.safetensors")) or any(
        model_dir.rglob("pytorch_model.bin")
    )
    return has_weights, model_dir, total


def _get_model() -> tuple[Any, Any]:
    """Lazily load + cache (model, feature_extractor)."""
    global _MODEL, _FEATURE_EXTRACTOR
    if _MODEL is not None:
        return _MODEL, _FEATURE_EXTRACTOR

    from transformers import AutoFeatureExtractor, WavLMForXVector

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cached, cache_path, cache_bytes = _is_model_cached()
    if cached:
        logger.info(
            "SpeakerSim · model %s already cached at %s (%.0f MB) — loading from disk · "
            "license=MIT (microsoft/wavlm-base-plus-sv)",
            SPEAKER_SIM_MODEL_ID, cache_path, cache_bytes / 1024**2,
        )
    else:
        logger.info(
            "SpeakerSim · model %s NOT cached — first run downloads ~380 MB to %s",
            SPEAKER_SIM_MODEL_ID, cache_path or "<HF default cache>",
        )

    logger.info(
        "SpeakerSim · initializing WavLM x-vector model · device=%s · dtype=float32", device
    )
    t0 = time.monotonic()
    fe = AutoFeatureExtractor.from_pretrained(SPEAKER_SIM_MODEL_ID)
    model = WavLMForXVector.from_pretrained(SPEAKER_SIM_MODEL_ID).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    _, _, bytes_after = _is_model_cached()
    logger.info(
        "SpeakerSim · model ready in %.1fs · %.1fM params · on-disk %.0f MB",
        time.monotonic() - t0, n_params / 1e6, bytes_after / 1024**2,
    )
    _MODEL, _FEATURE_EXTRACTOR = model, fe
    return _MODEL, _FEATURE_EXTRACTOR


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


def _embed(signal: np.ndarray, sr: int) -> torch.Tensor:
    """Compute the L2-normalised x-vector embedding for one waveform."""
    audio = _to_numpy_1d(signal)
    if sr != SPEAKER_SIM_SR:
        audio = _resample_if_needed(audio, sr, SPEAKER_SIM_SR)
    model, fe = _get_model()
    device = next(model.parameters()).device
    inputs = fe(
        [audio], sampling_rate=SPEAKER_SIM_SR, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        emb = model(**inputs).embeddings  # (1, D)
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu()


def speaker_similarity(reference: Any, hypothesis: Any, sr: int = SPEAKER_SIM_SR) -> float:
    """Cosine similarity of speaker embeddings for two waveforms.

    Args:
        reference: 1D waveform — the identity to preserve (clean studio reference on
            the studio-reference track; the noisy source on the real-podcast track).
        hypothesis: 1D waveform — the restored / enhanced output to check.
        sr: sample rate of both inputs (resampled to 16 kHz internally).

    Returns:
        Cosine similarity ∈ [-1, 1]; 1.0 = indistinguishable speaker, lower = the
        output drifted away from the reference voice.
    """
    t0 = time.monotonic()
    emb_ref = _embed(reference, sr)
    emb_hyp = _embed(hypothesis, sr)
    cos = float(torch.dot(emb_ref, emb_hyp).clamp(-1.0, 1.0))
    logger.info("SpeakerSim · cosine=%.4f · computed in %.2fs", cos, time.monotonic() - t0)
    return cos


__all__ = ["speaker_similarity", "SPEAKER_SIM_MODEL_ID", "SPEAKER_SIM_SR"]
