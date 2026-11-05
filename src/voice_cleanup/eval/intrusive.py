"""Intrusive (reference-based) speech quality metrics.

PESQ wideband (ITU-T P.862.2), Extended STOI, and SI-SDR.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

logger = logging.getLogger(__name__)

PESQ_WB_SR = 16000


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


def _align(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


def pesq_wb(ref: Any, deg: Any, sr: int = PESQ_WB_SR) -> float:
    """Wideband PESQ (ITU-T P.862.2). Score range ~[-0.5, 4.5]; higher is better."""
    if sr != PESQ_WB_SR:
        raise ValueError(f"PESQ wideband requires {PESQ_WB_SR} Hz; got {sr}")
    ref_a, deg_a = _align(_to_numpy(ref), _to_numpy(deg))
    logger.info(
        "PESQ-wb · sr=%d Hz · len=%.2fs (%d samples) · ref|deg-rms=%.4f|%.4f",
        sr, len(ref_a) / sr, len(ref_a), float(np.sqrt(np.mean(ref_a**2))),
        float(np.sqrt(np.mean(deg_a**2))),
    )
    t0 = time.monotonic()
    score = float(_pesq(sr, ref_a, deg_a, mode="wb"))
    logger.info("PESQ-wb = %.3f (computed in %.0f ms)", score, (time.monotonic() - t0) * 1000)
    return score


def estoi(ref: Any, deg: Any, sr: int = PESQ_WB_SR) -> float:
    """Extended STOI (Jensen & Taal 2016). Range ~[0, 1]; higher is better."""
    ref_a, deg_a = _align(_to_numpy(ref), _to_numpy(deg))
    logger.info("ESTOI · sr=%d Hz · len=%.2fs (%d samples)", sr, len(ref_a) / sr, len(ref_a))
    t0 = time.monotonic()
    score = float(_stoi(ref_a, deg_a, sr, extended=True))
    logger.info("ESTOI = %.4f (computed in %.0f ms)", score, (time.monotonic() - t0) * 1000)
    return score


def si_sdr(ref: Any, deg: Any, eps: float = 1e-8) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019). dB; higher is better.

    Both signals are mean-centered first so a DC offset doesn't dominate the projection.
    """
    ref_a, deg_a = _align(_to_numpy(ref), _to_numpy(deg))
    logger.info("SI-SDR · len=%d samples", len(ref_a))
    ref_c = ref_a - ref_a.mean()
    deg_c = deg_a - deg_a.mean()
    alpha = np.dot(deg_c, ref_c) / (np.dot(ref_c, ref_c) + eps)
    s_target = alpha * ref_c
    e_noise = deg_c - s_target
    score = float(
        10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps))
    )
    logger.info("SI-SDR = %+.2f dB · projection scale alpha=%.4f", score, alpha)
    return score
