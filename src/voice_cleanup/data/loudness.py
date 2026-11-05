"""Loudness normalization (ITU-R BS.1770 / EBU R128 LUFS).

Aviv (ex-sound-designer) flagged 2026-05-29 that EARS/DAPS clips are *not* level-
calibrated to each other. That biases two things: (1) the A/B listening pack —
the louder file always "sounds better", so you end up judging level not quality;
(2) the studio-reference set as a coherent "studio target" — every clip a
different loudness isn't a target. The fix is integrated-loudness normalization to
a common LUFS, the broadcast/podcast standard (not peak or RMS, which ignore
perceptual frequency weighting).

We use `pyloudnorm` (Steinmetz & Reiss — a BS.1770-4 meter). Default target
**-23 LUFS** (EBU R128 broadcast). Podcast delivery often targets -16 LUFS; the
target is a knob. Short/near-silent clips (below the meter's gating block size)
are returned unchanged with a warning rather than blowing up the gain.

CAVEAT (Aviv, ex-sound-designer, 2026-05-29): integrated LUFS equalizes a single
whole-clip loudness number — it does NOT make speech sound level *throughout*, on
two counts:
  - **Dynamics:** it's one number over the whole clip; moment-to-moment level
    (crest factor, gain-riding) is what compression exists for. Pure scalar gain
    can't fix it.
  - **Spectrum:** the ear isn't flat (equal-loudness contours) — it peaks in the
    ~1-4 kHz presence band and is far less sensitive to bass. So a bass-heavy
    voice needs *more* energy to sound as loud as a 600 Hz-forward voice at the
    same measured level. BS.1770's K-weighting (high-pass + HF shelf) is a coarse,
    fixed nod to this and beats RMS/peak — but it's a single curve, doesn't track
    level-dependent contours, and under-weights midrange presence. Two voices can
    match to the same LUFS and still not sound equally loud.
So this is **pure gain, no compression** — we only remove the gross *inter-clip*
offset (the ~12 dB spread between corpora) so it can't dominate the A/B listen or
level-sensitive metrics. It is one defensible, reproducible baseline, NOT a claim
of perceptual equivalence (the ear stays the arbiter) — and it's optional
(`target_lufs=None` disables it).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

TARGET_LUFS = -23.0  # EBU R128 broadcast integrated loudness
_MIN_SECONDS = 0.4   # pyloudnorm's gating block is 0.4 s; shorter → undefined


@lru_cache(maxsize=8)
def _meter(sr: int):
    """Cache a BS.1770 meter per sample rate (filter design is non-trivial)."""
    import pyloudnorm as pyln

    return pyln.Meter(sr)


def integrated_lufs(wav: np.ndarray, sr: int) -> float:
    """Integrated loudness (LUFS) of a 1-D signal; -inf for silence."""
    x = np.asarray(wav, dtype=np.float64)
    return float(_meter(sr).integrated_loudness(x))


def normalize_lufs(
    wav: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS, *, peak_ceiling: float = 0.98
) -> np.ndarray:
    """Scale `wav` to `target_lufs` integrated loudness.

    Pure gain (no compression), so timbre/dynamics are untouched — only the level
    moves. If applying the gain would clip, we back off so the peak sits at
    `peak_ceiling` (and the achieved loudness ends up below target — logged). Clips
    too short to measure, or silent, are returned unchanged.
    """
    x = np.asarray(wav, dtype=np.float32)
    if x.size < int(_MIN_SECONDS * sr):
        logger.warning(
            "Loudness · clip %.2fs shorter than %.2fs gating block — leaving level unchanged",
            x.size / sr, _MIN_SECONDS,
        )
        return x
    loudness = integrated_lufs(x, sr)
    if not np.isfinite(loudness):
        logger.warning("Loudness · non-finite loudness (silence?) — leaving level unchanged")
        return x

    gain_db = target_lufs - loudness
    gain = 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(x))) + 1e-12
    if peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
        logger.info(
            "Loudness · %.1f→%.1f LUFS would clip (peak %.3f); capping gain → ~%.1f LUFS",
            loudness, target_lufs, peak, loudness + 20.0 * np.log10(gain),
        )
    else:
        logger.debug("Loudness · %.1f→%.1f LUFS (gain %.2f dB)", loudness, target_lufs, gain_db)
    return (x * gain).astype(np.float32)


__all__ = ["integrated_lufs", "normalize_lufs", "TARGET_LUFS"]
