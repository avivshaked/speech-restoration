"""Taming post-filter — compose any enhancer with a high-shelf cut to pull down an
over-bright top end.

Aviv's ear on the first bazaar (2026-05-31): VoiceFixer is the front-runner at
*regenerating voice-like content from degraded input*, but it's "too excited" — whoever
trained it loved high frequencies. That's a tonal-balance flaw, not a generation flaw, so
it's fixable with a static high-shelf cut rather than retraining. The diagnostic agrees:
VoiceFixer's clean-input HF-ratio was ~+4–5 dB hotter in the top octave than the clean source.

`TamedEnhancer` runs a base enhancer, then applies a single high-shelf (torchaudio
`treble_biquad`, a shelving filter) with a negative gain above `freq`. Strengths are
registered as named presets so the bazaar can A/B them by ear; the right amount is the ear's
call, not a number's. This is deliberately a cheap, deterministic band-aid to *gauge whether a
tamed VoiceFixer is the one* before investing in a proper learned refiner / fine-tune.
"""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
import torch
import torchaudio.functional as AF

from voice_cleanup.eval.enhancers import Enhancer, get_enhancer, register

logger = logging.getLogger(__name__)


def high_shelf(audio: np.ndarray, sr: int, gain_db: float, freq: float,
               q: float = 0.707) -> np.ndarray:
    """Apply a high-shelf (treble) filter. Negative gain_db cuts content above `freq`."""
    y = AF.treble_biquad(torch.from_numpy(np.asarray(audio, dtype="float32")), sr,
                         gain_db, central_freq=freq, Q=q)
    return y.numpy().astype("float32")


class TamedEnhancer(Enhancer):
    """Run `base` enhancer, then high-shelf-cut the top end to tame brightness."""

    name = "tamed"

    def __init__(self, base: str = "voicefixer", gain_db: float = -5.0, freq: float = 5000.0,
                 q: float = 0.707, **base_kwargs) -> None:
        self._base_name = base
        self._base = get_enhancer(base, **base_kwargs)
        self.gain_db = float(gain_db)
        self.freq = float(freq)
        self.q = float(q)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        y = self._base.apply(audio, sr)
        logger.info("Tame · %s + high-shelf %.1f dB @ %.0f Hz (Q=%.2f)",
                    self._base_name, self.gain_db, self.freq, self.q)
        return high_shelf(y, sr, self.gain_db, self.freq, self.q)

    def label(self) -> str:
        return f"{self._base_name} + shelf {self.gain_db:+.0f}dB@{self.freq:.0f}Hz"


def adaptive_cut_db(src_hf_ratio_db: float, max_cut_db: float = -4.0,
                    hf_dull: float = -20.0, hf_bright: float = -7.0) -> float:
    """How much to high-shelf-cut, scaled by how bright the SOURCE is.

    Encodes Aviv's ear-rule (2026-05-31): VoiceFixer adds a roughly fixed top-end boost, so a
    DULL source (bandwidth-killed "radio voice", HF-ratio ≪ natural) is merely restored to
    natural → don't cut; an already-BRIGHT source is pushed over the top → cut. Linearly ramp
    the cut from 0 at `hf_dull` to `max_cut_db` at `hf_bright` (≈ the clean-speech top-octave
    level). `src_hf_ratio_db` is the source's 8–16 kHz-vs-speech-band ratio (eval.spectral).
    """
    frac = (src_hf_ratio_db - hf_dull) / (hf_bright - hf_dull)
    frac = min(1.0, max(0.0, frac))
    return max_cut_db * frac


class AdaptiveTamedEnhancer(Enhancer):
    """Run `base`, measure the SOURCE brightness, high-shelf-cut proportionally (Aviv's rule)."""

    name = "tamed-auto"

    def __init__(self, base: str = "voicefixer", max_cut_db: float = -4.0, freq: float = 5500.0,
                 hf_dull: float = -20.0, hf_bright: float = -7.0, q: float = 0.707,
                 **base_kwargs) -> None:
        self._base_name = base
        self._base = get_enhancer(base, **base_kwargs)
        self.max_cut_db = float(max_cut_db)
        self.freq = float(freq)
        self.hf_dull = float(hf_dull)
        self.hf_bright = float(hf_bright)
        self.q = float(q)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from voice_cleanup.eval.spectral import spectral_summary  # noqa: PLC0415
        src_hf = float(spectral_summary(np.asarray(audio, dtype="float32"), sr)["hf_ratio_db"])
        cut = adaptive_cut_db(src_hf, self.max_cut_db, self.hf_dull, self.hf_bright)
        y = self._base.apply(audio, sr)
        logger.info("Tame-auto · src HF-ratio %.1f dB → cut %.1f dB @ %.0f Hz",
                    src_hf, cut, self.freq)
        return high_shelf(y, sr, cut, self.freq, self.q) if cut < -0.1 else y

    def label(self) -> str:
        return f"{self._base_name} + adaptive shelf (≤{self.max_cut_db:+.0f}dB@{self.freq:.0f}Hz)"


# Named presets for the bazaar (partial is a valid registry value — get_enhancer calls it()).
# soft → med → hard taming of VoiceFixer's top octave.
_vf = partial(TamedEnhancer, base="voicefixer")
register("voicefixer-tame-soft", partial(_vf, gain_db=-3.0, freq=6000.0))
register("voicefixer-tame", partial(_vf, gain_db=-5.0, freq=5000.0))
register("voicefixer-tame-hard", partial(_vf, gain_db=-7.0, freq=4500.0))
# Source-adaptive tame — the one to listen to (encodes Aviv's "tame bright sources, leave dull
# ones" rule): cut ramps 0 → −4 dB as the source brightens. Max cut is intentionally modest
# (his "best is between voicefixer and -tame").
register("voicefixer-tame-auto", partial(AdaptiveTamedEnhancer, base="voicefixer"))

__all__ = ["AdaptiveTamedEnhancer", "TamedEnhancer", "adaptive_cut_db", "high_shelf"]
