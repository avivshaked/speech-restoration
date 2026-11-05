"""Spectral diagnostics — the blind spot DNSMOS and the whole non-intrusive panel lack.

DNSMOS can score a badly bandwidth- and EQ-damaged "radio voice" clip as clean, because there
is no NOISE: it is blind to spectral/coloration/bandwidth damage. These functions measure the
spectral envelope
directly so "radio voice" damage is catchable and top-octave regeneration is measurable.

All operate on 1-D float32 numpy at a given sample rate. Reference-free.
"""
from __future__ import annotations

import numpy as np
import torch


def ltas(x: np.ndarray, sr: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average spectrum: (freqs_hz, magnitude_db). Averaged over all STFT frames."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = torch.hann_window(n_fft)
    spec = torch.stft(torch.from_numpy(x), n_fft=n_fft, hop_length=n_fft // 2,
                      window=win, return_complex=True).abs()
    mag = spec.mean(dim=1).numpy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    db = 20.0 * np.log10(mag + 1e-10)
    return freqs, db


def band_energy(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float, n_fft: int = 4096) -> float:
    """Mean linear magnitude in [lo, hi] Hz."""
    freqs, db = ltas(x, sr, n_fft)
    lin = 10.0 ** (db / 20.0)
    sel = (freqs >= lo_hz) & (freqs < hi_hz)
    return float(lin[sel].mean()) if sel.any() else 0.0


def hf_energy_ratio(x: np.ndarray, sr: int, *, hf_lo: float = 8000.0, hf_hi: float = 16000.0,
                    ref_lo: float = 100.0, ref_hi: float = 8000.0) -> float:
    """Top-octave energy vs the speech band, in dB. Higher = more top octave present.

    The headline restoration diagnostic: a bandwidth-damaged "radio voice" sits very low;
    studio-clean speech and good regeneration sit high.
    """
    hi = band_energy(x, sr, hf_lo, hf_hi)
    lo = band_energy(x, sr, ref_lo, ref_hi)
    return float(20.0 * np.log10((hi + 1e-12) / (lo + 1e-12)))


def spectral_tilt_db_per_oct(x: np.ndarray, sr: int) -> float:
    """Overall spectral tilt (dB/octave) from a log-freq linear fit over 100 Hz–nyquist.

    A flat natural voice tilts gently; a resonant/boxy "radio voice" or a heavily damaged
    envelope deviates. Diagnostic, not a verdict.
    """
    freqs, db = ltas(x, sr)
    sel = (freqs >= 100.0) & (freqs < sr / 2.0 * 0.98)
    f, d = freqs[sel], db[sel]
    log2f = np.log2(f + 1e-9)
    a = np.polyfit(log2f, d, 1)[0]
    return float(a)


def estimate_snr_db(x: np.ndarray, sr: int) -> float:
    """Crude SNR estimate: ratio of speech-active (top-quartile-energy frames) to the noise
    floor (10th-percentile frames), in dB. For corpus quality-gating, not a calibrated SNR."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(sr * 0.02))
    nf = len(x) // hop
    if nf < 4:
        return float("nan")
    fe = np.sqrt((x[: nf * hop].reshape(nf, hop) ** 2).mean(axis=1) + 1e-12)
    floor = np.percentile(fe, 10)
    active = np.percentile(fe, 75)
    return float(20.0 * np.log10((active + 1e-9) / (floor + 1e-9)))


def crest_factor_db(x: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. Clipping/saturation crush it; clean speech is high (~15-22 dB)."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) + 1e-12
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return float(20.0 * np.log10(peak / rms))


def spectral_summary(x: np.ndarray, sr: int) -> dict:
    """All scalar spectral diagnostics for one signal."""
    return {
        "hf_ratio_db": hf_energy_ratio(x, sr),
        "tilt_db_per_oct": spectral_tilt_db_per_oct(x, sr),
        "snr_db": estimate_snr_db(x, sr),
        "crest_db": crest_factor_db(x),
    }


__all__ = [
    "ltas", "band_energy", "hf_energy_ratio", "spectral_tilt_db_per_oct",
    "estimate_snr_db", "crest_factor_db", "spectral_summary",
]
