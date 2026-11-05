"""Tests for intrusive metrics."""

from __future__ import annotations

import numpy as np
import pytest

from voice_cleanup.eval.intrusive import PESQ_WB_SR, estoi, pesq_wb, si_sdr

SR = PESQ_WB_SR


def _speech_like(duration: float = 3.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Synthetic speech-ish signal: voiced harmonics with a slow amplitude envelope.

    PESQ rejects pure noise / pure tones, so we need something with voiced structure.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 140.0
    sig = np.zeros_like(t)
    for harmonic in range(1, 8):
        sig += np.sin(2 * np.pi * f0 * harmonic * t) / harmonic
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    sig = sig * envelope
    sig += 0.01 * rng.standard_normal(sig.shape)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.5
    return sig.astype(np.float32)


def _add_noise(x: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape)
    sig_power = np.mean(x**2)
    noise_power = np.mean(noise**2)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-12))
    return (x + noise).astype(np.float32)


# --- SI-SDR ---


def test_si_sdr_identical_signal_is_high():
    x = _speech_like()
    assert si_sdr(x, x) > 60.0


def test_si_sdr_scale_invariance():
    x = _speech_like()
    noisy = _add_noise(x, snr_db=10.0)
    assert si_sdr(x, noisy) == pytest.approx(si_sdr(x, 3.7 * noisy), abs=1e-6)


def test_si_sdr_noisy_is_lower_than_clean():
    x = _speech_like()
    noisy = _add_noise(x, snr_db=5.0)
    assert si_sdr(x, noisy) < si_sdr(x, x)


def test_si_sdr_monotonic_in_snr():
    x = _speech_like()
    score_high = si_sdr(x, _add_noise(x, snr_db=20.0, seed=2))
    score_low = si_sdr(x, _add_noise(x, snr_db=0.0, seed=2))
    assert score_high > score_low


# --- ESTOI ---


def test_estoi_identical_signal_is_one():
    x = _speech_like()
    assert estoi(x, x) == pytest.approx(1.0, abs=1e-6)


def test_estoi_monotonic_in_snr():
    x = _speech_like()
    score_high = estoi(x, _add_noise(x, snr_db=20.0, seed=3))
    score_low = estoi(x, _add_noise(x, snr_db=-5.0, seed=3))
    assert score_high > score_low
    assert 0.0 <= score_low <= 1.0
    assert 0.0 <= score_high <= 1.0


# --- PESQ ---


def test_pesq_wb_identical_signal_is_near_max():
    x = _speech_like()
    score = pesq_wb(x, x)
    assert score > 4.0


def test_pesq_wb_monotonic_in_snr():
    x = _speech_like()
    score_high = pesq_wb(x, _add_noise(x, snr_db=20.0, seed=4))
    score_low = pesq_wb(x, _add_noise(x, snr_db=0.0, seed=4))
    assert score_high > score_low


def test_pesq_wb_rejects_wrong_sample_rate():
    x = _speech_like()
    with pytest.raises(ValueError, match="16000"):
        pesq_wb(x, x, sr=8000)


# --- input handling ---


def test_length_mismatch_is_truncated():
    x = _speech_like(duration=3.0)
    y = x[: len(x) - 1000]
    assert si_sdr(x, y) > 60.0
    assert estoi(x, y) == pytest.approx(1.0, abs=1e-3)


def test_rejects_2d_input():
    x = _speech_like()
    with pytest.raises(ValueError, match="1D"):
        si_sdr(x[None, :], x)
