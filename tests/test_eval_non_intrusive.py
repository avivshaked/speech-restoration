"""Tests for non-intrusive metrics (DNSMOS P.835, SQUIM-MOS).

These hit real models — DNSMOS ONNX on CPU and torchaudio's SQUIM_SUBJECTIVE.
Marked `slow` because loading SQUIM weights takes a few seconds on first run.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_cleanup.eval.non_intrusive import dnsmos_p835, squim_mos


def _speech_like(duration: float = 10.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Voiced-harmonics signal with a slow envelope — DNSMOS/SQUIM accept it.

    DNSMOS requires >= 9.01s, so default duration is 10s.
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


# --- DNSMOS P.835 ---


def test_dnsmos_returns_expected_keys():
    x = _speech_like()
    result = dnsmos_p835(x)
    for key in ("sig", "bak", "ovrl", "sig_raw", "bak_raw", "ovrl_raw", "num_hops"):
        assert key in result
    assert result["num_hops"] >= 1


def test_dnsmos_scores_in_mos_range():
    x = _speech_like()
    result = dnsmos_p835(x)
    for key in ("sig", "bak", "ovrl"):
        assert 1.0 <= result[key] <= 5.0, f"{key}={result[key]} outside [1, 5]"


def test_dnsmos_monotonic_in_snr():
    x = _speech_like()
    clean = dnsmos_p835(x)
    noisy = dnsmos_p835(_add_noise(x, snr_db=-5.0, seed=2))
    assert clean["ovrl"] > noisy["ovrl"]
    assert clean["bak"] > noisy["bak"]


def test_dnsmos_pads_short_signal():
    """Signals shorter than 9.01s are repeat-padded; should still produce a score."""
    short = _speech_like(duration=3.0)
    result = dnsmos_p835(short)
    assert 1.0 <= result["ovrl"] <= 5.0


def test_dnsmos_handles_torch_input():
    import torch

    x = torch.from_numpy(_speech_like())
    result = dnsmos_p835(x)
    assert 1.0 <= result["ovrl"] <= 5.0


def test_dnsmos_resamples_non_16k_input():
    x = _speech_like(duration=10.0, sr=48000)
    result = dnsmos_p835(x, sr=48000)
    assert 1.0 <= result["ovrl"] <= 5.0


# --- SQUIM-MOS ---


@pytest.mark.slow
def test_squim_mos_returns_float_in_range():
    x = _speech_like(duration=5.0)
    ref = _speech_like(duration=5.0, seed=99)
    score = squim_mos(x, ref)
    assert isinstance(score, float)
    assert 1.0 <= score <= 5.0


# Note: SQUIM monotonicity-in-SNR validation moved to test_eval_with_real_audio.py.
# SQUIM is calibrated on real speech and gives near-constant scores on synthetic
# harmonic stacks; the real-audio version of this test is the meaningful one.
