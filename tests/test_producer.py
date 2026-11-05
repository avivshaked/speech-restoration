"""Tests for the Producer (models/producer.py), the multi-res STFT loss (losses/spectral.py),
and the sentence-bounded eval window (data/daps.load_aligned_speech_window).

All fast (synthetic tensors / tiny temp wavs) — no real corpora, no GPU required.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from voice_cleanup.data.daps import SR, load_aligned_speech_window
from voice_cleanup.losses.spectral import multi_resolution_stft_loss
from voice_cleanup.models.producer import SpectralProducer, target_logmag


# ----------------------------------------------------------------------- model
def test_producer_identity_at_init() -> None:
    """Zero-init head ⇒ Δ≡0 ⇒ output reconstructs the input (the right prior)."""
    torch.manual_seed(0)
    m = SpectralProducer()
    x = torch.randn(2, SR) * 0.1
    out = m(x)
    assert out["delta"].abs().max().item() == 0.0  # exactly zero at init
    roundtrip = m._istft(m._stft(x), length=x.shape[-1])
    assert torch.allclose(out["wav_out"], roundtrip, atol=1e-5)


def test_producer_shapes_and_param_budget() -> None:
    m = SpectralProducer()
    x = torch.randn(3, SR // 2)
    out = m(x)
    assert out["wav_out"].shape == x.shape
    assert out["in_logmag"].shape == out["out_logmag"].shape == out["delta"].shape
    # "small" model — guard against an accidental blow-up in width/depth.
    assert m.num_params() < 10_000_000


def test_producer_delta_is_clamped() -> None:
    """After a few optimiser steps toward a loud target, Δ stays inside [delta_min, delta_max]."""
    torch.manual_seed(0)
    m = SpectralProducer(delta_min=-12.0, delta_max=3.0)
    opt = torch.optim.SGD(m.parameters(), lr=10.0)  # aggressive, to push the clamp
    x = torch.randn(2, SR // 2) * 0.05
    tgt = x * 8.0  # much louder target → wants a big positive Δ
    for _ in range(5):
        out = m(x)
        loss = (out["out_logmag"] - target_logmag(m, tgt)).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    d = m(x)["delta"].detach()
    assert float(d.min()) >= -12.0 - 1e-4
    assert float(d.max()) <= 3.0 + 1e-4


def test_producer_enhance_accepts_1d_and_2d() -> None:
    m = SpectralProducer()
    one = m.enhance(torch.randn(SR // 2))
    assert one.dim() == 1
    batch = m.enhance(torch.randn(2, SR // 2))
    assert batch.dim() == 2


# ------------------------------------------------------------------------ loss
def test_mrstft_zero_for_identical_positive_for_different() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, SR // 2)
    same = multi_resolution_stft_loss(a, a.clone())
    diff = multi_resolution_stft_loss(a, torch.randn_like(a))
    assert float(same) < 1e-4
    assert float(diff) > float(same)


# ---------------------------------------------------------------- eval window
def _tone(freq: float, secs: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def _silence(secs: float) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype="float32")


def test_speech_window_skips_lead_in_silence_and_aligns(tmp_path) -> None:
    """Build an aligned pair = [4s silence | 3s tone | 1s gap | 3s tone | 2s silence].
    A 6s window must skip the lead-in silence and land on the speech, identically on both."""
    target = np.concatenate([
        _silence(4.0), _tone(300, 3.0), _silence(1.0), _tone(300, 3.0), _silence(2.0),
    ])
    noise = np.random.default_rng(0).standard_normal(len(target)).astype("float32")
    degraded = target + 0.01 * noise
    tp, dp = tmp_path / "t.wav", tmp_path / "d.wav"
    sf.write(str(tp), target, SR)
    sf.write(str(dp), degraded, SR)

    tgt_win, deg_win = load_aligned_speech_window(tp, dp, 6.0)

    assert len(tgt_win) == len(deg_win)  # same crop on both streams
    # the window must be mostly speech, not the 4s lead-in silence
    rms = float(np.sqrt(np.mean(tgt_win**2)))
    assert rms > 0.1
    # and degraded is the same region (aligned): high correlation
    corr = float(np.corrcoef(tgt_win, deg_win)[0, 1])
    assert corr > 0.9


def test_speech_window_returns_whole_clip_when_short(tmp_path) -> None:
    target = _tone(200, 2.0)
    degraded = target.copy()
    tp, dp = tmp_path / "t.wav", tmp_path / "d.wav"
    sf.write(str(tp), target, SR)
    sf.write(str(dp), degraded, SR)
    tgt_win, deg_win = load_aligned_speech_window(tp, dp, 10.0)  # ask for more than exists
    assert len(tgt_win) == len(target)
