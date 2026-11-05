"""Tests for the restoration degradation pipeline (src/voice_cleanup/data/degrade.py).

Effects must be: finite, length-preserving, float32, 1-D, and reproducible from a
seed. The Codec round-trip needs an ffmpeg binary — those tests skip cleanly when
none is available (CI without imageio-ffmpeg), but run normally with the uv-managed
build present.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_cleanup.data.degrade import (
    RECIPE_NAMES,
    AdditiveNoise,
    BandwidthLimit,
    Clipping,
    Codec,
    DegradationChain,
    EqDamage,
    Saturation,
    SyntheticReverb,
    _ffmpeg_exe,
    make_chain,
)

SR = 48000


def _speechish(seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    """A deterministic voiced-ish signal (two harmonics) for effect tests."""
    t = np.arange(int(sr * seconds)) / sr
    wav = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 1400 * t)
    return wav.astype(np.float32)


def _assert_clean_output(out: np.ndarray, n: int) -> None:
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "effect",
    [
        Clipping(),
        Saturation(),
        BandwidthLimit(),
        EqDamage(),
        AdditiveNoise(color="white"),
        AdditiveNoise(color="pink"),
        SyntheticReverb(),
    ],
)
def test_effect_is_finite_and_length_preserving(effect) -> None:
    wav = _speechish()
    out = effect.apply(wav, SR, np.random.default_rng(0))
    _assert_clean_output(out, len(wav))


def test_saturation_adds_harmonics_without_flattopping() -> None:
    # A pure tone through soft saturation gains harmonic energy but, unlike hard
    # clipping, isn't flat-topped (its histogram shouldn't pile up at a single peak).
    t = np.arange(SR) / SR
    tone = (0.8 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = Saturation(drive_range=(6.0, 6.0)).apply(tone, SR, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    fund = spec[(freqs > 250) & (freqs < 350)].sum()
    harmonics = spec[(freqs > 500) & (freqs < 2000)].sum()
    assert harmonics > 0.01 * fund  # genuine harmonic generation
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_effect_is_reproducible_from_seed() -> None:
    wav = _speechish()
    eff = AdditiveNoise(color="pink")
    a = eff.apply(wav, SR, np.random.default_rng(7))
    b = eff.apply(wav, SR, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_clipping_actually_clips() -> None:
    wav = _speechish()
    out = Clipping(frac_range=(0.2, 0.2)).apply(wav, SR, np.random.default_rng(0))
    thresh = 0.2 * (np.max(np.abs(wav)) + 1e-9)
    assert np.max(np.abs(out)) <= thresh + 1e-5


def test_bandwidth_limit_removes_high_frequencies() -> None:
    wav = _speechish()
    out = BandwidthLimit(cutoff_range=(2000.0, 2000.0), order=8).apply(
        wav, SR, np.random.default_rng(0)
    )
    # The 1400 Hz tone survives; energy above ~2 kHz is strongly attenuated.
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    hi = spec[freqs > 4000].sum()
    lo = spec[(freqs > 100) & (freqs < 2000)].sum()
    assert hi < 0.05 * lo


def test_additive_noise_hits_target_snr() -> None:
    wav = _speechish()
    eff = AdditiveNoise(snr_db_range=(10.0, 10.0), color="white")
    out = eff.apply(wav, SR, np.random.default_rng(0))
    noise = out - wav
    snr = 10 * np.log10(np.mean(wav**2) / np.mean(noise**2))
    assert abs(snr - 10.0) < 1.0


@pytest.mark.parametrize("recipe", RECIPE_NAMES)
def test_all_recipes_finite_and_length_preserving(recipe: str) -> None:
    wav = _speechish()
    chain = make_chain(recipe)
    out = chain.apply(wav, SR, seed=1)
    _assert_clean_output(out, len(wav))


def test_chain_apply_is_reproducible() -> None:
    wav = _speechish()
    chain = make_chain("moderate")
    a = chain.apply(wav, SR, seed=3)
    b = chain.apply(wav, SR, seed=3)
    np.testing.assert_array_equal(a, b)


def test_make_chain_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="unknown degradation recipe"):
        make_chain("does-not-exist")


def test_clean_recipe_is_identity() -> None:
    wav = _speechish()
    out = make_chain("clean").apply(wav, SR, seed=0)
    np.testing.assert_array_equal(out, wav)


def test_chain_peak_guard_keeps_output_in_range() -> None:
    # Stacked boosts + clipping can blow past unity; the chain must rescale.
    wav = _speechish()
    out = make_chain("harsh").apply(wav, SR, seed=2)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


# --- codec round-trip (needs ffmpeg) ---------------------------------------

ffmpeg_required = pytest.mark.skipif(
    _ffmpeg_exe() is None, reason="ffmpeg binary not available (imageio-ffmpeg / PATH)"
)


@ffmpeg_required
@pytest.mark.parametrize("codec", ["mp3", "aac", "opus"])
def test_codec_roundtrip_is_clean_and_alters_signal(codec: str) -> None:
    wav = _speechish()
    eff = Codec(codecs=(codec,), bitrate_kbps_range=(32, 32))
    out = eff.apply(wav, SR, np.random.default_rng(0))
    _assert_clean_output(out, len(wav))
    # A lossy round-trip must change the waveform but preserve gross level.
    assert not np.array_equal(out, wav)
    assert 0.3 < np.max(np.abs(out)) < 1.01


@ffmpeg_required
def test_codec_chooses_from_pool_deterministically() -> None:
    wav = _speechish(0.25)
    eff = Codec()
    a = eff.apply(wav, SR, np.random.default_rng(11))
    b = eff.apply(wav, SR, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_codec_raises_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    import voice_cleanup.data.degrade as degrade_mod

    monkeypatch.setattr(degrade_mod, "_ffmpeg_exe", lambda: None)
    with pytest.raises(RuntimeError, match="Codec effect needs ffmpeg"):
        Codec().apply(_speechish(0.1), SR, np.random.default_rng(0))


def test_degradation_chain_describe() -> None:
    chain = DegradationChain("x", [Clipping(), AdditiveNoise()])
    assert chain.describe() == "clip → noise"
    assert make_chain("clean").describe() == "identity"
