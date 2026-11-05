"""Shared pytest fixtures.

Tests that need real audio load it from `data/test-sets/podcast-real/clips/`,
which is populated by `scripts/download_test_set.py`. If the directory is
missing (clean checkout, no audio fetched yet), tests requesting the fixture
are skipped — they're additive validation on top of the plumbing-level tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REAL_CLIPS_DIR = Path("data/test-sets/podcast-real/clips")


def _available_clips() -> list[Path]:
    if not REAL_CLIPS_DIR.is_dir():
        return []
    return sorted(REAL_CLIPS_DIR.glob("*.wav"))


@pytest.fixture(scope="session")
def real_clip_path() -> Path:
    """Path to a single canonical real-audio clip (16 kHz mono).

    Preference: the commercial-safe `biblecurious` clip if present (CC-BY-SA),
    else any commercial-safe clip, else any clip at all. Skips the test if
    nothing is on disk.
    """
    clips = _available_clips()
    if not clips:
        pytest.skip(
            "no real-audio clips in data/test-sets/podcast-real/clips/ — "
            "run scripts/download_test_set.py to populate"
        )
    preferred = [
        "biblecurious.wav",
        "TheInternetOfMoney-Volume1.wav",
    ]
    for name in preferred:
        for c in clips:
            if c.name == name:
                return c
    return clips[0]


@pytest.fixture(scope="session")
def real_clip_audio(real_clip_path: Path) -> tuple[np.ndarray, int]:
    """Return (waveform[T], sr) for the canonical real clip."""
    audio, sr = sf.read(str(real_clip_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


@pytest.fixture(scope="session")
def real_clip_nmr(real_clip_path: Path) -> tuple[np.ndarray, int]:
    """A different real clip, suitable as SQUIM's non-matching reference.

    Picks the first clip whose filename differs from `real_clip_path`.
    """
    clips = _available_clips()
    other = next((c for c in clips if c.name != real_clip_path.name), None)
    if other is None:
        pytest.skip("need at least two real clips for SQUIM NMR fixture")
    audio, sr = sf.read(str(other), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to `x` at a target SNR in dB."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape)
    sig_power = np.mean(x**2)
    noise_power = np.mean(noise**2)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-12))
    return (x + noise).astype(np.float32)
