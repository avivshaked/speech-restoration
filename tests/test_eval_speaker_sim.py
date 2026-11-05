"""Tests for the speaker-similarity metric (eval/speaker_sim.py).

Fast tests exercise the cosine math + resampling plumbing with the model
monkeypatched out. The slow test loads the real WavLM-SV model and asserts the
discrimination property (same speaker > different speaker) on real EARS audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_cleanup.eval import speaker_sim

EARS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpora" / "ears"


def test_module_surface() -> None:
    assert speaker_sim.SPEAKER_SIM_SR == 16000
    assert "wavlm" in speaker_sim.SPEAKER_SIM_MODEL_ID.lower()
    assert callable(speaker_sim.speaker_similarity)


def test_cosine_of_orthogonal_and_identical_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    # Map each input to a fixed unit vector by identity (first sample picks the axis).
    def fake_embed(signal, sr):  # noqa: ANN001, ARG001
        axis = int(np.asarray(signal).ravel()[0])
        v = torch.zeros(4)
        v[axis] = 1.0
        return v

    monkeypatch.setattr(speaker_sim, "_embed", fake_embed)
    same = speaker_sim.speaker_similarity(np.array([0.0, 9.0]), np.array([0.0, 1.0]), sr=16000)
    orth = speaker_sim.speaker_similarity(np.array([0.0, 9.0]), np.array([1.0, 1.0]), sr=16000)
    assert same == pytest.approx(1.0, abs=1e-6)
    assert orth == pytest.approx(0.0, abs=1e-6)


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="expected 1D"):
        speaker_sim._to_numpy_1d(np.zeros((2, 3)))


def test_resample_is_noop_at_target_rate() -> None:
    x = np.linspace(-1, 1, 100, dtype=np.float32)
    out = speaker_sim._resample_if_needed(x, 16000, 16000)
    np.testing.assert_array_equal(out, x)
    assert speaker_sim._resample_if_needed(x, 48000, 16000).shape[0] < x.shape[0]


@pytest.mark.slow
def test_discriminates_speakers_on_real_audio() -> None:
    import soundfile as sf

    if not EARS_DIR.is_dir() or not any(EARS_DIR.iterdir()):
        pytest.skip("EARS corpus not downloaded")
    speakers = sorted(p for p in EARS_DIR.iterdir() if p.is_dir())
    if len(speakers) < 2:
        pytest.skip("need ≥2 EARS speakers")

    def first_wavs(spk: Path, k: int = 2) -> list[Path]:
        return sorted(spk.rglob("*.wav"))[:k]

    spk1, spk2 = first_wavs(speakers[0]), first_wavs(speakers[1])

    def load(p: Path) -> tuple[np.ndarray, int]:
        a, sr = sf.read(str(p))
        return a.astype("float32")[: sr * 5], int(sr)

    a1, sr = load(spk1[0])
    b1, _ = load(spk1[1])
    c2, _ = load(spk2[0])

    same = speaker_sim.speaker_similarity(a1, a1, sr)
    same_spk = speaker_sim.speaker_similarity(a1, b1, sr)
    diff_spk = speaker_sim.speaker_similarity(a1, c2, sr)

    assert same == pytest.approx(1.0, abs=1e-4)
    assert same_spk > diff_spk + 0.2  # clear separation
