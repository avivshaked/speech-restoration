"""Real-audio validation of the metric wrappers.

These tests load actual podcast speech from data/test-sets/podcast-real/clips/
and exercise the metric panel against meaningful inputs. They complement the
plumbing-level synthetic-tone tests in `test_eval_intrusive.py` and
`test_eval_non_intrusive.py`.

Skipped if the test set is not on disk (fresh checkout, etc.).
"""

from __future__ import annotations

import pytest

from voice_cleanup.eval.intrusive import estoi, pesq_wb, si_sdr
from voice_cleanup.eval.non_intrusive import dnsmos_p835, squim_mos

from .conftest import add_noise

# --- intrusive metrics on real speech ---


def test_pesq_real_clean_identical(real_clip_audio):
    audio, sr = real_clip_audio
    assert sr == 16000, f"expected 16 kHz clip, got {sr}"
    score = pesq_wb(audio, audio)
    assert score > 4.0, f"PESQ on identical clean speech should be ~4.5, got {score}"


def test_pesq_real_monotonic_in_snr(real_clip_audio):
    audio, sr = real_clip_audio
    score_high = pesq_wb(audio, add_noise(audio, snr_db=20.0, seed=1))
    score_low = pesq_wb(audio, add_noise(audio, snr_db=0.0, seed=1))
    assert score_high > score_low, (
        f"PESQ should decrease as SNR drops; got {score_high} vs {score_low}"
    )


def test_estoi_real_clean_identical(real_clip_audio):
    audio, _ = real_clip_audio
    assert estoi(audio, audio) == pytest.approx(1.0, abs=1e-6)


def test_estoi_real_monotonic_in_snr(real_clip_audio):
    audio, _ = real_clip_audio
    score_high = estoi(audio, add_noise(audio, snr_db=20.0, seed=2))
    score_low = estoi(audio, add_noise(audio, snr_db=-5.0, seed=2))
    assert score_high > score_low
    assert 0.0 <= score_low <= 1.0
    assert 0.0 <= score_high <= 1.0


def test_si_sdr_real_scale_invariance(real_clip_audio):
    audio, _ = real_clip_audio
    noisy = add_noise(audio, snr_db=10.0, seed=3)
    assert si_sdr(audio, noisy) == pytest.approx(si_sdr(audio, 2.5 * noisy), abs=1e-5)


def test_si_sdr_real_monotonic_in_snr(real_clip_audio):
    audio, _ = real_clip_audio
    score_high = si_sdr(audio, add_noise(audio, snr_db=20.0, seed=4))
    score_low = si_sdr(audio, add_noise(audio, snr_db=0.0, seed=4))
    assert score_high > score_low


# --- non-intrusive metrics on real speech ---


def test_dnsmos_real_speech_in_sensible_range(real_clip_audio):
    """A real podcast clip should DNSMOS to a non-trivial score in [1, 5]."""
    audio, sr = real_clip_audio
    result = dnsmos_p835(audio, sr=sr)
    for key in ("sig", "bak", "ovrl"):
        assert 1.0 <= result[key] <= 5.0
    # SIG on a clean podcast clip should clear "completely degraded" territory.
    assert result["sig"] > 2.0, (
        f"SIG={result['sig']:.2f} on real speech — suspicious low"
    )


def test_dnsmos_real_degraded_scores_lower(real_clip_audio):
    audio, sr = real_clip_audio
    clean = dnsmos_p835(audio, sr=sr)
    noisy = dnsmos_p835(add_noise(audio, snr_db=-5.0, seed=5), sr=sr)
    assert clean["ovrl"] > noisy["ovrl"]
    assert clean["bak"] > noisy["bak"]


@pytest.mark.slow
def test_squim_mos_real_speech_in_sensible_range(real_clip_audio, real_clip_nmr):
    audio, sr = real_clip_audio
    nmr, _ = real_clip_nmr
    score = squim_mos(audio, nmr, sr=sr)
    assert 1.0 <= score <= 5.0
    assert score > 2.0, f"SQUIM on real speech = {score:.2f} — suspicious low"


@pytest.mark.slow
def test_squim_mos_responds_to_input_changes(real_clip_audio, real_clip_nmr):
    """SQUIM should produce different scores for materially different inputs.

    Note: SQUIM-MOS is trained on TTS / voice-conversion artifacts (BVCC + DAPS),
    not on additive-noise degradation. Empirically (see docs/learning/), it can
    score noisy speech HIGHER than clean speech when Gaussian noise is added —
    "more noise → lower score" is not a reliable property. So we only assert
    that SQUIM responds at all, and let DNSMOS carry the noise-monotonicity
    contract.
    """
    audio, sr = real_clip_audio
    nmr, _ = real_clip_nmr
    clean_score = squim_mos(audio, nmr, sr=sr)
    noisy_score = squim_mos(add_noise(audio, snr_db=-10.0, seed=6), nmr, sr=sr)
    assert abs(clean_score - noisy_score) > 0.1, (
        f"SQUIM produced near-identical scores ({clean_score:.3f} vs "
        f"{noisy_score:.3f}) — either the model is broken or the inputs are "
        "indistinguishable"
    )
