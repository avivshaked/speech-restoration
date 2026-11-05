"""Tests for Whisper-WER.

Loads Whisper-large-v3 (~3 GB) — gated behind `@pytest.mark.slow`. First run
downloads weights to ~/.cache/huggingface/.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_cleanup.eval.wer import transcribe, whisper_wer

from .conftest import add_noise


def test_wer_module_imports():
    """Cheap smoke check that doesn't load Whisper weights."""
    from voice_cleanup.eval import wer
    assert wer.WHISPER_SR == 16000
    assert callable(wer.whisper_wer)


@pytest.mark.slow
def test_whisper_transcribes_real_speech(real_clip_audio):
    """Whisper on a real podcast clip should produce non-empty English text."""
    audio, sr = real_clip_audio
    text = transcribe(audio, sr=sr)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    # English transcripts of 20s of podcast speech should have at least a few words
    assert len(text.split()) >= 5, f"suspiciously short transcript: {text!r}"


@pytest.mark.slow
def test_wer_zero_on_identical_audio(real_clip_audio):
    """Same audio → same transcript → WER = 0 (modulo Whisper determinism)."""
    audio, sr = real_clip_audio
    result = whisper_wer(audio, audio, sr=sr)
    assert result["wer"] == pytest.approx(0.0, abs=1e-6)
    assert result["ref_text"] == result["hyp_text"]
    assert result["n_ref_words"] >= 5


@pytest.mark.slow
def test_wer_increases_under_heavy_noise(real_clip_audio):
    """WER on -15 dB noisy speech should be substantially > clean-vs-clean."""
    audio, sr = real_clip_audio
    noisy = add_noise(audio, snr_db=-15.0, seed=42)
    result = whisper_wer(audio, noisy, sr=sr)
    # At -15 dB SNR Whisper will mishear / drop / hallucinate many words.
    # We just require that something visibly broke.
    assert result["wer"] > 0.1, (
        f"WER={result['wer']:.3f} suspiciously low at -15 dB; "
        f"ref={result['ref_text']!r} hyp={result['hyp_text']!r}"
    )


@pytest.mark.slow
def test_wer_accepts_precomputed_reference(real_clip_audio):
    """Passing `ref_text` should skip the reference-transcription pass."""
    audio, sr = real_clip_audio
    ref_text = transcribe(audio, sr=sr)
    # Use a different noise seed than the implicit transcription would; should still
    # match ref_text exactly because we forced it.
    result = whisper_wer(audio, audio, sr=sr, ref_text=ref_text)
    assert result["ref_text"] == ref_text


def test_wer_empty_reference_returns_nan():
    """If the reference transcribes to empty, WER is undefined → NaN."""
    # We can't easily induce an empty Whisper transcript without loading the model,
    # so we exercise the codepath by passing silence as the "audio" with a forced
    # empty ref_text.
    silence = np.zeros(16000, dtype=np.float32)
    # We need to short-circuit: pass ref_text="" so the function never calls Whisper
    # on the reference. The hypothesis still goes through Whisper though, which
    # would load the model. To keep this test FAST (no model load), monkey-patch
    # transcribe via the function-arg path. Simplest: rely on `ref_text=""` path
    # being hit BEFORE the hyp transcription.
    from voice_cleanup.eval import wer as wer_mod

    # Patch transcribe to a stub so we don't load Whisper.
    original = wer_mod.transcribe
    wer_mod.transcribe = lambda *a, **kw: ""
    try:
        result = wer_mod.whisper_wer(silence, silence, sr=16000, ref_text="")
    finally:
        wer_mod.transcribe = original

    assert np.isnan(result["wer"])
    assert result["n_ref_words"] == 0
