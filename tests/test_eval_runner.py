"""Tests for the eval panel runner.

Plumbing-level: spec parsing, manifest loading, and report shape. End-to-end
behavior is exercised via the CLI smoke test which runs against the real
test set (skipped if absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_cleanup.eval.runner import (
    Degradation,
    EvalReport,
    load_test_set,
    run_eval,
    write_report,
)

# --- Degradation spec parsing ---


def test_degradation_parses_none():
    d = Degradation.parse("none")
    assert d.kind == "none"
    assert d.snr_db is None


def test_degradation_parses_gaussian():
    d = Degradation.parse("gaussian:-5")
    assert d.kind == "gaussian"
    assert d.snr_db == -5.0


def test_degradation_parses_positive_snr():
    d = Degradation.parse("gaussian:20")
    assert d.snr_db == 20.0


def test_degradation_rejects_unknown_spec():
    with pytest.raises(ValueError, match="unknown degradation"):
        Degradation.parse("rir:church")


def test_degradation_none_is_passthrough():
    d = Degradation.parse("none")
    x = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    y = d.apply(x)
    np.testing.assert_array_equal(x, y)
    assert y is not x  # copy, not alias


def test_degradation_gaussian_reduces_snr():
    """Sanity: applying gaussian:0 produces a signal with empirical SNR ≈ 0 dB."""
    d = Degradation.parse("gaussian:0")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16000).astype(np.float32)
    y = d.apply(x)
    noise = y - x
    snr = 10 * np.log10(np.mean(x**2) / (np.mean(noise**2) + 1e-12))
    assert abs(snr) < 0.5, f"expected ~0 dB SNR, got {snr:.2f}"


def test_degradation_gaussian_is_seed_offset_deterministic():
    d = Degradation.parse("gaussian:-5")
    x = np.random.default_rng(2).standard_normal(16000).astype(np.float32)
    y0a = d.apply(x, seed_offset=0)
    y0b = d.apply(x, seed_offset=0)
    y1 = d.apply(x, seed_offset=1)
    np.testing.assert_array_equal(y0a, y0b)
    assert not np.array_equal(y0a, y1), "different seed_offsets must produce different noise"


# --- Manifest loading ---


def _make_fake_test_set(tmp_path: Path, n: int = 2) -> Path:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    clips = []
    for i in range(n):
        path = clips_dir / f"clip{i}.wav"
        sf.write(str(path), 0.1 * np.random.default_rng(i).standard_normal(16000 * 10), 16000)
        clips.append({
            "identifier": f"clip{i}",
            "clip_path": str(path),
            "license_short": "CC-BY",
            "commercial_safe": True,
            "status": "ok",
        })
    (tmp_path / "clips.json").write_text(json.dumps({"clips": clips}))
    return tmp_path


def test_load_test_set_reads_manifest(tmp_path):
    ts = _make_fake_test_set(tmp_path, n=3)
    records = load_test_set(ts)
    assert len(records) == 3
    assert records[0].identifier == "clip0"
    assert records[0].sr == 16000


def test_load_test_set_skips_non_ok_status(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    good = clips_dir / "good.wav"
    sf.write(str(good), np.zeros(16000), 16000)
    (tmp_path / "clips.json").write_text(json.dumps({
        "clips": [
            {"identifier": "good", "clip_path": str(good), "status": "ok"},
            {"identifier": "bad", "clip_path": "nonexistent.wav", "status": "decode_failed"},
        ],
    }))
    records = load_test_set(tmp_path)
    assert len(records) == 1
    assert records[0].identifier == "good"


def test_load_test_set_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="clips.json"):
        load_test_set(tmp_path)


# --- End-to-end on a tiny fake set ---


@pytest.mark.slow
def test_run_eval_end_to_end_on_fake_set(tmp_path):
    """Runs the full panel (incl. DNSMOS + SQUIM) on a fake 2-clip set."""
    ts = _make_fake_test_set(tmp_path, n=2)
    report = run_eval(
        ts,
        Degradation.parse("gaussian:0"),
        limit=2,
        run_squim=True,
        run_nisqa=False,  # NISQA needs vendored source; smoke separately
        run_wer=False,  # Whisper-large-v3 is too heavy for unit tests
    )
    assert isinstance(report, EvalReport)
    assert len(report.clips) == 2
    for clip in report.clips:
        assert clip.pesq is not None
        assert clip.estoi is not None
        assert clip.si_sdr is not None
        assert clip.dnsmos_ovrl is not None
        assert clip.squim_mos is not None

    out_dir = tmp_path / "out"
    results_path, summary_path = write_report(report, out_dir)
    assert results_path.is_file()
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["n_clips"] == 2
    assert "estoi" in summary["summary"]


def test_run_eval_none_degradation_skips_intrusive(tmp_path):
    """`none` degradation should only populate non-intrusive scores."""
    ts = _make_fake_test_set(tmp_path, n=2)
    report = run_eval(
        ts,
        Degradation.parse("none"),
        limit=2,
        run_squim=False,  # skip SQUIM to keep this fast
        run_nisqa=False,  # NISQA needs vendored source
        run_wer=False,  # WER also disabled under `none` anyway, but keep the test fast
    )
    for clip in report.clips:
        assert clip.pesq is None
        assert clip.estoi is None
        assert clip.si_sdr is None
        assert clip.dnsmos_ovrl is not None
        assert clip.squim_mos is None  # disabled
        assert clip.nisqa_mos is None  # disabled
        assert clip.wer is None  # WER skipped under `none` degradation


def test_eval_report_summary_aggregates_correctly():
    """Build a report by hand; check summary stats."""
    from voice_cleanup.eval.runner import ClipResult

    clips = [
        ClipResult(
            identifier=f"c{i}",
            license_short="CC-BY",
            commercial_safe=True,
            duration_s=20.0,
            pesq=float(i + 1),
            estoi=None,
            si_sdr=None,
            dnsmos_sig=None,
            dnsmos_bak=None,
            dnsmos_ovrl=None,
            squim_mos=None,
            nisqa_mos=None,
            nisqa_noi=None,
            nisqa_dis=None,
            nisqa_col=None,
            nisqa_loud=None,
            wer=None,
            wer_ref_text=None,
            wer_hyp_text=None,
            elapsed_s=0.1,
        )
        for i in range(4)
    ]
    report = EvalReport(
        degradation=Degradation.parse("gaussian:-5"),
        test_set_dir="/tmp/fake",
        started_at="2026-05-25T00:00:00+00:00",
        elapsed_s=1.0,
        clips=clips,
    )
    s = report.summary()
    assert "pesq" in s
    assert s["pesq"]["mean"] == 2.5  # mean of [1,2,3,4]
    assert s["pesq"]["n"] == 4
    assert "estoi" not in s  # all-None metrics excluded
