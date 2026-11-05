"""Tests for the two-track restoration eval (eval/restoration.py).

Fast tests cover clip discovery, the scoring panel's metric gating, loudness
normalization wiring, the A/B listening-pack writer, and the markdown scoreboard —
all with the heavy metric models monkeypatched out. A slow test runs a real
passthrough end-to-end over a couple of EARS clips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_cleanup.eval import restoration as R
from voice_cleanup.eval.enhancers import PassthroughEnhancer

EARS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpora" / "ears"


# --- discovery --------------------------------------------------------------


def _make_ears_tree(root: Path) -> None:
    sr = 48000
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype("float32")
    for spk in ("p001", "p002"):
        d = root / spk
        d.mkdir(parents=True)
        for name in (
            "sentences_01_regular.wav", "sentences_02_regular.wav",
            "rainbow_01_regular.wav", "emo_anger_sentences.wav",  # excluded by pattern
            "sentences_03_whisper.wav",  # excluded by pattern
        ):
            sf.write(str(d / name), tone, sr)


def test_discover_ears_filters_and_caps(tmp_path: Path) -> None:
    _make_ears_tree(tmp_path)
    clips = R.discover_ears_clips(tmp_path, per_speaker=2)
    assert len(clips) == 4  # 2 speakers × 2 each
    assert {c.speaker for c in clips} == {"p001", "p002"}
    # only neutral 'regular'/freeform material, never emo_/whisper
    assert all("emo_" not in c.path.name and "whisper" not in c.path.name for c in clips)
    assert all(c.corpus == "ears" for c in clips)


def test_discover_ears_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert R.discover_ears_clips(tmp_path / "nope") == []


def test_discover_daps_groups_by_speaker(tmp_path: Path) -> None:
    cond = tmp_path / "daps" / "clean"
    cond.mkdir(parents=True)
    sr = 44100
    tone = (0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype("float32")
    for name in ("f1_script1_clean.wav", "f1_script2_clean.wav", "m2_script1_clean.wav"):
        sf.write(str(cond / name), tone, sr)
    clips = R.discover_daps_clips(tmp_path, per_speaker=1)
    assert {c.speaker for c in clips} == {"f1", "m2"}
    assert len(clips) == 2  # capped at 1 per speaker
    assert all(c.corpus == "daps" for c in clips)


# --- scoring panel ----------------------------------------------------------


@pytest.fixture
def stub_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "pesq_wb", lambda r, h, sr: 3.0)
    monkeypatch.setattr(R, "estoi", lambda r, h, sr: 0.9)
    monkeypatch.setattr(R, "si_sdr", lambda r, h: 12.0)
    monkeypatch.setattr(R, "dnsmos_p835", lambda h, sr: {"sig": 3.1, "bak": 3.2, "ovrl": 3.0})
    monkeypatch.setattr(R, "nisqa_multidim", lambda h, sr: {
        "mos_pred": 4.0, "noi_pred": 4.1, "dis_pred": 4.2, "col_pred": 4.3, "loud_pred": 4.4,
    })
    monkeypatch.setattr(R, "speaker_similarity", lambda r, h, sr: 0.88)
    monkeypatch.setattr(R, "whisper_wer", lambda r, h, sr, ref_text=None: {
        "wer": 0.1, "ref_text": ref_text or "hello world", "hyp_text": "hello world",
    })
    monkeypatch.setattr(R, "transcribe", lambda x, sr: "hello world")


def test_score_panel_with_reference_runs_intrusive(stub_metrics: None) -> None:
    ref = np.random.default_rng(0).standard_normal(48000).astype("float32")
    out = R.score_panel(ref, ref, 48000, has_reference=True)
    assert out["pesq"] == 3.0
    assert out["si_sdr"] == 12.0
    assert out["nisqa_mos"] == 4.0
    assert out["speaker_sim"] == 0.88
    assert out["wer"] == 0.1


def test_score_panel_no_reference_skips_intrusive(stub_metrics: None) -> None:
    sig = np.random.default_rng(1).standard_normal(48000).astype("float32")
    out = R.score_panel(sig, sig, 48000, has_reference=False)
    assert out["pesq"] is None and out["estoi"] is None and out["si_sdr"] is None
    # non-intrusive + spk + wer still present
    assert out["nisqa_mos"] == 4.0 and out["speaker_sim"] == 0.88


def test_score_panel_flags_gate_metrics(stub_metrics: None) -> None:
    sig = np.zeros(48000, dtype="float32")
    out = R.score_panel(
        sig, sig, 48000, has_reference=True,
        run_dnsmos=False, run_nisqa=False, run_wer=False, run_spk=False,
    )
    assert out["dnsmos_ovrl"] is None
    assert out["nisqa_mos"] is None
    assert out["speaker_sim"] is None
    assert out["wer"] is None
    assert out["pesq"] == 3.0  # intrusive still on


# --- runner + pack + report -------------------------------------------------


def test_studio_runner_end_to_end_with_stubs(tmp_path: Path, stub_metrics: None) -> None:
    _make_ears_tree(tmp_path / "ears")
    clips = R.discover_ears_clips(tmp_path / "ears", per_speaker=1)
    pack = tmp_path / "pack"
    report = R.run_studio_reference_eval(
        clips, "moderate", [PassthroughEnhancer()],
        eval_sr=48000, max_seconds=1.0, listening_pack_dir=pack,
    )
    assert len(report.results) == len(clips)
    assert all(r.enhancer == "passthrough" for r in report.results)
    # A/B pack: clean + degraded + one enhancer per clip, plus an index.
    wavs = list(pack.glob("*.wav"))
    assert len(wavs) == len(clips) * 3
    assert (pack / "INDEX.md").is_file()
    # report renders a scoreboard with the headline columns
    out_dir = tmp_path / "out"
    results_path, report_path = R.write_studio_report(report, out_dir)
    assert results_path.is_file()
    text = report_path.read_text()
    assert "NISQA" in text and "SpkSim" in text and "passthrough" in text


def test_summary_by_enhancer_aggregates(stub_metrics: None) -> None:
    sig = np.random.default_rng(2).standard_normal(24000).astype("float32")
    panel = R.score_panel(sig, sig, 48000, has_reference=True)
    rows = [
        R.RestorationResult(
            identifier=f"c{i}", corpus="ears", speaker="p001", recipe="moderate",
            enhancer="passthrough", duration_s=0.5, elapsed_s=0.1,
            **{k: panel[k] for k in (
                "pesq", "estoi", "si_sdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
                "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
                "speaker_sim", "wer", "wer_ref_text", "wer_hyp_text",
            )},
        )
        for i in range(3)
    ]
    report = R.StudioReport("moderate", 48000, "t", 1.0, rows)
    summ = report.summary_by_enhancer()
    assert summ["passthrough"]["nisqa_mos"]["n"] == 3
    assert summ["passthrough"]["nisqa_mos"]["mean"] == pytest.approx(4.0)


@pytest.mark.slow
def test_real_passthrough_over_ears(tmp_path: Path) -> None:
    if not EARS_DIR.is_dir() or not any(EARS_DIR.iterdir()):
        pytest.skip("EARS corpus not downloaded")
    clips = R.discover_ears_clips(EARS_DIR, per_speaker=1)[:2]
    if not clips:
        pytest.skip("no EARS clips matched")
    report = R.run_studio_reference_eval(
        clips, "moderate", [PassthroughEnhancer()],
        eval_sr=48000, max_seconds=4.0,
        run_dnsmos=True, run_nisqa=True, run_wer=False, run_spk=True,
    )
    assert len(report.results) == len(clips)
    for r in report.results:
        assert r.pesq is not None
        assert r.speaker_sim is not None and -1.0 <= r.speaker_sim <= 1.0
