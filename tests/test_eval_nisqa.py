"""Tests for the NISQA wrapper.

Vendored upstream is loaded from external/nisqa/; the slow tests skip cleanly
if vendor files are absent (fresh checkout, fetch script not run).
"""

from __future__ import annotations

import pytest

from voice_cleanup.eval import nisqa as nisqa_mod


def test_nisqa_module_imports():
    """Cheap smoke check — no model load, no vendor required."""
    assert nisqa_mod.NISQA_SR == 48000
    assert callable(nisqa_mod.nisqa_multidim)
    assert nisqa_mod.NISQA_WEIGHTS_PATH.name == "nisqa.tar"


def test_vendor_present_reports_missing(tmp_path, monkeypatch):
    """When vendor dir is empty, _vendor_present returns (False, [...])."""
    monkeypatch.setattr(nisqa_mod, "EXTERNAL_NISQA_DIR", tmp_path)
    monkeypatch.setattr(nisqa_mod, "NISQA_WEIGHTS_PATH", tmp_path / "weights" / "nisqa.tar")
    monkeypatch.setattr(nisqa_mod, "NISQA_REQUIRED_SOURCES", (
        tmp_path / "nisqa" / "NISQA_lib.py",
        tmp_path / "nisqa" / "NISQA_model.py",
    ))
    ok, missing = nisqa_mod._vendor_present()
    assert ok is False
    assert len(missing) == 3


def test_ensure_vendor_raises_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(nisqa_mod, "EXTERNAL_NISQA_DIR", tmp_path)
    monkeypatch.setattr(nisqa_mod, "NISQA_WEIGHTS_PATH", tmp_path / "weights" / "nisqa.tar")
    monkeypatch.setattr(nisqa_mod, "NISQA_REQUIRED_SOURCES", (
        tmp_path / "nisqa" / "NISQA_lib.py",
    ))
    with pytest.raises(FileNotFoundError, match="fetch_nisqa.py"):
        nisqa_mod._ensure_vendor()


@pytest.mark.slow
def test_nisqa_real_speech_in_sensible_range(real_clip_audio):
    """NISQA on a real podcast clip should produce all five scores in [1, 5]."""
    ok, _missing = nisqa_mod._vendor_present()
    if not ok:
        pytest.skip("NISQA vendor files missing — run scripts/fetch_nisqa.py")
    audio, sr = real_clip_audio
    result = nisqa_mod.nisqa_multidim(audio, sr=sr)
    for key in ("mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"):
        assert key in result
        assert 1.0 <= result[key] <= 5.0, f"{key}={result[key]} outside [1, 5]"


@pytest.mark.slow
def test_nisqa_degraded_mos_drops(real_clip_audio):
    """Heavy noise should drop MOS and increase noisiness."""
    ok, _missing = nisqa_mod._vendor_present()
    if not ok:
        pytest.skip("NISQA vendor files missing — run scripts/fetch_nisqa.py")
    from .conftest import add_noise
    audio, sr = real_clip_audio
    clean = nisqa_mod.nisqa_multidim(audio, sr=sr)
    noisy = nisqa_mod.nisqa_multidim(add_noise(audio, snr_db=-5.0, seed=7), sr=sr)
    # Direction-only check. NISQA dimensions use P.804 convention: 1=annoying,
    # 5=imperceptible — so noi_pred DROPS under more noise (more annoying noise).
    # Similarly mos_pred (overall) drops under degradation.
    assert noisy["noi_pred"] < clean["noi_pred"], (
        f"NISQA NOI did not drop under -5 dB noise (higher = less noise per P.804): "
        f"clean={clean['noi_pred']:.2f} noisy={noisy['noi_pred']:.2f}"
    )
    assert noisy["mos_pred"] < clean["mos_pred"], (
        f"NISQA MOS did not drop under -5 dB noise: "
        f"clean={clean['mos_pred']:.2f} noisy={noisy['mos_pred']:.2f}"
    )
