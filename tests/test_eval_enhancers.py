"""Tests for the Enhancer abstraction + Passthrough baseline + runner integration."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from voice_cleanup.eval.enhancers import (
    Enhancer,
    PassthroughEnhancer,
    get_enhancer,
    register,
    registered_names,
)
from voice_cleanup.eval.runner import Degradation, run_eval


def test_passthrough_is_registered():
    assert "passthrough" in registered_names()


def test_get_enhancer_unknown_raises():
    with pytest.raises(ValueError, match="unknown enhancer"):
        get_enhancer("does-not-exist")


def test_passthrough_returns_copy_of_input():
    enh = PassthroughEnhancer()
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    y = enh.apply(x, sr=16000)
    np.testing.assert_array_equal(x, y)
    assert y is not x  # copy, not alias — protects callers from in-place writes


def test_passthrough_name_and_label():
    enh = PassthroughEnhancer()
    assert enh.name == "passthrough"
    assert "noisy floor" in enh.label()


def test_register_adds_to_lookup():
    class _DummyEnh(Enhancer):
        name = "test-dummy"

        def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
            return audio * 0.5

    try:
        register("test-dummy", _DummyEnh)
        assert "test-dummy" in registered_names()
        enh = get_enhancer("test-dummy")
        out = enh.apply(np.ones(8, dtype=np.float32), sr=16000)
        np.testing.assert_array_equal(out, np.full(8, 0.5, dtype=np.float32))
    finally:
        # Don't pollute the global registry for other tests.
        from voice_cleanup.eval import enhancers as enh_mod
        enh_mod._REGISTRY.pop("test-dummy", None)


# --- runner integration ---


def _make_fake_test_set(tmp_path, n: int = 2):
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


def test_runner_with_passthrough_produces_same_metrics_as_no_enhancer(tmp_path):
    """Sanity: passthrough enhancer should be indistinguishable from no enhancer.

    This validates that the enhancer slot is wired correctly — when the enhancer
    is identity, scoring should be unchanged. Any divergence here indicates a
    bug in the runner's enhancer hook.
    """
    ts = _make_fake_test_set(tmp_path, n=2)
    deg = Degradation.parse("gaussian:0")
    common_kwargs = dict(limit=2, run_squim=False, run_nisqa=False, run_wer=False)

    baseline = run_eval(ts, deg, enhancer=None, **common_kwargs)
    passthrough = run_eval(ts, deg, enhancer=PassthroughEnhancer(), **common_kwargs)

    for b_clip, p_clip in zip(baseline.clips, passthrough.clips, strict=True):
        # Intrusive metrics are exact-match because the math is deterministic.
        assert b_clip.pesq == pytest.approx(p_clip.pesq, abs=1e-3)
        assert b_clip.estoi == pytest.approx(p_clip.estoi, abs=1e-6)
        assert b_clip.si_sdr == pytest.approx(p_clip.si_sdr, abs=1e-3)
        # DNSMOS is also deterministic given identical input.
        assert b_clip.dnsmos_ovrl == pytest.approx(p_clip.dnsmos_ovrl, abs=1e-3)

    assert passthrough.enhancer_name == "passthrough"
    assert "noisy floor" in passthrough.enhancer_label


def test_runner_records_enhancer_metadata_in_report(tmp_path):
    ts = _make_fake_test_set(tmp_path, n=1)
    report = run_eval(
        ts, Degradation.parse("gaussian:-5"),
        limit=1, run_squim=False, run_nisqa=False, run_wer=False,
        enhancer=PassthroughEnhancer(),
    )
    d = report.to_dict()
    assert d["enhancer"]["name"] == "passthrough"
    assert d["enhancer"]["label"] == "passthrough (noisy floor)"


def test_runner_without_enhancer_has_null_metadata(tmp_path):
    ts = _make_fake_test_set(tmp_path, n=1)
    report = run_eval(
        ts, Degradation.parse("gaussian:-5"),
        limit=1, run_squim=False, run_nisqa=False, run_wer=False,
    )
    d = report.to_dict()
    assert d["enhancer"]["name"] is None
    assert d["enhancer"]["label"] is None


# --- MP-SENet enhancer (fast: no model load; slow: real forward) ---

from voice_cleanup.eval import mpsenet as mpsenet_mod  # noqa: E402
from voice_cleanup.eval.mpsenet import CHECKPOINTS, MPSENetEnhancer  # noqa: E402


def _mpsenet_vendored(ckpt: str = "dns") -> bool:
    ok, _ = mpsenet_mod._vendor_present(CHECKPOINTS[ckpt])
    return ok


def test_mpsenet_is_registered():
    assert "mp-senet" in registered_names()


def test_mpsenet_default_checkpoint_is_dns():
    assert MPSENetEnhancer().checkpoint == "dns"


def test_mpsenet_unknown_checkpoint_raises():
    with pytest.raises(ValueError, match="unknown MP-SENet checkpoint"):
        MPSENetEnhancer(checkpoint="nope")


def test_mpsenet_labels_name_the_checkpoint():
    assert MPSENetEnhancer("dns").name == "mp-senet"
    assert "DNS" in MPSENetEnhancer("dns").label()
    assert "VoiceBank" in MPSENetEnhancer("vb").label()


@pytest.mark.slow
def test_mpsenet_forward_preserves_length_and_is_finite():
    """Real MPNet forward on a short noisy signal — gated; needs vendored weights.

    Run `uv run python scripts/fetch_mpsenet.py` first. Asserts the enhancer
    returns a finite, same-length, same-sr signal (the runner relies on this).
    """
    if not _mpsenet_vendored("dns"):
        pytest.skip("MP-SENet not vendored — run scripts/fetch_mpsenet.py")
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(16000 * 2)).astype(np.float32)  # 2 s @ 16 kHz
    y = MPSENetEnhancer("dns").apply(x, sr=16000)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


@pytest.mark.slow
def test_mpsenet_near_silent_input_returns_passthrough():
    if not _mpsenet_vendored("dns"):
        pytest.skip("MP-SENet not vendored — run scripts/fetch_mpsenet.py")
    x = np.zeros(16000, dtype=np.float32)
    y = MPSENetEnhancer("dns").apply(x, sr=16000)
    np.testing.assert_array_equal(x, y)


# --- Resemble Enhance enhancer (fast: no model load; slow: real forward) ---

from voice_cleanup.eval import resemble_enhance as re_mod  # noqa: E402
from voice_cleanup.eval.resemble_enhance import ResembleEnhanceEnhancer  # noqa: E402


def _re_vendored() -> bool:
    ok, _ = re_mod._vendor_present()
    return ok


def test_resemble_enhance_is_registered():
    assert "resemble-enhance" in registered_names()


def test_resemble_enhance_defaults():
    enh = ResembleEnhanceEnhancer()
    assert enh.name == "resemble-enhance"
    assert (enh.nfe, enh.solver, enh.lambd, enh.tau) == (32, "midpoint", 0.5, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nfe": 0},
        {"nfe": 200},
        {"solver": "nope"},
        {"lambd": 1.5},
        {"tau": -0.1},
    ],
)
def test_resemble_enhance_validates_params(kwargs):
    with pytest.raises(ValueError):
        ResembleEnhanceEnhancer(**kwargs)


def test_resemble_enhance_label_names_config():
    label = ResembleEnhanceEnhancer(nfe=16, solver="rk4").label()
    assert "nfe=16" in label
    assert "rk4" in label


def test_deepspeed_stub_surface_is_inference_safe():
    """The stub must satisfy import-time symbols but refuse to run training paths."""
    import sys

    # Snapshot + clear so we exercise the stub even if a prior test imported RE.
    saved = {k: v for k, v in sys.modules.items() if k == "deepspeed" or k.startswith("deepspeed.")}
    for k in saved:
        del sys.modules[k]
    try:
        re_mod._install_deepspeed_stub()
        import deepspeed  # noqa: PLC0415
        from deepspeed.accelerator import get_accelerator  # noqa: PLC0415
        from deepspeed.runtime.engine import DeepSpeedEngine  # noqa: PLC0415

        # Subclassable (RE does `class Engine(DeepSpeedEngine)` at import) ...
        assert isinstance(DeepSpeedEngine, type)
        # ... but every callable refuses to run (would mean a training path executed).
        with pytest.raises(RuntimeError, match="inference"):
            deepspeed.init_distributed()
        with pytest.raises(RuntimeError):
            get_accelerator()
    finally:
        for k in list(sys.modules):
            if k == "deepspeed" or k.startswith("deepspeed."):
                del sys.modules[k]
        sys.modules.update(saved)


@pytest.mark.slow
def test_resemble_enhance_forward_preserves_length_and_is_finite():
    """Real RE enhance on a short noisy signal — gated; needs vendored weights.

    Run `uv run python scripts/fetch_resemble_enhance.py` first. Asserts the
    enhancer returns a finite, same-length, same-sr signal (the runner relies on
    this). Uses nfe=8 to keep the flow-matching solver quick.
    """
    if not _re_vendored():
        pytest.skip("Resemble Enhance not vendored — run scripts/fetch_resemble_enhance.py")
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(16000 * 2)).astype(np.float32)  # 2 s @ 16 kHz
    y = ResembleEnhanceEnhancer(nfe=8).apply(x, sr=16000)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
