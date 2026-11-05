"""NISQA non-intrusive speech quality + dimension predictor.

Vendored from gabrielmittag/NISQA at a pinned commit; see scripts/fetch_nisqa.py.
Loads the multidim checkpoint (`weights/nisqa.tar`) which outputs MOS + four
sub-dimensions: noisiness, discontinuity, coloration, loudness.

The upstream API is file-based (`nisqaModel(args).predict()` reads a WAV path
off disk), so the wrapper writes incoming audio to a temp 16 kHz mono WAV and
re-instantiates `nisqaModel` per call. Model re-instantiation is cheap (~100 ms
on a 1 MB checkpoint); the actual mel-spec + forward pass dominate.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

NISQA_SR = 48000  # NISQA was trained on 48 kHz; SpeechQualityDataset will resample if not

EXTERNAL_NISQA_DIR = (Path(__file__).resolve().parents[3] / "external" / "nisqa").resolve()
NISQA_WEIGHTS_PATH = EXTERNAL_NISQA_DIR / "weights" / "nisqa.tar"
NISQA_REQUIRED_SOURCES = (
    EXTERNAL_NISQA_DIR / "nisqa" / "NISQA_lib.py",
    EXTERNAL_NISQA_DIR / "nisqa" / "NISQA_model.py",
)

_NISQA_MODEL_CACHED: bool = False  # tracks whether we've verified vendor files & sys.path


def _vendor_present() -> tuple[bool, list[Path]]:
    missing = [p for p in (*NISQA_REQUIRED_SOURCES, NISQA_WEIGHTS_PATH) if not p.is_file()]
    return (not missing, missing)


def _ensure_vendor() -> None:
    """Verify vendored NISQA source + weights exist; instruct user to fetch if not."""
    ok, missing = _vendor_present()
    if not ok:
        raise FileNotFoundError(
            "NISQA vendor files missing: "
            + ", ".join(str(p.relative_to(EXTERNAL_NISQA_DIR.parent)) for p in missing)
            + "\nRun: uv run python scripts/fetch_nisqa.py"
        )
    # Add external/nisqa to sys.path so `from nisqa.NISQA_model import nisqaModel` resolves.
    vendor_path = str(EXTERNAL_NISQA_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        logger.debug("NISQA · inserted %s onto sys.path", vendor_path)


def _log_cache_preflight() -> None:
    """Pre-flight log per feedback_model_load_verbosity contract."""
    ok, missing = _vendor_present()
    if ok:
        total = sum(p.stat().st_size for p in (*NISQA_REQUIRED_SOURCES, NISQA_WEIGHTS_PATH))
        logger.info(
            "NISQA · vendored source + weights already on disk at %s (%.2f MB) — "
            "loading from disk · license=MIT (gabrielmittag/NISQA)",
            EXTERNAL_NISQA_DIR, total / 1024**2,
        )
    else:
        logger.info(
            "NISQA · vendor files MISSING — run `uv run python scripts/fetch_nisqa.py` "
            "to populate %s",
            EXTERNAL_NISQA_DIR,
        )


def _to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


def _resample_if_needed(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sr, target_sr)
    return resampled.squeeze(0).numpy()


def _build_predict_args(deg_path: Path) -> dict[str, Any]:
    """Minimal args dict that nisqaModel(args).predict() requires.

    The upstream constructor loads the checkpoint and merges its stored args
    over these, so all the model-hyperparam keys (ms_*, cnn_*, td_*, etc.) are
    populated automatically from the .tar.
    """
    return {
        "mode": "predict_file",
        "pretrained_model": str(NISQA_WEIGHTS_PATH),
        "deg": str(deg_path),
        "data_dir": str(deg_path.parent),
        "output_dir": None,
        "csv_file": None,
        "csv_deg": None,
        "ms_channel": None,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "tr_parallel": False,
    }


def _run_nisqa_predict(deg_path: Path) -> dict[str, float]:
    """Invoke vendored nisqaModel.predict() on `deg_path`, return its score dict."""
    _ensure_vendor()
    from nisqa.NISQA_model import nisqaModel  # type: ignore[import-not-found]

    args = _build_predict_args(deg_path)

    # NISQA prints copiously (yaml dump of args, progress lines, results table).
    # Capture stdout and route a single info-level summary through our logger so
    # the verbose-by-default contract still holds without 50 lines of upstream noise.
    captured = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(captured):
        model = nisqaModel(args)
        df = model.predict()
    elapsed = time.monotonic() - t0

    upstream_lines = [ln for ln in captured.getvalue().splitlines() if ln.strip()]
    logger.debug(
        "NISQA upstream stdout (%d lines):\n%s",
        len(upstream_lines), "\n".join(upstream_lines),
    )

    row = df.iloc[0].to_dict()
    scores = {
        "mos_pred": float(row["mos_pred"]),
        "noi_pred": float(row["noi_pred"]),
        "dis_pred": float(row["dis_pred"]),
        "col_pred": float(row["col_pred"]),
        "loud_pred": float(row["loud_pred"]),
    }
    logger.info(
        "NISQA = MOS %.2f · NOI %.2f · DIS %.2f · COL %.2f · LOUD %.2f (%.0f ms total)",
        scores["mos_pred"], scores["noi_pred"], scores["dis_pred"],
        scores["col_pred"], scores["loud_pred"], elapsed * 1000,
    )
    return scores


def nisqa_multidim(signal: Any, sr: int = NISQA_SR) -> dict[str, float]:
    """Predict NISQA multidim scores for a 1D waveform.

    Args:
        signal: 1D waveform (numpy or torch). Any sample rate; resampled to 48 kHz
            for NISQA's training distribution.
        sr: Sample rate of `signal`.

    Returns:
        Dict with keys mos_pred, noi_pred, dis_pred, col_pred, loud_pred in [1, 5].
    """
    global _NISQA_MODEL_CACHED
    if not _NISQA_MODEL_CACHED:
        _log_cache_preflight()
        _ensure_vendor()
        logger.info("NISQA · initializing model · device=cpu · weights=%s", NISQA_WEIGHTS_PATH.name)
        _NISQA_MODEL_CACHED = True

    audio = _to_numpy_1d(signal)
    original_sr = sr
    if sr != NISQA_SR:
        logger.info("NISQA · resampling %d → %d Hz", original_sr, NISQA_SR)
        audio = _resample_if_needed(audio, sr, NISQA_SR)

    logger.info(
        "NISQA · scoring %.2fs of audio (sr=%d Hz) via temp WAV", len(audio) / NISQA_SR, NISQA_SR,
    )
    # Force matplotlib to a non-interactive backend before NISQA_lib imports it.
    os.environ.setdefault("MPLBACKEND", "Agg")

    with tempfile.TemporaryDirectory(prefix="nisqa-") as td:
        wav_path = Path(td) / "input.wav"
        sf.write(str(wav_path), audio, NISQA_SR, subtype="PCM_16")
        return _run_nisqa_predict(wav_path)


__all__ = ["nisqa_multidim", "NISQA_SR", "EXTERNAL_NISQA_DIR", "NISQA_WEIGHTS_PATH"]
