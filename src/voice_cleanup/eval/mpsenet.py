"""MP-SENet zero-shot enhancer — Phase-1 baseline (Path B reference point).

MP-SENet (Lu et al., Interspeech 2023) denoises magnitude and phase spectra in
parallel: a Conformer-embedded codec encodes stacked compressed-magnitude +
wrapped-phase, a mask decoder recovers clean magnitude and a parallel phase
decoder predicts clean phase, recombined via ISTFT. See
docs/research-papers/MP-SENet-2023.pdf and docs/research-papers-md/MP-SENet-2023/.

The model + checkpoints are vendored from yxlu-0102/MP-SENet at a pinned SHA
(see scripts/fetch_mpsenet.py); we add external/mp-senet to sys.path and import
the released `MPNet` generator + STFT helpers verbatim — we do NOT reimplement
the architecture. The inference recipe (16 kHz, per-utterance RMS normalisation,
STFT n_fft=400/hop=100/win=400, power-law compression c=0.3) mirrors the repo's
inference.py exactly.

Two checkpoints ship: `vb` (VoiceBank+DEMAND, the paper's headline, PESQ 3.50)
and `dns` (DNS Challenge, trained on far broader real-world noise). We default to
`dns` for the real-podcast test set — VB+DEMAND is read speech mixed with a narrow
synthetic noise set and generalises poorly out of domain, whereas the DNS model is
the better zero-shot generaliser to real recordings. Override via
`get_enhancer("mp-senet", checkpoint="vb")`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from voice_cleanup.eval.enhancers import Enhancer, register

logger = logging.getLogger(__name__)

MPSENET_SR = 16000  # MP-SENet operates at 16 kHz; we resample in/out as needed.

EXTERNAL_MPSENET_DIR = (Path(__file__).resolve().parents[3] / "external" / "mp-senet").resolve()
MPSENET_CKPT_DIR = EXTERNAL_MPSENET_DIR / "best_ckpt"
MPSENET_CONFIG_PATH = MPSENET_CKPT_DIR / "config.json"

# Short checkpoint id → on-disk filename.
CHECKPOINTS = {
    "dns": MPSENET_CKPT_DIR / "g_best_dns",  # DNS Challenge — default for real noise
    "vb": MPSENET_CKPT_DIR / "g_best_vb",    # VoiceBank+DEMAND — paper headline
}

MPSENET_REQUIRED_SOURCES = (
    EXTERNAL_MPSENET_DIR / "models" / "model.py",
    EXTERNAL_MPSENET_DIR / "models" / "transformer.py",
    EXTERNAL_MPSENET_DIR / "utils.py",
    EXTERNAL_MPSENET_DIR / "dataset.py",
    EXTERNAL_MPSENET_DIR / "env.py",
)

# Cache of loaded (model, config) singletons, keyed by checkpoint id.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _vendor_present(ckpt_path: Path) -> tuple[bool, list[Path]]:
    required = (*MPSENET_REQUIRED_SOURCES, MPSENET_CONFIG_PATH, ckpt_path)
    missing = [p for p in required if not p.is_file()]
    return (not missing, missing)


def _ensure_vendor(ckpt_path: Path) -> None:
    """Verify vendored source + checkpoint exist; instruct user to fetch if not."""
    ok, missing = _vendor_present(ckpt_path)
    if not ok:
        raise FileNotFoundError(
            "MP-SENet vendor files missing: "
            + ", ".join(str(p.relative_to(EXTERNAL_MPSENET_DIR.parent)) for p in missing)
            + "\nRun: uv run python scripts/fetch_mpsenet.py"
        )
    vendor_path = str(EXTERNAL_MPSENET_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        logger.debug("MP-SENet · inserted %s onto sys.path", vendor_path)


def _log_cache_preflight(ckpt_id: str, ckpt_path: Path) -> None:
    """Pre-flight log per the feedback_model_load_verbosity contract."""
    ok, missing = _vendor_present(ckpt_path)
    if ok:
        total = sum(p.stat().st_size for p in (*MPSENET_REQUIRED_SOURCES, ckpt_path))
        logger.info(
            "MP-SENet · vendored source + '%s' checkpoint already on disk at %s "
            "(%.2f MB) — loading from disk · license=MIT (yxlu-0102/MP-SENet)",
            ckpt_id, EXTERNAL_MPSENET_DIR, total / 1024**2,
        )
    else:
        logger.info(
            "MP-SENet · vendor files MISSING — run `uv run python scripts/fetch_mpsenet.py` "
            "to populate %s",
            EXTERNAL_MPSENET_DIR,
        )


def _load_model(ckpt_id: str) -> tuple[Any, Any]:
    """Lazily load + cache the MPNet generator for `ckpt_id` ('dns' or 'vb')."""
    if ckpt_id in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_id]

    if ckpt_id not in CHECKPOINTS:
        raise ValueError(
            f"unknown MP-SENet checkpoint {ckpt_id!r}. Available: {sorted(CHECKPOINTS)}"
        )
    ckpt_path = CHECKPOINTS[ckpt_id]

    _log_cache_preflight(ckpt_id, ckpt_path)
    _ensure_vendor(ckpt_path)

    # Imports resolve against external/mp-senet on sys.path (set by _ensure_vendor).
    from env import AttrDict  # noqa: PLC0415
    from models.model import MPNet  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # MP-SENet checkpoints are fp32; STFT path is fp32-only.
    h = AttrDict(json.loads(MPSENET_CONFIG_PATH.read_text()))
    logger.info(
        "MP-SENet · initializing MPNet · checkpoint=%s · device=%s · dtype=%s · "
        "sr=%d n_fft=%d hop=%d win=%d c=%.2f",
        ckpt_id, device, str(dtype).replace("torch.", ""),
        h.sampling_rate, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )

    t0 = time.monotonic()
    model = MPNet(h).to(device).eval()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["generator"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "MP-SENet · MPNet ready in %.2fs · %.2fM params · checkpoint on-disk %.2f MB",
        time.monotonic() - t0, n_params / 1e6, ckpt_path.stat().st_size / 1024**2,
    )

    _MODEL_CACHE[ckpt_id] = (model, h)
    return model, h


def _to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


class MPSENetEnhancer(Enhancer):
    """Zero-shot MP-SENet denoiser. Defaults to the DNS Challenge checkpoint."""

    name = "mp-senet"

    def __init__(self, checkpoint: str = "dns") -> None:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(
                f"unknown MP-SENet checkpoint {checkpoint!r}. Available: {sorted(CHECKPOINTS)}"
            )
        self.checkpoint = checkpoint

    def label(self) -> str:
        pretty = {"dns": "DNS Challenge", "vb": "VoiceBank+DEMAND"}[self.checkpoint]
        return f"MP-SENet zero-shot ({pretty})"

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        model, h = _load_model(self.checkpoint)
        device = next(model.parameters()).device

        x = _to_numpy_1d(audio)
        n_in = len(x)
        logger.info(
            "MP-SENet · enhancing %.2fs (%d samples) · sr=%d Hz · checkpoint=%s",
            n_in / sr, n_in, sr, self.checkpoint,
        )

        # Resample to the model's 16 kHz if needed.
        wav = torch.from_numpy(x).to(device)
        if sr != MPSENET_SR:
            logger.info("MP-SENet · resampling %d → %d Hz for inference", sr, MPSENET_SR)
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, MPSENET_SR).squeeze(0)

        # Per-utterance RMS normalisation, verbatim from MP-SENet's inference.py
        # (https://github.com/yxlu-0102/MP-SENet, MIT licence).
        energy = torch.sum(wav**2)
        if energy <= 1e-12:
            logger.info("MP-SENet · near-silent input (energy≈0) — returning passthrough copy")
            return x.copy()
        norm_factor = torch.sqrt(wav.numel() / energy)
        wav_n = (wav * norm_factor).unsqueeze(0)

        from dataset import mag_pha_istft, mag_pha_stft  # noqa: PLC0415

        t0 = time.monotonic()
        with torch.no_grad():
            amp, pha, _ = mag_pha_stft(
                wav_n, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            amp_g, pha_g, _ = model(amp, pha)
            out = mag_pha_istft(
                amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            out = (out / norm_factor).squeeze(0)
        logger.info(
            "MP-SENet · forward+ISTFT done in %.2fs (%d output samples @ %d Hz)",
            time.monotonic() - t0, out.numel(), MPSENET_SR,
        )

        # Resample back to the caller's sample rate if we changed it.
        if sr != MPSENET_SR:
            out = torchaudio.functional.resample(out.unsqueeze(0), MPSENET_SR, sr).squeeze(0)

        result = out.detach().cpu().numpy().astype(np.float32)

        # Pad/trim to the input length so intrusive metrics align with the reference.
        if len(result) != n_in:
            if len(result) > n_in:
                result = result[:n_in]
            else:
                result = np.pad(result, (0, n_in - len(result)))
        return result


# Register on import so the CLI / runner can resolve "mp-senet".
register("mp-senet", MPSENetEnhancer)

__all__ = ["MPSENetEnhancer", "CHECKPOINTS"]
