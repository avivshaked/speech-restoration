"""Resemble Enhance zero-shot enhancer — Path-A generative-restoration baseline.

Resemble Enhance (Resemble AI, MIT) is a *generative restorer*, not a mask-based
suppressor: a denoiser UNet cleans the input, then a Latent Conditional Flow
Matching model (IRMAE-compressed latent) regenerates clean speech, vocoded by a
UnivNet GAN at 44.1 kHz. Unlike MP-SENet's multiplicative mask, it can *synthesise*
plausible speech detail rather than only attenuate the mixture — which is exactly
the property the Phase-1 baselines said suppression lacks (WER rose at low SNR).
See docs/research-papers/ (Resemble Enhance / LCFM) and docs/decisions/log.md
(2026-05-28: Path-A base model = Resemble Enhance, validate zero-shot first).

The model + weights are vendored from resemble-ai/resemble-enhance at a pinned SHA
(see scripts/fetch_resemble_enhance.py); we add external/resemble-enhance to
sys.path and call the released `enhance()` path verbatim — we do NOT reimplement
the architecture.

DeepSpeed is *never used at inference* (the checkpoint loads via plain
`torch.load(...)["module"]`, and `Engine(DeepSpeedEngine)` is only defined, never
instantiated, on the import path). But the upstream package `import deepspeed`s at
module load, and DeepSpeed won't import on this box (it probes for a CUDA toolkit
to compile ops, and there's none under WSL). So we inject a tiny `deepspeed` stub
into sys.modules before importing — no DeepSpeed install, no CUDA toolkit. If a
real DeepSpeed is ever installed, the stub defers to it.

The public `enhance()` runs denoise→enhance and resamples internally to 44.1 kHz,
returning 44.1 kHz audio; we resample back to the caller's rate (our 16 kHz panel)
and trim/pad to the input length so intrusive metrics stay aligned.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from voice_cleanup.eval.enhancers import Enhancer, register

logger = logging.getLogger(__name__)

RE_SR = 44100  # Resemble Enhance operates at 44.1 kHz; we resample in/out as needed.

EXTERNAL_RE_DIR = (Path(__file__).resolve().parents[3] / "external" / "resemble-enhance").resolve()
PKG_DIR = EXTERNAL_RE_DIR / "resemble_enhance"
# run_dir layout expected by resemble_enhance.enhancer.download.download().
RUN_DIR = EXTERNAL_RE_DIR / "model_repo" / "enhancer_stage2"
CKPT_PATH = RUN_DIR / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
HPARAMS_PATH = RUN_DIR / "hparams.yaml"

RE_REQUIRED_SOURCES = (
    PKG_DIR / "__init__.py",
    PKG_DIR / "inference.py",
    PKG_DIR / "enhancer" / "inference.py",
    PKG_DIR / "enhancer" / "train.py",
    PKG_DIR / "utils" / "distributed.py",
)

# Cache of the loaded enhancer model, keyed by inference config (so a re-config
# loads a fresh module copy only if needed — in practice one entry).
_MODEL_CACHE: dict[str, Any] = {}


def _build_deepspeed_stub() -> dict[str, types.ModuleType]:
    """Build the minimal `deepspeed` stub modules (not yet registered).

    Satisfies only the symbols touched at *import* time on the inference path:
    `DeepSpeedConfig`, `init_distributed`, `accelerator.get_accelerator`,
    `runtime.engine.DeepSpeedEngine` (subclassed but never instantiated),
    `runtime.utils.clip_grad_norm_`. Every callable raises if actually invoked —
    that would mean a training path ran, which inference never does. Each module
    gets a valid `__spec__` so `importlib.util.find_spec("deepspeed")` doesn't
    raise while the stub is live.
    """

    def _training_only(name: str):
        def _raise(*_a, **_k):
            raise RuntimeError(
                f"deepspeed stub: {name} is training-only and must not run during inference"
            )
        return _raise

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return m

    ds = _mod("deepspeed")
    ds.init_distributed = _training_only("init_distributed")

    class _DeepSpeedConfig:
        def __init__(self, *_a, **_k):
            raise RuntimeError("deepspeed stub: DeepSpeedConfig is training-only")

    ds.DeepSpeedConfig = _DeepSpeedConfig

    acc = _mod("deepspeed.accelerator")
    acc.get_accelerator = _training_only("get_accelerator")

    runtime = _mod("deepspeed.runtime")
    engine = _mod("deepspeed.runtime.engine")

    class DeepSpeedEngine:  # only `class Engine(DeepSpeedEngine)` at import; never built
        pass

    engine.DeepSpeedEngine = DeepSpeedEngine
    rutils = _mod("deepspeed.runtime.utils")
    rutils.clip_grad_norm_ = _training_only("clip_grad_norm_")

    ds.accelerator = acc
    ds.runtime = runtime
    runtime.engine = engine
    runtime.utils = rutils

    return {
        "deepspeed": ds,
        "deepspeed.accelerator": acc,
        "deepspeed.runtime": runtime,
        "deepspeed.runtime.engine": engine,
        "deepspeed.runtime.utils": rutils,
    }


def _install_deepspeed_stub() -> list[str]:
    """Register the stub in sys.modules; return the keys we added (for later removal).

    Returns [] if a real DeepSpeed is installed/imported (we leave it alone). The
    stub is deliberately TEMPORARY: callers should `_remove_deepspeed_stub()` the
    returned keys once RE's modules have bound their references, so other libraries
    (e.g. transformers, which probes `find_spec("deepspeed")` when loading Whisper)
    don't mistake the stub for a real DeepSpeed install.
    """
    if "deepspeed" in sys.modules:
        return []
    if importlib.util.find_spec("deepspeed") is not None:
        return []  # a real DeepSpeed is installed; let the package use it

    stub = _build_deepspeed_stub()
    sys.modules.update(stub)
    logger.debug("Resemble Enhance · injected minimal deepspeed stub (inference-only)")
    return list(stub)


def _remove_deepspeed_stub(keys: list[str]) -> None:
    """Remove stub modules we added, leaving already-bound RE references intact."""
    for key in keys:
        mod = sys.modules.get(key)
        if mod is not None and getattr(mod.__spec__, "loader", "x") is None:
            # Only pop our loader-less stub, never a real module that appeared since.
            del sys.modules[key]
    if keys:
        logger.debug("Resemble Enhance · removed temporary deepspeed stub from sys.modules")


def _vendor_present() -> tuple[bool, list[Path]]:
    required = (*RE_REQUIRED_SOURCES, HPARAMS_PATH, CKPT_PATH)
    missing = [p for p in required if not p.is_file()]
    return (not missing, missing)


def _ensure_vendor() -> None:
    """Verify vendored source + checkpoint exist; instruct user to fetch if not."""
    ok, missing = _vendor_present()
    if not ok:
        raise FileNotFoundError(
            "Resemble Enhance vendor files missing: "
            + ", ".join(str(p.relative_to(EXTERNAL_RE_DIR.parent)) for p in missing)
            + "\nRun: uv run python scripts/fetch_resemble_enhance.py"
        )
    vendor_path = str(EXTERNAL_RE_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        logger.debug("Resemble Enhance · inserted %s onto sys.path", vendor_path)


def _log_cache_preflight() -> None:
    """Pre-flight log per the feedback_model_load_verbosity contract."""
    ok, _ = _vendor_present()
    if ok:
        ckpt_mb = CKPT_PATH.stat().st_size / 1024**2
        logger.info(
            "Resemble Enhance · vendored source + enhancer_stage2 checkpoint already on disk "
            "at %s (checkpoint %.1f MB) — loading from disk · "
            "license=MIT (resemble-ai/resemble-enhance)",
            EXTERNAL_RE_DIR, ckpt_mb,
        )
    else:
        logger.info(
            "Resemble Enhance · vendor files MISSING — run "
            "`uv run python scripts/fetch_resemble_enhance.py` to populate %s",
            EXTERNAL_RE_DIR,
        )


def _load_enhancer() -> Any:
    """Lazily load + cache the Resemble Enhance model (denoiser+enhancer)."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]

    _log_cache_preflight()
    _ensure_vendor()

    # Stub deepspeed only across RE's import, then remove it so it can't fool
    # other libraries (transformers probes find_spec("deepspeed") when loading
    # Whisper). RE's modules keep their already-bound references after removal.
    stub_keys = _install_deepspeed_stub()
    try:
        from resemble_enhance.enhancer.inference import load_enhancer  # noqa: PLC0415
    finally:
        _remove_deepspeed_stub(stub_keys)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Resemble Enhance · initializing (LCFM + UnivNet vocoder) · "
        "device=%s · dtype=float32 · sr=%d",
        device, RE_SR,
    )
    t0 = time.monotonic()
    model = load_enhancer(RUN_DIR, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Resemble Enhance · model ready in %.2fs · %.2fM params · checkpoint on-disk %.1f MB",
        time.monotonic() - t0, n_params / 1e6, CKPT_PATH.stat().st_size / 1024**2,
    )
    _MODEL_CACHE["model"] = model
    return model


def _to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


class ResembleEnhanceEnhancer(Enhancer):
    """Zero-shot Resemble Enhance generative restorer (denoise → LCFM enhance).

    Inference knobs mirror the upstream `enhance()` defaults:
      nfe     — flow-matching function evals (denoising steps); higher = better, slower.
      solver  — ODE solver ('midpoint' | 'rk4' | 'euler').
      lambd   — denoiser/enhancer blend in [0, 1] (0.5 = balanced).
      tau     — CFM prior temperature in [0, 1].
    """

    name = "resemble-enhance"

    def __init__(
        self,
        nfe: int = 32,
        solver: str = "midpoint",
        lambd: float = 0.5,
        tau: float = 0.5,
    ) -> None:
        if not 0 < nfe <= 128:
            raise ValueError(f"nfe must be in (0, 128], got {nfe}")
        if solver not in ("midpoint", "rk4", "euler"):
            raise ValueError(f"solver must be one of midpoint/rk4/euler, got {solver!r}")
        if not 0 <= lambd <= 1:
            raise ValueError(f"lambd must be in [0, 1], got {lambd}")
        if not 0 <= tau <= 1:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.nfe = nfe
        self.solver = solver
        self.lambd = lambd
        self.tau = tau

    def label(self) -> str:
        return f"Resemble Enhance zero-shot (nfe={self.nfe}, {self.solver})"

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        model = _load_enhancer()
        device = next(model.parameters()).device

        x = _to_numpy_1d(audio)
        n_in = len(x)
        logger.info(
            "Resemble Enhance · enhancing %.2fs (%d samples) · sr=%d Hz · "
            "nfe=%d solver=%s lambd=%.2f tau=%.2f",
            n_in / sr, n_in, sr, self.nfe, self.solver, self.lambd, self.tau,
        )

        # enhance() resamples sr → 44.1 kHz internally and returns 44.1 kHz audio.
        from resemble_enhance.enhancer.inference import enhance  # noqa: PLC0415

        dwav = torch.from_numpy(x)
        t0 = time.monotonic()
        with torch.no_grad():
            hwav, out_sr = enhance(
                dwav=dwav, sr=sr, device=device,
                nfe=self.nfe, solver=self.solver, lambd=self.lambd, tau=self.tau,
            )
        logger.info(
            "Resemble Enhance · enhance() done in %.2fs (%d samples @ %d Hz)",
            time.monotonic() - t0, hwav.numel(), out_sr,
        )

        # Resample back to the caller's rate for the 16 kHz panel.
        if out_sr != sr:
            logger.info("Resemble Enhance · resampling output %d → %d Hz", out_sr, sr)
            hwav = torchaudio.functional.resample(hwav.unsqueeze(0), out_sr, sr).squeeze(0)

        result = hwav.detach().cpu().numpy().astype(np.float32)

        # Pad/trim to input length so intrusive metrics align with the reference.
        if len(result) != n_in:
            if len(result) > n_in:
                result = result[:n_in]
            else:
                result = np.pad(result, (0, n_in - len(result)))
        return result


# Register on import so the CLI / runner can resolve "resemble-enhance".
register("resemble-enhance", ResembleEnhanceEnhancer)

__all__ = ["ResembleEnhanceEnhancer"]
