"""SGMSE+ enhancer — score-based diffusion speech enhancement (Richter et al., sp-uhh).

A different generative mechanism for the bazaar: a score-based diffusion model that runs a
reverse SDE from the noisy spectrogram to a clean estimate. Vendored at `external/sgmse`
(gitignored). This wrapper calls the released `ScoreModel` + PC/ODE sampler verbatim (grounded
in their `enhancement.py`) — we do NOT reimplement the SDE.

⚠️ ENVIRONMENT: SGMSE+ pins `numpy<2.0` and needs pytorch-lightning / torchsde / sdeint /
torch-ema, which CONFLICT with this project's numpy-2.x env. So it is NOT installed in the main
env — it needs an isolated env (decision pending; see HANDOFF 2026-05-31). Until then this
module's import fails and `enhancers.py` skips registering it (the panel is unaffected). It is
also now a *lower-priority* candidate: SGMSE+ targets noise/reverb removal, not the bandwidth
extension / top-octave regeneration that is our gap (docs/research/restorer-candidates.md).

Set the checkpoint via env `SGMSE_CKPT` (e.g. the 48 kHz EARS-WHAM checkpoint). The model's
`backbone` tells us the native rate (ncsnpp_48k→48k, ncsnpp_v2→16k, else 16k).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from voice_cleanup.eval.enhancers import Enhancer, register

logger = logging.getLogger(__name__)

_SGMSE_DIR = Path(__file__).resolve().parents[3] / "external" / "sgmse"


class SGMSEEnhancer(Enhancer):
    """Zero-shot SGMSE+ diffusion enhancement (reverse SDE, PC sampler)."""

    name = "sgmse"

    def __init__(self, ckpt: str | None = None, n_steps: int = 30, snr: float = 0.5,
                 corrector_steps: int = 1) -> None:
        self.ckpt = ckpt or os.environ.get("SGMSE_CKPT", "")
        self.n_steps = int(n_steps)
        self.snr = float(snr)
        self.corrector_steps = int(corrector_steps)
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_sr: int | None = None
        self._pad_mode = "zero_pad"

    def _ensure_model(self):
        if self._model is None:
            if not self.ckpt or not Path(self.ckpt).is_file():
                raise FileNotFoundError(
                    f"SGMSE+ checkpoint not found (ckpt={self.ckpt!r}). Set env SGMSE_CKPT to a "
                    "downloaded .ckpt (e.g. the 48 kHz EARS-WHAM model from the sp-uhh repo).")
            if str(_SGMSE_DIR) not in sys.path:
                sys.path.insert(0, str(_SGMSE_DIR))
            from sgmse.model import ScoreModel  # noqa: PLC0415
            from sgmse.util.other import set_torch_cuda_arch_list  # noqa: PLC0415
            set_torch_cuda_arch_list()
            t0 = time.monotonic()
            logger.info("SGMSE+ · loading %s on %s …", Path(self.ckpt).name, self._device)
            m = ScoreModel.load_from_checkpoint(self.ckpt, map_location=self._device)
            m.eval()
            self._model = m
            bb = getattr(m, "backbone", "")
            self._target_sr = 48000 if bb == "ncsnpp_48k" else 16000
            self._pad_mode = "reflection" if bb in ("ncsnpp_48k", "ncsnpp_v2") else "zero_pad"
            logger.info("SGMSE+ · ready in %.1fs (backbone=%s, sr=%d)",
                        time.monotonic() - t0, bb, self._target_sr)
        return self._model

    @torch.no_grad()
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from sgmse.util.other import pad_spec  # noqa: PLC0415
        model = self._ensure_model()
        tsr = self._target_sr
        x = np.asarray(audio, dtype="float32")
        n_in = len(x)
        y = torch.from_numpy(x).unsqueeze(0)
        if sr != tsr:
            y = torchaudio.functional.resample(y, sr, tsr)
        y = y.to(self._device)
        T_orig = y.size(1)
        # Inference sequence adapted from SGMSE+'s enhancement.py
        # (https://github.com/sp-uhh/sgmse, MIT licence).
        norm = y.abs().max().clamp(min=1e-8)
        y = y / norm
        Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
        Y = pad_spec(Y, mode=self._pad_mode)
        logger.info("SGMSE+ · reverse SDE N=%d snr=%.2f on %.2fs …", self.n_steps, self.snr,
                    n_in / sr)
        t0 = time.monotonic()
        sampler = model.get_pc_sampler("reverse_diffusion", "ald", Y, N=self.n_steps,
                                       corrector_steps=self.corrector_steps, snr=self.snr)
        sample, _ = sampler()
        x_hat = model.to_audio(sample.squeeze(), T_orig) * norm
        out = x_hat.detach().cpu().float()
        if tsr != sr:
            out = torchaudio.functional.resample(out, tsr, sr)
        out = out.numpy().astype("float32").squeeze()
        out = out[:n_in] if len(out) >= n_in else np.pad(out, (0, n_in - len(out)))
        logger.info("SGMSE+ · done in %.1fs", time.monotonic() - t0)
        return out


register("sgmse", SGMSEEnhancer)

__all__ = ["SGMSEEnhancer"]
