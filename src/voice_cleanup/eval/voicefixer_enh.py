"""VoiceFixer enhancer — a generative speech-restoration candidate for the model bazaar.

VoiceFixer (Liu et al., MIT-licensed, pip `voicefixer`) is a *generative restorer* in the
same spirit as the goal here: a ResUNet analysis stage predicts a clean mel from a degraded
input, then a neural vocoder (TFGAN) *re-synthesises* the waveform at 44.1 kHz. Because it
regenerates from a learned prior rather than masking the mixture, it is purpose-built for
*severe* degradation (de-noising, de-reverb, de-clipping, and bandwidth/super-resolution) —
exactly the "regenerate something plausibly human from a trashed observation" property that
RE (a faithful reconstructor conditioned on the degraded mel) lacks. See decision log
2026-05-31 (retire RE; model bazaar) and HANDOFF.md.

We call the released `restore_inmem` path verbatim — we do NOT reimplement the architecture.
Weights auto-download to ~/.cache/voicefixer on first construction.

`mode`: 0 = standard restore; 1 = add the un-restored residual back (less aggressive); 2 = a
training-time variant. We default to 0 and expose it so the bazaar can A/B modes by ear.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torchaudio

from voice_cleanup.eval.enhancers import Enhancer, register

logger = logging.getLogger(__name__)

VF_SR = 44100  # VoiceFixer works at 44.1 kHz end-to-end: restore_inmem expects a 44.1 kHz mono
# input (its `wav_10k` arg name is a misnomer — the library's own restore() loads at 44100 and
# saves at 44100; seg_length=44100*30 and the mel framing assume it) and the vocoder renders at
# 44.1 kHz. Feeding 16 kHz here makes the model read the samples as 44.1 kHz → ~2.76× pitch-up
# (chipmunk). So we resample to 44.1 kHz in, not down to 16 kHz.


class VoiceFixerEnhancer(Enhancer):
    """Zero-shot VoiceFixer generative restoration. Lazy, GPU if available."""

    name = "voicefixer"

    def __init__(self, mode: int = 0) -> None:
        self.mode = int(mode)
        self._model = None
        self._cuda = torch.cuda.is_available()

    def _ensure_model(self):
        if self._model is None:
            from voicefixer import VoiceFixer
            t0 = time.monotonic()
            logger.info("VoiceFixer · loading model (cuda=%s, mode=%d) …", self._cuda, self.mode)
            self._model = VoiceFixer()
            logger.info("VoiceFixer · model ready in %.1fs", time.monotonic() - t0)
        return self._model

    @torch.no_grad()
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        model = self._ensure_model()
        x = np.asarray(audio, dtype="float32")
        n_in = len(x)
        # restore_inmem works at 44.1 kHz (its `wav_10k` arg name is a misnomer — see VF_SR note).
        x44 = x
        if sr != VF_SR:
            x44 = torchaudio.functional.resample(
                torch.from_numpy(x), sr, VF_SR).numpy().astype("float32")
        logger.info("VoiceFixer · restoring %.2fs (mode=%d) …", n_in / sr, self.mode)
        t0 = time.monotonic()
        out44 = model.restore_inmem(x44, cuda=self._cuda, mode=self.mode)
        out44 = np.asarray(out44, dtype="float32").squeeze()
        # Resample VoiceFixer's 44.1 kHz output back to the caller's rate, match length.
        out = out44
        if sr != VF_SR:
            out = torchaudio.functional.resample(
                torch.from_numpy(out44), VF_SR, sr).numpy().astype("float32")
        out = out[:n_in] if len(out) >= n_in else np.pad(out, (0, n_in - len(out)))
        logger.info("VoiceFixer · done in %.1fs · out %.2fs @ %d Hz",
                    time.monotonic() - t0, len(out) / sr, sr)
        return out

    def label(self) -> str:
        return f"voicefixer (mode={self.mode})"


register("voicefixer", VoiceFixerEnhancer)

__all__ = ["VoiceFixerEnhancer"]
