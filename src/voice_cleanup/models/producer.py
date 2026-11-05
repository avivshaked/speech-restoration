"""Producer — a small from-scratch spectral model that masters a clean restoration.

Maps ``cleanraw`` → ``produced``: the second stage of the two-model pipeline (Restorer first,
then Producer). The transform is near-deterministic (verified on DAPS): a static presence EQ
(+2–3 dB through 0.4–6 kHz, low-cut below ~100 Hz), ~4–6 dB of compression (crest 20→16 dB),
loudness to ~−16 LUFS, plus the de-breathing inherited from the ``clean`` edit. All of that is
a time–frequency *gain* on an already-clean signal — no content to regenerate — so we model it
as a **residual log-magnitude mask** and reuse the input phase, rather than a heavy generative
vocoder (the wrong, artifact-prone tool for a clean→clean transform).

Design choices:
- **Residual in log-magnitude.** The net predicts Δ(t,f); output magnitude is exp(logmag + Δ).
  EQ is a (time-constant) Δ ridge per frequency; compression is a level-dependent Δ; de-breathing
  is a strongly negative Δ localized to breath frames. The final conv is **zero-initialized**, so
  the model *starts as identity* (Δ≡0, output == input) — the right prior for a gentle transform,
  and it only has to learn the (small) departure from passthrough.
- **Input phase reused.** The mastering chain barely touches phase; predicting magnitude only is
  stable and standard. A complex/phase head can be added later if the ear/metrics demand it.
- **Compact U-Net** over the spectrogram (both axes downsampled) with skip connections — enough
  time context for compression envelopes and ~300 ms breaths, cheap on the 4090.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 1024
    hop: int = 256
    sr: int = 44100

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1  # 513


DEFAULT_STFT = STFTConfig()


class _ConvBlock(nn.Module):
    """Two 3×3 convs with GroupNorm + GELU. Keeps F,T (padding='same')."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralProducer(nn.Module):
    """Residual log-magnitude mask U-Net: waveform → mastered waveform.

    `channels` sets encoder width per level; `len(channels)` is the U-Net depth. Δ is clamped to
    [delta_min, delta_max] (natural-log units) so de-breathing can drive bins toward silence
    (−12 ≈ −104 dB) while boosts stay sane (+3 ≈ +26 dB).
    """

    def __init__(
        self,
        stft: STFTConfig = DEFAULT_STFT,
        channels: tuple[int, ...] = (32, 64, 128),
        *,
        delta_min: float = -12.0,
        delta_max: float = 3.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.stft = stft
        self.delta_min, self.delta_max, self.eps = delta_min, delta_max, eps
        self.register_buffer("window", torch.hann_window(stft.n_fft))
        self.depth = len(channels)
        self._mult = 2 ** self.depth  # freq/time padded to a multiple of this for clean strides

        # Encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        c_prev = 1
        for c in channels:
            self.enc.append(_ConvBlock(c_prev, c))
            self.downs.append(nn.Conv2d(c, c, 4, stride=2, padding=1))
            c_prev = c
        self.bottleneck = _ConvBlock(c_prev, c_prev * 2)
        # Decoder (mirror, with skips)
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        c_up = c_prev * 2
        for c in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(c_up, c, 4, stride=2, padding=1))
            self.dec.append(_ConvBlock(c * 2, c))  # *2 from skip concat
            c_up = c
        self.head = nn.Conv2d(channels[0], 1, 1)
        nn.init.zeros_(self.head.weight)  # identity at init: Δ ≡ 0 → output == input
        nn.init.zeros_(self.head.bias)

    # ---- spectral transforms -------------------------------------------------
    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav, self.stft.n_fft, self.stft.hop, window=self.window,
            return_complex=True, center=True,
        )  # (B, F, T) complex

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec, self.stft.n_fft, self.stft.hop, window=self.window,
            center=True, length=length,
        )

    def predict_delta(self, logmag: torch.Tensor) -> torch.Tensor:
        """logmag (B,1,F,T) → Δ (B,1,F,T), clamped. Pads F,T to a clean stride multiple."""
        _, _, Fdim, Tdim = logmag.shape
        pf = (self._mult - Fdim % self._mult) % self._mult
        pt = (self._mult - Tdim % self._mult) % self._mult
        x = F.pad(logmag, (0, pt, 0, pf))  # pad high-freq and tail (reflect-free, zeros)

        skips = []
        for block, down in zip(self.enc, self.downs, strict=True):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.ups, self.dec, reversed(skips), strict=True):
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        delta = self.head(x)[..., :Fdim, :Tdim]
        return delta.clamp(self.delta_min, self.delta_max)

    # ---- end-to-end ----------------------------------------------------------
    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        """wav (B, N) → dict with the mastered waveform plus the log-mags for the loss.

        Returns in_logmag / out_logmag so the trainer can put its primary L1 on the log-mag
        (the actual learning target) and a multi-res STFT loss on `wav_out`.
        """
        spec = self._stft(wav)  # (B,F,T) complex
        mag = spec.abs()
        phase = spec / (mag + self.eps)  # unit-modulus complex phase
        in_logmag = torch.log(mag + self.eps).unsqueeze(1)  # (B,1,F,T)
        delta = self.predict_delta(in_logmag)
        out_logmag = in_logmag + delta
        out_mag = torch.exp(out_logmag).squeeze(1)
        out_spec = out_mag * phase  # reuse input phase
        wav_out = self._istft(out_spec, length=wav.shape[-1])
        return {
            "wav_out": wav_out,
            "in_logmag": in_logmag.squeeze(1),
            "out_logmag": out_logmag.squeeze(1),
            "delta": delta.squeeze(1),
        }

    @torch.no_grad()
    def enhance(self, wav: torch.Tensor) -> torch.Tensor:
        """Inference convenience: (N,) or (B,N) → same-shape mastered waveform."""
        self.eval()
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)
        out = self.forward(wav)["wav_out"]
        return out.squeeze(0) if squeeze else out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def target_logmag(producer: SpectralProducer, wav: torch.Tensor) -> torch.Tensor:
    """Log-magnitude of `wav` under the producer's STFT — the L1 target for training."""
    spec = producer._stft(wav)
    return torch.log(spec.abs() + producer.eps)
