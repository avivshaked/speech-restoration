"""Multi-resolution STFT loss (Yamamoto et al., Parallel WaveGAN, 2020).

Sums spectral-convergence + log-magnitude L1 over several STFT resolutions, so the model is
penalised on spectral shape at multiple time/frequency trade-offs rather than one. Used as the
waveform-domain complement to the Producer's primary log-magnitude L1 (which lives at the model's
own analysis resolution); the extra resolutions catch errors that one window size hides.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _stft_mag(
    wav: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor, eps: float
) -> torch.Tensor:
    spec = torch.stft(wav, n_fft, hop, window=win, return_complex=True, center=True)
    return spec.abs() + eps


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean over resolutions of (spectral convergence + log-mag L1). `pred`,`target`: (B, N)."""
    total = pred.new_zeros(())
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        mp = _stft_mag(pred, n_fft, hop, win, eps)
        mt = _stft_mag(target, n_fft, hop, win, eps)
        sc = torch.norm(mt - mp, p="fro") / (torch.norm(mt, p="fro") + eps)  # spectral convergence
        logmag = F.l1_loss(torch.log(mp), torch.log(mt))
        total = total + sc + logmag
    return total / len(fft_sizes)
