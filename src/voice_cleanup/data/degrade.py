"""Restoration degradation pipeline — clean studio speech → realistically damaged.

Phase-2-lite. Per the 2026-05-28 eval re-architecture (docs/decisions/log.md): the
studio-reference track takes genuinely clean speech (EARS/DAPS), degrades it with a
known recipe, restores it, and scores the restoration against the *true* studio
reference. This module is the degradation half.

Design: small composable `Effect`s, each `apply(wav, sr, rng) -> wav` on float32
1-D numpy at the clip's native rate (full-band; we no longer crush to 16 kHz).
Effects are seeded via a passed `np.random.Generator` so a (clip, recipe, seed)
triple is fully reproducible. `DegradationChain` runs them in order; named
`RECIPES` bundle them into light/moderate/harsh mixes + single-effect diagnostics.

Grounded in Literature Review §10 (URGENT-2024 distortion set: noise, reverb,
clipping, bandwidth limitation, codec, EQ) and Resemble Enhance's own
`data/distorter/` framework — but implemented on torch/torchaudio/scipy only (no
WavAugment/sox dependency).

Codec artifacts (MP3/AAC/Opus) are a real encode→decode round-trip via the
uv-managed imageio-ffmpeg binary (Aviv-authorised 2026-05-28) — see `Codec`.

NOT YET IMPLEMENTED (out of scope tonight):
- **Real-RIR reverb**: we use a synthetic exponential-decay RIR here; pyroomacoustics
  (CLAUDE.md's documented RIR tool) or a measured RIR bank is the upgrade.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy import signal

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ffmpeg_exe() -> str | None:
    """Locate an ffmpeg binary: the uv-managed imageio-ffmpeg build, else PATH."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def _as_1d_float32(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D waveform, got shape {arr.shape}")
    return arr


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


# --- effects ---------------------------------------------------------------


class Effect:
    """Base degradation effect. Subclasses implement `apply`."""

    name: str = "effect"

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class Clipping(Effect):
    """Hard-clip at a fraction of peak amplitude → harmonic/intermodulation distortion.

    `frac` in (0, 1): the clip threshold as a fraction of the signal's peak. Lower =
    harsher (more samples clipped). This is the digital-overdrive / "too hot a gain
    stage" failure Aviv called out as genuinely destructive (data is removed).
    """

    name = "clip"
    frac_range: tuple[float, float] = (0.1, 0.4)

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        peak = float(np.max(np.abs(wav))) + 1e-9
        frac = float(rng.uniform(*self.frac_range))
        thresh = frac * peak
        return np.clip(wav, -thresh, thresh).astype(np.float32)


@dataclass
class Saturation(Effect):
    """Soft-clipping / analog-style recording saturation (a hot preamp/converter/tape).

    Distinct from hard `Clipping`: a smooth `tanh` knee adds *graduated* harmonics —
    "warmth" at low drive, grit as drive rises — instead of flat-topping the peaks.
    This is the recording-saturation failure Aviv named as a normal podcast case
    (input gain pushed too hot), perceptually gentler and more natural than digital
    clipping. Drive is applied to a peak-normalized signal so its meaning is
    level-independent, then the output is RMS-matched back to the input.
    """

    name = "sat"
    drive_range: tuple[float, float] = (2.0, 8.0)

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        peak = float(np.max(np.abs(wav))) + 1e-9
        drive = float(rng.uniform(*self.drive_range))
        x = wav / peak
        shaped = np.tanh(drive * x) / np.tanh(drive)
        # RMS-match to input so saturation changes timbre, not gross level.
        in_rms = _rms(wav)
        out_rms = _rms(shaped)
        if out_rms > 1e-9:
            shaped = shaped * (in_rms / out_rms)
        return shaped.astype(np.float32)


@dataclass
class BandwidthLimit(Effect):
    """Low-pass to a random cutoff → telephone/codec/cheap-mic high-frequency loss.

    Butterworth low-pass (zero-phase) to a cutoff in `cutoff_range` Hz. This is the
    band-limitation that 16 kHz eval was blind to and that a generative restorer is
    supposed to *extend back* (bandwidth extension).
    """

    name = "bwl"
    cutoff_range: tuple[float, float] = (2000.0, 7000.0)
    order: int = 8

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        nyq = sr / 2.0
        cutoff = float(rng.uniform(*self.cutoff_range))
        cutoff = min(cutoff, nyq * 0.98)
        sos = signal.butter(self.order, cutoff / nyq, btype="low", output="sos")
        return signal.sosfiltfilt(sos, wav).astype(np.float32)


@dataclass
class EqDamage(Effect):
    """Random peaking-EQ boosts/cuts → tonal/coloration damage (bad mic/room/EQ).

    Applies `n_bands` random peaking biquads (torchaudio.functional.equalizer_biquad)
    at random center freqs with random gains/Q. Models the "EQ damage" degradation.
    """

    name = "eq"
    n_bands_range: tuple[int, int] = (2, 4)
    freq_range: tuple[float, float] = (120.0, 8000.0)
    gain_db_range: tuple[float, float] = (-15.0, 15.0)
    q_range: tuple[float, float] = (1.0, 5.0)

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        t = torch.from_numpy(wav)
        n_bands = int(rng.integers(self.n_bands_range[0], self.n_bands_range[1] + 1))
        nyq = sr / 2.0
        for _ in range(n_bands):
            f0 = float(rng.uniform(self.freq_range[0], min(self.freq_range[1], nyq * 0.95)))
            gain = float(rng.uniform(*self.gain_db_range))
            q = float(rng.uniform(*self.q_range))
            t = AF.equalizer_biquad(t, sr, center_freq=f0, gain=gain, Q=q)
        out = t.numpy().astype(np.float32)
        # Guard against the rare resonant blow-up from stacked boosts.
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak
        return out


@dataclass
class AdditiveNoise(Effect):
    """Add white or pink noise at a random SNR (dB), measured on RMS power."""

    name = "noise"
    snr_db_range: tuple[float, float] = (5.0, 20.0)
    color: str = "white"  # "white" | "pink"

    def _noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        white = rng.standard_normal(n).astype(np.float64)
        if self.color == "white":
            return white
        # Pink: 1/f magnitude shaping in the frequency domain.
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        spec = spec / np.sqrt(freqs)
        return np.fft.irfft(spec, n=n)

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        snr_db = float(rng.uniform(*self.snr_db_range))
        noise = self._noise(len(wav), rng)
        sig_p = float(np.mean(wav.astype(np.float64) ** 2)) + 1e-12
        noise_p = float(np.mean(noise**2)) + 1e-12
        target_noise_p = sig_p / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_p / noise_p)
        return (wav + noise).astype(np.float32)


@dataclass
class SyntheticReverb(Effect):
    """Convolve with a synthetic exponential-decay RIR (RT60-parameterised).

    A dependency-free stand-in for real-RIR reverb: a direct impulse + an
    exponentially-decaying Gaussian tail (early energy + diffuse decay) with a random
    RT60. Output is RMS-matched to the input and trimmed to the original length.
    Upgrade path: pyroomacoustics image-source RIRs or a measured RIR bank.
    """

    name = "reverb"
    rt60_range: tuple[float, float] = (0.3, 0.9)
    direct_gain: float = 1.0

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        rt60 = float(rng.uniform(*self.rt60_range))
        n = max(1, int(rt60 * sr))
        t = np.arange(n) / sr
        # Diffuse tail: white noise * exp decay reaching -60 dB at rt60.
        decay = np.exp(-6.908 * t / rt60)  # ln(1000) ≈ 6.908 → -60 dB
        tail = rng.standard_normal(n) * decay
        rir = tail.astype(np.float64)
        rir[0] += self.direct_gain  # direct path
        rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-12)  # unit-energy → preserve loudness
        wet = signal.fftconvolve(wav.astype(np.float64), rir, mode="full")[: len(wav)]
        # RMS-match to input so reverb doesn't change apparent level.
        in_rms = _rms(wav)
        wet_rms = _rms(wet)
        if wet_rms > 1e-9:
            wet = wet * (in_rms / wet_rms)
        return wet.astype(np.float32)


# Encoder spec per lossy codec: (ffmpeg encoder, container extension).
_CODEC_SPECS: dict[str, tuple[str, str]] = {
    "mp3": ("libmp3lame", "mp3"),   # the canonical podcast/legacy-download codec
    "aac": ("aac", "m4a"),          # Apple Podcasts / AAC-LC delivery
    "opus": ("libopus", "opus"),    # modern streaming / WebRTC (resamples to 48 kHz internally)
}


@dataclass
class Codec(Effect):
    """Round-trip through a lossy codec (MP3/AAC/Opus) → real perceptual-coder artifacts.

    Encodes the clip to a low bitrate and decodes it straight back, so the signal
    carries the *actual* psychoacoustic-coding damage (pre-echo, transient smearing,
    high-frequency hole-punching, stereo/joint artifacts) that podcast audio picks up
    on delivery — not a hand-rolled approximation of it. Uses the uv-managed
    imageio-ffmpeg binary (Aviv-authorised 2026-05-28); no system install.

    `codecs` is the pool to draw from; `bitrate_kbps_range` the (low) bitrate window.
    Output is resampled back to `sr` mono and length-matched to the input (codecs add
    encoder-delay/priming samples, so we trim/pad to keep the clip frame-aligned).
    """

    name = "codec"
    codecs: tuple[str, ...] = ("mp3", "aac", "opus")
    bitrate_kbps_range: tuple[int, int] = (24, 64)

    def apply(self, wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        wav = _as_1d_float32(wav)
        exe = _ffmpeg_exe()
        if exe is None:
            raise RuntimeError(
                "Codec effect needs ffmpeg; none found. Install the in-project binary "
                "with `uv add imageio-ffmpeg` (Aviv-authorised) or put ffmpeg on PATH."
            )
        codec = str(rng.choice(self.codecs))
        encoder, ext = _CODEC_SPECS[codec]
        bitrate = int(rng.integers(self.bitrate_kbps_range[0], self.bitrate_kbps_range[1] + 1))

        n = len(wav)
        with tempfile.TemporaryDirectory(prefix="vc_codec_") as td:
            tmp = Path(td)
            src, enc, dec = tmp / "src.wav", tmp / f"enc.{ext}", tmp / "dec.wav"
            sf.write(src, wav, sr, subtype="FLOAT")
            self._run(exe, ["-i", str(src), "-c:a", encoder, "-b:a", f"{bitrate}k", str(enc)])
            self._run(exe, ["-i", str(enc), "-ar", str(sr), "-ac", "1", "-f", "wav", str(dec)])
            out, _ = sf.read(dec, dtype="float32", always_2d=False)
        out = _as_1d_float32(np.atleast_1d(out))
        # Codec delay shifts/lengthens the signal; restore the original frame count.
        out = out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))
        return out.astype(np.float32)

    @staticmethod
    def _run(exe: str, args: list[str]) -> None:
        cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}): {' '.join(args)}\n{proc.stderr.strip()}"
            )


# --- chain + recipes -------------------------------------------------------


@dataclass
class DegradationChain:
    """Ordered list of effects applied as a pipeline, with a peak-safety guard."""

    name: str
    effects: list[Effect] = field(default_factory=list)

    def apply(self, wav: np.ndarray, sr: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = _as_1d_float32(wav)
        for eff in self.effects:
            out = _as_1d_float32(eff.apply(out, sr, rng))
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:  # avoid hard wrap on write; keep relative dynamics
            out = out / peak
        return out

    def describe(self) -> str:
        return " → ".join(e.describe() for e in self.effects) or "identity"


# Severity-graded mixes (order matters: spectral/reverb before additive noise, like a
# real signal chain — room/mic coloration happens, THEN ambient noise sums in).
def _moderate() -> list[Effect]:
    return [
        EqDamage(),
        SyntheticReverb(rt60_range=(0.3, 0.6)),
        BandwidthLimit(cutoff_range=(3500.0, 7000.0)),
        Clipping(frac_range=(0.25, 0.5)),
        AdditiveNoise(snr_db_range=(10.0, 20.0), color="pink"),
        Codec(codecs=("mp3", "aac"), bitrate_kbps_range=(48, 96)),  # delivery encode, last
    ]


def _light() -> list[Effect]:
    return [
        EqDamage(gain_db_range=(-8.0, 8.0)),
        BandwidthLimit(cutoff_range=(6000.0, 9000.0)),
        AdditiveNoise(snr_db_range=(20.0, 30.0), color="pink"),
    ]


def _harsh() -> list[Effect]:
    return [
        EqDamage(gain_db_range=(-18.0, 18.0)),
        SyntheticReverb(rt60_range=(0.6, 1.0)),
        BandwidthLimit(cutoff_range=(2000.0, 4000.0)),
        Clipping(frac_range=(0.1, 0.3)),
        AdditiveNoise(snr_db_range=(2.0, 10.0), color="pink"),
        Codec(codecs=("mp3", "aac", "opus"), bitrate_kbps_range=(24, 48)),  # low-bitrate delivery
    ]


# --- realistic "normal podcast" profile (Aviv 2026-05-29) ------------------
# The dominant real-world failure is bad room acoustics + bad micing +
# clipping/saturation — NOT heavy additive noise / codec crush / telephone
# bandwidth. These recipes stay fully intelligible (the realistic target band,
# ~ DAPS real device/room conditions), which is where the generative restorer's
# faithful zero-shot range already lives. This is the *product* target; the
# severity-graded light/moderate/harsh mixes above are the stress ladder.
def _podcast() -> list[Effect]:
    return [
        SyntheticReverb(rt60_range=(0.3, 0.6)),                     # untreated room
        EqDamage(n_bands_range=(2, 3), gain_db_range=(-9.0, 9.0)),  # off-axis / cheap mic
        Saturation(drive_range=(2.0, 5.0)),                         # input gain a touch hot
        AdditiveNoise(snr_db_range=(24.0, 38.0), color="pink"),     # mild room/electrical hiss
    ]


def _room() -> list[Effect]:
    # Room + mic only (no level damage) — isolates the acoustics/coloration half.
    return [
        SyntheticReverb(rt60_range=(0.3, 0.7)),
        EqDamage(n_bands_range=(2, 3), gain_db_range=(-9.0, 9.0)),
    ]


def make_chain(recipe: str) -> DegradationChain:
    """Build a DegradationChain by recipe name.

    Realistic profile: 'podcast' (room + mic + saturation + mild noise — the normal
    use case) | 'room' (room + mic only). Severity stress ladder: 'light' |
    'moderate' | 'harsh'. Single-effect diagnostics: 'clip' | 'sat' | 'bwl' | 'eq' |
    'noise' | 'reverb' | 'codec'. 'clean' = identity (passthrough).
    """
    if recipe == "clean":
        return DegradationChain("clean", [])
    mixes = {
        "podcast": _podcast, "room": _room,
        "light": _light, "moderate": _moderate, "harsh": _harsh,
    }
    if recipe in mixes:
        return DegradationChain(recipe, mixes[recipe]())
    singles: dict[str, Effect] = {
        "clip": Clipping(),
        "sat": Saturation(),
        "bwl": BandwidthLimit(),
        "eq": EqDamage(),
        "noise": AdditiveNoise(),
        "reverb": SyntheticReverb(),
        "codec": Codec(),
    }
    if recipe in singles:
        return DegradationChain(recipe, [singles[recipe]])
    raise ValueError(
        f"unknown degradation recipe {recipe!r}. Realistic: podcast/room; "
        f"stress ladder: light/moderate/harsh; "
        f"singles: clip/sat/bwl/eq/noise/reverb/codec; or 'clean'."
    )


RECIPE_NAMES = [
    "clean", "podcast", "room", "light", "moderate", "harsh",
    "clip", "sat", "bwl", "eq", "noise", "reverb", "codec",
]

__all__ = [
    "Effect", "Clipping", "Saturation", "BandwidthLimit", "EqDamage", "AdditiveNoise",
    "SyntheticReverb", "Codec", "DegradationChain", "make_chain", "RECIPE_NAMES",
]
