"""Tier-1 commercial-safe clean speech → on-the-fly synthetic (degraded → clean) pairs.

The Restorer rebuild (decision log 2026-05-30 late-evening) trains on commercially-licensed,
full-band, studio-grade clean speech, synthetically degraded — NOT DAPS (CC BY-NC, and only
one degradation family). The clean recording IS the target `y`; we manufacture the input
`x = degrade(y)` per access (dynamic mixing, so the model never sees the same degradation
twice — URGENT-2024 protocol, Lit Review §10).

Tier-1 corpora (both verified commercial-safe, both >=44.1 kHz so they CONTAIN the top octave
the restorer must learn to regenerate):
- **VCTK 0.92** — 48 kHz, CC BY 4.0. `wav48_silence_trimmed/pXXX/pXXX_NNN_mic1.flac`.
- **Hi-Fi TTS** — 44.1 kHz, CC BY 4.0. `hi_fi_tts_v0/audio/<book>/<spk>/*.flac`.

Speaker-disjoint split (same discipline as DAPS): whole speakers held out for eval.

The degradation POLICY (`build_training_chain`) leads with EQ-damage / de-coloration
(resonant "radio voice" mids, which the DAPS fine-tune was never trained on) with
bandwidth-limit as the companion
(top-octave extension), plus the rest of the URGENT set at lower frequency so the restorer
stays GENERAL rather than narrowing onto one family again.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import degrade
from .loudness import normalize_lufs

log = logging.getLogger(__name__)

SR = 44100  # we train at 44.1 kHz; VCTK 48 kHz is resampled down on load

REPO = Path(__file__).resolve().parents[3]
VCTK_DIR = REPO / "data" / "corpora" / "vctk"
HIFI_DIR = REPO / "data" / "corpora" / "hifi_tts"


@dataclass(frozen=True)
class CleanClip:
    """One clean studio recording (the restoration TARGET)."""
    speaker: str   # corpus-prefixed, e.g. "vctk:p225" / "hifi:92"
    path: Path
    corpus: str    # "vctk" | "hifi"


def enumerate_vctk(root: Path = VCTK_DIR) -> list[CleanClip]:
    """All VCTK mic1 flacs (fall back to mic2/any flac), speaker = parent dir name."""
    base = root / "wav48_silence_trimmed"
    if not base.is_dir():
        base = root  # tolerate a flatter layout
    clips: list[CleanClip] = []
    for spk_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        spk = spk_dir.name
        flacs = sorted(spk_dir.glob("*mic1.flac")) or sorted(spk_dir.glob("*mic2.flac")) \
            or sorted(spk_dir.glob("*.flac"))
        for f in flacs:
            clips.append(CleanClip(speaker=f"vctk:{spk}", path=f, corpus="vctk"))
    log.info("VCTK · %d clips from %d speakers", len(clips),
             len({c.speaker for c in clips}))
    return clips


def enumerate_hifi(root: Path = HIFI_DIR) -> list[CleanClip]:
    """All Hi-Fi TTS flacs; speaker id derived from the path under audio/.

    Hi-Fi layout is audio/<book_id>/<speaker_id>/<utt>.flac (book and speaker dirs); the
    speaker id is the numeric reader id. We glob all flacs and take the *numeric* path
    component as the speaker, which is robust to whichever of book/speaker nests first.
    """
    base = root / "hi_fi_tts_v0"
    if not base.is_dir():
        base = root
    audio = base / "audio"
    search = audio if audio.is_dir() else base
    clips: list[CleanClip] = []
    for f in sorted(search.rglob("*.flac")):
        rel_parts = f.relative_to(search).parts
        # speaker = the numeric reader id among the path components (Hi-Fi reader ids are ints)
        numeric = [p for p in rel_parts[:-1] if p.isdigit()]
        spk = numeric[0] if numeric else (rel_parts[0] if len(rel_parts) > 1 else "unknown")
        clips.append(CleanClip(speaker=f"hifi:{spk}", path=f, corpus="hifi"))
    log.info("Hi-Fi TTS · %d clips from %d speakers", len(clips),
             len({c.speaker for c in clips}))
    return clips


def enumerate_tier1() -> list[CleanClip]:
    return enumerate_vctk() + enumerate_hifi()


def split_by_speaker(
    clips: list[CleanClip], *, n_held_out_per_corpus: int = 3, seed: int = 42,
) -> tuple[list[CleanClip], list[CleanClip], list[str]]:
    """(train, val, held_out_speakers) — whole speakers held out, balanced per corpus.

    Deterministic: held-out speakers are the seed-shuffled first `n` of each corpus, so a
    re-run picks the same eval voices.
    """
    rng = np.random.default_rng(seed)
    held: set[str] = set()
    for corpus in ("vctk", "hifi"):
        spk = sorted({c.speaker for c in clips if c.corpus == corpus})
        rng.shuffle(spk)
        held.update(spk[:n_held_out_per_corpus])
    train = [c for c in clips if c.speaker not in held]
    val = [c for c in clips if c.speaker in held]
    log.info("Tier-1 split · train=%d clips (%d spk) · val=%d clips (%d held-out: %s)",
             len(train), len({c.speaker for c in train}),
             len(val), len(held), ",".join(sorted(held)))
    return train, val, sorted(held)


# --- degradation policy ----------------------------------------------------
# Per-item dynamic mixing. EQ-damage leads (the de-coloration headline target); bandwidth
# limit is the companion (top-octave extension). The rest of the URGENT set appears at lower
# probability so the restorer generalizes instead of narrowing onto one family. SEVERITY
# RANGES are calibrated/validated separately (scripts/calibrate_degradations.py); the values
# here mirror degrade.py's URGENT-grounded defaults and are overridden once calibration lands.


def build_training_chain(rng: np.random.Generator) -> degrade.DegradationChain:
    """Compose a random per-item degradation chain (signal-chain order)."""
    effects: list[degrade.Effect] = []
    # Room/mic coloration first (like a real chain), THEN level damage, THEN ambient noise,
    # THEN delivery codec.
    if rng.random() < 0.45:
        effects.append(degrade.SyntheticReverb(rt60_range=(0.2, 0.7)))
    # EQ-damage — the LEAD. High probability, full mid-range coverage incl. resonant peaks.
    if rng.random() < 0.85:
        effects.append(degrade.EqDamage(
            n_bands_range=(2, 4), freq_range=(150.0, 8000.0),
            gain_db_range=(-14.0, 14.0), q_range=(1.0, 6.0)))
    # Bandwidth limit — the companion (top-octave extension). VARY slope + cutoff to SPAN the
    # real range: gentle cheap-mic rolloff (order 1, high cutoff) → moderate "radio voice"
    # → steep telephone (order 4, low cutoff). NB sosfiltfilt is zero-phase
    # → DOUBLES the effective order, so order 1-4 here behaves like 2-8 in magnitude (the 05-30
    # calibration finding: order-8 was a -95 dB brick wall, far harsher than real bandwidth loss).
    if rng.random() < 0.60:
        bw_order = int(rng.integers(1, 5))
        effects.append(degrade.BandwidthLimit(cutoff_range=(3000.0, 12000.0), order=bw_order))
    if rng.random() < 0.25:
        effects.append(degrade.Saturation(drive_range=(2.0, 6.0)))
    if rng.random() < 0.15:
        effects.append(degrade.Clipping(frac_range=(0.2, 0.5)))
    if rng.random() < 0.45:
        effects.append(degrade.AdditiveNoise(snr_db_range=(12.0, 35.0), color="pink"))
    if rng.random() < 0.12:
        effects.append(degrade.Codec(codecs=("mp3", "aac", "opus"),
                                     bitrate_kbps_range=(32, 96)))
    if not effects:  # guarantee at least one degradation
        effects.append(degrade.EqDamage())
    return degrade.DegradationChain("tier1_dynamic", effects)


def _read_clean(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import torchaudio
        a = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return a.astype("float32")


def _read_clean_window(path: Path, start: int, frames: int) -> np.ndarray:
    info = sf.info(str(path))
    if info.samplerate == SR:
        a, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(1)
        return a.astype("float32")
    # Mismatched rate (VCTK 48k): read whole, resample, then window in resampled domain.
    full = _read_clean(path)
    return full[start:start + frames].astype("float32")


def build_eval_clips(
    val_clips: list[CleanClip], *, n_speakers: int = 4, target_seconds: float = 30.0,
    gap_s: float = 0.3,
) -> list[tuple[str, np.ndarray]]:
    """Held-out clean eval clips, ~``target_seconds`` each, built by concatenating WHOLE
    utterances per speaker (never cutting mid-word — the eval-WER discipline that bit us on
    DAPS). Returns [(speaker, clean_audio_44k), ...], one clip per speaker (up to n_speakers).

    VCTK utterances are single sentences; Hi-Fi are longer reads — either way we only ever
    join complete files, with a short silence between, so both clip edges sit in deep silence
    and Whisper has no fragment to resolve differently on clean vs restored.
    """
    by_spk: dict[str, list[CleanClip]] = {}
    for c in val_clips:
        by_spk.setdefault(c.speaker, []).append(c)
    gap = np.zeros(int(gap_s * SR), dtype="float32")
    target_n = int(target_seconds * SR)
    out: list[tuple[str, np.ndarray]] = []
    for spk in sorted(by_spk)[:n_speakers]:
        parts: list[np.ndarray] = []
        total = 0
        for c in sorted(by_spk[spk], key=lambda c: c.path.name):
            try:
                a = _read_clean(c.path)
            except Exception:  # noqa: BLE001
                continue
            if len(a) < int(0.3 * SR):  # skip near-empty files
                continue
            parts.append(a)
            parts.append(gap)
            total += len(a) + len(gap)
            if total >= target_n:
                break
        if parts:
            out.append((spk, np.concatenate(parts).astype("float32")))
    log.info("Tier-1 eval clips · %d speakers · ~%.0fs each", len(out), target_seconds)
    return out


class SyntheticPairDataset(torch.utils.data.Dataset):
    """On-the-fly (degraded, clean) pairs from Tier-1 clean speech, 44.1 kHz, LUFS-normalized.

    Each clip is tiled into ``floor(len/crop)`` windows (>=1). Per access: load the clean
    window, build a random degradation chain (dynamic mixing), apply it, LUFS-normalize BOTH
    streams (kills inter-clip level, matches the Restorer's DAPSPairDataset contract), and
    return ``(degraded, clean)`` tensors.

    Degradation RNG is seeded per access from the worker's torch RNG (seeded deterministically
    via the DataLoader generator + worker_init_fn), so degradations vary across epochs yet the
    whole run stays reproducible for a fixed seed.
    """

    def __init__(self, clips: list[CleanClip], *, crop_seconds: float = 3.0,
                 min_seconds: float = 1.0):
        if not clips:
            raise ValueError("SyntheticPairDataset got 0 clips")
        self.clips = clips
        self.crop = int(round(crop_seconds * SR))
        self.min_frames = int(round(min_seconds * SR))
        self.lengths: list[int] = []
        self.index: list[tuple[int, int]] = []  # (clip_idx, window_idx)
        skipped = 0
        for ci, c in enumerate(clips):
            try:
                info = sf.info(str(c.path))
                n = info.frames if info.samplerate == SR else \
                    int(info.frames * SR / info.samplerate)
            except Exception:  # noqa: BLE001  unreadable file
                n = 0
            self.lengths.append(n)
            if n < self.min_frames:
                skipped += 1
                continue
            n_win = max(1, n // self.crop)
            self.index.extend((ci, w) for w in range(n_win))
        log.info("SyntheticPairDataset · %d clips → %d windows of %.1f s (%d clips too short)",
                 len(clips), len(self.index), self.crop / SR, skipped)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_idx, w = self.index[i]
        c = self.clips[clip_idx]
        n = self.lengths[clip_idx]
        if n <= self.crop:
            start, frames = 0, n
        else:
            n_win = max(1, n // self.crop)
            leftover = n - n_win * self.crop
            offset = int(torch.randint(0, leftover + 1, (1,)).item()) if leftover > 0 else 0
            start, frames = offset + w * self.crop, self.crop
        clean = _read_clean_window(c.path, start, frames)
        if len(clean) < self.crop:  # pad tail with silence
            clean = np.pad(clean, (0, self.crop - len(clean)))
        # Dynamic mixing: per-item degradation seeded from the worker's (deterministic) RNG.
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        chain = build_training_chain(rng)
        try:
            deg = chain.apply(clean, SR, seed=seed)
        except Exception as e:  # noqa: BLE001  e.g. a transient codec failure → fall back clean-ish
            log.warning("degradation failed (%s) — using EQ-only fallback", e)
            deg = degrade.EqDamage().apply(clean, SR, rng)
        deg = normalize_lufs(deg.astype("float32"), SR)
        clean_n = normalize_lufs(clean.astype("float32"), SR)
        m = min(len(deg), len(clean_n))
        return torch.from_numpy(deg[:m]), torch.from_numpy(clean_n[:m])


__all__ = [
    "SR", "CleanClip", "enumerate_vctk", "enumerate_hifi", "enumerate_tier1",
    "split_by_speaker", "build_training_chain", "SyntheticPairDataset",
]
