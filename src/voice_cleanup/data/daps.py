"""DAPS real paired (degraded → clean) data for fine-tuning.

DAPS ships, for each of 20 speakers × 5 scripts, a clean studio take plus the *same*
performance re-recorded through real devices in real rooms (~12 conditions). The pairs are
ALREADY time-aligned (same length, envelope corr ~0.85 at zero lag — verified 2026-05-29),
so we do **no** cross-correlation alignment here: a previous `_align` step latched onto
spurious ~2 s lags and collapsed a fine-tune. Crop both streams from the same offset and
they stay frame-aligned.

Split strategy (decided 2026-05-29): **hold out whole speakers**. Train on 16 speakers
across ALL conditions (incl. the hard balcony), validate on 4 unseen speakers — the honest
"does it generalize to a new voice?" gauge. Conditions are shared between train and val;
only the *speaker* is held out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .loudness import normalize_lufs

log = logging.getLogger(__name__)

SR = 44100

# The 12 real device/room conditions — these are the degraded INPUTS, each paired to the
# clean studio target. (clean/cleanraw/produced/sample/supplementary are not inputs.)
DEGRADED_CONDITIONS: tuple[str, ...] = (
    "ipad_balcony1", "ipad_bedroom1", "ipad_confroom1", "ipad_confroom2",
    "ipad_livingroom1", "ipad_office1", "ipad_office2",
    "ipadflat_confroom1", "ipadflat_office1",
    "iphone_balcony1", "iphone_bedroom1", "iphone_livingroom1",
)
# Studio target versions, in faithfulness order. cleanraw = raw studio incl. breaths/mouth
# sounds (the most FAITHFUL restoration target — verified 2026-05-30 that the device inputs
# contain the breaths, +13.9 dB over noise, so they're restorable, not hallucinated). clean
# = cleanraw with breaths editorially removed. produced = clean + EQ/compression. For a
# restorer we default to cleanraw; clean/produced bake in editorial/creative choices.
TARGET_VERSIONS: tuple[str, ...] = ("cleanraw", "clean", "produced")
DEFAULT_TARGET = "cleanraw"
SCRIPTS: tuple[str, ...] = ("script1", "script2", "script3", "script4", "script5")
ALL_SPEAKERS: tuple[str, ...] = (
    tuple(f"f{i}" for i in range(1, 11)) + tuple(f"m{i}" for i in range(1, 11))
)

# Validation speakers — never seen in training (2F + 2M). f8 is kept here because we
# already have an ear-reference for it from the Phase-2 spike.
HELD_OUT_SPEAKERS: tuple[str, ...] = ("f8", "f10", "m8", "m10")

# Bounded eval set: easy → mid → hard condition, to keep held-out scoring cheap.
EVAL_CONDITIONS: tuple[str, ...] = ("ipad_office1", "iphone_livingroom1", "iphone_balcony1")


@dataclass(frozen=True)
class Pair:
    """One (degraded → studio target) DAPS pair. Same performance, same length, aligned.

    `target` is the chosen studio version (cleanraw / clean / produced).
    """
    speaker: str
    script: str
    condition: str
    degraded: Path
    target: Path

    @property
    def identifier(self) -> str:
        return f"{self.speaker}_{self.script}_{self.condition}"


def _daps_root(daps_dir: Path) -> Path:
    # Repo layout nests the payload one level down: data/corpora/daps/daps/<condition>/...
    inner = daps_dir / "daps"
    return inner if inner.is_dir() else daps_dir


def enumerate_pairs(
    daps_dir: Path,
    *,
    speakers: tuple[str, ...] | list[str] | None = None,
    conditions: tuple[str, ...] = DEGRADED_CONDITIONS,
    target: str = DEFAULT_TARGET,
) -> list[Pair]:
    """All (degraded → `target`) pairs for the given speakers × conditions that exist on disk."""
    if target not in TARGET_VERSIONS:
        raise ValueError(f"target must be one of {TARGET_VERSIONS}, got {target!r}")
    root = _daps_root(daps_dir)
    speakers = tuple(speakers) if speakers is not None else ALL_SPEAKERS
    pairs: list[Pair] = []
    missing = 0
    for spk in speakers:
        for script in SCRIPTS:
            tgt = root / target / f"{spk}_{script}_{target}.wav"
            if not tgt.is_file():
                continue
            for cond in conditions:
                deg = root / cond / f"{spk}_{script}_{cond}.wav"
                if deg.is_file():
                    pairs.append(Pair(spk, script, cond, deg, tgt))
                else:
                    missing += 1
    if missing:
        log.warning("DAPS: %d expected degraded files missing under %s", missing, root)
    return pairs


def split_by_speaker(
    daps_dir: Path,
    *,
    held_out: tuple[str, ...] = HELD_OUT_SPEAKERS,
    conditions: tuple[str, ...] = DEGRADED_CONDITIONS,
    target: str = DEFAULT_TARGET,
) -> tuple[list[Pair], list[Pair]]:
    """Return (train_pairs, val_pairs) split by speaker — disjoint speaker sets."""
    train_spk = tuple(s for s in ALL_SPEAKERS if s not in held_out)
    train = enumerate_pairs(daps_dir, speakers=train_spk, conditions=conditions, target=target)
    val = enumerate_pairs(daps_dir, speakers=held_out, conditions=conditions, target=target)
    log.info(
        "DAPS split · target=%s · train=%d pairs (%d speakers) · val=%d pairs (%d held-out: %s)",
        target, len(train), len(train_spk), len(val), len(held_out), ",".join(held_out),
    )
    return train, val


def enumerate_studio_pairs(
    daps_dir: Path,
    *,
    speakers: tuple[str, ...] | list[str] | None = None,
    input_version: str = "cleanraw",
    target_version: str = "produced",
) -> list[Pair]:
    """All (`input_version` → `target_version`) STUDIO pairs — both are studio takes of the
    same performance, sample-aligned (verified 0-sample length diff, envelope corr 0.97).

    This is the Producer's data: cleanraw → produced (EQ + compression + loudness + the
    de-breathing inherited from `clean`). Reuses `Pair`, with `degraded` holding the input
    studio version (the field name is "the input" here, not a device recording).
    """
    for v in (input_version, target_version):
        if v not in TARGET_VERSIONS:
            raise ValueError(f"version must be one of {TARGET_VERSIONS}, got {v!r}")
    root = _daps_root(daps_dir)
    speakers = tuple(speakers) if speakers is not None else ALL_SPEAKERS
    pairs: list[Pair] = []
    for spk in speakers:
        for script in SCRIPTS:
            src = root / input_version / f"{spk}_{script}_{input_version}.wav"
            tgt = root / target_version / f"{spk}_{script}_{target_version}.wav"
            if src.is_file() and tgt.is_file():
                pairs.append(Pair(spk, script, input_version, src, tgt))
    return pairs


def split_studio_by_speaker(
    daps_dir: Path,
    *,
    held_out: tuple[str, ...] = HELD_OUT_SPEAKERS,
    input_version: str = "cleanraw",
    target_version: str = "produced",
) -> tuple[list[Pair], list[Pair]]:
    """(train, val) studio pairs split by speaker — same held-out speakers as the Restorer."""
    train_spk = tuple(s for s in ALL_SPEAKERS if s not in held_out)
    kw = dict(input_version=input_version, target_version=target_version)
    train = enumerate_studio_pairs(daps_dir, speakers=train_spk, **kw)
    val = enumerate_studio_pairs(daps_dir, speakers=held_out, **kw)
    log.info(
        "DAPS studio split · %s→%s · train=%d pairs (%d spk) · val=%d pairs (%d held-out: %s)",
        input_version, target_version, len(train), len(train_spk), len(val),
        len(held_out), ",".join(held_out),
    )
    return train, val


EVAL_SCRIPT = "script1"  # one fixed script per (speaker, condition) keeps the ear-set small


def select_eval_pairs(
    daps_dir: Path,
    *,
    held_out: tuple[str, ...] = HELD_OUT_SPEAKERS,
    conditions: tuple[str, ...] = EVAL_CONDITIONS,
    target: str = DEFAULT_TARGET,
) -> list[Pair]:
    """Curated, representative held-out eval set: one clip per (speaker × condition).

    With 4 held-out speakers × 3 conditions (easy/mid/hard) that's 12 clips — a set the
    METRICS score in full automatically, and small enough that the EAR can spot-check the
    rows that matter (the hard balcony condition, plus whatever the metrics flag). We fix a
    single script so the ear-set doesn't balloon. Ordered speaker-major.
    """
    pairs = enumerate_pairs(daps_dir, speakers=held_out, conditions=conditions, target=target)
    return [p for p in pairs if p.script == EVAL_SCRIPT]


def _frames(path: Path) -> tuple[int, int]:
    info = sf.info(str(path))
    return info.frames, info.samplerate


def _read_crop(path: Path, start: int, frames: int) -> np.ndarray:
    a, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:  # DAPS is 44.1 kHz; guard anyway
        import torchaudio
        a = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return a.astype("float32")


def _read_mono(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import torchaudio
        a = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return a.astype("float32")


def load_aligned_speech_window(
    target_path: Path, degraded_path: Path, seconds: float,
    *, min_pause_s: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a ~``seconds`` span bounded by SENTENCE pauses from an aligned (target, degraded)
    pair, taking the SAME offset from both so they stay frame-aligned.

    Why this shape:

    1. DAPS recordings open with several seconds of lead-in silence, so a fixed first-N-seconds
       crop yields ~half silence — useless for an ear eval and noisy for WER.
    2. An arbitrary cut bisects a word at each edge; Whisper resolves the fragment differently
       on the clean (reference) vs enhanced (hypothesis) audio, injecting a spurious WER error
       per cut edge — and heterogeneously, which corrupts per-clip comparisons. Cutting only at
       SENTENCE boundaries (long, unambiguous pauses) removes that entirely: whole sentences in,
       deep silence at both edges.

    Method: smooth the target's 10ms RMS envelope (so plosive closures don't read as pauses),
    mark sustained sub-threshold runs (≥ ``min_pause_s``) as sentence gaps, and pick the run of
    whole sentences whose total length is closest to ``seconds`` (tie-break: most speech-dense).
    If a clip has no qualifying pause, relax ``min_pause_s`` toward clause/word gaps; only if
    nothing is found at all do we fall back to the densest fixed-length window.
    """
    tgt = _read_mono(target_path)
    deg = _read_mono(degraded_path)
    n = min(len(tgt), len(deg))
    tgt, deg = tgt[:n], deg[:n]
    win = int(SR * seconds)
    if n <= win:
        return tgt, deg

    # Smoothed 10ms RMS envelope of the (clean) target — 50ms smoothing erases sub-50ms plosive
    # closures so only real pauses survive as sustained sub-threshold runs.
    hop = max(1, int(SR * 0.01))
    nf = n // hop
    fe = np.sqrt((tgt[: nf * hop].reshape(nf, hop) ** 2).mean(axis=1) + 1e-12)
    fe = np.convolve(fe, np.ones(5) / 5, mode="same")
    thr = max(0.02, 0.12 * float(np.percentile(fe, 95)))
    energy = np.concatenate([[0.0], np.cumsum(tgt.astype(np.float64) ** 2)])

    def _gap_centers(min_run_frames: int) -> list[int]:
        """Sample-index centers of every sustained sub-threshold (pause) run."""
        centers, i = [], 0
        silent = fe < thr
        while i < nf:
            if silent[i]:
                j = i
                while j < nf and silent[j]:
                    j += 1
                if j - i >= min_run_frames:
                    centers.append(((i + j) // 2) * hop)
                i = j
            else:
                i += 1
        return centers

    # Prefer sentence-length pauses; relax toward clause/word gaps only if the clip has none.
    cuts: list[int] = []
    for mp in (min_pause_s, 0.22, 0.15):
        cuts = _gap_centers(max(1, int(mp / 0.01)))
        if cuts:
            break

    pts = sorted({0, *cuts, n})
    # Pick the whole-sentence span whose length is closest to `win`; tie-break by speech density.
    best = None
    for ai in range(len(pts)):
        for bi in range(ai + 1, len(pts)):
            length = pts[bi] - pts[ai]
            if length < win * 0.5:
                continue
            if length > win * 2.0:
                break
            dev = round(abs(length - win) / (SR * 0.5))  # 0.5s buckets so density can tie-break
            density = (energy[pts[bi]] - energy[pts[ai]]) / length
            key = (dev, -density)
            if best is None or key < best[0]:
                best = (key, pts[ai], pts[bi])

    if best is None:  # no usable pause structure — densest fixed-length window
        stride = max(1, int(SR * 0.25))
        off = max(range(0, n - win + 1, stride), key=lambda o: energy[o + win] - energy[o])
        return tgt[off : off + win].astype("float32"), deg[off : off + win].astype("float32")

    _, s, e = best
    return tgt[s:e].astype("float32"), deg[s:e].astype("float32")


class DAPSPairDataset(torch.utils.data.Dataset):
    """Windowed (degraded, target) pairs at 44.1 kHz, LUFS-normalized, frame-aligned.

    Each ~2–3 min clip is tiled into ``floor(len / crop)`` non-overlapping windows, so
    ``len(dataset)`` reflects the real data volume (~16 k windows at 10 s, ~55 k at 3 s for
    the 16-speaker train split) rather than the 960 full-clip count. Per access, the whole
    tiling grid is shifted by a random offset within the clip's leftover samples, so windows
    still vary across epochs while every part of the clip is reachable.

    Reads only the cropped window (soundfile frame-range seek), never the whole file. The
    pair is NOT time-aligned here — DAPS ships it aligned; both streams crop from the same
    offset and stay frame-aligned.
    """

    def __init__(self, pairs: list[Pair], *, crop_seconds: float = 10.0,
                 normalize: str = "each"):
        if not pairs:
            raise ValueError("DAPSPairDataset got 0 pairs")
        if normalize not in ("each", "none"):
            raise ValueError(f"normalize must be 'each' or 'none', got {normalize!r}")
        self.pairs = pairs
        self.normalize = normalize  # "each": LUFS-norm both (restorer); "none": raw (producer)
        self.crop = int(round(crop_seconds * SR))
        # Metadata-only pass: usable length (min of the two streams) + window count per pair.
        self.lengths: list[int] = []
        self.index: list[tuple[int, int]] = []  # (pair_idx, window_idx)
        for pi, p in enumerate(pairs):
            n = min(_frames(p.degraded)[0], _frames(p.target)[0])
            self.lengths.append(n)
            n_win = max(1, n // self.crop)
            self.index.extend((pi, w) for w in range(n_win))
        log.info(
            "DAPSPairDataset · %d clip-pairs → %d windows of %.1f s",
            len(pairs), len(self.index), self.crop / SR,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx, w = self.index[i]
        p = self.pairs[pair_idx]
        n = self.lengths[pair_idx]
        if n <= self.crop:
            start, frames = 0, n
        else:
            n_win = max(1, n // self.crop)
            leftover = n - n_win * self.crop  # shift the whole grid for per-epoch variety
            offset = int(torch.randint(0, leftover + 1, (1,)).item()) if leftover > 0 else 0
            start, frames = offset + w * self.crop, self.crop
        deg = _read_crop(p.degraded, start, frames)
        tgt = _read_crop(p.target, start, frames)
        m = min(len(deg), len(tgt))
        deg, tgt = deg[:m], tgt[:m]
        if m < self.crop:  # short clip → pad tail with silence (both, identically)
            pad = self.crop - m
            deg = np.pad(deg, (0, pad))
            tgt = np.pad(tgt, (0, pad))
        if self.normalize == "each":  # restorer: kill inter-clip level, both to TARGET_LUFS
            deg = normalize_lufs(deg, SR)
            tgt = normalize_lufs(tgt, SR)
        # "none": producer — DAPS is already peak-normalized; keep the native cleanraw→produced
        # loudness/compression relationship intact (normalizing would distort the very thing
        # we're learning).
        return torch.from_numpy(deg), torch.from_numpy(tgt)
