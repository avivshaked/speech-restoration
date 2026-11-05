"""Automated quality gate for the Tier-1 clean corpora — drop the bad clips before training.

VCTK (studio chamber) and Hi-Fi TTS (NVIDIA-curated >=13 kHz / >=32 dB SNR) are already
clean by construction, so this is a SAFETY NET, not a wholesale filter: it catches the
occasional outlier (a near-silent file, a noisy/dull take, a Miipher-style restoration
artifact) using the spectral diagnostics DNSMOS lacks (eval/spectral.py). No listening.

Per clip (a bounded window, for speed): HF-energy ratio (is the top octave actually there?),
crude SNR, crest factor. Writes a manifest of all scores + a drop-list of clips below
percentile/absolute thresholds. The drop-list is consumed by training via --drop-list.

  uv run python scripts/quality_gate_tier1.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Pin BLAS/torch intra-op threads to 1 BEFORE numpy/torch import. With a process pool, each
# worker spawning multithreaded STFTs oversubscribes the cores and thrashes (measured: 10
# workers each at ~325% CPU on 32 cores → ~12 clips/s, an 8 h run). One thread per worker +
# many workers is the right shape for this embarrassingly-parallel scan.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data import tier1  # noqa: E402
from voice_cleanup.eval.spectral import (  # noqa: E402
    crest_factor_db,
    estimate_snr_db,
    ltas,
)

setup_logging()
log = logging.getLogger("quality-gate")
SR = tier1.SR
OUT = REPO / "runs" / "quality-gate"
GATE_SECONDS = 10.0


def _init_worker() -> None:
    """Pin torch to a single thread in each pool worker (belt-and-suspenders with the env vars)."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _score_clip(path_str: str) -> dict | None:
    """Read a bounded window and compute spectral diagnostics. Top-level for pool pickling."""
    try:
        info = sf.info(path_str)
        frames = min(int(GATE_SECONDS * info.samplerate), info.frames)
        a, sr = sf.read(path_str, frames=frames, dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(1)
        if len(a) < int(0.3 * sr):
            return {"path": path_str, "ok": False, "reason": "too_short"}
        # Single STFT per clip (hf_energy_ratio would run two); derive both bands from it.
        # Matches hf_energy_ratio's defaults exactly (hf 8-16k vs ref 100-8k, at native SR).
        freqs, db = ltas(a, sr)
        lin = 10.0 ** (db / 20.0)

        def _band(lo: float, hi: float) -> float:
            sel = (freqs >= lo) & (freqs < hi)
            return float(lin[sel].mean()) if sel.any() else 0.0

        hf_ratio = float(20.0 * np.log10((_band(8000.0, 16000.0) + 1e-12)
                                         / (_band(100.0, 8000.0) + 1e-12)))
        return {
            "path": path_str, "ok": True,
            "hf_ratio_db": hf_ratio,
            "snr_db": estimate_snr_db(a, sr),
            "crest_db": crest_factor_db(a),
            "sr": sr,
        }
    except Exception as e:  # noqa: BLE001
        return {"path": path_str, "ok": False, "reason": str(e)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = tier1.enumerate_tier1()
    if not clips:
        log.error("No Tier-1 clips — run fetch_tier1_corpora.py first.")
        return 2
    paths = [str(c.path) for c in clips]
    corpus_of = {str(c.path): c.corpus for c in clips}
    log.info("Quality gate · scoring %d clips (%.0fs window each) with a process pool …",
             len(paths), GATE_SECONDS)

    workers = min(24, max(1, (os.cpu_count() or 8) - 4))
    log.info("Process pool · %d single-threaded workers (32-core box, headroom left)", workers)
    scored: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for r in ex.map(_score_clip, paths, chunksize=64):
            scored.append(r)
            done += 1
            if done % 5000 == 0:
                log.info("  … %d/%d scored (%.0f%%)", done, len(paths), 100 * done / len(paths))

    good = [s for s in scored if s.get("ok")]
    bad = [s for s in scored if not s.get("ok")]
    log.info("Scored %d · %d readable · %d unreadable/too-short", len(scored), len(good), len(bad))

    hf = np.array([s["hf_ratio_db"] for s in good])
    snr = np.array([s["snr_db"] for s in good if np.isfinite(s["snr_db"])])
    # Thresholds: relative (bottom percentile of HF) OR absolute (clearly bad SNR). Conservative
    # — we only want to drop genuine outliers in an already-clean corpus.
    hf_floor = float(np.percentile(hf, 3))
    snr_floor = 15.0
    def _is_bad(s: dict) -> bool:
        return s["hf_ratio_db"] < hf_floor or (
            np.isfinite(s["snr_db"]) and s["snr_db"] < snr_floor)
    drop = [s["path"] for s in good if _is_bad(s)]
    drop += [s["path"] for s in bad]

    for label, arr in [("HF-ratio dB", hf), ("SNR dB", snr)]:
        log.info("  %-12s p3=%.1f p10=%.1f p50=%.1f p90=%.1f",
                 label, np.percentile(arr, 3), np.percentile(arr, 10),
                 np.percentile(arr, 50), np.percentile(arr, 90))
    by_corpus: dict[str, int] = {}
    for p in drop:
        by_corpus[corpus_of.get(p, "?")] = by_corpus.get(corpus_of.get(p, "?"), 0) + 1
    log.info("Drop-list: %d clips (%.1f%%) · HF floor=%.1f dB, SNR floor=%.1f dB · by corpus: %s",
             len(drop), 100 * len(drop) / max(1, len(clips)), hf_floor, snr_floor, by_corpus)

    (OUT / "scores.json").write_text(json.dumps(scored))
    (OUT / "drop_list.json").write_text(json.dumps(sorted(set(drop)), indent=2))
    log.info("Manifest → %s · drop-list → %s", OUT / "scores.json", OUT / "drop_list.json")
    log.info("NOTE: corpora are pre-curated, so a small drop-list is expected and healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
