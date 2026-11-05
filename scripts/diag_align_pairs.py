"""Diagnostic: did `_align` actually line up the real DAPS training pairs?

The Phase-2 fine-tune collapsed (voice removed, noise kept) while the CFM's latent L1
loss went DOWN. One prime suspect: `_align` cross-correlates a reverberant/noisy device
take against a clean studio take — spectrally very different signals — so the estimated
lag could be wrong. If it is, the CFM was trained to map input@t -> clean-latent@t', i.e.
a temporally mismatched target, which would memorize as garbage.

This reuses the REAL `_load` / `_align` from the spike (no reimplementation, so we test
the actual code path) and:
  1. logs the estimated lag (in ms) for every training pair,
  2. writes aligned `degraded` next to `clean` so you can EAR-check they line up.

No model, no DeepSpeed, no Whisper — just I/O + numpy. CPU-light, seconds.

Run:  uv run python scripts/diag_align_pairs.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from voice_cleanup._logging import setup_logging  # noqa: E402

# Import the ACTUAL functions under test from the spike (don't reimplement).
from spike_re_finetune_overfit import DAPS, SR, _align, _load  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402

setup_logging()
log = logging.getLogger("diag-align")

OUT = REPO / "runs" / "re-finetune-spike-diag" / "aligned_pairs"

CONDITIONS = ["ipad_office1", "iphone_balcony1"]
TRAIN_SPK = ["f1_script1", "m3_script1"]


def _lag_ms(deg: np.ndarray, clean: np.ndarray) -> float:
    """Recompute the lag `_align` would use, in milliseconds (for logging only)."""
    d = 50
    a = deg[::d] - deg[::d].mean()
    b = clean[::d] - clean[::d].mean()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = np.correlate(a, b, mode="full")
    lag = (np.argmax(corr) - (n - 1)) * d
    peak = float(corr.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return lag / SR * 1000.0, peak


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seconds = 6.0  # a bit longer than the 3-s training crop, easier to ear-check
    log.info("Dumping aligned pairs to %s (%.0f-s crops)", OUT, seconds)
    for spk in TRAIN_SPK:
        for cond in CONDITIONS:
            cf = DAPS / "clean" / f"{spk}_clean.wav"
            df = DAPS / cond / f"{spk}_{cond}.wav"
            if not (cf.is_file() and df.is_file()):
                log.warning("MISSING %s | %s — skipping", spk, cond)
                continue
            clean = _load(cf, seconds)
            deg_raw = _load(df, seconds)
            lag_ms, peak = _lag_ms(deg_raw, clean)
            deg = _align(deg_raw, clean)
            tag = f"{spk}-{cond}"
            sf.write(str(OUT / f"{tag}__clean.wav"), normalize_lufs(clean, SR), SR)
            sf.write(str(OUT / f"{tag}__degraded_aligned.wav"), normalize_lufs(deg, SR), SR)
            flag = "  <-- suspicious (|lag|>250ms or weak peak)" if (
                abs(lag_ms) > 250 or peak < 0.05) else ""
            log.info("%-28s lag=%+7.1f ms · xcorr_peak=%.3f%s", tag, lag_ms, peak, flag)
    log.info("Done. EAR-check each pair: do __clean and __degraded_aligned line up in time?")
    log.info("If they DON'T, the training target was wrong and the collapse is an align bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
