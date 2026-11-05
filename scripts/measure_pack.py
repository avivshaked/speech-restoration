"""Offline measurement of the fine-tune A/B pack — the primary eval panel.

Runs on the ALREADY-RENDERED listening_pack (no re-training, no re-inference), so it's
cheap and reproducible. Computes the project's primary panel per clip for degraded /
RE-baseline / RE-finetuned, with the clean studio take as the reference:

  - WER          (Whisper-large-v3) — faithfulness / hallucination guard (lower better)
  - NISQA MOS    — non-intrusive predicted MOS proxy (higher better; known to overstate)
  - DNSMOS OVR   — P.835 overall predicted MOS proxy (higher better)
  - spk-sim      — speaker-embedding cosine vs clean (higher better; voice preserved?)

Thread-capped BEFORE torch import so the CPU-side Whisper orchestration can't peg the
cores (the fry we fixed earlier). The metrics are PROXIES — the ear is the verdict.

Run:  uv run python scripts/measure_pack.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "runs" / "re-finetune-spike" / "listening_pack"

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(REPO / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402

setup_logging()
log = logging.getLogger("measure")

CLIPS = [
    ("f1_script1-iphone_balcony1", "SEEN, hard"),
    ("f8_script1-iphone_balcony1", "HELD-OUT, hard"),
    ("f1_script1-ipad_office1", "SEEN, easy (canary)"),
]
VARIANTS = [("1_degraded", "degraded"), ("2_RE_baseline", "baseline"), ("3_RE_finetuned", "finetuned")]
SR = 44100


def _read(p: Path) -> np.ndarray:
    a, sr = sf.read(str(p))
    a = a.mean(1) if a.ndim > 1 else a
    assert sr == SR, f"{p} sr={sr}"
    return a.astype("float32")


def main() -> int:
    from voice_cleanup.eval.non_intrusive import dnsmos_p835
    from voice_cleanup.eval.nisqa import nisqa_multidim
    from voice_cleanup.eval.speaker_sim import speaker_similarity
    from voice_cleanup.eval.wer import transcribe, whisper_wer

    if not PACK.is_dir():
        log.error("No pack at %s", PACK)
        return 2

    results = {}
    for key, role in CLIPS:
        clean = _read(PACK / f"{key}__0_clean.wav")
        ref_text = transcribe(clean, sr=SR)  # cache the reference transcript once
        log.info("=== %s (%s) · ref words=%d ===", key, role, len(ref_text.split()))
        results[key] = {"role": role, "variants": {}}
        for fname, vlabel in VARIANTS:
            x = _read(PACK / f"{key}__{fname}.wav")
            wer = float(whisper_wer(clean, x, sr=SR, ref_text=ref_text)["wer"])
            nisqa = nisqa_multidim(x, sr=SR)
            dns = dnsmos_p835(x, sr=SR)
            spk = float(speaker_similarity(clean, x, sr=SR))
            row = dict(wer=wer, nisqa_mos=float(nisqa.get("mos", float("nan"))),
                       dnsmos_ovr=float(dns.get("ovr", dns.get("p808_mos", float("nan")))),
                       spk_sim=spk)
            results[key]["variants"][vlabel] = row
            log.info("  %-10s WER=%.3f  NISQA=%.2f  DNSMOS-OVR=%.2f  spk-sim=%.3f",
                     vlabel, wer, row["nisqa_mos"], row["dnsmos_ovr"], spk)

    out = REPO / "runs" / "re-finetune-spike" / "measurements.json"
    out.write_text(json.dumps(results, indent=2))

    # Compact comparison table: baseline -> finetuned, the question that matters.
    log.info("=== PANEL: baseline -> finetuned (degraded for context) ===")
    log.info("%-30s | %-21s | %-21s | %-21s | %-21s",
             "clip", "WER (deg/base/ft)", "NISQA", "DNSMOS-OVR", "spk-sim")
    for key, role in CLIPS:
        v = results[key]["variants"]
        def trip(m):
            return f"{v['degraded'][m]:.2f}/{v['baseline'][m]:.2f}/{v['finetuned'][m]:.2f}"
        log.info("%-30s | %-21s | %-21s | %-21s | %-21s",
                 key, trip("wer"), trip("nisqa_mos"), trip("dnsmos_ovr"), trip("spk_sim"))
    log.info("Saved %s. Metrics are PROXIES — the ear is the verdict.", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
