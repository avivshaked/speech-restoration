"""Proof that `producer-natural` (clean→produced) preserves breaths while `producer-cleanup`
(cleanraw→produced) suppresses them — both fed the SAME cleanraw input (the real deployment input
from the Restorer). Measures breath/gap-region energy and renders an A/B pack for the ear.

Breath regions = frames the engineer pushed to near-silence in `produced` (de-breathed) but which
still carried real energy in `cleanraw` (i.e. an actual breath, not a true pause). The verdict:
  cleanup_out  ≈ produced   → breaths SUPPRESSED
  natural_out  ≈ cleanraw   → breaths PRESERVED (+ the mastering loudness gain)

  uv run python scripts/validate_breath_preservation.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAPS = REPO / "data" / "corpora" / "daps" / "daps"
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.daps import (  # noqa: E402
    HELD_OUT_SPEAKERS,
    SR,
    enumerate_studio_pairs,
    load_aligned_speech_window,
)
from voice_cleanup.models.producer import SpectralProducer  # noqa: E402

setup_logging()
log = logging.getLogger("breath-val")

CLEANUP = REPO / "runs" / "producer" / "20260530-122752" / "producer_best.pt"
NATURAL = REPO / "runs" / "producer" / "natural-20260530-185900" / "producer_best.pt"
OUT = REPO / "runs" / "producer" / "natural-20260530-185900" / "breath_validation"


def _load(ckpt: Path, device) -> SpectralProducer:
    m = SpectralProducer().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model"])
    m.eval()
    log.info("loaded %s (step %d)", ckpt.name, state["step"])
    return m


def _frames_db(x: np.ndarray, win: float = 0.020) -> np.ndarray:
    n = int(SR * win)
    m = len(x) // n
    x = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((x**2).mean(1) + 1e-12) + 1e-12)


def main() -> int:
    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)

    cleanup = _load(CLEANUP, device)
    natural = _load(NATURAL, device)

    # Held-out script1, cleanraw input (deployment reality) vs produced target (de-breathed ref).
    pairs = [p for p in enumerate_studio_pairs(
        DAPS, speakers=HELD_OUT_SPEAKERS,
        input_version="cleanraw", target_version="produced") if p.script == "script1"]

    log.info("Breath/gap-region RMS (dB) — regions the engineer silenced but cleanraw kept:")
    log.info("  %-12s %10s %10s %12s %12s   %s", "clip", "cleanraw", "produced",
             "cleanup_out", "natural_out", "natural verdict")
    rows = []
    for p in pairs:
        produced, cleanraw = load_aligned_speech_window(p.target, p.degraded, 30.0)
        with torch.no_grad():
            c_out = cleanup.enhance(torch.from_numpy(cleanraw).to(device)).cpu().numpy()
            n_out = natural.enhance(torch.from_numpy(cleanraw).to(device)).cpu().numpy()
        m = min(len(produced), len(cleanraw), len(c_out), len(n_out))
        produced, cleanraw, c_out, n_out = (a[:m] for a in (produced, cleanraw, c_out, n_out))

        dc, dp = _frames_db(cleanraw), _frames_db(produced)
        dco, dno = _frames_db(c_out), _frames_db(n_out)
        k = min(map(len, (dc, dp, dco, dno)))
        dc, dp, dco, dno = dc[:k], dp[:k], dco[:k], dno[:k]
        reg = (dp < np.percentile(dp, 25)) & (dc > np.median(dc) - 25)
        if reg.sum() < 5:
            log.info("  %-12s (too few breath frames)", p.speaker)
            continue
        mc, mp, mco, mno = dc[reg].mean(), dp[reg].mean(), dco[reg].mean(), dno[reg].mean()
        # natural preserves if its breath energy tracks cleanraw, not the silenced produced.
        verdict = ("PRESERVED" if (mno - mp) > 6 and (mc - mno) < 8
                   else "suppressed" if (mno - mp) < 3 else "partial")
        log.info("  %-12s %10.1f %10.1f %12.1f %12.1f   %s",
                 p.speaker, mc, mp, mco, mno, verdict)
        rows.append((p.speaker, mc, mp, mco, mno))

        # render for the ear: cleanraw in, produced target, both model outputs
        sf.write(str(OUT / f"{p.speaker}__0_cleanraw_in.wav"), cleanraw, SR)
        sf.write(str(OUT / f"{p.speaker}__1_produced_target.wav"), produced, SR)
        sf.write(str(OUT / f"{p.speaker}__2_cleanup_out.wav"), c_out, SR)
        sf.write(str(OUT / f"{p.speaker}__3_natural_out.wav"), n_out, SR)

    if rows:
        a = np.array([r[1:] for r in rows])
        log.info("  %-12s %10.1f %10.1f %12.1f %12.1f   (means)",
                 "MEAN", *a.mean(0))
        gap_cleanup = a[:, 2].mean() - a[:, 1].mean()  # cleanup_out − produced
        gap_natural = a[:, 3].mean() - a[:, 1].mean()  # natural_out − produced
        log.info("Breath energy above the de-breathed target: cleanup +%.1f dB, natural +%.1f dB",
                 gap_cleanup, gap_natural)
    log.info("Pack: %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
