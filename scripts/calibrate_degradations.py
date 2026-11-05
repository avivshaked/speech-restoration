"""Objectively validate the synthetic degradations BEFORE training — the make-or-break check.

The 2026-05-29 failure was synthetic recipes harsher than reality (synthetic 'podcast' →
RE-WER 0.58 vs real room+mic 0.03), which silently narrowed the project. The fix, per the
guardrails for working alone: don't trust my taste on realism — calibrate to URGENT ranges
and validate the synthetic distribution against DAPS-REAL *objectively*, with the spectral
diagnostics DNSMOS lacks (eval/spectral.py).

What this does (no ear required for the objective part; an A/B pack IS rendered for Aviv's ear):
  1. Measure DAPS-real degraded clips + DAPS clean (the realism reference distribution).
  2. Measure Tier-1 clean + Tier-1 degraded with build_training_chain (the training mix) and
     with each fixed eval condition.
  3. Compare distributions (HF-ratio, spectral tilt, SNR, crest) — flag if synthetic is
     harsher/out-of-range vs DAPS for the families DAPS HAS (room/mic/noise). For families
     DAPS LACKS (bandwidth, strong EQ-damage), place an OPTIONAL real radio-voice anchor
     (a consent-clean reference, supplied via --anchor-wav) in the space so we can see the
     training mix COVERS the radio-voice region without being dominated by it.
  4. Render a degradation A/B pack (clean vs each degradation) for the morning ear-check.
  5. Plot the distributions.

  uv run python scripts/calibrate_degradations.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data import degrade, tier1  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402
from voice_cleanup.eval.spectral import spectral_summary  # noqa: E402

setup_logging()
log = logging.getLogger("calibrate")
SR = tier1.SR
DAPS = REPO / "data" / "corpora" / "daps" / "daps"
OUT = REPO / "runs" / "calibration"
METRICS = ["hf_ratio_db", "tilt_db_per_oct", "snr_db", "crest_db"]


def _read_window(path: Path, seconds: float = 20.0) -> np.ndarray:
    """A representative speech-dense window: skip lead-in, take `seconds` from ~25% in."""
    info = sf.info(str(path))
    sr = info.samplerate
    start = int(info.frames * 0.25)
    frames = min(int(seconds * sr), info.frames - start)
    a, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import torch
        import torchaudio
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a.astype("float32")


def _anchor_summary(path: Path | None) -> dict | None:
    """Spectral summary of an OPTIONAL consent-clean real radio-voice anchor (audio or video).

    No anchor is fine — placement is skipped. Supply your own consent-clean reference via
    --anchor-wav.
    """
    if path is None:
        return None
    if not path.exists():
        log.warning("Anchor not found at %s — skipping placement", path)
        return None
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "a.wav"
        subprocess.run([ff, "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "wav", str(tmp)], check=True)
        a, _ = sf.read(str(tmp), dtype="float32")
    return spectral_summary(a.astype("float32"), SR)


def _summaries(signals: list[np.ndarray]) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {m: [] for m in METRICS}
    for x in signals:
        s = spectral_summary(x, SR)
        for m in METRICS:
            if np.isfinite(s[m]):
                acc[m].append(s[m])
    return acc


def _pctiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    a = np.array(vals)
    return {"n": len(a), "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-wav", type=Path, default=None,
                    help="optional consent-clean real radio-voice reference (audio or video) "
                         "to place as an anchor in the distribution plot")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_clips = 80
    anchor_s = _anchor_summary(args.anchor_wav)

    # 1. DAPS real (reference distribution) — degraded conditions + clean targets.
    from voice_cleanup.data.daps import enumerate_pairs
    daps_pairs = enumerate_pairs(DAPS)
    rng.shuffle(daps_pairs)
    daps_pairs = daps_pairs[:n_clips]
    log.info("Measuring %d DAPS-real degraded + clean ...", len(daps_pairs))
    daps_deg = _summaries([_read_window(p.degraded) for p in daps_pairs])
    daps_clean = _summaries([_read_window(p.target) for p in daps_pairs[:40]])

    # 2. Tier-1 clean + synthetic. (Tier-1 must be downloaded; tolerate VCTK-only.)
    clips = tier1.enumerate_tier1()
    if not clips:
        log.error("No Tier-1 clips — run fetch_tier1_corpora.py first.")
        return 2
    rng.shuffle(clips)
    sample = clips[:n_clips]
    log.info("Measuring %d Tier-1 clean + synthetic ...", len(sample))
    clean_sigs = [tier1._read_clean(c.path) for c in sample]
    clean_sigs = [x for x in clean_sigs if len(x) > int(2 * SR)]
    tier1_clean = _summaries(clean_sigs)

    return _run(daps_deg, daps_clean, tier1_clean, clean_sigs, rng, anchor_s)


def _run(daps_deg, daps_clean, tier1_clean, clean_sigs, rng, anchor_s):
    # training-mix degradation (dynamic)
    train_deg_sigs = []
    for x in clean_sigs:
        seed = int(rng.integers(0, 2**31 - 1))
        chain = tier1.build_training_chain(np.random.default_rng(seed))
        with contextlib.suppress(Exception):
            train_deg_sigs.append(chain.apply(x, SR, seed=seed))
    tier1_train_deg = _summaries(train_deg_sigs)

    # fixed eval conditions (the held-out scoreboard degradations)
    conds = {
        "eq_damage": degrade.DegradationChain("eq_damage", [
            degrade.EqDamage(n_bands_range=(3, 4), freq_range=(200.0, 6000.0),
                             gain_db_range=(-12.0, 12.0), q_range=(2.0, 6.0))]),
        "radio": degrade.DegradationChain("radio", [
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-10.0, 12.0), q_range=(2.0, 5.0)),
            degrade.BandwidthLimit(cutoff_range=(3800.0, 4500.0), order=2)]),
        "bandwidth": degrade.DegradationChain("bandwidth", [
            degrade.BandwidthLimit(cutoff_range=(4500.0, 6000.0), order=2)]),
        "broadband": degrade.DegradationChain("broadband", [
            degrade.SyntheticReverb(rt60_range=(0.3, 0.5)),
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-9.0, 9.0)),
            degrade.AdditiveNoise(snr_db_range=(15.0, 25.0), color="pink")]),
    }
    cond_summ = {}
    for cname, chain in conds.items():
        sigs = []
        for i, x in enumerate(clean_sigs):
            with contextlib.suppress(Exception):
                sigs.append(chain.apply(x, SR, seed=1000 + i))
        cond_summ[cname] = _summaries(sigs)

    # 3. Optional real radio-voice anchor (consent-clean, supplied via --anchor-wav)
    # 4. Report
    report = {
        "daps_real_degraded": {m: _pctiles(daps_deg[m]) for m in METRICS},
        "daps_clean": {m: _pctiles(daps_clean[m]) for m in METRICS},
        "tier1_clean": {m: _pctiles(tier1_clean[m]) for m in METRICS},
        "tier1_train_degraded": {m: _pctiles(tier1_train_deg[m]) for m in METRICS},
        "tier1_eval_conditions": {c: {m: _pctiles(s[m]) for m in METRICS}
                                  for c, s in cond_summ.items()},
        "anchor": anchor_s,
    }
    (OUT / "calibration_report.json").write_text(json.dumps(report, indent=2))

    log.info("── Spectral distributions (p10 / p50 / p90) ──")
    for m in METRICS:
        log.info("  %s:", m)
        for label, summ in [("DAPS-real degraded", daps_deg), ("DAPS clean", daps_clean),
                            ("Tier1 clean", tier1_clean), ("Tier1 train-degraded", tier1_train_deg),
                            *[(f"  cond/{c}", s) for c, s in cond_summ.items()]]:
            p = _pctiles(summ[m])
            if p:
                log.info("    %-22s %7.1f / %7.1f / %7.1f", label, p["p10"], p["p50"], p["p90"])
        if anchor_s is not None:
            log.info("    %-22s %7.1f  ← real radio-voice anchor", "ANCHOR", anchor_s[m])

    # Realism flag: training-mix HF-ratio / SNR vs DAPS-real range (families DAPS shares).
    _flag(report)

    # 5. A/B pack for the ear
    pack = OUT / "degradation_ab"
    pack.mkdir(exist_ok=True)
    for i, x in enumerate(clean_sigs[:4]):
        sf.write(str(pack / f"clip{i}__clean.wav"), normalize_lufs(x, SR), SR)
        seed = 5000 + i
        chain = tier1.build_training_chain(np.random.default_rng(seed))
        try:
            d = chain.apply(x, SR, seed=seed)
            sf.write(str(pack / f"clip{i}__train_dynamic.wav"), normalize_lufs(d, SR), SR)
        except Exception:  # noqa: BLE001
            pass
        for cname, chain in conds.items():
            try:
                d = chain.apply(x, SR, seed=6000 + i)
                sf.write(str(pack / f"clip{i}__{cname}.wav"), normalize_lufs(d, SR), SR)
            except Exception:  # noqa: BLE001
                pass
    log.info("Degradation A/B pack for the ear: %s", pack)

    # 6. Plot
    _plot(daps_deg, tier1_train_deg, cond_summ, anchor_s)
    log.info("Calibration report + plot + A/B pack under %s", OUT)
    return 0


def _flag(report: dict) -> None:
    """Loud warning if the training mix sits outside the DAPS-real range on shared families."""
    d = report["daps_real_degraded"]
    t = report["tier1_train_degraded"]
    for m in ("hf_ratio_db", "snr_db", "crest_db"):
        if not d.get(m) or not t.get(m):
            continue
        # training median should sit within ~ DAPS [p10, p90] (roughly comparable severity)
        if t[m]["p50"] < d[m]["p10"] - 3 or t[m]["p50"] > d[m]["p90"] + 3:
            log.warning("REALISM FLAG · %s: train-mix p50=%.1f outside DAPS-real [%.1f, %.1f] "
                        "— synthetic may be mis-calibrated (the 05-29 failure mode). Review.",
                        m, t[m]["p50"], d[m]["p10"], d[m]["p90"])


def _plot(daps_deg, tier1_train_deg, cond_summ, anchor_s) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for ax, m in zip(axes, METRICS, strict=True):
        data = [daps_deg[m], tier1_train_deg[m]] + [cond_summ[c][m] for c in cond_summ]
        labels = ["DAPS-real", "train-mix"] + list(cond_summ)
        ax.boxplot([d if d else [0] for d in data], tick_labels=labels, showfliers=False)
        if anchor_s is not None:
            ax.axhline(anchor_s[m], color="red", ls="--", lw=1, label="anchor")
            ax.legend(fontsize=7)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Synthetic degradation vs DAPS-real (+ optional radio-voice anchor)")
    fig.tight_layout()
    fig.savefig(str(OUT / "calibration_distributions.png"), dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
