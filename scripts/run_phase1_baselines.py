"""Phase-1 baseline driver: passthrough + MP-SENet + Resemble Enhance across an SNR sweep.

Runs the full eval panel for the Phase-1 baselines over a sweep of degradation
conditions, in a SINGLE process so every model (DNSMOS, SQUIM, NISQA, Whisper,
MP-SENet, Resemble Enhance) loads once and is reused across all condition ×
enhancer combinations. Three reference points (per docs/decisions/log.md 2026-05-28,
restoration re-centering): the **noisy floor** (passthrough), a **mask-based
suppressor** (MP-SENet), and a **generative restorer** (Resemble Enhance).

Writes per-combo JSON under runs/phase1-baselines/<condition>-<enhancer>/ and a
consolidated runs/phase1-baselines/REPORT.md with one comparison table per
condition.

Usage:
    uv run python scripts/run_phase1_baselines.py [--limit N] [--no-wer]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/run_phase1_baselines.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.eval.enhancers import get_enhancer  # noqa: E402
from voice_cleanup.eval.runner import (  # noqa: E402
    Degradation,
    EvalReport,
    run_eval,
    write_report,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SET_DIR = REPO_ROOT / "data" / "test-sets" / "podcast-real"
OUT_DIR = REPO_ROOT / "runs" / "phase1-baselines"

# SNR sweep: clean(none) → mild → moderate → harsh. The `none` condition probes
# whether an enhancer harms already-clean(ish) real podcast audio; the noisy
# conditions probe how far each enhancer lifts the noisy floor.
CONDITIONS = ["none", "gaussian:10", "gaussian:5", "gaussian:0"]
ENHANCERS = ["passthrough", "mp-senet", "resemble-enhance"]

# (key, header, fmt, higher_is_better) for the REPORT tables.
METRIC_COLS = [
    ("pesq", "PESQ", "{:.3f}", True),
    ("estoi", "ESTOI", "{:.3f}", True),
    ("si_sdr", "SI-SDR", "{:+.2f}", True),
    ("dnsmos_sig", "DNSMOS-SIG", "{:.2f}", True),
    ("dnsmos_bak", "DNSMOS-BAK", "{:.2f}", True),
    ("dnsmos_ovrl", "DNSMOS-OVRL", "{:.2f}", True),
    ("nisqa_mos", "NISQA-MOS", "{:.2f}", True),
    ("nisqa_noi", "NISQA-noi", "{:.2f}", True),
    ("squim_mos", "SQUIM", "{:.2f}", True),
    ("wer", "WER", "{:.3f}", False),
]


def _combo_slug(condition: str, enhancer: str) -> str:
    cond = condition.replace(":", "-") + ("db" if condition.startswith("gaussian") else "")
    return f"{cond}-{enhancer}"


def _fmt(summary: dict, key: str, fmt: str) -> str:
    cell = summary.get(key)
    if not cell:
        return "—"
    return fmt.format(cell["mean"])


def _render_report(
    reports: dict[tuple[str, str], EvalReport],
    *,
    n_clips: int,
    run_wer: bool,
) -> str:
    lines: list[str] = []
    lines.append(
        "# Phase 1 baselines — noisy floor vs suppression (MP-SENet) "
        "vs generative restoration (Resemble Enhance)\n"
    )
    lines.append(
        "Three zero-shot reference points per the 2026-05-28 restoration "
        "re-centering (`docs/decisions/log.md`): the **noisy floor** (passthrough), "
        "a **mask-based suppressor** (MP-SENet), and a **generative restorer** "
        "(Resemble Enhance). The question this answers: *where does each family sit "
        "on our panel — and does generative restoration avoid the low-SNR "
        "intelligibility regression that suppression showed — so Phase-3a "
        "fine-tuning has a clear bar to beat?*\n"
    )
    wer_note = "" if run_wer else " **WER skipped in this run (`--no-wer`).**"
    lines.append("## Setup\n")
    lines.append(
        f"- **Test set:** `data/test-sets/podcast-real/` — {n_clips} real podcast "
        "clips (20 s each, 16 kHz mono, mixed CC licenses)."
    )
    lines.append(
        "- **Degradation:** additive Gaussian noise at the clip's natural level, "
        "seeded per clip; `none` = the real clip untouched."
    )
    lines.append(
        "- **MP-SENet:** zero-shot, vendored from `yxlu-0102/MP-SENet` at a pinned "
        "SHA (`scripts/fetch_mpsenet.py`), **DNS Challenge** checkpoint "
        "(`g_best_dns`) — the better real-noise generaliser than the "
        "VoiceBank+DEMAND headline checkpoint. 16 kHz, n_fft=400/hop=100/win=400, "
        "power-law compression c=0.3, per-utterance RMS normalisation (verbatim "
        "from the repo's `inference.py`)."
    )
    lines.append(
        "- **Resemble Enhance:** zero-shot, vendored from `resemble-ai/resemble-enhance` "
        "at a pinned SHA (`scripts/fetch_resemble_enhance.py`), the bundled "
        "`enhancer_stage2` checkpoint (denoiser + LCFM latent flow-matching + UnivNet "
        "GAN vocoder). Generative restorer, not a mask: it regenerates speech at "
        "44.1 kHz (resampled in/out from our 16 kHz panel). Defaults: nfe=32, "
        "solver=midpoint, lambd=0.5, tau=0.5."
    )
    lines.append(
        "- **Panel:** PESQ-wb, ESTOI, SI-SDR (intrusive; undefined under `none`); "
        "DNSMOS P.835, NISQA multidim, SQUIM-MOS (non-intrusive); Whisper-large-v3 "
        "WER (intelligibility regression; skipped under `none`)." + wer_note
    )
    lines.append(
        "- Intrusive metrics treat the **original real clip as the clean "
        "reference**; the enhancer is scored on its attempt to restore the noised "
        "input.\n"
    )
    lines.append(
        "> Reminder (from `docs/learning/log.md`): SQUIM-MOS is OOD on additive "
        "noise and is only expected to be *responsive*, not noise-monotonic — "
        "DNSMOS carries the noise-monotonicity contract.\n"
    )

    header = "| enhancer | " + " | ".join(h for _, h, _, _ in METRIC_COLS) + " |"
    sep = "|" + "---|" * (len(METRIC_COLS) + 1)

    lines.append("## Results by condition\n")
    for condition in CONDITIONS:
        if condition == "none":
            cond_label = "clean (none)"
        else:
            cond_label = condition.replace("gaussian:", "") + " dB SNR"
        lines.append(f"### {cond_label}\n")
        lines.append(header)
        lines.append(sep)
        for enhancer in ENHANCERS:
            rep = reports.get((condition, enhancer))
            if rep is None:
                continue
            s = rep.summary()
            row = "| " + (rep.enhancer_label or enhancer) + " | "
            row += " | ".join(_fmt(s, k, f) for k, _, f, _ in METRIC_COLS)
            row += " |"
            lines.append(row)
        lines.append("")

    lines.append("## Observations\n")
    lines.append("_(filled in after the run — see deltas above.)_\n")
    lines.append("---\n")
    lines.append("Regenerate: `uv run python scripts/run_phase1_baselines.py`. "
                 "Per-combo raw scores live in the sibling `*/results.json` files.\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap clips (smoke runs)")
    parser.add_argument("--no-wer", action="store_true", help="skip Whisper-WER (much faster)")
    parser.add_argument("--no-squim", action="store_true", help="skip SQUIM-MOS")
    parser.add_argument("--no-nisqa", action="store_true", help="skip NISQA")
    args = parser.parse_args(argv)

    run_wer = not args.no_wer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Phase-1 baselines · conditions=%s · enhancers=%s · out=%s",
                CONDITIONS, ENHANCERS, OUT_DIR)

    reports: dict[tuple[str, str], EvalReport] = {}
    n_clips = 0
    total = len(CONDITIONS) * len(ENHANCERS)
    i = 0
    for condition in CONDITIONS:
        deg = Degradation.parse(condition)
        for enhancer_name in ENHANCERS:
            i += 1
            logger.info("=== [%d/%d] condition=%s · enhancer=%s ===",
                        i, total, condition, enhancer_name)
            enhancer = get_enhancer(enhancer_name)
            rep = run_eval(
                TEST_SET_DIR, deg,
                limit=args.limit,
                run_squim=not args.no_squim,
                run_nisqa=not args.no_nisqa,
                run_wer=run_wer,
                enhancer=enhancer,
            )
            reports[(condition, enhancer_name)] = rep
            n_clips = len(rep.clips)
            write_report(rep, OUT_DIR / _combo_slug(condition, enhancer_name))

    report_md = _render_report(reports, n_clips=n_clips, run_wer=run_wer)
    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text(report_md)
    logger.info("Phase-1 baselines · wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
