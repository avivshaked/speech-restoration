"""Run the studio-reference restoration scoreboard (the headline Path-A eval).

Takes clean studio speech (EARS / DAPS), applies a known degradation recipe,
restores it with each enhancer (passthrough = degraded floor, MP-SENet =
suppression baseline, Resemble Enhance = generative restorer), and scores every
restoration against the *true studio reference* — full-band (48 kHz). Emits a
markdown scoreboard + a loudness-matched A/B listening pack for Aviv's ear.

This is the restoration-era replacement for run_phase1_baselines.py (which was the
16 kHz suppression scoreboard). See docs/decisions/log.md (2026-05-28 eval
re-architecture).

Usage:
    uv run python scripts/run_restoration_scoreboard.py
    uv run python scripts/run_restoration_scoreboard.py --recipe harsh --per-speaker 1
    uv run python scripts/run_restoration_scoreboard.py --corpus daps --no-wer
    uv run python scripts/run_restoration_scoreboard.py --enhancers passthrough resemble-enhance
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.eval.enhancers import get_enhancer  # noqa: E402
from voice_cleanup.eval.restoration import (  # noqa: E402
    discover_daps_clips,
    discover_ears_clips,
    run_studio_reference_eval,
    write_studio_report,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ROOT / "data" / "corpora"
DEFAULT_ENHANCERS = ["passthrough", "mp-senet", "resemble-enhance"]


def gather_clips(corpus: str, per_speaker: int, limit: int | None):
    clips = []
    if corpus in ("ears", "both"):
        clips += discover_ears_clips(CORPORA / "ears", per_speaker=per_speaker)
    if corpus in ("daps", "both"):
        clips += discover_daps_clips(CORPORA / "daps", condition="clean", per_speaker=per_speaker)
    if limit is not None:
        clips = clips[:limit]
    return clips


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Studio-reference restoration scoreboard.")
    p.add_argument("--recipe", default="moderate", help="degradation recipe (see data/degrade.py)")
    p.add_argument("--corpus", default="ears", choices=["ears", "daps", "both"])
    p.add_argument("--per-speaker", type=int, default=1, help="clips per speaker")
    p.add_argument("--limit", type=int, default=6, help="max clips total (None for all)")
    p.add_argument("--max-seconds", type=float, default=12.0, help="trim clips to this length")
    p.add_argument("--eval-sr", type=int, default=48000)
    p.add_argument("--enhancers", nargs="+", default=DEFAULT_ENHANCERS)
    p.add_argument("--no-wer", action="store_true", help="skip Whisper-WER (slow)")
    p.add_argument("--no-spk", action="store_true", help="skip speaker-similarity")
    p.add_argument("--no-pack", action="store_true", help="skip the A/B listening pack")
    args = p.parse_args(argv)

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    clips = gather_clips(args.corpus, args.per_speaker, limit)
    if not clips:
        logger.error("No clips found — run scripts/fetch_clean_corpora.py first.")
        return 1
    logger.info(
        "Scoreboard · %d clips · corpus=%s · recipe=%s", len(clips), args.corpus, args.recipe
    )

    enhancers = [get_enhancer(name) for name in args.enhancers]

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "runs" / "restoration-scoreboard" / f"{ts}-{args.recipe}-{args.corpus}"
    pack_dir = None if args.no_pack else out_dir / "listening_pack"

    report = run_studio_reference_eval(
        clips, args.recipe, enhancers,
        eval_sr=args.eval_sr, max_seconds=args.max_seconds,
        run_dnsmos=True, run_nisqa=True, run_wer=not args.no_wer, run_spk=not args.no_spk,
        listening_pack_dir=pack_dir,
    )
    results_path, report_path = write_studio_report(report, out_dir)
    logger.info("Scoreboard · wrote %s", report_path)
    logger.info("Scoreboard · results %s", results_path)
    if pack_dir is not None:
        logger.info("Scoreboard · A/B listening pack → %s", pack_dir)

    # Echo the scoreboard to the console for an at-a-glance read.
    print("\n" + report_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
