"""voice-cleanup CLI entry point.

Subcommands:
  eval — run the metric panel on a test set under a controlled degradation

More subcommands (train/infer/data) will land as those phases come online.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from voice_cleanup._logging import get_console, setup_logging
from voice_cleanup.eval.enhancers import Enhancer, get_enhancer, registered_names
from voice_cleanup.eval.runner import (
    ClipRecord,
    ClipResult,
    Degradation,
    EvalReport,
    run_eval,
    write_report,
)

logger = logging.getLogger(__name__)


# --- formatting helpers -----------------------------------------------------


def _fmt(val: float | None, fmt: str = "{:.3f}") -> str:
    return "—" if val is None else fmt.format(val)


def _build_results_table(report_label: str) -> Table:
    table = Table(title=f"Per-clip metrics — {report_label}", show_lines=False, expand=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("clip", overflow="ellipsis", no_wrap=True, max_width=28, min_width=10)
    table.add_column("lic", justify="center", style="dim", no_wrap=True, width=11)
    table.add_column("PESQ", justify="right", no_wrap=True, width=5)
    table.add_column("ESTOI", justify="right", no_wrap=True, width=6)
    table.add_column("SI-SDR", justify="right", no_wrap=True, width=7)
    table.add_column("DNS·S", justify="right", no_wrap=True, width=5)
    table.add_column("DNS·B", justify="right", no_wrap=True, width=5)
    table.add_column("DNS·O", justify="right", no_wrap=True, width=5)
    table.add_column("SQUIM", justify="right", no_wrap=True, width=5)
    table.add_column("NISQA", justify="right", no_wrap=True, width=5)
    table.add_column("WER", justify="right", no_wrap=True, width=5)
    table.add_column("s", justify="right", style="dim", no_wrap=True, width=4)
    return table


def _append_clip_row(table: Table, idx: int, rec: ClipRecord, res: ClipResult) -> None:
    table.add_row(
        str(idx + 1),
        rec.identifier,
        rec.license_short,
        _fmt(res.pesq, "{:.2f}"),
        _fmt(res.estoi, "{:.3f}"),
        _fmt(res.si_sdr, "{:+.1f}"),
        _fmt(res.dnsmos_sig, "{:.2f}"),
        _fmt(res.dnsmos_bak, "{:.2f}"),
        _fmt(res.dnsmos_ovrl, "{:.2f}"),
        _fmt(res.squim_mos, "{:.2f}"),
        _fmt(res.nisqa_mos, "{:.2f}"),
        _fmt(res.wer, "{:.2f}"),
        f"{res.elapsed_s:.1f}",
    )


def _build_summary_table(report: EvalReport) -> Table:
    table = Table(title="Aggregate summary", show_lines=False)
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("n", justify="right", style="dim")
    pretty = {
        "pesq": "PESQ-wb",
        "estoi": "ESTOI",
        "si_sdr": "SI-SDR (dB)",
        "dnsmos_sig": "DNSMOS SIG",
        "dnsmos_bak": "DNSMOS BAK",
        "dnsmos_ovrl": "DNSMOS OVRL",
        "squim_mos": "SQUIM-MOS",
        "nisqa_mos": "NISQA MOS",
        "nisqa_noi": "NISQA NOI",
        "nisqa_dis": "NISQA DIS",
        "nisqa_col": "NISQA COL",
        "nisqa_loud": "NISQA LOUD",
        "wer": "Whisper-WER",
    }
    for key, label in pretty.items():
        s = report.summary().get(key)
        if not s:
            continue
        table.add_row(
            label,
            f"{s['mean']:.3f}",
            f"{s['std']:.3f}",
            f"{s['min']:.3f}",
            f"{s['max']:.3f}",
            str(s["n"]),
        )
    return table


# --- eval subcommand --------------------------------------------------------


def _run_eval_cmd(args: argparse.Namespace) -> int:
    console = get_console()
    setup_logging(level=args.log_level)

    test_set_dir = Path(args.test_set)
    degradation = Degradation.parse(args.degradation)
    enhancer = get_enhancer(args.enhancer) if args.enhancer else None
    out_dir = Path(args.output) if args.output else _default_output_dir(degradation, enhancer)

    console.print(
        Panel.fit(
            f"[bold]Voice Cleanup — eval[/bold]\n"
            f"test-set:    {test_set_dir}\n"
            f"degradation: {degradation.label()}\n"
            f"enhancer:    {enhancer.label() if enhancer else 'none (score degraded directly)'}\n"
            f"output:      {out_dir}\n"
            f"squim: {'OFF (--no-squim)' if args.no_squim else 'ON'}\n"
            f"nisqa: {'OFF (--no-nisqa)' if args.no_nisqa else 'ON'}\n"
            f"wer:   {'OFF (--no-wer)' if args.no_wer else 'ON'}"
            + (f"\nlimit: {args.limit}" if args.limit else ""),
            title="config",
            border_style="cyan",
        )
    )

    results_table = _build_results_table(degradation.label())

    def on_clip_done(idx: int, _total: int, rec: ClipRecord, res: ClipResult) -> None:
        _append_clip_row(results_table, idx, rec, res)

    try:
        report = run_eval(
            test_set_dir,
            degradation,
            limit=args.limit,
            run_squim=not args.no_squim,
            run_nisqa=not args.no_nisqa,
            run_wer=not args.no_wer,
            enhancer=enhancer,
            on_clip_done=on_clip_done,
        )
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        return 2

    console.print()
    console.print(results_table)
    console.print()
    console.print(_build_summary_table(report))

    results_path, summary_path = write_report(report, out_dir)
    console.print(
        Panel.fit(
            f"results:  {results_path}\n"
            f"summary:  {summary_path}\n"
            f"elapsed:  {report.elapsed_s:.1f}s · {len(report.clips)} clips",
            title="written",
            border_style="green",
        )
    )
    return 0


def _default_output_dir(degradation: Degradation, enhancer: Enhancer | None = None) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    deg_slug = degradation.kind if degradation.kind == "none" else (
        f"gaussian-{degradation.snr_db:g}db"
    )
    enh_slug = f"-{enhancer.name}" if enhancer is not None else ""
    return Path("runs") / "eval" / f"{stamp}-{deg_slug}{enh_slug}"


# --- argument parser --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-cleanup",
        description="Voice Cleanup CLI — eval, train, infer (eval available now).",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Log level (DEBUG/INFO/WARNING). Defaults to VOICE_CLEANUP_LOG_LEVEL or INFO.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser(
        "eval",
        help="Run the metric panel on a test set under a controlled degradation.",
    )
    eval_p.add_argument(
        "--test-set",
        default="data/test-sets/podcast-real",
        help="Path to a test-set directory containing clips.json (default: %(default)s).",
    )
    eval_p.add_argument(
        "--degradation",
        default="gaussian:-5",
        help=(
            "Degradation spec applied to every clip. Options: 'none' (non-intrusive "
            "only, no degradation), 'gaussian:SNR_DB' (additive white noise at given "
            "SNR, e.g. 'gaussian:-5'). Default: %(default)s."
        ),
    )
    eval_p.add_argument(
        "--output",
        default=None,
        help="Output directory for results.json + summary.json (default: runs/eval/<ts>-<slug>/).",
    )
    eval_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N clips (useful for smoke runs).",
    )
    eval_p.add_argument(
        "--no-squim",
        action="store_true",
        help="Skip SQUIM-MOS (saves ~5s setup and N×500ms per clip).",
    )
    eval_p.add_argument(
        "--no-nisqa",
        action="store_true",
        help="Skip NISQA multidim (saves N×~12s per clip; requires vendored source).",
    )
    eval_p.add_argument(
        "--no-wer",
        action="store_true",
        help="Skip Whisper-WER (very slow; saves ~5-15s per clip on CPU + model load).",
    )
    eval_p.add_argument(
        "--enhancer",
        choices=registered_names(),
        default=None,
        help=(
            "Apply this enhancer to the degraded audio before scoring. "
            "Without this flag, the panel scores the degraded audio directly "
            "(useful for validating the panel itself). With one of "
            f"{registered_names()}, this becomes a baseline / model evaluation."
        ),
    )
    eval_p.set_defaults(func=_run_eval_cmd)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
