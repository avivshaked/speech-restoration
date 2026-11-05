"""Paired comparisons for a held-out evaluation written by scripts/eval_heldout.py.

Reads results.json and prints, for each comparison, the mean per-clip difference, a 95%
percentile bootstrap interval over clips (10,000 resamples, fixed seed) and how many clips
came out lower, equal and higher. The tables in docs/findings.md come from this script.

  uv run python scripts/heldout_stats.py runs/heldout-eval/<timestamp>/results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESAMPLES = 10_000
SEED = 0


def bootstrap(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(label: str, a: list[float], b: list[float], rng: np.random.Generator) -> None:
    diff = np.asarray(a) - np.asarray(b)
    lo, hi = bootstrap(diff, rng)
    lower, equal = int((diff < 0).sum()), int((diff == 0).sum())
    higher = len(diff) - lower - equal
    print(f"| {label} | {diff.mean():+.3f} | {lo:+.3f} to {hi:+.3f} "
          f"| {lower} / {equal} / {higher} |")


def by_system(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["system"]][r["clip"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--restorer", default="last", help="fine-tuned restorer to compare")
    ap.add_argument("--producer", default="cleanup", help="producer to compare")
    ap.add_argument("--producer-skip", default=None, metavar="TEXT",
                    help="leave out producer clips whose name contains TEXT, e.g. _script1_ "
                         "for the clips train_producer.py used to choose its checkpoint")
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    rng = np.random.default_rng(SEED)
    header = "| Comparison | Mean difference | 95% interval | Clips lower / equal / higher |"
    rule = "|---|---|---|---|"

    if results.get("restorer", {}).get("rows"):
        r = by_system(results["restorer"]["rows"])
        ft = r[args.restorer]
        clips = sorted(ft)
        groups = {
            "all clips": clips,
            "balcony": [c for c in clips if "balcony" in c],
            "other rooms": [c for c in clips if "balcony" not in c],
        }
        print(f"Restorer `{args.restorer}`, WER\n\n{header}\n{rule}")
        for ref in [s for s in r if s != args.restorer]:
            for name, subset in groups.items():
                compare(f"against {ref}, {name}", [ft[c]["wer"] for c in subset],
                        [r[ref][c]["wer"] for c in subset], rng)
        print(f"\nRestorer `{args.restorer}`, speaker similarity\n\n{header}\n{rule}")
        for ref in [s for s in r if s != args.restorer]:
            compare(f"against {ref}", [ft[c]["spk_sim"] for c in clips],
                    [r[ref][c]["spk_sim"] for c in clips], rng)

    if results.get("producer", {}).get("rows"):
        p = by_system(results["producer"]["rows"])
        pr = p[args.producer]
        clips = sorted(c for c in pr if not (args.producer_skip and args.producer_skip in c))
        print(f"\nProducer `{args.producer}`, log-magnitude L1 to target, {len(clips)} clips")
        print("\n| System | Mean |\n|---|---|")
        for name, rows in p.items():
            print(f"| {name} | {np.mean([rows[c]['logmag_l1'] for c in clips]):.3f} |")
        print(f"\n{header}\n{rule}")
        for ref in [s for s in p if s != args.producer]:
            compare(f"against {ref}", [pr[c]["logmag_l1"] for c in clips],
                    [p[ref][c]["logmag_l1"] for c in clips], rng)


if __name__ == "__main__":
    main()
