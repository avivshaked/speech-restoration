"""Model bazaar — run several restoration models on the SAME reference clips and lay the
outputs out for the ear, side by side.

Why this exists (decision log 2026-05-31): RE was retired as the base after it failed the ear
despite good proxy metrics. The lesson — metrics (spk-sim, WER) can't hear naturalness — is
baked into this tool: the **LUFS-matched A/B pack is the verdict**; the LTAS/spectral panel is
a *diagnostic only*, never the score. We're shopping for a generative restorer that sounds
plausibly human on severely degraded input; the only way to know is to listen.

Drop any consent-clean reference wavs into --refs (your own clean+degraded recordings,
public-domain old-time radio, CC bad-audio podcasts, synthetic-degraded clean) and run. Every
registered enhancer runs on every clip; outputs are loudness-matched so level isn't a
confound.

  uv run python scripts/model_bazaar.py --refs data/bazaar-refs
  uv run python scripts/model_bazaar.py --enhancers passthrough voicefixer resemble-enhance
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# RE enhancer needs the fake-cuda shim + cuda libpath, same as the rest of the panel.
import os  # noqa: E402

os.environ.setdefault("CUDA_HOME", str(REPO / "external" / "fake-cuda"))
from voice_cleanup.eval._cuda_libs import ensure_cuda_libpath  # noqa: E402

ensure_cuda_libpath()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402
from voice_cleanup.eval import enhancers as enh  # noqa: E402
from voice_cleanup.eval.spectral import ltas, spectral_summary  # noqa: E402

setup_logging()
log = logging.getLogger("bazaar")
SR = 44100  # full-band: the whole question is the top octave + envelope, so never 16 kHz here.
MATCH_LUFS = -16.0
DEFAULT_ENHANCERS = ["passthrough", "resemble-enhance", "voicefixer"]


def _load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a.astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", type=Path, default=REPO / "data" / "bazaar-refs",
                    help="dir of consent-clean reference wavs (any sr; resampled to 44.1 kHz)")
    ap.add_argument("--enhancers", nargs="+", default=None,
                    help=f"enhancer names (default: {DEFAULT_ENHANCERS}; registered: "
                         f"{enh.registered_names()})")
    ap.add_argument("--limit", type=int, default=None, help="cap number of reference clips")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="trim each clip for speed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    refs = sorted(p for p in args.refs.glob("*.wav")) if args.refs.is_dir() else []
    if not refs:
        log.error("No reference wavs in %s — add consent-clean clips first "
                  "(see scripts/fetch_bazaar_refs.py).", args.refs)
        return 2
    if args.limit:
        refs = refs[: args.limit]

    names = args.enhancers or [n for n in DEFAULT_ENHANCERS if n in enh.registered_names()]
    bad = [n for n in names if n not in enh.registered_names()]
    if bad:
        log.error("Unknown enhancer(s) %s. Registered: %s", bad, enh.registered_names())
        return 2

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (REPO / "runs" / "bazaar" / ts)
    pack = out / "pack"
    ltas_dir = out / "ltas"
    pack.mkdir(parents=True, exist_ok=True)
    ltas_dir.mkdir(parents=True, exist_ok=True)
    log.info("Bazaar · %d reference clips × %d models %s → %s",
             len(refs), len(names), names, out)

    # Instantiate enhancers once (models are heavy).
    models = {}
    for n in names:
        log.info("Loading enhancer: %s", n)
        models[n] = enh.get_enhancer(n)

    table: dict[str, dict] = {}
    for ci, ref in enumerate(refs):
        clip = ref.stem
        x = _load(ref)
        if args.max_seconds:
            x = x[: int(args.max_seconds * SR)]
        log.info("── clip %d/%d · %s (%.1fs) ──", ci + 1, len(refs), clip, len(x) / SR)
        outputs = {"0_input": x}
        for n in names:
            try:
                y = models[n].apply(x, SR)
            except Exception as e:  # noqa: BLE001
                log.warning("  %s FAILED on %s: %s", n, clip, e)
                continue
            outputs[n] = np.asarray(y, dtype="float32")

        # LUFS-match everything so level isn't a confound, write the A/B pack.
        for tag, sig in outputs.items():
            m = normalize_lufs(sig, SR, MATCH_LUFS)
            sf.write(str(pack / f"{clip}__{tag}.wav"), m, SR)
            summ = spectral_summary(m, SR)
            table.setdefault(clip, {})[tag] = summ

        # LTAS overlay (diagnostic only).
        plt.figure(figsize=(11, 6))
        for tag, sig in outputs.items():
            m = normalize_lufs(sig, SR, MATCH_LUFS)
            f, db = ltas(m, SR)
            plt.semilogx(f, db, label=tag, lw=1.3)
        plt.axvspan(8000, 16000, color="grey", alpha=0.12)
        plt.axvspan(400, 4000, color="orange", alpha=0.07)
        plt.xlim(50, 22000)
        plt.ylim(-90, 0)
        plt.xlabel("Hz (log)")
        plt.ylabel("avg magnitude (dB)")
        plt.title(f"{clip} — LTAS (grey=top octave, orange=boxy mids) · DIAGNOSTIC only")
        plt.legend(fontsize=8)
        plt.grid(True, which="both", alpha=0.2)
        plt.tight_layout()
        plt.savefig(str(ltas_dir / f"{clip}.png"), dpi=130)
        plt.close()

    (out / "spectral.json").write_text(json.dumps(table, indent=2))
    _report(out, refs, names, table)
    log.info("Bazaar done · A/B pack: %s · LTAS: %s", pack, ltas_dir)
    log.info("EARS DECIDE. The spectral table/plots are diagnostics — not a quality score.")
    return 0


def _report(out: Path, refs, names, table) -> None:
    lines = ["# Model bazaar", "",
             "**The A/B pack is the verdict. The numbers below are diagnostics only** "
             "(HF-ratio = top octave present; tilt = envelope shape). Neither hears naturalness.",
             "", f"Models: {', '.join(names)}", f"Clips: {len(refs)}", ""]
    for clip, rows in table.items():
        lines.append(f"## {clip}")
        lines.append("")
        lines.append("| version | HF-ratio dB | tilt dB/oct |")
        lines.append("|---|---|---|")
        for tag, s in rows.items():
            lines.append(f"| {tag} | {s['hf_ratio_db']:+.1f} | {s['tilt_db_per_oct']:+.2f} |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
