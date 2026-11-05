"""Held-out evaluation of the restorer and the producer against baselines, with saved results.

Restorer: DAPS device recordings of the four held-out speakers (f8, f10, m8, m10) in all 12
device/room conditions, one script each, cut to ~30 s speech-dense windows at sentence
boundaries. Systems: the untreated input, off-the-shelf Resemble Enhance, and any fine-tuned
weights passed with --restorer. WER compares Whisper's transcript of each output with its
transcript of the clean studio take (cleanraw); speaker similarity is against the same take.

Producer: studio takes of the same held-out speakers, all five scripts, with cleanraw as input
and the engineer's "produced" version as target. Systems: pass-through, gain only (each clip
scaled to its target's loudness, which uses the target and so favours this baseline), a static
EQ plus gain fitted on the training speakers, and any weights passed with --producer.

  uv run python scripts/eval_heldout.py \\
      --restorer last=models/restorer-daps/model.safetensors \\
      --producer cleanup=models/producer-cleanup/model.safetensors

Writes runs/heldout-eval/<timestamp>/results.json and REPORT.md. No audio is written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_HOME", str(REPO / "external" / "fake-cuda"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(REPO / "src"))
from voice_cleanup.eval._cuda_libs import ensure_cuda_libpath  # noqa: E402

ensure_cuda_libpath()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

# The locked environment carries two cuDNN 9 builds: torch's own (CUDA 13) and
# nvidia-cudnn-cu12 for faster-whisper. Mixed in one process, torch's GPU convolutions fail
# with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Torch falls back to its own CUDA kernels.
torch.backends.cudnn.enabled = False

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.daps import (  # noqa: E402
    DEGRADED_CONDITIONS,
    HELD_OUT_SPEAKERS,
    SR,
    enumerate_pairs,
    load_aligned_speech_window,
    split_studio_by_speaker,
)
from voice_cleanup.infer.pretrained import load_producer, load_restorer  # noqa: E402
from voice_cleanup.models.producer import SpectralProducer, target_logmag  # noqa: E402

setup_logging()
log = logging.getLogger("eval-heldout")


def _named(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        out.append((name, Path(path)))
    return out


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(st.mean(vals)) if vals else float("nan")


def _median(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(st.median(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------- restorer


@torch.inference_mode()
def _restore(model, deg: np.ndarray, device: str) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)
    torch.manual_seed(0)
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().float()
    if out_sr != SR:
        out = torchaudio.functional.resample(out.unsqueeze(0), out_sr, SR).squeeze(0)
    return out.numpy().astype("float32")


def eval_restorer(args, device: str) -> dict:
    from voice_cleanup.eval.speaker_sim import speaker_similarity
    from voice_cleanup.eval.wer import transcribe, whisper_wer

    pairs = [p for p in enumerate_pairs(args.daps, speakers=HELD_OUT_SPEAKERS,
                                        conditions=DEGRADED_CONDITIONS, target="cleanraw")
             if p.script == args.script]
    if args.limit:
        pairs = pairs[: args.limit]
    log.info("Restorer · %d held-out clips (%s, %s, %.0f s windows)",
             len(pairs), ",".join(HELD_OUT_SPEAKERS), args.script, args.seconds)

    clips = []
    for p in pairs:
        clean, deg = load_aligned_speech_window(p.target, p.degraded, args.seconds)
        clips.append((p, clean, deg))
    ref_text = {p.identifier: transcribe(clean, sr=SR) for p, clean, _ in clips}

    def score(system: str, p, clean, out) -> dict:
        n = min(len(clean), len(out))
        w = float(whisper_wer(clean[:n], out[:n], sr=SR, ref_text=ref_text[p.identifier])["wer"])
        s = float(speaker_similarity(clean[:n], out[:n], sr=SR))
        log.info("  [%-14s] %-30s WER=%.3f spk=%.3f", system, p.identifier, w, s)
        return dict(system=system, clip=p.identifier, speaker=p.speaker,
                    condition=p.condition, wer=w, spk_sim=s)

    rows = [score("untreated", p, clean, deg) for p, clean, deg in clips]
    systems = [("off-the-shelf", None), *_named(args.restorer)]
    for name, weights in systems:
        model = load_restorer(weights, device=device)
        rows += [score(name, p, clean, _restore(model, deg, device)) for p, clean, deg in clips]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {}
    for name in ["untreated"] + [n for n, _ in systems]:
        sr_rows = [r for r in rows if r["system"] == name]
        bal = [r for r in sr_rows if "balcony" in r["condition"]]
        rest = [r for r in sr_rows if "balcony" not in r["condition"]]
        summary[name] = dict(
            n=len(sr_rows), mean_wer=_mean(sr_rows, "wer"), median_wer=_median(sr_rows, "wer"),
            mean_spk=_mean(sr_rows, "spk_sim"),
            balcony_mean_wer=_mean(bal, "wer"), other_mean_wer=_mean(rest, "wer"),
        )
    return dict(rows=rows, summary=summary)


# ---------------------------------------------------------------------------- producer


def _crest_db(x: np.ndarray) -> float:
    peak = 20 * np.log10(np.max(np.abs(x)) + 1e-9)
    rms = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)
    return float(peak - rms)


@torch.no_grad()
def _static_eq(ref: SpectralProducer, wav: np.ndarray, delta: torch.Tensor, device: str):
    """Apply a fixed per-frequency log-magnitude offset, reusing the input phase."""
    x = torch.from_numpy(wav).unsqueeze(0).to(device)
    spec = ref._stft(x)
    mag = spec.abs()
    phase = spec / (mag + ref.eps)
    out = torch.exp(torch.log(mag + ref.eps) + delta[None, :, None]) * phase
    return ref._istft(out, length=x.shape[-1]).squeeze(0).cpu().numpy()


@torch.no_grad()
def _fit_static_eq(ref: SpectralProducer, train_pairs, seconds: float, device: str):
    """Mean (target - input) log-magnitude per frequency bin over non-silent training frames."""
    total, count = None, None
    for p in train_pairs:
        produced, cleanraw = load_aligned_speech_window(p.target, p.degraded, seconds)
        n = min(len(produced), len(cleanraw))
        lt = target_logmag(ref, torch.from_numpy(produced[:n]).unsqueeze(0).to(device))[0]
        li = target_logmag(ref, torch.from_numpy(cleanraw[:n]).unsqueeze(0).to(device))[0]
        frame_db = 10 * torch.log10(torch.exp(2 * li).mean(0) + 1e-12)
        keep = frame_db > frame_db.max() - 40.0  # skip near-silent frames
        d = (lt - li)[:, keep]
        total = d.sum(1) if total is None else total + d.sum(1)
        count = keep.sum() if count is None else count + keep.sum()
    log.info("Producer · static EQ fitted on %d training clips", len(train_pairs))
    return (total / count).clamp(-12.0, 3.0)


def eval_producer(args, device: str) -> dict:
    import pyloudnorm as pyln

    from voice_cleanup.eval.speaker_sim import speaker_similarity
    from voice_cleanup.eval.wer import transcribe, whisper_wer

    meter = pyln.Meter(SR)
    train, val = split_studio_by_speaker(args.daps)
    if args.limit:
        val = val[: args.limit]
    ref = SpectralProducer().to(device).eval()
    delta = _fit_static_eq(ref, train, args.seconds, device)

    clips = []
    for p in val:
        produced, cleanraw = load_aligned_speech_window(p.target, p.degraded, args.seconds)
        n = min(len(produced), len(cleanraw))
        clips.append((p, produced[:n], cleanraw[:n]))
    ref_text = {p.identifier: transcribe(produced, sr=SR) for p, produced, _ in clips}
    log.info("Producer · %d held-out clips", len(clips))

    def score(system: str, p, produced, cleanraw, out) -> dict:
        out = out[: len(produced)]
        lt = target_logmag(ref, torch.from_numpy(produced).unsqueeze(0).to(device))
        lo = target_logmag(ref, torch.from_numpy(out).unsqueeze(0).to(device))
        row = dict(
            system=system, clip=p.identifier, speaker=p.speaker,
            logmag_l1=float((lo - lt).abs().mean()),
            wer=float(whisper_wer(produced, out, sr=SR, ref_text=ref_text[p.identifier])["wer"]),
            spk_sim=float(speaker_similarity(produced, out, sr=SR)),
            d_lufs=float(meter.integrated_loudness(out) - meter.integrated_loudness(cleanraw)),
            d_crest=_crest_db(out) - _crest_db(cleanraw),
        )
        log.info("  [%-12s] %-14s L1=%.3f WER=%.3f spk=%.3f dLUFS=%+.1f dcrest=%+.1f", system,
                 p.identifier, row["logmag_l1"], row["wer"], row["spk_sim"], row["d_lufs"],
                 row["d_crest"])
        return row

    rows = []
    for p, produced, cleanraw in clips:
        rows.append(score("pass-through", p, produced, cleanraw, cleanraw))
        gain_db = meter.integrated_loudness(produced) - meter.integrated_loudness(cleanraw)
        rows.append(score("gain only", p, produced, cleanraw, cleanraw * 10 ** (gain_db / 20)))
        rows.append(score("static EQ", p, produced, cleanraw,
                          _static_eq(ref, cleanraw, delta, device)))
    for name, weights in _named(args.producer):
        model = load_producer(weights, device=device)
        with torch.no_grad():
            for p, produced, cleanraw in clips:
                out = model.enhance(torch.from_numpy(cleanraw).to(device)).cpu().numpy()
                rows.append(score(name, p, produced, cleanraw, out))

    summary = {}
    for name in dict.fromkeys(r["system"] for r in rows):
        sr_rows = [r for r in rows if r["system"] == name]
        summary[name] = {k: _mean(sr_rows, k)
                         for k in ("logmag_l1", "wer", "spk_sim", "d_lufs", "d_crest")}
        summary[name]["n"] = len(sr_rows)
    target_rows = [dict(d_lufs=meter.integrated_loudness(pr) - meter.integrated_loudness(cr),
                        d_crest=_crest_db(pr) - _crest_db(cr)) for _, pr, cr in clips]
    summary["engineer's version (target)"] = dict(
        d_lufs=_mean(target_rows, "d_lufs"), d_crest=_mean(target_rows, "d_crest"),
        n=len(target_rows))
    return dict(rows=rows, summary=summary)


# ---------------------------------------------------------------------------- report


def _report(results: dict) -> str:
    lines = ["# Held-out evaluation", "", f"Generated {results['generated_at']}.", ""]
    if "restorer" in results:
        lines += ["## Restorer (DAPS device recordings, held-out speakers)", "",
                  "| System | Clips | Mean WER | Median WER | WER, balcony | WER, other rooms "
                  "| Speaker similarity |", "|---|---|---|---|---|---|---|"]
        for name, s in results["restorer"]["summary"].items():
            lines.append(f"| {name} | {s['n']} | {s['mean_wer']:.3f} | {s['median_wer']:.3f} | "
                         f"{s['balcony_mean_wer']:.3f} | {s['other_mean_wer']:.3f} | "
                         f"{s['mean_spk']:.3f} |")
        lines.append("")
    if "producer" in results:
        lines += ["## Producer (studio takes, held-out speakers)", "",
                  "| System | Clips | Log-mag L1 to target | WER | Speaker similarity "
                  "| Loudness change (LU) | Crest change (dB) |", "|---|---|---|---|---|---|---|"]
        for name, s in results["producer"]["summary"].items():
            cells = []
            for key, fmt in (("logmag_l1", ".3f"), ("wer", ".3f"), ("spk_sim", ".3f"),
                             ("d_lufs", "+.1f"), ("d_crest", "+.1f")):
                v = s.get(key, float("nan"))
                cells.append("" if v != v else format(v, fmt))
            lines.append(f"| {name} | {s['n']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--daps", type=Path, default=REPO / "data" / "corpora" / "daps" / "daps")
    ap.add_argument("--restorer", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--producer", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--script", default="script1", help="DAPS script for the restorer clips")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0, help="cap clips per part (smoke test)")
    ap.add_argument("--no-restorer", action="store_true")
    ap.add_argument("--no-producer", action="store_true")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
    out = args.out or REPO / "runs" / "heldout-eval" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    results: dict = dict(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        config=dict(script=args.script, seconds=args.seconds, limit=args.limit,
                    held_out=list(HELD_OUT_SPEAKERS),
                    restorer=[n for n, _ in _named(args.restorer)],
                    producer=[n for n, _ in _named(args.producer)]),
    )
    if not args.no_restorer:
        results["restorer"] = eval_restorer(args, device)
    if not args.no_producer:
        results["producer"] = eval_producer(args, device)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(_report(results))
    log.info("Wrote %s and %s", out / "results.json", out / "REPORT.md")
    print(_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
