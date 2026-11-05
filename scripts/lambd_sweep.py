"""Sweep RE's denoiser-strength knob `lambd` on the extreme tail (balcony), inference-only.

`lambd` blends RE's internal denoiser output with the original at the mel stage
(`enhancer.py`: `x_mel = lambd*denoised + (1-lambd)*original`). Because the fine-tune trained
`lambd` as a random `Beta(0.2,0.2)` draw per example, the model is robust to ANY strength → an
inference-time sweep is in-distribution and free (no retrain). lambd=0 = no pre-clean (raw into
the CFM); lambd=1 = full denoise; we ship 0.5. This is the "amount of pre-cleaning" knob, broadband.

Question: does a different `lambd` help on balcony (the outdoor/far-field extreme tail), where the
denoiser has the most to remove but also the most chance to over-scrub? Scores WER (vs clean ref,
the faithfulness/intelligibility guard) + speaker-sim per lambd, on the held-out balcony clips.

  uv run python scripts/lambd_sweep.py --run-dir runs/re-finetune-daps/20260530-015903 --tag best
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"
FAKE_CUDA = REPO / "external" / "fake-cuda"
DAPS = REPO / "data" / "corpora" / "daps" / "daps"

os.environ.setdefault("CUDA_HOME", str(FAKE_CUDA))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
sys.path.insert(0, str(REPO / "src"))
from voice_cleanup.eval._cuda_libs import ensure_cuda_libpath  # noqa: E402

ensure_cuda_libpath()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(RE_ROOT))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.daps import SR, load_aligned_speech_window, select_eval_pairs  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402

setup_logging()
log = logging.getLogger("lambd-sweep")


def _build_engine(run_dir: Path, tag: str):
    import dataclasses

    from deepspeed import DeepSpeedConfig
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.utils import Engine

    hp = HParams.load(STAGE2)
    hp = dataclasses.replace(hp, batch_size_per_gpu=1, nj=1, praat_augment_prob=0.0)
    cfg = hp.deepspeed_config
    cfg["optimizer"]["params"]["torch_adam"] = True
    cfg["optimizer"]["params"]["adam_w_mode"] = False
    engine = Engine(model=Enhancer(hp), config_class=DeepSpeedConfig(cfg),
                    ckpt_dir=STAGE2 / "ds" / "G")
    engine.load_checkpoint()  # pretrained base
    engine._ckpt_dir = run_dir / "ds" / "G"
    engine.load_checkpoint(tag=tag)  # fine-tuned tag over the base
    log.info("Loaded fine-tuned checkpoint tag=%s from %s", tag, engine._ckpt_dir)
    return engine


@torch.no_grad()
def _enhance(model, deg: np.ndarray, device, *, lambd: float) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.eval()
    model.configurate_(nfe=32, solver="midpoint", lambd=lambd, tau=0.0)  # deterministic
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().numpy().astype("float32")
    if out_sr != SR:
        out = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), out_sr, SR).squeeze(0).numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--tag", default="best")
    ap.add_argument("--target", default="cleanraw")
    ap.add_argument("--lambdas", default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated lambd values to sweep")
    ap.add_argument("--conditions", default="balcony",
                    help="substring filter on condition ('balcony' = the extreme tail; 'all')")
    ap.add_argument("--eval-clips", type=int, default=12)
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--no-wer", action="store_true")
    ap.add_argument("--write-audio", action="store_true",
                    help="render {clip}__lambd{v}.wav per value for the ear (off by default)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    lambdas = [float(x) for x in args.lambdas.split(",")]
    do_wer = not args.no_wer
    pack = args.run_dir / "lambd_sweep"
    pack.mkdir(parents=True, exist_ok=True)

    pairs = select_eval_pairs(DAPS, target=args.target)[: args.eval_clips]
    if args.conditions != "all":
        pairs = [p for p in pairs if args.conditions in p.condition]
    log.info("Sweep · %d clips · λ∈%s · %.0fs window · conditions=%s",
             len(pairs), lambdas, args.eval_seconds, args.conditions)

    from voice_cleanup.eval.speaker_sim import speaker_similarity
    if do_wer:
        from voice_cleanup.eval.wer import transcribe, whisper_wer

    # Pick the speech-dense window once per clip; reuse for every lambd (fair A/B).
    windows, ref_text = {}, {}
    for p in pairs:
        key = f"{p.speaker}-{p.condition}"
        clean, deg = load_aligned_speech_window(p.target, p.degraded, args.eval_seconds)
        windows[key] = (clean, deg)
        if do_wer:
            ref_text[key] = transcribe(clean, sr=SR)

    engine = _build_engine(args.run_dir, args.tag)
    model = engine.module

    results = []
    for lambd in lambdas:
        log.info("── λ=%.2f ───────────────────────────────────", lambd)
        rows = []
        for p in pairs:
            key = f"{p.speaker}-{p.condition}"
            clean, deg = windows[key]
            out = _enhance(model, deg, device, lambd=lambd)
            spk = float(speaker_similarity(clean, out, sr=SR))
            w = float("nan")
            if do_wer:
                w = float(whisper_wer(clean, out, sr=SR, ref_text=ref_text[key])["wer"])
            if args.write_audio:
                sf.write(str(pack / f"{key}__lambd{lambd:.2f}.wav"),
                         normalize_lufs(out, SR), SR)
            rows.append(dict(clip=key, wer=w, spk_sim=spk))
            log.info("  %-26s WER=%s spk=%.3f", key,
                     f"{w:.3f}" if do_wer else "—", spk)
        mean_wer = float(np.nanmean([r["wer"] for r in rows]))
        mean_spk = float(np.nanmean([r["spk_sim"] for r in rows]))
        log.info("λ=%.2f · mean WER=%.3f · mean spk=%.3f", lambd, mean_wer, mean_spk)
        results.append(dict(lambd=lambd, mean_wer=mean_wer, mean_spk=mean_spk, per_clip=rows))

    # Summary table — the whole point of the sweep.
    log.info("══ λ sweep summary (conditions=%s, %d clips) ══", args.conditions, len(pairs))
    log.info("  %-6s  %-9s  %-9s", "λ", "mean WER", "mean spk")
    for r in results:
        flag = "  ← we ship" if abs(r["lambd"] - 0.5) < 1e-9 else ""
        log.info("  %-6.2f  %-9.3f  %-9.3f%s", r["lambd"], r["mean_wer"], r["mean_spk"], flag)
    best = min(results, key=lambda r: r["mean_wer"]) if do_wer else \
        max(results, key=lambda r: r["mean_spk"])
    log.info("Best by %s: λ=%.2f", "WER" if do_wer else "spk", best["lambd"])

    out_json = pack / f"sweep_{args.conditions}_{args.tag}.json"
    out_json.write_text(json.dumps(
        dict(run_dir=str(args.run_dir), tag=args.tag, conditions=args.conditions,
             eval_seconds=args.eval_seconds, ts=time.strftime("%Y%m%d-%H%M%S"),
             results=results), indent=2))
    log.info("Done · %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
