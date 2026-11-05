"""Render a saved fine-tune checkpoint's enhanced audio into a run's listening pack.

Periodic scaling-curve checkpoints (`best`, `last`) are metrics-only — no audio is written
at the eval that found them. This loads a given checkpoint tag and renders its enhanced
output for the held-out eval clips, dropping `{clip}__4_RE_{tag}.wav` alongside the existing
`0_clean` / `1_degraded` / `2_RE_baseline` / `3_RE_finetuned` files for a direct A/B.

  uv run python scripts/render_checkpoint_pack.py --run-dir runs/re-finetune-daps/<ts> --tag best
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
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
log = logging.getLogger("render-ckpt")


def _build_engine():
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
    return engine


@torch.no_grad()  # NOT inference_mode: we load_checkpoint() AFTER the baseline render pass,
def _enhance(model, deg: np.ndarray, device) -> np.ndarray:  # and inference tensors block that
    from resemble_enhance.inference import inference

    model.eval()
    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)
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
    ap.add_argument("--eval-clips", type=int, default=12)
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--conditions", default="all", help="'all' or substring filter, e.g. 'balcony'")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip re-rendering the pretrained baseline (render only the tag)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    pack = args.run_dir / "listening_pack"
    pack.mkdir(parents=True, exist_ok=True)

    pairs = select_eval_pairs(DAPS, target=args.target)[: args.eval_clips]
    if args.conditions != "all":
        pairs = [p for p in pairs if args.conditions in p.condition]
    log.info("Rendering %d clips · tag=%s · %.1fs speech-dense window", len(pairs), args.tag,
             args.eval_seconds)

    # Pick the speech-dense window ONCE per clip and reuse it for every render, so clean,
    # degraded, baseline and the tag are all the same 12 s of audio — a fair A/B.
    from voice_cleanup.eval.speaker_sim import speaker_similarity
    windows = {}
    for p in pairs:
        key = f"{p.speaker}-{p.condition}"
        clean, deg = load_aligned_speech_window(p.target, p.degraded, args.eval_seconds)
        windows[key] = (clean, deg)
        sf.write(str(pack / f"{key}__0_clean.wav"), normalize_lufs(clean, SR), SR)
        sf.write(str(pack / f"{key}__1_degraded.wav"), normalize_lufs(deg, SR), SR)

    engine = _build_engine()  # pretrained base = the zero-shot baseline
    model = engine.module

    if not args.no_baseline:
        log.info("Rendering pretrained baseline (zero-shot RE) ...")
        for p in pairs:
            key = f"{p.speaker}-{p.condition}"
            clean, deg = windows[key]
            out = _enhance(model, deg, device)
            sf.write(str(pack / f"{key}__2_RE_baseline.wav"), normalize_lufs(out, SR), SR)
            spk = float(speaker_similarity(clean, out, sr=SR))
            log.info("  %-26s baseline spk-sim=%.3f", key, spk)

    engine._ckpt_dir = args.run_dir / "ds" / "G"
    engine.load_checkpoint(tag=args.tag)  # load the fine-tuned tag over the base
    log.info("Loaded checkpoint tag=%s from %s", args.tag, engine._ckpt_dir)
    model = engine.module

    for p in pairs:
        key = f"{p.speaker}-{p.condition}"
        clean, deg = windows[key]
        out = _enhance(model, deg, device)
        sf.write(str(pack / f"{key}__4_RE_{args.tag}.wav"), normalize_lufs(out, SR), SR)
        spk = float(speaker_similarity(clean, out, sr=SR))
        log.info("  %-26s %s spk-sim=%.3f → %s__4_RE_%s.wav", key, args.tag, spk, key, args.tag)
    log.info("Done · pack at %s", pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
