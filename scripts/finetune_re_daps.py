"""Restoration fine-tune: RE's CFM on real DAPS pairs, held-out-speaker generalization.

This is the real run the Phase-2 spike was a feasibility test for. It fine-tunes RE's CFM
(`z=None` → denoiser frozen, vocoder untrained) on the 16-speaker DAPS train split, and its
PRIMARY output is the honest question: **does fine-tuning beat zero-shot RE on HELD-OUT
speakers?** So it captures a zero-shot baseline A/B *before* training and a fine-tuned A/B
*after*, on the 4 unseen speakers — same clips, same deterministic inference (tau=0).

Grounded in the spike (HANDOFF / learning log 2026-05-29):
- DeepSpeed forced to plain torch Adam + stub nvcc (no CUDA toolkit here).
- Flat LR (min==max, warmup=1): the pretrained checkpoint resumes at global_step=870000,
  which would sit at the tail of RE's WarmupDecayLR — a constant LR sidesteps that cleanly.
- No `_align`: DAPS pairs ship aligned; the dataset crops both streams from one offset.
- cfm mode: this targets content FAITHFULNESS, not the vocoder resynthesis texture.

Main-process torch threads are capped (the heavy compute is on the GPU); DataLoader workers
do audio decode/LUFS on the CPU intentionally, and the held-out Whisper eval is CPU-heavy
by nature — both are deliberate CPU use, logged.

Run:  uv run python scripts/finetune_re_daps.py --steps 4000 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"
FAKE_CUDA = REPO / "external" / "fake-cuda"
DAPS = REPO / "data" / "corpora" / "daps" / "daps"
OUT = REPO / "runs" / "re-finetune-daps"

os.environ.setdefault("CUDA_HOME", str(FAKE_CUDA))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# Put the CUDA-12 cublas/cudnn libs on LD_LIBRARY_PATH for ctranslate2 (faster-whisper
# WER) — must happen before torch import; may re-exec the process once.
sys.path.insert(0, str(REPO / "src"))
from voice_cleanup.eval._cuda_libs import ensure_cuda_libpath  # noqa: E402

ensure_cuda_libpath()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(RE_ROOT))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.daps import (  # noqa: E402
    DEFAULT_TARGET,
    SR,
    TARGET_VERSIONS,
    DAPSPairDataset,
    load_aligned_speech_window,
    select_eval_pairs,
    split_by_speaker,
)
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402

setup_logging()
log = logging.getLogger("re-ft-daps")


def _ensure_fake_nvcc() -> None:
    nvcc = FAKE_CUDA / "bin" / "nvcc"
    if nvcc.exists():
        return
    nvcc.parent.mkdir(parents=True, exist_ok=True)
    nvcc.write_text('#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n')
    nvcc.chmod(0o755)


def _build_engine(lr: float, total_steps: int, batch_size: int, nj: int):
    import dataclasses

    from deepspeed import DeepSpeedConfig
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.utils import Engine

    hp = HParams.load(STAGE2)
    hp = dataclasses.replace(
        hp, batch_size_per_gpu=batch_size, nj=nj, praat_augment_prob=0.0,
        min_lr=lr, max_lr=lr, warmup_steps=1, max_steps=total_steps,
    )
    ds_cfg = hp.deepspeed_config
    ds_cfg["optimizer"]["params"]["torch_adam"] = True
    ds_cfg["optimizer"]["params"]["adam_w_mode"] = False
    model = Enhancer(hp)
    engine = Engine(model=model, config_class=DeepSpeedConfig(ds_cfg), ckpt_dir=STAGE2 / "ds" / "G")
    engine.load_checkpoint()
    dev = next(engine.module.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError(f"Engine landed on {dev}, not CUDA — aborting (would grind on CPU).")
    log.info("GPU guard OK · model on %s · resumed global_step=%d · mem=%.2f GB",
             dev, engine.global_step, torch.cuda.memory_allocated() / 1e9)
    return engine


@torch.inference_mode()
def _enhance(model, deg: np.ndarray, device) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.eval()
    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)  # deterministic A/B
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().numpy().astype("float32")
    if out_sr != SR:
        out = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), out_sr, SR).squeeze(0).numpy()
    return out


def _perceptual_eval(model, device, eval_pairs, pack: Path, tag: str, *, eval_seconds: float,
                     wer: bool, ref_cache: dict, write_audio: bool = True) -> list[dict]:
    """Score `tag` on held-out clips: WER + spk-sim vs target. Renders audio iff write_audio.

    `write_audio=False` is the cheap metrics-only path used at intermediate scaling-curve
    checkpoints (no point rewriting 12 reference/degraded files every 1000 steps); the full
    A/B pack is written only for the baseline and the final fine-tuned eval.
    """
    from voice_cleanup.eval.speaker_sim import speaker_similarity
    rows = []
    for p in eval_pairs:
        clean, deg = load_aligned_speech_window(p.target, p.degraded, eval_seconds)
        out = _enhance(model, deg, device)
        key = f"{p.speaker}-{p.condition}"
        if write_audio:
            sf.write(str(pack / f"{key}__0_clean.wav"), normalize_lufs(clean, SR), SR)
            sf.write(str(pack / f"{key}__1_degraded.wav"), normalize_lufs(deg, SR), SR)
            sf.write(str(pack / f"{key}__{tag}.wav"), normalize_lufs(out, SR), SR)
        spk = float(speaker_similarity(clean, out, sr=SR))
        w = float("nan")
        if wer:
            from voice_cleanup.eval.wer import transcribe, whisper_wer
            if key not in ref_cache:
                ref_cache[key] = transcribe(clean, sr=SR)
            w = float(whisper_wer(clean, out, sr=SR, ref_text=ref_cache[key])["wer"])
        rows.append(dict(clip=key, speaker=p.speaker, condition=p.condition, wer=w, spk_sim=spk))
        log.info("  [%s] %-26s WER=%s spk-sim=%.3f", tag, key,
                 f"{w:.3f}" if wer else "—", spk)
    return rows


def _curve_point(step: int, rows: list[dict]) -> dict:
    """Summarise a held-out eval into one scaling-curve point (overall + balcony-only means)."""
    import statistics as st
    bal = [r for r in rows if "balcony" in r["condition"]]
    def _mean(rs, k): return float(st.mean(r[k] for r in rs)) if rs else float("nan")
    return dict(
        step=step,
        mean_wer=_mean(rows, "wer"), mean_spk=_mean(rows, "spk_sim"),
        balcony_wer=_mean(bal, "wer"), balcony_spk=_mean(bal, "spk_sim"),
        per_clip={r["clip"]: dict(wer=r["wer"], spk=r["spk_sim"]) for r in rows},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000,
                    help="Total optimizer steps. Ignored if --epochs is set.")
    ap.add_argument("--epochs", type=float, default=None,
                    help="Train this many passes over the windowed dataset (overrides --steps). "
                         "The scaling-curve knob: e.g. 8 = ~8 passes, with eval along the way.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--eval-clips", type=int, default=6, help="held-out clips for the A/B")
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Eval held-out + checkpoint every N steps (scaling curve). 0 = only "
                         "baseline + final. WER eval is ~20s now, so periodic eval is cheap.")
    ap.add_argument("--no-wer", action="store_true", help="skip Whisper (spk-sim + ear only)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--target", default=DEFAULT_TARGET, choices=TARGET_VERSIONS,
                    help="Studio target version. cleanraw = faithful restoration (breaths kept); "
                         "clean = de-breathed; produced = mastered. Default: faithful restoration.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Salt for full reproducibility: shuffle order + crop offsets + "
                         "flow-matching noise. Same seed = identical run; change for a 2nd seed.")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="Cap this process to a fraction of total VRAM (e.g. 0.6 ≈ 14 GB on a "
                         "24 GB card). Use for COEXIST runs so the desktop/compositor keeps "
                         "headroom and doesn't glitch. Omit for AFK runs that use the whole card.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    if args.gpu_mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU VRAM capped to %.0f%% (~%.1f GB) — leaving headroom for the desktop.",
                 args.gpu_mem_fraction * 100, args.gpu_mem_fraction * total)
    # Reproducible-with-salt: seed every RNG that touches the run.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    log.info("Seed=%d (reproducible run)", args.seed)
    _ensure_fake_nvcc()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT / ts
    pack = run_dir / "listening_pack"
    pack.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    log.info("Run dir: %s", run_dir)

    # --- Data --------------------------------------------------------------------------
    log.info("Restoration target: %s", args.target)
    train_pairs, _val_pairs = split_by_speaker(DAPS, target=args.target)
    ds = DAPSPairDataset(train_pairs, crop_seconds=args.crop_seconds)

    def _seed_worker(wid: int) -> None:  # deterministic per-worker RNG (crop offsets)
        s = (torch.initial_seed() + wid) % 2**32
        np.random.seed(s)
        random.seed(s)

    gen = torch.Generator()
    gen.manual_seed(args.seed)  # deterministic shuffle order
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        generator=gen, worker_init_fn=_seed_worker,
    )
    eval_pairs = select_eval_pairs(DAPS, target=args.target)[: args.eval_clips]
    log.info("Eval (held-out) clips: %s", [f"{p.speaker}/{p.condition}" for p in eval_pairs])

    steps_per_epoch = max(1, len(loader))
    total_steps = round(args.epochs * steps_per_epoch) if args.epochs else args.steps
    log.info("Schedule · %d steps (%.2f epochs · %d steps/epoch · batch %d)",
             total_steps, total_steps / steps_per_epoch, steps_per_epoch, args.batch_size)

    # --- Engine ------------------------------------------------------------------------
    engine = _build_engine(args.lr, total_steps, args.batch_size, args.num_workers)
    # Save fine-tuned checkpoints into THIS run's dir, not the vendored pretrained release the
    # engine loaded from (its _ckpt_dir). The Engine override forces save_dir=self._ckpt_dir,
    # so we repoint it post-load; tags ("best"/"last") then land under run_dir/ds/G/.
    engine._ckpt_dir = run_dir / "ds" / "G"
    model = engine.module
    do_wer = not args.no_wer

    # --- BASELINE A/B (zero-shot, before fine-tune) — the comparison that matters -------
    log.info("=== BASELINE (zero-shot) held-out A/B%s ===", " + WER" if do_wer else "")
    ref_cache: dict = {}
    base_rows = _perceptual_eval(model, device, eval_pairs, pack, "2_RE_baseline",
                                 eval_seconds=args.eval_seconds, wer=do_wer, ref_cache=ref_cache)

    # Scaling curve + checkpoint selection. Lower score = better (WER if scored, else 1-spk).
    curve: list[dict] = [_curve_point(0, base_rows)]
    def _score(pt: dict) -> float:
        return pt["mean_wer"] if do_wer else (1.0 - pt["mean_spk"])
    best = {"step": 0, "score": float("inf")}  # step 0 = zero-shot release; never re-saved

    def _save(tag: str) -> None:
        try:
            engine.save_checkpoint(tag=tag)  # Engine forces save_dir=run_dir/ds/G
        except Exception as e:  # noqa: BLE001
            log.warning("Checkpoint save (tag=%s) failed: %s", tag, e)

    def _checkpoint_eval(step: int, *, write_audio: bool, tag_audio: str) -> list[dict]:
        torch.cuda.empty_cache()  # reclaim training cache before the eval inference pass
        rows = _perceptual_eval(model, device, eval_pairs, pack, tag_audio,
                                eval_seconds=args.eval_seconds, wer=do_wer,
                                ref_cache=ref_cache, write_audio=write_audio)
        pt = _curve_point(step, rows)
        curve.append(pt)
        (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))  # persist each eval
        log.info("curve · step %5d · mean WER=%.3f · balcony WER=%.3f · mean spk=%.3f",
                 step, pt["mean_wer"], pt["balcony_wer"], pt["mean_spk"])
        _save("last")
        if _score(pt) < best["score"]:
            best.update(step=step, score=_score(pt))
            _save("best")
            log.info("  ↳ new BEST (step %d, score %.3f) — saved tag=best", step, _score(pt))
        engine.train()
        torch.cuda.empty_cache()
        return rows

    # --- Train -------------------------------------------------------------------------
    log.info("=== Fine-tuning CFM · %d steps · batch=%d · %.1fs · lr=%.1e · eval-every=%s ===",
             total_steps, args.batch_size, args.crop_seconds, args.lr, args.eval_every or "off")
    engine.train()
    t0 = time.time()
    step = 0
    data_iter = iter(loader)
    while step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        engine(x, y, None)  # cfm mode → stashes losses
        losses = engine.gather_attribute("losses")
        loss = sum(losses.values())
        engine.backward(loss)
        engine.step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == total_steps:
            rate = step / (time.time() - t0)
            log.info("step %5d/%d · loss=%.4f · %s · %.1f it/s · gpu=%.2f GB",
                     step, total_steps, float(loss),
                     {k: round(float(v), 4) for k, v in losses.items()},
                     rate, torch.cuda.memory_allocated() / 1e9)
        if args.eval_every and step % args.eval_every == 0 and step != total_steps:
            log.info("=== held-out eval @ step %d/%d ===", step, total_steps)
            _checkpoint_eval(step, write_audio=False, tag_audio="3_RE_finetuned")

    log.info("Training done in %.1f min · peak gpu %.2f GB",
             (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 1e9)

    # --- FINAL FINE-TUNED A/B (writes the listening pack) ------------------------------
    log.info("=== FINAL FINE-TUNED held-out A/B ===")
    ft_rows = _checkpoint_eval(total_steps, write_audio=True, tag_audio="3_RE_finetuned")
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))
    log.info("Scaling curve (%d points) → curve.json · best=step %d (score %.3f)",
             len(curve), best["step"], best["score"])

    # --- Report: zero-shot vs fine-tuned on HELD-OUT -----------------------------------
    summary = []
    log.info("=== HELD-OUT: zero-shot → fine-tuned ===")
    for b, f in zip(base_rows, ft_rows, strict=True):
        d_wer = f["wer"] - b["wer"] if do_wer else float("nan")
        d_spk = f["spk_sim"] - b["spk_sim"]
        log.info("%-26s WER %s→%s  spk-sim %.3f→%.3f",
                 b["clip"], f"{b['wer']:.3f}" if do_wer else "—",
                 f"{f['wer']:.3f}" if do_wer else "—", b["spk_sim"], f["spk_sim"])
        summary.append(dict(clip=b["clip"], wer_base=b["wer"], wer_ft=f["wer"], d_wer=d_wer,
                            spk_base=b["spk_sim"], spk_ft=f["spk_sim"], d_spk=d_spk))
    (run_dir / "results.json").write_text(json.dumps(
        dict(args=vars(args), total_steps=total_steps, best=best, held_out=summary), indent=2))
    log.info("Audio A/B in %s · results.json + curve.json written. Best checkpoint: tag=best "
             "(step %d). Metrics are PROXIES — the ear is the verdict.", pack, best["step"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
