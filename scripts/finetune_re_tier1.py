"""Restorer rebuild: fine-tune RE's CFM on Tier-1 commercial-safe clean speech, synthetically
degraded on the fly. From the PRETRAINED BASE (not our DAPS fine-tune, which was trained only
on degradations that never lacked the top octave).

Goal: teach DE-COLORATION (flatten resonant "radio voice" mids) + preserve TOP-OCTAVE
regeneration, on full-band
commercial data (VCTK/Hi-Fi), so the result is commercially usable AND covers the degradation
families DAPS lacked.

Model selection: held-out CFM **validation loss** (does the restored latent match the clean
target), NOT WER — a passthrough has perfect WER but restores nothing. WER is kept as a
HALLUCINATION GUARD (generative restoration can invent words — the balcony clip did) and
HF-energy ratio as the headline top-octave diagnostic. Metrics are PROXIES — the ear is the
verdict (results land provisional for Aviv).

Run:  uv run python scripts/finetune_re_tier1.py --epochs 2 --batch-size 12 --eval-every 1500
Smoke: uv run python scripts/finetune_re_tier1.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"
FAKE_CUDA = REPO / "external" / "fake-cuda"
OUT = REPO / "runs" / "re-finetune-tier1"

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
from torch.utils.data import DataLoader  # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(RE_ROOT))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data import degrade, tier1  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402
from voice_cleanup.eval.spectral import hf_energy_ratio  # noqa: E402

setup_logging()
log = logging.getLogger("re-ft-tier1")
SR = tier1.SR


def _ensure_fake_nvcc() -> None:
    nvcc = FAKE_CUDA / "bin" / "nvcc"
    if nvcc.exists():
        return
    nvcc.parent.mkdir(parents=True, exist_ok=True)
    nvcc.write_text('#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n')
    nvcc.chmod(0o755)


# Fixed held-out eval conditions — isolate the two target skills + a realistic broadband mix.
# Each scored with a FIXED seed per (speaker, condition) so the eval is reproducible.
def eval_conditions() -> dict[str, degrade.DegradationChain]:
    return {
        # De-coloration test: strong resonant mid-range EQ damage.
        "eq_damage": degrade.DegradationChain("eq_damage", [
            degrade.EqDamage(n_bands_range=(3, 4), freq_range=(200.0, 6000.0),
                             gain_db_range=(-12.0, 12.0), q_range=(2.0, 6.0))]),
        # "Radio voice" synthetic proxy: bandwidth loss + resonant mids.
        # Gentle slope (order 2, doubled by zero-phase filtering).
        "radio": degrade.DegradationChain("radio", [
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-10.0, 12.0), q_range=(2.0, 5.0)),
            degrade.BandwidthLimit(cutoff_range=(3800.0, 4500.0), order=2)]),
        # Top-octave extension test: clean bandwidth limit only (moderate, realistic rolloff).
        "bandwidth": degrade.DegradationChain("bandwidth", [
            degrade.BandwidthLimit(cutoff_range=(4500.0, 6000.0), order=2)]),
        # Realistic broadband: room + mild mic EQ + mild noise.
        "broadband": degrade.DegradationChain("broadband", [
            degrade.SyntheticReverb(rt60_range=(0.3, 0.5)),
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-9.0, 9.0)),
            degrade.AdditiveNoise(snr_db_range=(15.0, 25.0), color="pink")]),
    }


def _build_engine(lr: float, total_steps: int, batch_size: int, nj: int, mem_fraction):
    import dataclasses

    from deepspeed import DeepSpeedConfig
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.utils import Engine

    if mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
    hp = HParams.load(STAGE2)
    hp = dataclasses.replace(
        hp, batch_size_per_gpu=batch_size, nj=nj, praat_augment_prob=0.0,
        min_lr=lr, max_lr=lr, warmup_steps=1, max_steps=total_steps)
    ds_cfg = hp.deepspeed_config
    ds_cfg["optimizer"]["params"]["torch_adam"] = True
    ds_cfg["optimizer"]["params"]["adam_w_mode"] = False
    engine = Engine(model=Enhancer(hp), config_class=DeepSpeedConfig(ds_cfg),
                    ckpt_dir=STAGE2 / "ds" / "G")
    engine.load_checkpoint()  # PRETRAINED BASE only — no DAPS fine-tune
    dev = next(engine.module.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError(f"Engine on {dev}, not CUDA — aborting.")
    log.info("GPU guard OK · base loaded on %s · resumed global_step=%d · mem=%.2f GB",
             dev, engine.global_step, torch.cuda.memory_allocated() / 1e9)
    return engine


@torch.inference_mode()
def _enhance(model, deg: np.ndarray, device) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.eval()
    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().numpy().astype("float32")
    if out_sr != SR:
        out = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), out_sr, SR).squeeze(0).numpy()
    return out


def _build_val_loss_set(val_clips, crop_seconds, n_windows, device, seed=1234):
    """A FIXED set of (deg, clean) 3s windows for a reproducible held-out CFM val loss.

    Deterministic degradation (fixed per-window seed) so the val loss is comparable across
    checkpoints — it measures the model, not a fresh random degradation each time.
    """
    rng = np.random.default_rng(seed)
    crop = int(crop_seconds * SR)
    usable = [c for c in val_clips]
    rng.shuffle(usable)
    degs, cleans = [], []
    for c in usable:
        if len(degs) >= n_windows:
            break
        try:
            a = tier1._read_clean(c.path)
        except Exception:  # noqa: BLE001
            continue
        if len(a) < crop:
            a = np.pad(a, (0, crop - len(a)))
        off = int(rng.integers(0, max(1, len(a) - crop + 1)))
        clean = a[off:off + crop]
        wseed = int(rng.integers(0, 2**31 - 1))
        chain = tier1.build_training_chain(np.random.default_rng(wseed))
        try:
            deg = chain.apply(clean, SR, seed=wseed)
        except Exception:  # noqa: BLE001
            deg = degrade.EqDamage().apply(clean, SR, np.random.default_rng(wseed))
        degs.append(normalize_lufs(deg.astype("float32"), SR)[:crop])
        cleans.append(normalize_lufs(clean.astype("float32"), SR)[:crop])
    x = torch.from_numpy(np.stack(degs)).to(device)
    y = torch.from_numpy(np.stack(cleans)).to(device)
    log.info("Val-loss set · %d fixed windows of %.1fs", len(degs), crop_seconds)
    return x, y


@torch.no_grad()
def _val_loss(engine, x, y, batch_size, val_seed=777) -> float:
    """Mean held-out CFM loss over the fixed val set. Seeds the flow-matching t/noise draws
    identically each call so checkpoints are compared on the same targets."""
    was_training = engine.module.training
    engine.module.eval()
    torch.manual_seed(val_seed)  # identical (t, noise) draws across checkpoints
    tot, nb = 0.0, 0
    for i in range(0, x.shape[0], batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        engine(xb, yb, None)
        losses = engine.gather_attribute("losses")
        tot += float(sum(losses.values()))
        nb += 1
    if was_training:
        engine.module.train()
    return tot / max(1, nb)


def _perceptual_eval(model, device, eval_clips, conds, pack, tag, *, wer, ref_cache,
                     write_audio=True) -> list[dict]:
    """Score the model on held-out (speaker × condition): WER (faithfulness guard), spk-sim,
    and HF-energy recovery (top octave). Degradation is fixed-seed per (speaker, condition)."""
    from voice_cleanup.eval.speaker_sim import speaker_similarity
    rows = []
    for spk, clean in eval_clips:
        if spk not in ref_cache and wer:
            from voice_cleanup.eval.wer import transcribe
            ref_cache[spk] = transcribe(clean, sr=SR)
        hf_clean = hf_energy_ratio(clean, SR)
        for cname, chain in conds.items():
            seed = zlib.crc32(f"{spk}|{cname}".encode()) % (2**31)
            deg = chain.apply(clean, SR, seed=seed)
            deg = normalize_lufs(deg.astype("float32"), SR)
            out = _enhance(model, deg, device)
            key = f"{spk.replace(':', '-')}__{cname}"
            if write_audio:
                sf.write(str(pack / f"{key}__0_clean.wav"), normalize_lufs(clean, SR), SR)
                sf.write(str(pack / f"{key}__1_degraded.wav"), deg, SR)
                sf.write(str(pack / f"{key}__{tag}.wav"), normalize_lufs(out, SR), SR)
            spk_sim = float(speaker_similarity(clean, out, sr=SR))
            hf_deg, hf_out = hf_energy_ratio(deg, SR), hf_energy_ratio(out, SR)
            w = float("nan")
            if wer:
                from voice_cleanup.eval.wer import whisper_wer
                w = float(whisper_wer(clean, out, sr=SR, ref_text=ref_cache[spk])["wer"])
            rows.append(dict(speaker=spk, condition=cname, wer=w, spk_sim=spk_sim,
                             hf_clean=hf_clean, hf_deg=hf_deg, hf_out=hf_out,
                             hf_recovery=hf_out - hf_deg))
            log.info("  [%s] %-22s WER=%s spk=%.3f HF: deg%.1f→out%.1f (clean%.1f) Δ%+.1f",
                     tag, key, f"{w:.3f}" if wer else "—", spk_sim,
                     hf_deg, hf_out, hf_clean, hf_out - hf_deg)
    return rows


def _curve_point(step, val_loss, rows) -> dict:
    import statistics as st
    def _m(k): return float(st.mean(r[k] for r in rows)) if rows else float("nan")
    return dict(step=step, val_loss=val_loss, mean_wer=_m("wer"), mean_spk=_m("spk_sim"),
                mean_hf_out=_m("hf_out"), mean_hf_recovery=_m("hf_recovery"),
                per_cond={f"{r['speaker']}/{r['condition']}":
                          dict(wer=r["wer"], hf_out=r["hf_out"], hf_rec=r["hf_recovery"])
                          for r in rows})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--crop-seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-clips", type=int, default=4)
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--val-windows", type=int, default=48)
    ap.add_argument("--no-wer", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="VRAM cap fraction. Omit for full-card AFK runs (default).")
    ap.add_argument("--max-vctk", type=int, default=None, help="cap VCTK clips (smoke/debug)")
    ap.add_argument("--max-hifi", type=int, default=None, help="cap Hi-Fi clips (smoke/debug)")
    ap.add_argument("--drop-list", type=Path,
                    default=REPO / "runs" / "quality-gate" / "drop_list.json",
                    help="quality-gate clips to exclude (skipped if missing)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: few clips, 30 steps, one eval — catches bugs cheap")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    if args.smoke:
        args.max_vctk, args.max_hifi = 60, 60
        args.steps, args.eval_every, args.val_windows = 30, 0, 8
        args.eval_clips, args.epochs = 2, None
        log.info("SMOKE MODE · tiny run to validate the pipeline end-to-end")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _ensure_fake_nvcc()
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT / (f"smoke-{ts}" if args.smoke else ts)
    pack = run_dir / "listening_pack"
    pack.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    log.info("Run dir: %s", run_dir)

    # --- Data ---
    clips = tier1.enumerate_tier1()
    if not clips:
        log.error("No Tier-1 clips found — is data/corpora/{vctk,hifi_tts} populated?")
        return 2
    train_clips, val_clips, held = tier1.split_by_speaker(clips, seed=args.seed)
    if args.drop_list and args.drop_list.is_file():
        dropped = set(json.loads(args.drop_list.read_text()))
        before = len(train_clips)
        train_clips = [c for c in train_clips if str(c.path) not in dropped]
        log.info("Quality gate · dropped %d/%d train clips (drop-list %s)",
                 before - len(train_clips), before, args.drop_list)
    if args.max_vctk is not None or args.max_hifi is not None:
        v = [c for c in train_clips if c.corpus == "vctk"][: args.max_vctk or 10**9]
        h = [c for c in train_clips if c.corpus == "hifi"][: args.max_hifi or 10**9]
        train_clips = v + h
        log.info("Capped train clips → %d (vctk=%d hifi=%d)", len(train_clips), len(v), len(h))
    ds = tier1.SyntheticPairDataset(train_clips, crop_seconds=args.crop_seconds)

    def _seed_worker(wid: int) -> None:
        s = (torch.initial_seed() + wid) % 2**32
        np.random.seed(s)
        random.seed(s)

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        generator=gen, worker_init_fn=_seed_worker)
    eval_clips = tier1.build_eval_clips(val_clips, n_speakers=args.eval_clips,
                                        target_seconds=args.eval_seconds)
    conds = eval_conditions()

    steps_per_epoch = max(1, len(loader))
    total_steps = round(args.epochs * steps_per_epoch) if args.epochs else args.steps
    log.info("Schedule · %d steps (%.2f epochs · %d steps/epoch · batch %d)",
             total_steps, total_steps / steps_per_epoch, steps_per_epoch, args.batch_size)

    # --- Engine (from base) ---
    engine = _build_engine(args.lr, total_steps, args.batch_size, args.num_workers,
                           args.gpu_mem_fraction)
    engine._ckpt_dir = run_dir / "ds" / "G"
    model = engine.module
    do_wer = not args.no_wer
    val_x, val_y = _build_val_loss_set(val_clips, args.crop_seconds, args.val_windows, device,
                                       seed=args.seed + 1)

    # --- Baseline (zero-shot base) ---
    log.info("=== BASELINE (zero-shot base) held-out eval ===")
    ref_cache: dict = {}
    base_vl = _val_loss(engine, val_x, val_y, args.batch_size)
    base_rows = _perceptual_eval(model, device, eval_clips, conds, pack, "2_base",
                                 wer=do_wer, ref_cache=ref_cache, write_audio=True)
    curve = [_curve_point(0, base_vl, base_rows)]
    log.info("curve · step 0 (base) · val_loss=%.4f · WER=%.3f · HF out=%.1f rec=%+.1f",
             base_vl, curve[0]["mean_wer"], curve[0]["mean_hf_out"], curve[0]["mean_hf_recovery"])
    best = {"step": 0, "val_loss": base_vl}

    def _save(tag: str) -> None:
        try:
            engine.save_checkpoint(tag=tag)
        except Exception as e:  # noqa: BLE001
            log.warning("Checkpoint save (tag=%s) failed: %s", tag, e)

    def _checkpoint_eval(step, *, write_audio, tag_audio):
        torch.cuda.empty_cache()
        vl = _val_loss(engine, val_x, val_y, args.batch_size)
        rows = _perceptual_eval(model, device, eval_clips, conds, pack, tag_audio,
                                wer=do_wer, ref_cache=ref_cache, write_audio=write_audio)
        pt = _curve_point(step, vl, rows)
        curve.append(pt)
        (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))
        log.info("curve · step %5d · val_loss=%.4f · WER=%.3f · HF out=%.1f rec=%+.1f spk=%.3f",
                 step, vl, pt["mean_wer"], pt["mean_hf_out"], pt["mean_hf_recovery"],
                 pt["mean_spk"])
        _save("last")
        if vl < best["val_loss"]:
            best.update(step=step, val_loss=vl)
            _save("best")
            log.info("  ↳ new BEST (step %d, val_loss %.4f) — saved tag=best", step, vl)
        engine.train()
        torch.cuda.empty_cache()
        return rows

    # --- Train ---
    log.info("=== Fine-tuning CFM from base · %d steps · batch=%d · %.1fs · lr=%.1e ===",
             total_steps, args.batch_size, args.crop_seconds, args.lr)
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
        engine(x, y, None)
        losses = engine.gather_attribute("losses")
        loss = sum(losses.values())
        engine.backward(loss)
        engine.step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == total_steps:
            rate = step / (time.time() - t0)
            log.info("step %5d/%d · loss=%.4f · %.1f it/s · gpu=%.2f GB",
                     step, total_steps, float(loss.detach()), rate,
                     torch.cuda.memory_allocated() / 1e9)
        if args.eval_every and step % args.eval_every == 0 and step != total_steps:
            log.info("=== held-out eval @ step %d/%d ===", step, total_steps)
            _checkpoint_eval(step, write_audio=False, tag_audio="3_finetuned")

    log.info("Training done in %.1f min · peak gpu %.2f GB",
             (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 1e9)

    # --- Final eval ---
    log.info("=== FINAL FINE-TUNED held-out eval ===")
    ft_rows = _checkpoint_eval(total_steps, write_audio=True, tag_audio="3_finetuned")
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))

    summary = []
    log.info("=== HELD-OUT: base → fine-tuned ===")
    for b, f in zip(base_rows, ft_rows, strict=True):
        log.info("%-30s WER %s→%s  HF_out %.1f→%.1f  spk %.3f→%.3f",
                 f"{b['speaker']}/{b['condition']}",
                 f"{b['wer']:.3f}" if do_wer else "—", f"{f['wer']:.3f}" if do_wer else "—",
                 b["hf_out"], f["hf_out"], b["spk_sim"], f["spk_sim"])
        summary.append(dict(speaker=b["speaker"], condition=b["condition"],
                            wer_base=b["wer"], wer_ft=f["wer"],
                            hf_base=b["hf_out"], hf_ft=f["hf_out"],
                            spk_base=b["spk_sim"], spk_ft=f["spk_sim"]))
    safe_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (run_dir / "results.json").write_text(json.dumps(
        dict(args=safe_args, total_steps=total_steps, best=best,
             held_out_speakers=held, held_out=summary), indent=2))
    log.info("A/B in %s · best=tag=best (step %d, val_loss %.4f). Metrics are PROXIES — "
             "the ear is the verdict.", pack, best["step"], best["val_loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
