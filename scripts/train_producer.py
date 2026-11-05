"""Train the Producer: cleanraw → produced (the mastering + de-breathing stage).

Small from-scratch spectral residual-mask net (src/voice_cleanup/models/producer.py) on the
sample-aligned DAPS studio pairs. Primary loss = L1 on log-magnitude (the actual learning
target); plus a multi-resolution STFT loss on the waveform. Held-out by SPEAKER (same f8/f10/
m8/m10 as the Restorer), so we measure generalisation to a new voice.

Eval (the target EXISTS and the transform is a gentle mask, so intrusive metrics are honest here,
unlike the generative Restorer): log-mag L1 (primary), Whisper-WER (faithfulness guard — mastering
must not change words), speaker-similarity, and measured loudness/crest deltas to confirm it
learned the compression + loudness lift. Renders a cleanraw / produced / producer A/B pack.

  uv run python scripts/train_producer.py --epochs 30 --batch-size 8 --eval-every 500
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAPS = REPO / "data" / "corpora" / "daps" / "daps"
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(REPO / "src"))
from voice_cleanup.eval._cuda_libs import ensure_cuda_libpath  # noqa: E402

ensure_cuda_libpath()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader  # noqa: E402

torch.set_num_threads(8)

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.daps import (  # noqa: E402
    HELD_OUT_SPEAKERS,
    SR,
    DAPSPairDataset,
    enumerate_studio_pairs,
    load_aligned_speech_window,
    split_studio_by_speaker,
)
from voice_cleanup.losses.spectral import multi_resolution_stft_loss  # noqa: E402
from voice_cleanup.models.producer import SpectralProducer, target_logmag  # noqa: E402

setup_logging()
log = logging.getLogger("train-producer")


def _crest_db(x: np.ndarray) -> float:
    peak = 20 * np.log10(np.max(np.abs(x)) + 1e-9)
    rms = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)
    return float(peak - rms)


@torch.no_grad()
def _evaluate(model, device, eval_pairs, pack: Path, *, eval_seconds: float, wer: bool,
              ref_cache: dict, write_audio: bool) -> dict:
    import pyloudnorm as pyln

    from voice_cleanup.eval.speaker_sim import speaker_similarity
    meter = pyln.Meter(SR)
    model.eval()
    rows = []
    for p in eval_pairs:
        # p.degraded = cleanraw (input), p.target = produced. Same 30s sentence window from both
        # (energy picked on `produced`, the louder/cleaner ref).
        produced, cleanraw = load_aligned_speech_window(p.target, p.degraded, eval_seconds)
        out = model.enhance(torch.from_numpy(cleanraw).to(device)).cpu().numpy()
        out = out[: len(produced)]
        produced = produced[: len(out)]
        cleanraw = cleanraw[: len(out)]
        # log-mag L1 (primary)
        t = torch.from_numpy(out).unsqueeze(0).to(device)
        g = torch.from_numpy(produced).unsqueeze(0).to(device)
        lm = float((target_logmag(model, t) - target_logmag(model, g)).abs().mean())
        spk = float(speaker_similarity(produced, out, sr=SR))
        w = float("nan")
        if wer:
            from voice_cleanup.eval.wer import transcribe, whisper_wer
            key = f"{p.speaker}-produced"
            if key not in ref_cache:
                ref_cache[key] = transcribe(produced, sr=SR)
            w = float(whisper_wer(produced, out, sr=SR, ref_text=ref_cache[key])["wer"])
        d_lufs = float(meter.integrated_loudness(out) - meter.integrated_loudness(cleanraw))
        d_crest = _crest_db(out) - _crest_db(cleanraw)
        key = f"{p.speaker}-{p.script}"
        if write_audio:
            sf.write(str(pack / f"{key}__0_cleanraw_in.wav"), cleanraw, SR)
            sf.write(str(pack / f"{key}__1_produced_target.wav"), produced, SR)
            sf.write(str(pack / f"{key}__2_producer_out.wav"), out, SR)
        rows.append(dict(clip=key, logmag_l1=lm, wer=w, spk_sim=spk,
                         d_lufs=d_lufs, d_crest=d_crest))
        log.info("  %-14s logmagL1=%.3f WER=%s spk=%.3f  ΔLUFS=%+.1f Δcrest=%+.1f",
                 key, lm, f"{w:.3f}" if wer else "—", spk, d_lufs, d_crest)
    agg = {k: float(np.nanmean([r[k] for r in rows]))
           for k in ("logmag_l1", "wer", "spk_sim", "d_lufs", "d_crest")}
    model.train()
    return {"mean": agg, "per_clip": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=None, help="override: total steps (else epochs)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-seconds", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mrstft-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--input-version", default="cleanraw")
    ap.add_argument("--target-version", default="produced")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="cap VRAM (e.g. 0.6) to leave headroom on the interactive workstation")
    ap.add_argument("--gpu-duty", type=float, default=1.0,
                    help="target GPU duty cycle in (0,1]; <1 sleeps per step to hold util down "
                         "(0.5 ≈ half util) so the display stays smooth while training in bg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        log.info("VRAM capped at %.0f%% of the 4090", args.gpu_mem_fraction * 100)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out or (REPO / "runs" / "producer" / ts)
    pack = run_dir / "listening_pack"
    pack.mkdir(parents=True, exist_ok=True)
    log.info("Run dir: %s", run_dir)

    train_pairs, _ = split_studio_by_speaker(
        DAPS, input_version=args.input_version, target_version=args.target_version)
    eval_pairs = [p for p in enumerate_studio_pairs(
        DAPS, speakers=HELD_OUT_SPEAKERS,
        input_version=args.input_version, target_version=args.target_version)
        if p.script == "script1"]
    ds = DAPSPairDataset(train_pairs, crop_seconds=args.crop_seconds, normalize="none")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    steps_per_epoch = len(dl)
    total_steps = args.steps or args.epochs * steps_per_epoch
    log.info("Schedule · %d steps (%.2f epochs · %d steps/epoch · batch %d · crop %.0fs)",
             total_steps, total_steps / steps_per_epoch, steps_per_epoch,
             args.batch_size, args.crop_seconds)

    model = SpectralProducer().to(device)
    log.info("SpectralProducer · %.2fM params · identity at init", model.num_params() / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ref_cache: dict = {}
    best = float("inf")

    def _checkpoint_eval(step: int, *, write_audio: bool):
        nonlocal best
        torch.cuda.empty_cache()
        log.info("── eval @ step %d ─────────────────────────────", step)
        res = _evaluate(model, device, eval_pairs, pack, eval_seconds=args.eval_seconds,
                        wer=True, ref_cache=ref_cache, write_audio=write_audio)
        m = res["mean"]
        log.info("eval · logmagL1=%.3f WER=%.3f spk=%.3f ΔLUFS=%+.1f Δcrest=%+.1f",
                 m["logmag_l1"], m["wer"], m["spk_sim"], m["d_lufs"], m["d_crest"])
        torch.save({"model": model.state_dict(), "step": step, "mean": m},
                   run_dir / "producer_last.pt")
        if m["logmag_l1"] < best:
            best = m["logmag_l1"]
            torch.save({"model": model.state_dict(), "step": step, "mean": m},
                       run_dir / "producer_best.pt")
            log.info("  ✓ new best (logmagL1=%.3f) → producer_best.pt", best)

    model.train()
    step = 0
    progress = Progress(
        TextColumn("[bold blue]producer"), BarColumn(),
        TextColumn("{task.completed}/{task.total} steps"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeElapsedColumn(), TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("train", total=total_steps, loss=float("nan"))
        done = False
        while not done:
            for cleanraw, produced in dl:
                t0 = time.perf_counter()
                cleanraw, produced = cleanraw.to(device), produced.to(device)
                out = model(cleanraw)
                tgt_lm = target_logmag(model, produced)
                loss_lm = (out["out_logmag"] - tgt_lm).abs().mean()
                loss_wav = multi_resolution_stft_loss(out["wav_out"], produced)
                loss = loss_lm + args.mrstft_weight * loss_wav
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                # GPU duty-cycle throttle: sleep proportional to step time so util stays near
                # `--gpu-duty` — protects the interactive display when training while Aviv works.
                if args.gpu_duty < 1.0:
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    time.sleep(dt * (1.0 / args.gpu_duty - 1.0))
                step += 1
                progress.update(task, advance=1, loss=float(loss))
                if step % 50 == 0:
                    log.info("step %d · loss=%.4f (logmag=%.4f mrstft=%.4f)",
                             step, float(loss), float(loss_lm), float(loss_wav))
                if step % args.eval_every == 0 or step == total_steps:
                    _checkpoint_eval(step, write_audio=(step == total_steps))
                if step >= total_steps:
                    done = True
                    break

    log.info("Done · best logmagL1=%.3f · checkpoints + pack in %s", best, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
