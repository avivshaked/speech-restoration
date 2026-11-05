"""Phase-2 feasibility spike — does fine-tuning RE on REAL pairs move anything?

Phase 1 proved RE's trainer takes a gradient step. This goes one step further: can the
CFM latent generator actually LEARN from real paired (degraded, clean) data, and does
that move the hallucination / faithfulness on real degradation? It is still a SPIKE, not
a training run — a tiny overfit on a handful of DAPS pairs, with a before/after relisten.

Design decisions (grounded in the RE code, see HANDOFF / learning log 2026-05-29):
- cfm training mode trains the CFM latent generator (lcfm/cfm/l1); the LCFM returns None
  in this mode so the VOCODER never runs / never trains. So this targets content
  FAITHFULNESS (a latent problem), NOT the vocoder resynthesis texture.
- z=None in forward(x, y, z): the denoiser computes no loss and is left as a fixed
  front-end, so ONLY the CFM is fine-tuned (isolates the hallucination culprit).
- Real DAPS device recordings can be latency-offset vs the clean reference; we
  cross-correlate-align each pair before cropping (matters for a mel-domain target).
- DeepSpeed forced to plain torch Adam + stub nvcc (no CUDA toolkit here) — see Phase 1.
- before/after inference uses the SAME in-memory model object (capture baseline, then
  fine-tune, then capture again) via RE's own inference() util — a true A/B.

Honest scope: tiny overfit, single seed, your EAR is the verdict (WER is the faithfulness
guard, NISQA is a predicted-MOS proxy — Claude can't hear any of it).

Run:  uv run python scripts/spike_re_finetune_overfit.py [--steps N] [--lr LR]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"
FAKE_CUDA = REPO / "external" / "fake-cuda"
DAPS = REPO / "data" / "corpora" / "daps" / "daps"
OUT = REPO / "runs" / "re-finetune-spike"

# DeepSpeed's import-time op probe needs CUDA_HOME + a (stub) nvcc. Set BEFORE torch
# import so torch.utils.cpp_extension picks it up. We never actually compile.
os.environ.setdefault("CUDA_HOME", str(FAKE_CUDA))
# Cap CPU thread pools BEFORE torch import. The real work (RE + Whisper) is on the GPU;
# torch's default pool (~93 threads on this WSL box) just oversubscribes the cores —
# it melts the CPU AND thrashes Whisper to ~38 s/clip. A small cap fixes both.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

torch.set_num_threads(8)

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(RE_ROOT))

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402

setup_logging()
log = logging.getLogger("re-ft-spike")
SR = 44100


def _ensure_fake_nvcc() -> None:
    nvcc = FAKE_CUDA / "bin" / "nvcc"
    if nvcc.exists():
        return
    nvcc.parent.mkdir(parents=True, exist_ok=True)
    nvcc.write_text(
        '#!/usr/bin/env bash\n'
        'echo "nvcc: NVIDIA (R) Cuda compiler driver"\n'
        'echo "Cuda compilation tools, release 13.0, V13.0.88"\n'
    )
    nvcc.chmod(0o755)


def _load(path: Path, seconds: float | None = None) -> np.ndarray:
    a, sr = sf.read(str(path))
    a = a.astype("float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        t = torch.from_numpy(a).unsqueeze(0)
        a = torchaudio.functional.resample(t, sr, SR).squeeze(0).numpy()
    if seconds is not None:
        a = a[: int(SR * seconds)]
    return normalize_lufs(a, SR)


def _align(deg: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """DISPROVEN (2026-05-29) — kept only as the reproducer for diag_align_pairs.py.

    Coarsely shift `deg` to line up with `clean` via decimated cross-correlation. This was
    a mistake: DAPS pairs are ALREADY aligned, and raw-waveform xcorr between a clean and a
    noisy/reverberant signal finds spurious ~2 s lags (corr ~0.02), so this INTRODUCED a
    ~2 s misalignment and collapsed the first fine-tune. The training path no longer calls
    it; use the pairs as-shipped. See diag_align_pairs.py and the decisions log.
    """
    d = 50  # decimation for a cheap coarse lag estimate
    a = deg[::d] - deg[::d].mean()
    b = clean[::d] - clean[::d].mean()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = np.correlate(a, b, mode="full")
    lag = (np.argmax(corr) - (n - 1)) * d  # samples: deg leads clean by `lag`
    if lag > 0:
        deg = deg[lag:]
    elif lag < 0:
        deg = np.pad(deg, (-lag, 0))
    # match length to clean
    deg = np.pad(deg, (0, len(clean) - len(deg))) if len(deg) < len(clean) else deg[: len(clean)]
    return deg.astype("float32")


def _build_engine(lr: float, total_steps: int):
    import dataclasses

    from deepspeed import DeepSpeedConfig
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.utils import Engine

    hp = HParams.load(STAGE2)
    # Flat fine-tune LR (min==max => WarmupDecayLR is constant); tiny batch.
    hp = dataclasses.replace(
        hp, batch_size_per_gpu=1, nj=1, praat_augment_prob=0.0,
        min_lr=lr, max_lr=lr, warmup_steps=1, max_steps=total_steps,
    )
    ds_cfg = hp.deepspeed_config
    ds_cfg["optimizer"]["params"]["torch_adam"] = True
    ds_cfg["optimizer"]["params"]["adam_w_mode"] = False
    model = Enhancer(hp)
    engine = Engine(model=model, config_class=DeepSpeedConfig(ds_cfg), ckpt_dir=STAGE2 / "ds" / "G")
    engine.load_checkpoint()
    # HARD GPU GUARD: the previous detached run silently ran on CPU and pegged 8 cores.
    # Fail loudly here rather than grind for hours on CPU.
    dev = next(engine.module.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError(f"Engine landed on {dev}, not CUDA — aborting (would grind on CPU).")
    log.info("GPU guard OK · model on %s · mem=%.2f GB", dev, torch.cuda.memory_allocated() / 1e9)
    return engine


@torch.inference_mode()
def _enhance_inmem(model, deg: np.ndarray, device) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.eval()
    # tau=0 => deterministic CFM prior, so before/after WER reflects the fine-tune, not
    # sampling noise (the confound flagged in the first run).
    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().numpy().astype("float32")
    if out_sr != SR:
        out = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), out_sr, SR
        ).squeeze(0).numpy()
    return out


def _wer(clean: np.ndarray, hyp: np.ndarray, ref_text) -> float:
    from voice_cleanup.eval.wer import whisper_wer

    return float(whisper_wer(clean, hyp, sr=SR, ref_text=ref_text)["wer"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop-seconds", type=float, default=3.0)
    ap.add_argument("--eval-seconds", type=float, default=12.0)
    ap.add_argument("--wer", action="store_true",
                    help="Also run the Whisper WER faithfulness guard (CPU-heavy; off by "
                         "default — render audio on the GPU, run WER separately if wanted).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    _ensure_fake_nvcc()
    OUT.mkdir(parents=True, exist_ok=True)
    pack = OUT / "listening_pack"
    pack.mkdir(exist_ok=True)
    device = "cuda"

    # --- Real DAPS pairs: train on f1+m3, hold out f8; typical + hard condition --------
    conditions = ["ipad_office1", "iphone_balcony1"]
    train_spk = ["f1_script1", "m3_script1"]
    # Explicit 3-clip eval set (fewer Whisper calls): seen-hard (learns?), held-out-hard
    # (generalizes?), seen-easy (damage?). 3 clips x 3 transcriptions = 9 Whisper calls.
    eval_pairs = [
        ("f1_script1", "iphone_balcony1"),  # SEEN, hard — does it learn?
        ("f8_script1", "iphone_balcony1"),  # HELD-OUT, hard — does it generalize?
        ("f1_script1", "ipad_office1"),     # SEEN, easy — does it damage a good case?
    ]

    def pair(spk: str, cond: str, seconds: float | None):
        cf = DAPS / "clean" / f"{spk}_clean.wav"
        df = DAPS / cond / f"{spk}_{cond}.wav"
        if not (cf.is_file() and df.is_file()):
            return None
        clean = _load(cf, seconds)
        deg = _load(df, seconds)
        # DAPS ships device recordings ALREADY time-aligned to the clean reference (same
        # length, envelope corr ~0.85 at zero lag — verified 2026-05-29). Both are cropped
        # from sample 0 by _load, so they stay frame-aligned. We deliberately do NOT
        # cross-correlate-shift: raw-waveform xcorr between a clean and a noisy/reverberant
        # signal latched onto spurious ~2 s lags and BROKE the alignment, which collapsed
        # the first fine-tune (degraded ended up starting ~2 s into the clip). See
        # diag_align_pairs.py + decisions log 2026-05-29.
        if len(deg) != len(clean):
            m = min(len(deg), len(clean))
            deg, clean = deg[:m], clean[:m]
        return deg, clean

    log.info("Loading + aligning training pairs (%d-s crops) ...", args.crop_seconds)
    train = []
    for spk in train_spk:
        for cond in conditions:
            p = pair(spk, cond, args.crop_seconds)
            if p:
                train.append((f"{spk}|{cond}", *p))
    log.info("Train pairs: %s", [t[0] for t in train])
    if not train:
        log.error("No training pairs found under %s", DAPS)
        return 2

    log.info("Loading eval clips (%d-s) ...", args.eval_seconds)
    evals = []
    for spk, cond in eval_pairs:
        p = pair(spk, cond, args.eval_seconds)
        if p:
            seen = spk in train_spk
            evals.append((f"{spk}|{cond}{'' if seen else ' (HELD-OUT)'}", spk, cond, seen, *p))

    # --- Build engine, capture BASELINE inference on eval clips ------------------------
    engine = _build_engine(args.lr, args.steps)
    model = engine.module
    log.info("Engine ready · global_step=%d · mem=%.2f GB",
             engine.global_step, torch.cuda.memory_allocated() / 1e9)

    if args.wer:
        from voice_cleanup.eval.wer import transcribe

    log.info("=== BASELINE inference (pretrained, before fine-tune)%s ===",
             " + WER guard" if args.wer else " (audio only)")
    base_rows = []
    for name, spk, cond, _seen, deg, clean in evals:
        ref = transcribe(clean, sr=SR) if args.wer else None
        before = _enhance_inmem(model, deg, device)
        w = _wer(clean, before, ref) if args.wer else float("nan")
        base_rows.append((name, ref, w))
        key = f"{spk}-{cond}"
        sf.write(str(pack / f"{key}__0_clean.wav"), normalize_lufs(clean, SR), SR)
        sf.write(str(pack / f"{key}__1_degraded.wav"), normalize_lufs(deg, SR), SR)
        sf.write(str(pack / f"{key}__2_RE_baseline.wav"), normalize_lufs(before, SR), SR)
        log.info("BASELINE %-28s WER=%s", name, f"{w:.3f}" if args.wer else "—")

    # --- Fine-tune the CFM on real pairs (z=None => CFM only) --------------------------
    log.info("=== Fine-tuning CFM on %d real pairs for %d steps (lr=%.1e) ===",
             len(train), args.steps, args.lr)
    engine.train()
    xs = torch.from_numpy(np.stack([t[1] for t in train])).to(device)  # degraded
    ys = torch.from_numpy(np.stack([t[2] for t in train])).to(device)  # clean
    n = xs.shape[0]
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (1,)).item()  # one pair per step (tiny overfit)
        x = xs[idx : idx + 1]
        y = ys[idx : idx + 1]
        engine(x, y, None)  # cfm mode: returns None, stashes losses
        losses = engine.gather_attribute("losses")
        loss = sum(losses.values())
        engine.backward(loss)
        engine.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            log.info("step %4d/%d · loss=%.4f · %s · gpu=%.2f GB", step, args.steps,
                     float(loss), {k: round(float(v), 4) for k, v in losses.items()},
                     torch.cuda.memory_allocated() / 1e9)

    # --- Capture FINE-TUNED inference on the same eval clips ---------------------------
    log.info("=== FINE-TUNED inference (after) · peak gpu so far %.2f GB ===",
             torch.cuda.max_memory_allocated() / 1e9)
    ft_rows = []
    for (name, spk, cond, _seen, deg, clean), (_, ref, _) in zip(evals, base_rows, strict=True):
        after = _enhance_inmem(model, deg, device)
        w = _wer(clean, after, ref) if args.wer else float("nan")
        ft_rows.append((name, w))
        sf.write(str(pack / f"{spk}-{cond}__3_RE_finetuned.wav"), normalize_lufs(after, SR), SR)
        log.info("FINETUNED %-28s WER=%s", name, f"{w:.3f}" if args.wer else "—")

    # --- Report -----------------------------------------------------------------------
    summary = []
    if args.wer:
        log.info("=== WER: baseline -> fine-tuned (faithfulness guard) ===")
        for (name, _ref, wb), (_, wf) in zip(base_rows, ft_rows, strict=True):
            delta = wf - wb
            arrow = "↓ better" if delta < -0.02 else ("↑ worse" if delta > 0.02 else "≈ flat")
            log.info("%-30s %.3f -> %.3f  (%s)", name, wb, wf, arrow)
            summary.append(dict(clip=name, wer_baseline=wb, wer_finetuned=wf, delta=delta))
    else:
        log.info("WER guard skipped (--wer off) — render is audio-only; run WER separately.")
        summary = [dict(clip=name, wer_baseline=None, wer_finetuned=None, delta=None)
                   for name, _ref, _ in base_rows]
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"# Phase-2 fine-tune spike A/B ({args.steps} steps, {len(train)} real DAPS pairs)",
        "",
        "tau=0 (deterministic) so before/after reflects the fine-tune, not sampling.",
        "CFM trained only (vocoder untrained). WER = faithfulness guard, NOT quality.",
        "Per clip: 0_clean | 1_degraded | 2_RE_baseline | 3_RE_finetuned (all -23 LUFS).",
        "",
    ]
    if args.wer:
        lines += ["| clip | baseline WER | finetuned WER |", "|------|------|------|"]
        for (name, _ref, wb), (_, wf) in zip(base_rows, ft_rows, strict=True):
            lines.append(f"| {name} | {wb:.3f} | {wf:.3f} |")
    else:
        lines += ["WER not computed (audio-only render). Clips:", ""]
        lines += [f"- {name}" for name, _ref, _ in base_rows]
    (pack / "index.md").write_text("\n".join(lines) + "\n")

    log.info("Audio A/B in %s  (0_clean | 1_degraded | 2_RE_baseline | 3_RE_finetuned)", pack)
    log.info("DONE — your EAR is the verdict; WER is only the faithfulness guard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
