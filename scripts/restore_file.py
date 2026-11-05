"""Restore one real-world audio/video file end-to-end and (optionally) A/B it against another
tool's output. The in-the-wild test: no clean reference exists, so the
scoreboard is NON-INTRUSIVE MOS (DNSMOS P.835 + NISQA — reference-free) + the ear, plus a
transcript check that the restoration didn't change the words.

Pipeline: <file> → decode/resample to 44.1k mono → Restorer (RE fine-tuned best, λ=0.5)
→ Producer (natural, breaths kept) → master. Renders an A/B pack, all LUFS-matched for a fair
listen (loudness removed as a confound), plus the master at its native level.

  uv run python scripts/restore_file.py --input "<clip.mp4>" --compare "<other_tool.wav>"
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"
FAKE_CUDA = REPO / "external" / "fake-cuda"

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
from voice_cleanup.data.daps import SR  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402
from voice_cleanup.models.producer import SpectralProducer  # noqa: E402

setup_logging()
log = logging.getLogger("restore-file")

RESTORER = REPO / "runs" / "re-finetune-daps" / "20260530-015903"
PRODUCER_NATURAL = REPO / "runs" / "producer" / "natural-20260530-185900" / "producer_best.pt"


def _decode_to_mono44k(path: Path) -> np.ndarray:
    """Decode any audio/video file to 44.1k mono float32 via the bundled ffmpeg (imageio-ffmpeg)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "audio.wav"
        subprocess.run([ff, "-y", "-v", "error", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", str(SR), "-f", "wav", str(tmp)], check=True)
        x, sr = sf.read(str(tmp), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x.astype("float32")


def _build_restorer(tag: str):
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
    engine.load_checkpoint()
    engine._ckpt_dir = RESTORER / "ds" / "G"
    engine.load_checkpoint(tag=tag)
    log.info("Restorer loaded · fine-tuned tag=%s", tag)
    return engine.module


@torch.no_grad()
def _restore(model, deg: np.ndarray, device) -> np.ndarray:
    from resemble_enhance.inference import inference

    model.eval()
    model.configurate_(nfe=32, solver="midpoint", lambd=0.5, tau=0.0)
    hwav, out_sr = inference(model=model, dwav=torch.from_numpy(deg), sr=SR, device=device)
    out = hwav.detach().cpu().numpy().astype("float32")
    if out_sr != SR:
        out = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), out_sr, SR).squeeze(0).numpy()
    return out


def _metrics(name: str, x: np.ndarray, meter, ref_text: str | None) -> dict:
    import contextlib
    import io

    from voice_cleanup.eval.non_intrusive import dnsmos_p835
    row = {"name": name, "lufs": float(meter.integrated_loudness(x))}
    try:
        d = dnsmos_p835(x, sr=SR)
        row.update(dnsmos_sig=d["sig"], dnsmos_bak=d["bak"], dnsmos_ovrl=d["ovrl"])
    except Exception as e:  # noqa: BLE001
        log.warning("DNSMOS failed for %s: %s", name, e)
    try:
        from voice_cleanup.eval.nisqa import nisqa_multidim
        with contextlib.redirect_stdout(io.StringIO()):
            nq = nisqa_multidim(x, sr=SR)
        row["nisqa_mos"] = nq.get("mos") or nq.get("nisqa_mos") or next(iter(nq.values()))
    except Exception as e:  # noqa: BLE001
        log.warning("NISQA failed for %s: %s", name, e)
    if ref_text is not None:
        try:
            from voice_cleanup.eval.wer import whisper_wer
            # transcribe x (the hypothesis) and compare to the original's transcript (ref_text);
            # the 1st arg is ignored when ref_text is supplied.
            row["wer_vs_orig"] = float(whisper_wer(x, x, sr=SR, ref_text=ref_text)["wer"])
        except Exception as e:  # noqa: BLE001
            log.warning("WER failed for %s: %s", name, e)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="degraded audio/video file")
    ap.add_argument("--compare", type=Path, default=None,
                    help="another tool's output to A/B against")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--match-lufs", type=float, default=-16.0,
                    help="LUFS to normalise every rendered file to for a fair A/B")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.6,
                    help="cap VRAM to protect the interactive display (default 0.6)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "restore-file")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
    pack = args.out
    pack.mkdir(parents=True, exist_ok=True)

    import pyloudnorm as pyln
    meter = pyln.Meter(SR)

    log.info("Decoding %s", args.input.name)
    original = _decode_to_mono44k(args.input)
    log.info("  → %.1f s of audio @ %d Hz mono", len(original) / SR, SR)

    restorer = _build_restorer(args.tag)
    log.info("Restoring (RE fine-tuned, λ=0.5) — this scales with clip length ...")
    restored = _restore(restorer, original, device)

    natural = SpectralProducer().to(device)
    natural.load_state_dict(torch.load(PRODUCER_NATURAL, map_location=device,
                                       weights_only=False)["model"])
    natural.eval()
    with torch.no_grad():
        mastered = natural.enhance(torch.from_numpy(restored).to(device)).cpu().numpy()
    mastered = mastered[: len(restored)]

    signals = {"0_original": original, "1_restored": restored, "2_ours_mastered": mastered}
    if args.compare and args.compare.exists():
        other = _decode_to_mono44k(args.compare)
        signals["3_compare"] = other
        log.info("Comparison file: %.1f s", len(other) / SR)

    # Render every version LUFS-matched (fair quality A/B) + the master at native loudness.
    stem = args.input.stem
    for name, x in signals.items():
        matched = normalize_lufs(x, SR, args.match_lufs)
        sf.write(str(pack / f"{stem}__{name}.wav"), matched, SR)
    sf.write(str(pack / f"{stem}__2b_ours_mastered_native.wav"), mastered, SR)

    # Transcript faithfulness: did either restoration change the words vs the original?
    from voice_cleanup.eval.wer import transcribe
    ref_text = None
    try:
        ref_text = transcribe(original, sr=SR)
        log.info("Original transcript (%d words) captured as the faithfulness reference.",
                 len(ref_text.split()))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not transcribe original: %s", e)

    log.info("── Scoreboard (non-intrusive: higher MOS = better; WER = transcript divergence) ──")
    rows = [_metrics(n, x, meter, ref_text) for n, x in signals.items()]
    hdr = (f"  {'version':<18} {'ovrl':>5} {'sig':>5} {'bak':>5} "
           f"{'NISQA':>6} {'WERΔorig':>9} {'LUFS':>7}")
    log.info(hdr)
    for r in rows:
        log.info("  %-18s %11s %5s %5s %6s %9s %7.1f", r["name"],
                 f"{r.get('dnsmos_ovrl', float('nan')):.2f}",
                 f"{r.get('dnsmos_sig', float('nan')):.2f}",
                 f"{r.get('dnsmos_bak', float('nan')):.2f}",
                 f"{r.get('nisqa_mos', float('nan')):.2f}",
                 f"{r.get('wer_vs_orig', float('nan')):.3f}", r["lufs"])
    import json
    (pack / f"{stem}__scoreboard.json").write_text(json.dumps(rows, indent=2))
    log.info("Pack + scoreboard: %s", pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
