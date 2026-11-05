"""Full pipeline end-to-end: degraded device audio → Restorer (RE fine-tuned) → Producer → master.

Stage 1 (Restorer): RE fine-tuned `best` restores a real degraded device recording toward cleanraw
(breaths kept, 44.1 kHz). Stage 2 (Producer): masters it (EQ/comp/loudness). Renders BOTH producer
variants so the chain is auditioned end-to-end:
  0_device_in → 1_restored (RE) → 3_mastered_natural (breaths) / 4_mastered_cleanup (debreathed)
plus the true 2_cleanraw_ref for comparison.

Per stage we log WER (faithfulness through the chain — a generative restorer is the hallucination
risk, mastering shouldn't add any) + LUFS (confirm the master lands near −16) + spk-sim.

  uv run python scripts/chain_pipeline.py --conditions ipad_office1,iphone_balcony1
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
from voice_cleanup.models.producer import SpectralProducer  # noqa: E402

setup_logging()
log = logging.getLogger("chain")

RESTORER = REPO / "runs" / "re-finetune-daps" / "20260530-015903"
PRODUCERS = {
    "natural": REPO / "runs" / "producer" / "natural-20260530-185900" / "producer_best.pt",
    "cleanup": REPO / "runs" / "producer" / "20260530-122752" / "producer_best.pt",
}


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


def _load_producer(ckpt: Path, device) -> SpectralProducer:
    m = SpectralProducer().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model"])
    m.eval()
    return m


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="best")
    ap.add_argument("--conditions", default="ipad_office1,iphone_livingroom1,iphone_balcony1")
    ap.add_argument("--eval-clips", type=int, default=12)
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--no-wer", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "chain")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("No CUDA — aborting.")
        return 2
    device = "cuda"
    do_wer = not args.no_wer
    conds = args.conditions.split(",")
    pack = args.out
    pack.mkdir(parents=True, exist_ok=True)

    import pyloudnorm as pyln

    from voice_cleanup.eval.speaker_sim import speaker_similarity
    meter = pyln.Meter(SR)
    if do_wer:
        from voice_cleanup.eval.wer import transcribe, whisper_wer

    # device→cleanraw pairs (degraded = real device recording, target = cleanraw studio ref).
    pairs = select_eval_pairs(DAPS, target="cleanraw")[: args.eval_clips]
    pairs = [p for p in pairs if any(c in p.condition for c in conds)]
    log.info("Chaining %d clips · conditions=%s", len(pairs), conds)

    restorer = _build_restorer(args.tag)
    producers = {name: _load_producer(p, device) for name, p in PRODUCERS.items()}

    def _lufs(x):
        return float(meter.integrated_loudness(x))

    rows = []
    for p in pairs:
        key = f"{p.speaker}-{p.condition}"
        cleanraw, device_in = load_aligned_speech_window(p.target, p.degraded, args.eval_seconds)
        restored = _restore(restorer, device_in, device)
        n = min(len(cleanraw), len(device_in), len(restored))
        cleanraw, device_in, restored = cleanraw[:n], device_in[:n], restored[:n]
        outs = {}
        for name, m in producers.items():
            with torch.no_grad():
                o = m.enhance(torch.from_numpy(restored).to(device)).cpu().numpy()
            outs[name] = o[: len(restored)]

        # WER through the chain (ref = clean transcript of the studio cleanraw)
        ref = transcribe(cleanraw, sr=SR) if do_wer else None

        def _wer(x, _ref=cleanraw, _txt=ref):
            if not do_wer:
                return float("nan")
            return float(whisper_wer(_ref, x, sr=SR, ref_text=_txt)["wer"])
        w_dev, w_res = _wer(device_in), _wer(restored)
        w_nat, w_cln = _wer(outs["natural"]), _wer(outs["cleanup"])
        spk = float(speaker_similarity(cleanraw, outs["natural"], sr=SR))

        sf.write(str(pack / f"{key}__0_device_in.wav"), normalize_lufs(device_in, SR), SR)
        sf.write(str(pack / f"{key}__1_restored.wav"), normalize_lufs(restored, SR), SR)
        sf.write(str(pack / f"{key}__2_cleanraw_ref.wav"), normalize_lufs(cleanraw, SR), SR)
        # masters at NATIVE level (no LUFS-norm) so you hear the loudness the master actually set
        sf.write(str(pack / f"{key}__3_mastered_natural.wav"), outs["natural"], SR)
        sf.write(str(pack / f"{key}__4_mastered_cleanup.wav"), outs["cleanup"], SR)

        log.info("%-26s WER device %.3f → restored %.3f → natural %.3f / cleanup %.3f | "
                 "LUFS restored %.1f → natural %.1f | spk %.3f", key,
                 w_dev, w_res, w_nat, w_cln, _lufs(restored), _lufs(outs["natural"]), spk)
        rows.append(dict(clip=key, wer_device=w_dev, wer_restored=w_res,
                         wer_natural=w_nat, wer_cleanup=w_cln,
                         lufs_restored=_lufs(restored), lufs_natural=_lufs(outs["natural"]),
                         spk_natural=spk))

    if rows:
        def _m(k):
            return float(np.nanmean([r[k] for r in rows]))
        means = {f"mean_{k}": _m(k) for k in rows[0] if k != "clip"}
        log.info("══ MEAN · WER device %.3f → restored %.3f → natural %.3f / cleanup %.3f · "
                 "LUFS restored %.1f → natural %.1f · spk %.3f ══",
                 _m("wer_device"), _m("wer_restored"), _m("wer_natural"), _m("wer_cleanup"),
                 _m("lufs_restored"), _m("lufs_natural"), _m("spk_natural"))
        import json
        (pack / "chain_results.json").write_text(
            json.dumps({"conditions": conds, "means": means, "per_clip": rows}, indent=2))
    log.info("Pack: %s", pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
