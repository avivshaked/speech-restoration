"""Fine-tune feasibility spike — can Resemble Enhance take ONE gradient step?

This is a DERISK, not training. It answers the binary question that gates the entire
Path-A fine-tune plan: does RE's DeepSpeed engine initialize, load the pretrained
stage-2 checkpoint, run a forward + flow-matching/vocoder loss, backward, and optimizer
step — on the local RTX 4090 (24 GB) — without blowing up on memory or DeepSpeed setup?

It is deliberately SURGICAL: it feeds the model a synthetic random batch shaped exactly
like the dataset's output, so it tests the *engine + model + step* mechanics in isolation
from the data pipeline (sox / parselmouth / RIR deps), which is separate plumbing.

It does NOT save a checkpoint (so the vendored release in model_repo is never mutated)
and it does NOT claim anything about quality — one step tells us "the trainer runs,"
nothing more. The artefact/transparency question (does fine-tuning actually move the
synthetic-resynthesis texture Aviv heard) is a separate, larger Phase-2 spike that needs
real paired data and many steps.

Key finding this spike sits on top of: RE's native trainer (enhancer/train.py feed_G,
cfm mode) synthesizes its degraded input at runtime from clean fg + noise bg + RE's own
Distorter — i.e. it trains on SYNTHETIC degradation, the exact realism gap we found makes
RE hallucinate. So a real fine-tune will also need the data path reworked to feed real
degraded<->clean pairs. This spike does not address that; it only derisks the mechanics.

Run (once a GPU + the vendored external/resemble-enhance are present):
    uv run python scripts/spike_re_finetune_step.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RE_ROOT = REPO / "external" / "resemble-enhance"
STAGE2 = RE_ROOT / "model_repo" / "enhancer_stage2"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(RE_ROOT))

from voice_cleanup._logging import setup_logging  # noqa: E402

setup_logging()
log = logging.getLogger("re-spike")


def main() -> int:
    if not torch.cuda.is_available():
        log.error("No CUDA device — RE training is DeepSpeed/GPU only. Aborting spike.")
        return 2
    if not STAGE2.is_dir():
        log.error("Vendored RE stage-2 not found at %s — run scripts/fetch_* first.", STAGE2)
        return 2

    # Imports from the vendored package (after sys.path insert).
    import dataclasses

    from deepspeed import DeepSpeedConfig
    from resemble_enhance.data import mix_fg_bg
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.utils import Engine

    log.info("Loading hparams from %s", STAGE2)
    hp = HParams.load(STAGE2)
    # HParams is a frozen dataclass — replace() to get a tiny/deterministic variant.
    # GAN already off (gan_training_start_step null in stage2).
    hp = dataclasses.replace(hp, batch_size_per_gpu=1, nj=1, praat_augment_prob=0.0)
    log.info(
        "Config: mode=%s wav_rate=%d training_seconds=%.1f bs=%d gan_start=%s",
        hp.lcfm_training_mode, hp.wav_rate, hp.training_seconds,
        hp.batch_size_per_gpu, hp.gan_training_start_step,
    )

    log.info("Building Enhancer + DeepSpeed engine and loading pretrained G ...")
    t0 = time.time()
    # Build the engine ourselves (instead of enhancer.train.load_G) so we can force
    # DeepSpeed to use plain torch Adam: its default maps "Adam" -> FusedAdam, which
    # JIT-compiles a CUDA op we can't build (no CUDA toolkit / nvcc here). torch_adam
    # needs no compiled op and is fine for a single-GPU feasibility step.
    ds_cfg = hp.deepspeed_config
    ds_cfg["optimizer"]["params"]["torch_adam"] = True
    ds_cfg["optimizer"]["params"]["adam_w_mode"] = False  # plain Adam, matches RE intent
    model = Enhancer(hp)
    # ckpt_dir=STAGE2/ds/G so load_checkpoint() resumes the pretrained weights.
    # We NEVER call save_checkpoint, so the vendored release on disk stays read-only.
    engine_G = Engine(
        model=model,
        config_class=DeepSpeedConfig(ds_cfg),
        ckpt_dir=STAGE2 / "ds" / "G",
    )
    engine_G.load_checkpoint()
    log.info("Engine ready in %.1fs · global_step=%d", time.time() - t0, engine_G.global_step)
    log.info("CUDA mem after load: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    # --- Build a synthetic batch shaped like Dataset.collate_fn output ----------
    # cfm-mode feed_G uses fg_wavs (clean target), fg_dwavs + bg_dwavs (distorted),
    # mixes the two distorted streams, and calls engine(mx, fg_wavs, fg_dwavs).
    device = "cuda"
    n = int(hp.training_seconds * hp.wav_rate)
    bs = hp.batch_size_per_gpu

    def _wavish(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        t = torch.arange(n) / hp.wav_rate
        base = 0.4 * torch.sin(2 * torch.pi * 180 * t) + 0.2 * torch.sin(2 * torch.pi * 900 * t)
        noise = 0.02 * torch.randn(n, generator=g)
        w = (base + noise).repeat(bs, 1)
        return (w / w.abs().amax()).to(device)

    fg_wavs = _wavish(1)
    fg_dwavs = _wavish(2)
    bg_dwavs = _wavish(3)
    log.info("Synthetic batch: fg/fg_d/bg_d each %s on %s", tuple(fg_wavs.shape), device)

    # --- One generator step (mirrors enhancer/train.py feed_G, cfm branch) -------
    engine_G.train()
    step0 = engine_G.global_step
    log.info("Running ONE generator step ...")
    t1 = time.time()

    mx_dwavs = mix_fg_bg(fg_dwavs, bg_dwavs)
    _pred = engine_G(mx_dwavs, fg_wavs, fg_dwavs)
    losses = engine_G.gather_attribute("losses")
    loss_G = sum(losses.values())

    log.info("Forward OK · losses: %s", {k: round(float(v), 4) for k, v in losses.items()})
    if torch.isnan(loss_G):
        log.error("Loss is NaN — forward produced no usable gradient signal.")
        return 1

    engine_G.backward(loss_G)
    engine_G.step()

    dt = time.time() - t1
    step1 = engine_G.global_step
    grad_norm = engine_G.get_grad_norm() or 0.0
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info("Backward + step OK in %.2fs", dt)
    log.info("global_step %d -> %d · loss=%.4f · grad_norm=%.3f · peak_mem=%.2f GB",
             step0, step1, float(loss_G), float(grad_norm), peak)

    if step1 <= step0:
        log.error("global_step did not advance — optimizer.step() was a no-op.")
        return 1

    log.info("SPIKE PASS ✅ — RE took a real gradient step on this GPU. "
             "Mechanics are viable; next gate is the real-paired-data feed (Phase 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
