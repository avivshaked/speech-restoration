"""Load the trained weights published at https://huggingface.co/shakeda/speech-restoration-models.

The files are safetensors. They are licensed CC BY-NC 4.0 because the models were trained on
DAPS; see the model card. The restorer is a fine-tune of Resemble Enhance, so it is built from
the released model's hyperparameters: run ``uv run python scripts/fetch_resemble_enhance.py``
once before calling :func:`load_restorer`.

Every call builds a fresh model. Resemble Enhance's own ``load_enhancer`` is cached and its
``inference`` strips weight normalisation from the model it runs, so reusing one instance for
two different weight sets would silently mix them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from voice_cleanup.models.producer import SpectralProducer

logger = logging.getLogger(__name__)


def load_producer(weights: str | Path, device: str = "cpu") -> SpectralProducer:
    """A :class:`SpectralProducer` with published weights, in eval mode on ``device``."""
    model = SpectralProducer()
    model.load_state_dict(load_file(str(weights)), strict=True)
    logger.info("Producer · loaded %s · %.2fM params", Path(weights).name, model.num_params() / 1e6)
    return model.to(device).eval()


def load_restorer(weights: str | Path | None = None, device: str | None = None) -> Any:
    """Resemble Enhance with fine-tuned ``weights`` (safetensors), or off the shelf if None.

    Returns the ``Enhancer`` module in eval mode on ``device`` (CUDA if available).
    """
    from voice_cleanup.eval import resemble_enhance as vendor

    vendor._ensure_vendor()
    stub_keys = vendor._install_deepspeed_stub()
    try:
        from resemble_enhance.enhancer.enhancer import Enhancer
        from resemble_enhance.enhancer.hparams import HParams
        from resemble_enhance.inference import remove_weight_norm_recursively
    finally:
        vendor._remove_deepspeed_stub(stub_keys)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Enhancer(HParams.load(vendor.RUN_DIR))
    if weights is None:
        state = torch.load(vendor.CKPT_PATH, map_location="cpu", weights_only=False)["module"]
        label = "off-the-shelf release"
    else:
        # The fine-tuned weights were saved after an evaluation pass had already removed weight
        # normalisation (Resemble Enhance's inference does this in place), so the model must
        # have it removed too before the keys line up.
        remove_weight_norm_recursively(model)
        state = load_file(str(weights))
        label = Path(weights).name
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Restorer · loaded %s · %.1fM params · device=%s", label, n_params / 1e6, device)
    return model.eval().to(device)
