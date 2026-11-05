"""Fetch Resemble Enhance source + pretrained checkpoint at a pinned SHA.

Resemble Enhance (resemble-ai/resemble-enhance) is the Path-A generative restorer
(LCFM latent flow-matching + UnivNet GAN vocoder, 44.1 kHz, MIT-licensed code AND
weights). Its PyPI release (`resemble-enhance==0.0.1`) pins conflicting deps and
hard-requires `deepspeed`, which won't even *import* on this box (it probes for a
CUDA toolkit / CUDA_HOME at import time to compile ops, and there is none under
WSL — torch ships its own runtime but no nvcc). So — like NISQA and MP-SENet — we
vendor the source at a known-good commit and run inference against it.

Inference never touches DeepSpeed: the checkpoint loads via plain
`torch.load(...)["module"]` and `Engine(DeepSpeedEngine)` is only *defined* (never
instantiated) on the import path. `eval/resemble_enhance.py` therefore injects a
tiny `deepspeed` stub into sys.modules before importing the package — no DeepSpeed
install, no CUDA toolkit. The only genuinely-missing import-time third-party dep is
`omegaconf` (config loading); everything else (torch, torchaudio, numpy, scipy,
librosa, rich, tqdm, pandas, matplotlib, soundfile) is already in the project.

We vendor the whole `resemble_enhance/` package (≈50 small files — the import graph
is interconnected, so cherry-picking is more fragile than vendoring all of it).

Checkpoint: the single `enhancer_stage2` model (`mp_rank_00_model_states.pt`,
~713 MB) bundles BOTH the denoiser and the enhancer — the public `enhance()` path
runs denoise→enhance from this one file. Downloaded from the public, ungated HF
repo `ResembleAI/resemble-enhance` and SHA-256-pinned.

Idempotent — re-runs verify size + SHA-256 and skip anything already present.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
from pathlib import Path

# Allow `python scripts/fetch_resemble_enhance.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Pinned to a known-good commit of resemble-ai/resemble-enhance (HEAD as of
# 2025; the model code + released weights have been stable since the 1.0 release).
RE_PIN_SHA = "8e978149bfe8abab3eb77d965d579a111afdb0ff"
RE_RAW_BASE = f"https://raw.githubusercontent.com/resemble-ai/resemble-enhance/{RE_PIN_SHA}"

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external" / "resemble-enhance"
# Vendored package root: external/resemble-enhance/resemble_enhance/...
PKG_DIR = EXTERNAL_DIR / "resemble_enhance"

# Full `resemble_enhance/` package tree at RE_PIN_SHA (paths relative to the package).
SOURCE_FILES = [
    "__init__.py",
    "common.py",
    "hparams.py",
    "inference.py",
    "melspec.py",
    "data/__init__.py",
    "data/dataset.py",
    "data/utils.py",
    "data/distorter/__init__.py",
    "data/distorter/base.py",
    "data/distorter/custom.py",
    "data/distorter/distorter.py",
    "data/distorter/sox.py",
    "denoiser/__init__.py",
    "denoiser/__main__.py",
    "denoiser/denoiser.py",
    "denoiser/hparams.py",
    "denoiser/inference.py",
    "denoiser/train.py",
    "denoiser/unet.py",
    "enhancer/__init__.py",
    "enhancer/__main__.py",
    "enhancer/download.py",
    "enhancer/enhancer.py",
    "enhancer/hparams.py",
    "enhancer/inference.py",
    "enhancer/train.py",
    "enhancer/lcfm/__init__.py",
    "enhancer/lcfm/cfm.py",
    "enhancer/lcfm/irmae.py",
    "enhancer/lcfm/lcfm.py",
    "enhancer/lcfm/wn.py",
    "enhancer/univnet/__init__.py",
    "enhancer/univnet/amp.py",
    "enhancer/univnet/discriminator.py",
    "enhancer/univnet/lvcnet.py",
    "enhancer/univnet/mrstft.py",
    "enhancer/univnet/univnet.py",
    "enhancer/univnet/alias_free_torch/__init__.py",
    "enhancer/univnet/alias_free_torch/filter.py",
    "enhancer/univnet/alias_free_torch/resample.py",
    "utils/__init__.py",
    "utils/control.py",
    "utils/distributed.py",
    "utils/engine.py",
    "utils/logging.py",
    "utils/train_loop.py",
    "utils/utils.py",
]

# Public, ungated HF repo. The enhancer_stage2 checkpoint bundles denoiser+enhancer.
HF_BASE = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2"
# run_dir layout expected by resemble_enhance.enhancer.download.download():
#   <run_dir>/hparams.yaml, <run_dir>/ds/G/latest, <run_dir>/ds/G/default/mp_rank_00_model_states.pt
CKPT_DIR = EXTERNAL_DIR / "model_repo" / "enhancer_stage2"
CKPT_FILES = {
    "hparams.yaml": None,  # 717 B plain YAML; not worth pinning
    "ds/G/latest": None,   # 7 B pointer file
    # ~713 MB — the model. SHA-256 verified on every re-fetch.
    "ds/G/default/mp_rank_00_model_states.pt": (
        "f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6"
    ),
}


# NumPy-2.x compatibility patches applied to the vendored source after download.
# RE predates NumPy 2.0, where `float(arr)` on a size-1 array raises instead of
# coercing. Each patch is an idempotent (old -> new) string swap; re-fetch re-applies.
COMPAT_PATCHES: list[tuple[str, str, str]] = [
    (
        "enhancer/lcfm/cfm.py",
        "a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0))",
        "a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)[0])",
    ),
]


def _apply_compat_patches() -> None:
    for rel, old, new in COMPAT_PATCHES:
        path = PKG_DIR / rel
        text = path.read_text()
        if new in text:
            logger.info("Resemble Enhance · compat patch already applied: %s", rel)
            continue
        if old not in text:
            raise RuntimeError(
                f"compat patch target not found in {rel}; vendored source may have changed "
                f"(expected to find: {old!r})"
            )
        path.write_text(text.replace(old, new))
        logger.info("Resemble Enhance · applied NumPy-2.x compat patch to %s", rel)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        logger.info(
            "Resemble Enhance · already on disk: %s (%.1f KB)", dest, dest.stat().st_size / 1024
        )
        return
    logger.info("Resemble Enhance · downloading %s → %s", url, dest)
    t0 = time.monotonic()
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - known GitHub raw / HF URL
    logger.info(
        "Resemble Enhance · downloaded %s in %.2fs (%.1f KB on disk)",
        dest.name, time.monotonic() - t0, dest.stat().st_size / 1024,
    )


def fetch_all(*, force: bool = False) -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Resemble Enhance · pinned SHA = %s", RE_PIN_SHA)
    logger.info("Resemble Enhance · target dir = %s", EXTERNAL_DIR)

    for rel in SOURCE_FILES:
        dest = PKG_DIR / rel
        if force and dest.is_file():
            dest.unlink()
        _download(f"{RE_RAW_BASE}/resemble_enhance/{rel}", dest)

    for rel, expected_sha in CKPT_FILES.items():
        dest = CKPT_DIR / rel
        if force and dest.is_file():
            dest.unlink()
        _download(f"{HF_BASE}/{rel}?download=true", dest)
        if expected_sha is not None:
            actual = _sha256(dest)
            if actual != expected_sha:
                raise RuntimeError(
                    f"hash mismatch for {dest}: got {actual}, expected {expected_sha}"
                )
            logger.info("Resemble Enhance · %s SHA-256 OK (%s)", rel, expected_sha[:12])

    _apply_compat_patches()
    logger.info("Resemble Enhance · all files ready under %s", EXTERNAL_DIR)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fetch vendored Resemble Enhance source + checkpoint."
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    args = parser.parse_args(argv)
    fetch_all(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
