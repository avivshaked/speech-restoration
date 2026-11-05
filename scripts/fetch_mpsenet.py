"""Fetch MP-SENet source + pretrained checkpoints at a pinned SHA into external/mp-senet/.

MP-SENet (yxlu-0102/MP-SENet on GitHub) is research code. Its PyPI footprint is
nonexistent and `requirements.txt` pins torch==1.8.1 / librosa==0.7.2 etc. that
conflict with this project's locked deps, so — like NISQA — we vendor the source
files we actually import at inference time, at a known-good commit, and add
external/mp-senet to sys.path in voice_cleanup.eval.mpsenet.

We deliberately vendor only the inference path. The released `MPNet` generator
imports `models.transformer` (GRU-FFN transformer blocks), NOT `models.conformer`
(vestigial), so conformer.py + its einops dependency are skipped. discriminator.py
and train.py are training-only and skipped too. The remaining files
(model.py, transformer.py, utils.py, dataset.py, env.py) drag only deps we already
have (torch, numpy, pesq, joblib, librosa, matplotlib) — no new dependency.

Both pretrained checkpoints are vendored: g_best_vb (VoiceBank+DEMAND, the paper's
headline, PESQ 3.50) and g_best_dns (DNS Challenge, the broader real-noise model).
The enhancer defaults to g_best_dns for real podcast audio; see eval/mpsenet.py.

Idempotent — re-runs are cheap (size + SHA-256 checks; no re-download if present).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
from pathlib import Path

# Allow `python scripts/fetch_mpsenet.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Pinned to the current HEAD of yxlu-0102/MP-SENet (2025-05-19). The model code
# and checkpoints have been stable since 2023; recent commits are README-only.
MPSENET_PIN_SHA = "89932cfe90d1dacb8e170e4a331d762462c21792"
MPSENET_RAW_BASE = f"https://raw.githubusercontent.com/yxlu-0102/MP-SENet/{MPSENET_PIN_SHA}"

# Source files imported at inference time (the MPNet generator + STFT helpers).
SOURCE_FILES = [
    "models/__init__.py",  # may be missing upstream; we create an empty stub if so
    "models/model.py",
    "models/transformer.py",
    "utils.py",
    "dataset.py",
    "env.py",
]

# Checkpoints + config. Hashes computed once from the pinned SHA; re-fetches verify
# against them so an upstream re-publish under the same SHA would be caught.
WEIGHTS_FILES = {
    "best_ckpt/config.json": None,  # tiny, plain JSON; not worth pinning a hash
    "best_ckpt/g_best_vb": "aedfb1aa549159f71b39613d94db831dfa983b3a68b0deea02e84e0ad563f4f9",
    "best_ckpt/g_best_dns": "97a77ba67c5c484c65363bb703ea85962f773ca0819e22ce81b4ec33db5e7206",
}

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external" / "mp-senet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        size_kb = dest.stat().st_size / 1024
        logger.info("MP-SENet · already on disk: %s (%.1f KB)", dest, size_kb)
        return
    logger.info("MP-SENet · downloading %s → %s", url, dest)
    t0 = time.monotonic()
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - GitHub raw, known URL
    size_kb = dest.stat().st_size / 1024
    logger.info(
        "MP-SENet · downloaded %s in %.2fs (%.1f KB on disk)",
        dest.name, time.monotonic() - t0, size_kb,
    )


def fetch_all(*, force: bool = False) -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("MP-SENet · pinned SHA = %s", MPSENET_PIN_SHA)
    logger.info("MP-SENet · target dir = %s", EXTERNAL_DIR)

    # Source files
    for rel in SOURCE_FILES:
        url = f"{MPSENET_RAW_BASE}/{rel}"
        dest = EXTERNAL_DIR / rel
        if force and dest.is_file():
            dest.unlink()
        try:
            _download(url, dest)
        except Exception as exc:
            # models/__init__.py does not exist upstream (it's a plain dir); create
            # an empty one so `from models.model import MPNet` resolves as a package.
            if rel.endswith("__init__.py") and "404" in str(exc):
                logger.info("MP-SENet · %s missing upstream, creating empty stub", rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("")
            else:
                raise

    # Checkpoints + config
    for rel, expected_sha in WEIGHTS_FILES.items():
        url = f"{MPSENET_RAW_BASE}/{rel}"
        dest = EXTERNAL_DIR / rel
        if force and dest.is_file():
            dest.unlink()
        _download(url, dest)
        if expected_sha is not None:
            actual = _sha256(dest)
            if actual != expected_sha:
                raise RuntimeError(
                    f"hash mismatch for {dest}: got {actual}, expected {expected_sha}"
                )
            logger.info("MP-SENet · %s SHA-256 OK (%s)", rel, expected_sha[:12])

    logger.info("MP-SENet · all files ready under %s", EXTERNAL_DIR)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch vendored MP-SENet source + checkpoints.")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    args = parser.parse_args(argv)
    fetch_all(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
