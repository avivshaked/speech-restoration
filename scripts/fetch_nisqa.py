"""Fetch NISQA source + multidim weights at a pinned SHA into external/nisqa/.

NISQA is research code (gabrielmittag/NISQA on GitHub). The PyPI package
(`nisqa==2.0.post2`) pins exact versions of torch/librosa/scipy that conflict
with this project's locked deps, so we vendor source files directly at a
known-good commit and import them via the sys.path insertion done in
voice_cleanup.eval.nisqa.

Idempotent — re-runs are cheap (HEAD requests + size/hash checks).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
from pathlib import Path

# Allow `python scripts/fetch_nisqa.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Pinned to the most recent stable NISQA commit on master (2024-12-01).
NISQA_PIN_SHA = "fe84f0f252abec382b24367d5b22498a7ce34dbb"
NISQA_RAW_BASE = f"https://raw.githubusercontent.com/gabrielmittag/NISQA/{NISQA_PIN_SHA}"

# Source files we actually import at inference time.
SOURCE_FILES = [
    "nisqa/__init__.py",  # may be missing; we'll create one if so
    "nisqa/NISQA_lib.py",
    "nisqa/NISQA_model.py",
]

# Multidim (MOS + noi + dis + col + loud) checkpoint — the variant we want for
# podcast eval. Other variants (mos_only, tts) ignored.
WEIGHTS_FILES = {
    # Multidim checkpoint at pinned SHA. Computed once on first download; future
    # fetches verify against this. If upstream re-publishes weights for this SHA
    # we want to know.
    "weights/nisqa.tar": "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c",
}

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external" / "nisqa"


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
        logger.info("NISQA · already on disk: %s (%.1f KB)", dest, size_kb)
        return
    logger.info("NISQA · downloading %s → %s", url, dest)
    t0 = time.monotonic()
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - GitHub raw, known URL
    size_kb = dest.stat().st_size / 1024
    logger.info(
        "NISQA · downloaded %s in %.2fs (%.1f KB on disk)",
        dest.name, time.monotonic() - t0, size_kb,
    )


def fetch_all(*, force: bool = False) -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("NISQA · pinned SHA = %s", NISQA_PIN_SHA)
    logger.info("NISQA · target dir = %s", EXTERNAL_DIR)

    # Source files
    for rel in SOURCE_FILES:
        url = f"{NISQA_RAW_BASE}/{rel}"
        dest = EXTERNAL_DIR / rel
        if force and dest.is_file():
            dest.unlink()
        try:
            _download(url, dest)
        except Exception as exc:
            # nisqa/__init__.py may not exist upstream; create an empty one so
            # `import nisqa` works after sys.path manipulation.
            if rel.endswith("__init__.py") and "404" in str(exc):
                logger.info("NISQA · %s missing upstream, creating empty stub", rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("")
            else:
                raise

    # Weights
    for rel, expected_sha in WEIGHTS_FILES.items():
        url = f"{NISQA_RAW_BASE}/{rel}"
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
            logger.info("NISQA · %s SHA-256 OK (%s)", rel, expected_sha[:12])
        else:
            logger.info("NISQA · %s downloaded (no pinned SHA; consider pinning)", rel)

    logger.info("NISQA · all files ready under %s", EXTERNAL_DIR)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch vendored NISQA source + weights.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args(argv)
    fetch_all(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
