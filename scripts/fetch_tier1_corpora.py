"""Fetch the Tier-1 commercial-safe, full-band clean-speech corpora for the Restorer rebuild.

Per the 2026-05-30 (late-evening) decision (docs/decisions/log.md): the rebuild trains on
commercially-licensed, studio-grade, HIGH-sample-rate clean speech, synthetically degraded.
Tier-1 = the only data that can teach top-octave (8-16 kHz) regeneration because it actually
contains it:

- **VCTK 0.92** — 48 kHz, 110 speakers, ~44 h, **CC BY 4.0** (commercial-safe). Studio chamber,
  full-band to ~20 kHz. FLAC, silence-trimmed, two mics per utterance (we keep mic1).
  ~10.94 GB zip from Edinburgh DataShare.
- **Hi-Fi TTS** — 44.1 kHz, 10 speakers, ~292 h, **CC BY 4.0** (commercial-safe). NVIDIA-curated
  >=13 kHz bandwidth + >=32 dB SNR. FLAC. ~41 GB tar.gz from OpenSLR 109.

Reuses the resumable, progress-logged streamer from fetch_clean_corpora.py (survives the machine
sleeping mid-download). Idempotent: skips archives/extractions already on disk.

Usage:
    uv run python scripts/fetch_tier1_corpora.py            # both
    uv run python scripts/fetch_tier1_corpora.py --vctk-only
    uv run python scripts/fetch_tier1_corpora.py --hifi-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fetch_clean_corpora import _stream_download  # noqa: E402  reuse resumable downloader

from voice_cleanup._logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CORPORA_DIR = REPO / "data" / "corpora"
VCTK_DIR = CORPORA_DIR / "vctk"
HIFI_DIR = CORPORA_DIR / "hifi_tts"

VCTK_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/"
    "VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y"
)
VCTK_ZIP = VCTK_DIR / "VCTK-Corpus-0.92.zip"  # ~10.94 GB

HIFI_URL = "https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz"
HIFI_MIRROR = "https://openslr.elda.org/resources/109/hi_fi_tts_v0.tar.gz"
HIFI_TGZ = HIFI_DIR / "hi_fi_tts_v0.tar.gz"  # ~41 GB


def fetch_vctk() -> None:
    VCTK_DIR.mkdir(parents=True, exist_ok=True)
    marker = VCTK_DIR / "wav48_silence_trimmed"
    if marker.is_dir() and any(marker.iterdir()):
        logger.info("VCTK · already extracted at %s — skipping", marker)
        return
    _stream_download(VCTK_URL, VCTK_ZIP)
    logger.info("VCTK · extracting %s (~11 GB) …", VCTK_ZIP.name)
    t0 = time.monotonic()
    with zipfile.ZipFile(VCTK_ZIP, "r") as zf:
        zf.extractall(VCTK_DIR)
    logger.info("VCTK · extracted in %.1fs", time.monotonic() - t0)
    VCTK_ZIP.unlink()
    n = sum(1 for _ in marker.rglob("*mic1.flac")) if marker.is_dir() else 0
    logger.info("VCTK · done · %d mic1 flacs · license=ODC-By 1.0 (commercial-safe)", n)


def fetch_hifi() -> None:
    HIFI_DIR.mkdir(parents=True, exist_ok=True)
    marker = HIFI_DIR / "hi_fi_tts_v0"
    if marker.is_dir() and any(marker.iterdir()):
        logger.info("Hi-Fi TTS · already extracted at %s — skipping", marker)
        return
    try:
        _stream_download(HIFI_URL, HIFI_TGZ)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hi-Fi TTS · primary URL failed (%s); trying EU mirror", exc)
        _stream_download(HIFI_MIRROR, HIFI_TGZ)
    logger.info("Hi-Fi TTS · extracting %s (~41 GB, takes a while) …", HIFI_TGZ.name)
    t0 = time.monotonic()
    with tarfile.open(HIFI_TGZ, "r:gz") as tf:
        tf.extractall(HIFI_DIR, filter="data")
    logger.info("Hi-Fi TTS · extracted in %.1fs", time.monotonic() - t0)
    HIFI_TGZ.unlink()
    n = sum(1 for _ in marker.rglob("*.flac")) if marker.is_dir() else 0
    logger.info("Hi-Fi TTS · done · %d flacs · license=CC BY 4.0 (commercial-safe)", n)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description="Fetch VCTK + Hi-Fi TTS (Tier-1 commercial-safe).")
    ap.add_argument("--vctk-only", action="store_true")
    ap.add_argument("--hifi-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.hifi_only:
        fetch_vctk()
    if not args.vctk_only:
        fetch_hifi()
    logger.info("Tier-1 · all requested downloads complete under %s", CORPORA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
