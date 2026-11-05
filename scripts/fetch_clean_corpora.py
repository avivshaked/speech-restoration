"""Fetch studio-clean reference corpora (EARS + DAPS) for the restoration eval.

Per the 2026-05-28 "Eval re-architecture" decision (docs/decisions/log.md): the
restoration eval needs a genuinely studio-grade clean reference, not the mediocre
real-podcast clips. Lit Review §10 names both:

- **EARS** (Richter et al. 2024) — anechoic, full-band **48 kHz**, 107 speakers,
  100 h. Per-speaker zips from the official Meta GitHub release. ~592 MB/speaker
  (~63 GB for all 107). We pull a **diverse held-out eval subset** by default; the
  full corpus (`--all`) is only needed once Phase-3a training starts.
- **DAPS** (Mysore 2015) — parallel **studio** recordings + 12 real device/room
  conditions of the same scripts. One ~16 GB `daps.tar.gz` from Zenodo.

Both are **CC BY-NC 4.0** — fine for eval, FLAGGED for the commercial path
(CLAUDE.md licensing stance; VCTK/HiFi-TTS are the commercial-safe fallbacks).

Downloads stream to disk (never load multi-GB files into memory). Idempotent:
re-runs skip speakers / archives already extracted.

Usage:
    uv run python scripts/fetch_clean_corpora.py                 # EARS eval subset + DAPS
    uv run python scripts/fetch_clean_corpora.py --ears-only
    uv run python scripts/fetch_clean_corpora.py --daps-only
    uv run python scripts/fetch_clean_corpora.py --all-speakers  # full 107-speaker EARS
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_cleanup._logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CORPORA_DIR = Path(__file__).resolve().parents[1] / "data" / "corpora"
EARS_DIR = CORPORA_DIR / "ears"
DAPS_DIR = CORPORA_DIR / "daps"

EARS_RELEASE_BASE = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"

# Diverse held-out eval subset: balanced over gender / age / ethnicity / accent
# (incl. non-native English — realistic for podcasts). From speaker_statistics.json.
EARS_EVAL_SUBSET = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12]

DAPS_URL = "https://zenodo.org/api/records/4660670/files/daps.tar.gz/content"
DAPS_TGZ = DAPS_DIR / "daps.tar.gz"  # ~16 GB; size verified against server Content-Length


def _content_length(url: str) -> int | None:
    """Total size of `url` via a HEAD request (follows redirects); None if unknown."""
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310 - known release/Zenodo URL
    try:
        with urllib.request.urlopen(req) as r:  # noqa: S310
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as exc:  # noqa: BLE001
        logger.info("Corpora · HEAD failed (%s); proceeding without a size target", exc)
        return None


def _stream_download(url: str, dest: Path) -> None:
    """Resumable stream download (HTTP Range) with progress logging.

    Survives interruptions (e.g. the machine sleeping mid-download): a re-run
    continues from the bytes already on disk instead of restarting. Verifies the
    final size against the server's Content-Length when available.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url)
    existing = dest.stat().st_size if dest.is_file() else 0

    if total is not None and existing == total:
        logger.info("Corpora · %s already complete (%.0f MB) — skipping", dest.name, total / 1e6)
        return
    if total is not None and existing > total:
        logger.info("Corpora · %s larger than expected — restarting", dest.name)
        dest.unlink()
        existing = 0

    req = urllib.request.Request(url)  # noqa: S310 - known release/Zenodo URL
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        logger.info("Corpora · resuming %s from %.0f MB", dest.name, existing / 1e6)
    else:
        logger.info("Corpora · downloading %s → %s", url, dest)

    t0 = time.monotonic()
    next_pct = [0.0]
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        # If we asked for a range but got 200 (server ignored it), start clean.
        mode = "ab"
        if existing and getattr(resp, "status", 200) == 200:
            logger.info("Corpora · server ignored Range — restarting %s", dest.name)
            existing = 0
            mode = "wb"
        full = total if total is not None else (
            int(resp.headers.get("Content-Length", 0)) + existing or None
        )
        got = existing
        with open(dest, mode) as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
                if full:
                    pct = min(100.0, got * 100.0 / full)
                    if pct >= next_pct[0]:
                        logger.info(
                            "  … %.0f%% (%.0f / %.0f MB, %.1f MB/s)",
                            pct, got / 1e6, full / 1e6,
                            got / 1e6 / max(1e-6, time.monotonic() - t0),
                        )
                        next_pct[0] = pct + 10.0

    size = dest.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(
            f"{dest.name}: downloaded {size} bytes but expected {total} — "
            f"re-run to resume the remainder"
        )
    logger.info(
        "Corpora · downloaded %s in %.1fs (%.0f MB on disk)",
        dest.name, time.monotonic() - t0, size / 1e6,
    )


def fetch_ears(speakers: list[int]) -> None:
    EARS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("EARS · target dir = %s · %d speaker(s): %s", EARS_DIR, len(speakers), speakers)
    for i in speakers:
        spk = f"p{i:03d}"
        spk_dir = EARS_DIR / spk
        if spk_dir.is_dir() and any(spk_dir.iterdir()):
            logger.info("EARS · %s already extracted — skipping", spk)
            continue
        zip_path = EARS_DIR / f"{spk}.zip"
        _stream_download(f"{EARS_RELEASE_BASE}/{spk}.zip", zip_path)
        logger.info("EARS · extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(EARS_DIR)
        zip_path.unlink()
        logger.info("EARS · %s ready (%d files)", spk, sum(1 for _ in spk_dir.rglob("*.wav")))
    logger.info("EARS · done · license=CC BY-NC 4.0 (eval-only; flag for commercial)")


def fetch_daps() -> None:
    DAPS_DIR.mkdir(parents=True, exist_ok=True)
    # DAPS extracts to a top-level `daps/` dir with clean/produced/device subdirs.
    extracted_marker = DAPS_DIR / "daps"
    if extracted_marker.is_dir() and any(extracted_marker.iterdir()):
        logger.info("DAPS · already extracted at %s — skipping", extracted_marker)
        return
    # Resumable: continues a partial archive (e.g. after the machine slept) instead
    # of restarting, and self-skips if already complete.
    _stream_download(DAPS_URL, DAPS_TGZ)
    logger.info("DAPS · extracting %s (~16 GB, this takes a while) …", DAPS_TGZ.name)
    t0 = time.monotonic()
    with tarfile.open(DAPS_TGZ, "r:gz") as tf:
        # filter='data' = safe extraction (rejects path-traversal / unsafe members).
        tf.extractall(DAPS_DIR, filter="data")
    logger.info("DAPS · extracted in %.1fs", time.monotonic() - t0)
    # DAPS was tarred on macOS: it carries AppleDouble sidecars (._foo.wav, ~120 B)
    # and .DS_Store files alongside the real audio. They aren't playable audio and
    # clutter clean/ + cleanraw/ — strip them so the dirs hold only real wavs.
    junk = [p for p in DAPS_DIR.rglob("._*")] + [p for p in DAPS_DIR.rglob(".DS_Store")]
    for p in junk:
        p.unlink()
    if junk:
        logger.info("DAPS · removed %d macOS AppleDouble/.DS_Store sidecar files", len(junk))
    DAPS_TGZ.unlink()
    logger.info("DAPS · done · license=CC BY-NC 4.0 (eval-only; flag for commercial)")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch EARS + DAPS studio-clean corpora.")
    parser.add_argument("--ears-only", action="store_true", help="fetch EARS only")
    parser.add_argument("--daps-only", action="store_true", help="fetch DAPS only")
    parser.add_argument(
        "--all-speakers", action="store_true",
        help="fetch all 107 EARS speakers (~63 GB) instead of the eval subset",
    )
    args = parser.parse_args(argv)

    do_ears = not args.daps_only
    do_daps = not args.ears_only

    if do_ears:
        speakers = list(range(1, 108)) if args.all_speakers else EARS_EVAL_SUBSET
        fetch_ears(speakers)
    if do_daps:
        fetch_daps()
    logger.info("Corpora · all requested downloads complete under %s", CORPORA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
