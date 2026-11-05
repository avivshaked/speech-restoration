"""Fetch public-domain old-time-radio reference clips for the model bazaar.

Genuine "radio voice" degradation (bandwidth-limited, boxy, AM-noisy) — the timbre the
synthetic refs under-represent. Source: archive.org
old-time-radio items that carry an EXPLICIT public-domain / CC0 license marker in their
metadata. We re-check each item's licenseurl at fetch time and SKIP anything that isn't clearly
PD/CC0/CC-BY — consent-clean only (decision log 2026-05-31). Provenance + license is recorded
per clip in PROVENANCE.json.

Downloads from a known public source (allowed); nothing is sent out. Each episode is decoded
with the in-project imageio-ffmpeg binary, a mid-show speech window is extracted to 44.1 kHz
mono wav, and the source episode is deleted (we keep only the short clip).

  uv run python scripts/fetch_pd_radio_refs.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import soundfile as sf  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data.degrade import _ffmpeg_exe  # noqa: E402

setup_logging()
log = logging.getLogger("pd-radio")
OUT = REPO / "data" / "bazaar-refs"

# Curated archive.org identifiers, each confirmed (probe 2026-05-31) to carry a
# public-domain / CC0 license marker. Speech-heavy (dialogue / comedy), not music.
ITEMS = [
    "FibberMcGeeAndMolly_201610",   # publicdomain/mark/1.0 — comedy dialogue
    "DuffysTavernWithVincentPrice",  # publicdomain/zero/1.0 (CC0) — dialogue
    "AdventureAhead",               # publicdomain/mark/1.0 — drama narration
    "MomsMabley",                   # publicdomain/mark/1.0 — stand-up speech
]
# Only these license families are accepted as consent-clean.
_OK_LIC = ("publicdomain", "/zero/", "/by/", "/by-sa/")
START_S = 75.0   # skip intro music/announcer; land in dialogue
LEN_S = 25.0


def _meta(identifier: str) -> dict:
    url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
        return json.load(r)


def _pick_audio(files: list[dict]) -> str | None:
    """Prefer a compact MP3; fall back to any audio format."""
    audio = [f for f in files if str(f.get("format", "")).lower().endswith(("mp3", "ogg vorbis"))
             or "mp3" in str(f.get("format", "")).lower()]
    # Prefer smallest (fastest) MP3.
    audio.sort(key=lambda f: int(f.get("size", "0") or 0))
    return audio[0]["name"] if audio else None


def main() -> int:
    exe = _ffmpeg_exe()
    if not exe:
        log.error("No ffmpeg (imageio-ffmpeg). Cannot decode downloads.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    prov_path = OUT / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}

    n_ok = 0
    for ident in ITEMS:
        try:
            m = _meta(ident)
        except Exception as e:  # noqa: BLE001
            log.warning("%s · metadata fetch failed: %s", ident, e)
            continue
        lic = str(m.get("metadata", {}).get("licenseurl", ""))
        if not any(tok in lic for tok in _OK_LIC):
            log.warning("%s · license %r not clearly PD/CC — SKIPPING", ident, lic or "(none)")
            continue
        fname = _pick_audio(m.get("files", []))
        if not fname:
            log.warning("%s · no audio file found — skipping", ident)
            continue
        src_url = f"https://archive.org/download/{ident}/{urllib.parse.quote(fname)}"
        title = m.get("metadata", {}).get("title", ident)
        log.info("%s · %s · lic=%s", ident, title, lic)

        tmp = OUT / f"_tmp_{ident}{Path(fname).suffix or '.mp3'}"
        out_wav = OUT / f"otr_{ident}.wav"
        try:
            log.info("  downloading %s …", fname)
            urllib.request.urlretrieve(src_url, tmp)  # noqa: S310
            cmd = [exe, "-nostdin", "-loglevel", "error", "-ss", str(START_S), "-t", str(LEN_S),
                   "-i", str(tmp), "-ac", "1", "-ar", "44100", "-y", str(out_wav)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            dur = len(sf.read(str(out_wav))[0]) / 44100
            log.info("  → %s (%.1fs)", out_wav.name, dur)
            if dur < 5.0:
                log.warning("  clip too short (%.1fs) — removing", dur)
                out_wav.unlink(missing_ok=True)
                continue
            prov[out_wav.name] = {"identifier": ident, "title": title, "licenseurl": lic,
                                  "source": src_url, "source_file": fname,
                                  "window": f"{START_S:.0f}-{START_S + LEN_S:.0f}s"}
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            log.warning("  %s · fetch/decode failed: %s", ident, e)
        finally:
            tmp.unlink(missing_ok=True)

    prov_path.write_text(json.dumps(prov, indent=2))
    log.info("Wrote %d PD-radio clips to %s · provenance: %s", n_ok, OUT, prov_path.name)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
