"""Assemble consent-clean reference clips for the model bazaar.

Sources (all consent-clean; see decision log 2026-05-31):
  - synthetic: VCTK clean speech (ODC-By, full-band) degraded with controlled "radio voice"
    chains (radio / bandwidth / broadband) — PAIRED (we keep the clean target too) and fully
    reproducible. The workhorse first reference until real material lands.
  - (TODO, separate fetch) public-domain old-time radio + CC bad-audio podcasts; and Aviv's
    own clean+degraded recordings dropped straight into --out.

Writes 44.1 kHz mono wavs into data/bazaar-refs/. The bazaar (`model_bazaar.py`) just globs
that dir, so any consent-clean wav added later is picked up automatically.

  uv run python scripts/fetch_bazaar_refs.py --vctk-speakers 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_cleanup._logging import setup_logging  # noqa: E402
from voice_cleanup.data import degrade, tier1  # noqa: E402
from voice_cleanup.data.loudness import normalize_lufs  # noqa: E402

setup_logging()
log = logging.getLogger("bazaar-refs")
SR = tier1.SR
OUT = REPO / "data" / "bazaar-refs"


def _conditions() -> dict[str, degrade.DegradationChain]:
    """The controlled 'radio voice' damage families (mirror calibrate_degradations.py)."""
    return {
        "radio": degrade.DegradationChain("radio", [
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-10.0, 12.0), q_range=(2.0, 5.0)),
            degrade.BandwidthLimit(cutoff_range=(3800.0, 4500.0), order=2)]),
        "bandwidth": degrade.DegradationChain("bandwidth", [
            degrade.BandwidthLimit(cutoff_range=(4500.0, 6000.0), order=2)]),
        "broadband": degrade.DegradationChain("broadband", [
            degrade.SyntheticReverb(rt60_range=(0.3, 0.5)),
            degrade.EqDamage(n_bands_range=(2, 3), gain_db_range=(-9.0, 9.0)),
            degrade.AdditiveNoise(snr_db_range=(15.0, 25.0), color="pink")]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vctk-speakers", type=int, default=3, help="how many VCTK speakers to use")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="target clip length (utterances concatenated per speaker to reach it)")
    ap.add_argument("--keep-clean", action="store_true", default=True,
                    help="also write the undamaged clean reference (paired)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    clips = tier1.enumerate_vctk()
    if not clips:
        log.error("No VCTK clips — run scripts/fetch_tier1_corpora.py first.")
        return 2
    # Group utterances per speaker (deterministic order), keep the first N speakers.
    by_spk: dict[str, list[tier1.CleanClip]] = {}
    for c in sorted(clips, key=lambda c: str(c.path)):
        by_spk.setdefault(c.speaker, []).append(c)
    chosen_spk = list(by_spk)[: args.vctk_speakers]
    conds = _conditions()
    log.info("Synthesising %d speakers × {clean + %s} (~%.0fs/clip) → %s",
             len(chosen_spk), list(conds), args.seconds, OUT)

    target = int(args.seconds * SR)
    n = 0
    for spk_name in chosen_spk:
        spk = spk_name.replace(":", "_")
        # Concatenate consecutive utterances (0.2s gap) until we reach the target length.
        gap = np.zeros(int(0.2 * SR), dtype="float32")
        parts: list[np.ndarray] = []
        total = 0
        for c in by_spk[spk_name]:
            seg = tier1._read_clean(c.path)
            parts.extend([seg, gap])
            total += len(seg) + len(gap)
            if total >= target:
                break
        clean = np.concatenate(parts)[:target] if parts else np.zeros(target, dtype="float32")
        if args.keep_clean:
            sf.write(str(OUT / f"vctk_{spk}__clean.wav"), normalize_lufs(clean, SR), SR)
            n += 1
        for cname, chain in conds.items():
            seed = 7000 + hash(spk) % 1000
            try:
                deg = chain.apply(clean, SR, seed=seed)
            except Exception as e:  # noqa: BLE001
                log.warning("  %s/%s failed: %s", spk, cname, e)
                continue
            sf.write(str(OUT / f"vctk_{spk}__{cname}.wav"),
                     normalize_lufs(np.asarray(deg, dtype="float32"), SR), SR)
            n += 1
            log.info("  %s · %s", spk, cname)
    log.info("Wrote %d reference wavs to %s", n, OUT)
    log.info("Add your own clean+degraded recordings / PD radio / CC podcast clips to the same "
             "dir — the bazaar globs *.wav.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
