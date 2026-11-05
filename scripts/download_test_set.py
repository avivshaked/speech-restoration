#!/usr/bin/env python
"""Download + segment the curated podcast test set.

Reads manifests/podcast-real-candidates.json, downloads each kept=true
entry's audio to raw/, decodes + resamples + extracts a fixed-length segment
from the midpoint into clips/, and writes clips.json with provenance.

Idempotent: skips items whose raw download AND clip already exist.

Usage:
    uv run python scripts/download_test_set.py
    uv run python scripts/download_test_set.py --segment-seconds 30 --offset-from-start 60
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
import torch
import torchaudio
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

USER_AGENT = "voice-cleanup/0.1 (test-set download)"
TARGET_SR = 16000
TARGET_CHANNELS = 1
DOWNLOAD_TIMEOUT_S = 300.0
DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB


def safe_name(s: str) -> str:
    """Sanitize identifier to safe filename component."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_-.")
    return out[:80] or "item"


def encode_url(url: str) -> str:
    """Percent-encode the path portion of a URL (handles spaces in filenames)."""
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment)
    )


def http_download(
    url: str, dest: Path, console: Console, expected_size: int | None
) -> int:
    """Stream-download `url` to `dest`. Returns bytes written."""
    encoded = encode_url(url)
    console.log(
        f"  [bold]downloading[/] {encoded}"
        + (f"  (expected {expected_size / 1e6:.1f} MB)" if expected_size else "")
    )
    t0 = time.monotonic()
    req = Request(encoded, headers={"User-Agent": USER_AGENT})
    written = 0
    last_log = t0
    with urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        total = int(resp.headers.get("content-length", 0)) or expected_size or 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            while True:
                data = resp.read(DOWNLOAD_CHUNK_BYTES)
                if not data:
                    break
                f.write(data)
                written += len(data)
                now = time.monotonic()
                if now - last_log >= 2.0:
                    speed = written / (now - t0) / 1e6
                    pct = (written / total * 100) if total else 0
                    console.log(
                        f"    [dim]progress[/] {written / 1e6:.1f} MB"
                        + (f" / {total / 1e6:.1f} MB ({pct:.0f}%)" if total else "")
                        + f"  @ {speed:.2f} MB/s"
                    )
                    last_log = now
        tmp.replace(dest)
    elapsed = time.monotonic() - t0
    console.log(
        f"  [green]downloaded[/] {written / 1e6:.1f} MB in {elapsed:.1f}s "
        f"({written / elapsed / 1e6:.2f} MB/s)"
    )
    return written


def load_audio_any(path: Path, console: Console) -> tuple[np.ndarray, int]:
    """Decode any audio file path via soundfile. Returns (waveform[T,C], sr)."""
    console.log(f"  [bold]decoding[/] {path.name}")
    t0 = time.monotonic()
    data, sr = sf.read(str(path), always_2d=True)
    console.log(
        f"  loaded sr={sr} · shape={data.shape} · "
        f"dtype={data.dtype} · {time.monotonic() - t0:.2f}s"
    )
    return data, sr


def to_mono_16k(data: np.ndarray, sr: int, console: Console) -> np.ndarray:
    """Average channels to mono, resample to TARGET_SR."""
    if data.shape[1] > 1:
        console.log(f"  averaging {data.shape[1]} channels → mono")
        mono = data.mean(axis=1)
    else:
        mono = data[:, 0]
    if sr != TARGET_SR:
        console.log(f"  resampling {sr} → {TARGET_SR} Hz")
        t0 = time.monotonic()
        tensor = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, sr, TARGET_SR)
        mono = resampled.squeeze(0).numpy()
        console.log(f"  resampled in {time.monotonic() - t0:.2f}s")
    return mono.astype(np.float32)


def cut_midpoint_segment(
    mono: np.ndarray, segment_s: float, console: Console
) -> tuple[np.ndarray, float, float]:
    """Cut a segment centered at the midpoint of `mono`. Returns (segment, t_start, t_end)."""
    total_s = len(mono) / TARGET_SR
    seg_samples = int(segment_s * TARGET_SR)
    if len(mono) <= seg_samples:
        console.log(
            f"  [yellow]warning[/] audio ({total_s:.1f}s) shorter than segment "
            f"({segment_s:.1f}s); using full audio"
        )
        return mono, 0.0, total_s
    mid = len(mono) // 2
    half = seg_samples // 2
    start = max(0, mid - half)
    end = start + seg_samples
    console.log(
        f"  [green]segment[/] {start / TARGET_SR:.2f}s → {end / TARGET_SR:.2f}s "
        f"(midpoint cut, {segment_s:.1f}s of {total_s:.1f}s)"
    )
    return mono[start:end], start / TARGET_SR, end / TARGET_SR


def process_one(
    cand: dict[str, Any],
    raw_dir: Path,
    clips_dir: Path,
    segment_s: float,
    console: Console,
) -> dict[str, Any]:
    """Download + segment a single candidate. Returns a record for clips.json."""
    ident = cand["identifier"]
    name = safe_name(ident)
    src_ext = Path(cand["file_name"]).suffix or ".audio"
    raw_path = raw_dir / f"{name}{src_ext}"
    clip_path = clips_dir / f"{name}.wav"

    record: dict[str, Any] = {
        "identifier": ident,
        "source_url": cand["download_url"],
        "license_url": cand["license_url"],
        "license_short": cand["license_short"],
        "commercial_safe": cand["commercial_safe"],
        "raw_path": str(raw_path),
        "clip_path": None,
        "clip_duration_s": None,
        "clip_offset_s": None,
        "source_duration_s": cand["duration_s"],
        "title": cand["title"],
        "status": "pending",
        "error": None,
    }

    # download
    if raw_path.exists() and raw_path.stat().st_size > 0:
        console.log(
            f"  [dim]raw cached[/] {raw_path.name} "
            f"({raw_path.stat().st_size / 1e6:.1f} MB)"
        )
    else:
        try:
            http_download(
                cand["download_url"], raw_path, console, cand.get("file_size_bytes")
            )
        except (HTTPError, URLError, TimeoutError) as e:
            console.log(f"  [red]download failed[/]: {e}")
            record["status"] = "download_failed"
            record["error"] = str(e)
            return record

    # decode + segment (skip if clip already on disk)
    if clip_path.exists() and clip_path.stat().st_size > 0:
        info = sf.info(str(clip_path))
        console.log(
            f"  [dim]clip cached[/] {clip_path.name} "
            f"({info.duration:.2f}s @ {info.samplerate} Hz)"
        )
        record["clip_path"] = str(clip_path)
        record["clip_duration_s"] = info.duration
        record["status"] = "ok"
        return record

    try:
        data, sr = load_audio_any(raw_path, console)
        mono = to_mono_16k(data, sr, console)
        seg, t_start, t_end = cut_midpoint_segment(mono, segment_s, console)
        sf.write(str(clip_path), seg, TARGET_SR, subtype="PCM_16")
        console.log(
            f"  [bold green]wrote[/] {clip_path} "
            f"({(t_end - t_start):.2f}s @ {TARGET_SR} Hz, "
            f"{clip_path.stat().st_size / 1e6:.2f} MB)"
        )
        record["clip_path"] = str(clip_path)
        record["clip_duration_s"] = float(t_end - t_start)
        record["clip_offset_s"] = float(t_start)
        record["status"] = "ok"
    except Exception as e:  # noqa: BLE001 — surface anything for verbose audit
        console.log(f"  [red]decode/segment failed[/]: {type(e).__name__}: {e}")
        record["status"] = "decode_failed"
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/podcast-real-candidates.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/test-sets/podcast-real"),
        help="Root of the test set: raw/, clips/, clips.json live here",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=20.0,
        help="Length of each segment in seconds (centered on midpoint).",
    )
    args = parser.parse_args(argv)

    console = Console()
    console.rule("[bold cyan]Voice Cleanup · test-set download + segmentation[/]")
    console.print(f"  manifest:           [bold]{args.manifest}[/]")
    console.print(f"  output root:        [bold]{args.out_dir}[/]")
    console.print(f"  segment length:     [bold]{args.segment_seconds:.1f}s[/]")
    console.print(f"  target sample rate: [bold]{TARGET_SR} Hz mono[/]")
    console.print()

    if not args.manifest.exists():
        console.print(f"[red]manifest not found:[/] {args.manifest}")
        return 1
    manifest = json.loads(args.manifest.read_text())
    all_cands = manifest["candidates"]
    kept = [c for c in all_cands if c.get("kept", True)]
    console.log(
        f"manifest has {len(all_cands)} candidates · "
        f"[bold]{len(kept)}[/] marked kept=True"
    )

    raw_dir = args.out_dir / "raw"
    clips_dir = args.out_dir / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    t_start = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("processing items", total=len(kept))
        for i, cand in enumerate(kept, start=1):
            console.log(
                f"[bold][{i}/{len(kept)}][/] [cyan]{cand['identifier']}[/] · "
                f"{cand['license_short']} · {cand['duration_s'] / 60:.1f}m"
            )
            rec = process_one(cand, raw_dir, clips_dir, args.segment_seconds, console)
            records.append(rec)
            progress.update(task, advance=1)

    # write clips.json
    clips_manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_manifest": str(args.manifest),
        "segment_seconds": args.segment_seconds,
        "target_sr": TARGET_SR,
        "stats": {
            "total": len(records),
            "ok": sum(1 for r in records if r["status"] == "ok"),
            "download_failed": sum(1 for r in records if r["status"] == "download_failed"),
            "decode_failed": sum(1 for r in records if r["status"] == "decode_failed"),
            "elapsed_s": time.monotonic() - t_start,
        },
        "clips": records,
    }
    clips_json = args.out_dir / "clips.json"
    clips_json.write_text(json.dumps(clips_manifest, indent=2))

    # final summary table
    table = Table(
        title=f"Clips ({clips_manifest['stats']['ok']} ok / {len(records)} total)",
        title_style="bold cyan",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("identifier", style="cyan")
    table.add_column("status")
    table.add_column("license", style="green")
    table.add_column("use")
    table.add_column("clip", justify="right")
    table.add_column("note", overflow="fold")
    for i, r in enumerate(records, 1):
        status_color = {"ok": "green", "download_failed": "red", "decode_failed": "red"}.get(
            r["status"], "yellow"
        )
        use_tag = "[green]pub[/]" if r["commercial_safe"] else "[yellow]eval[/]"
        clip_disp = (
            f"{r['clip_duration_s']:.1f}s" if r["clip_duration_s"] is not None else "—"
        )
        table.add_row(
            str(i),
            r["identifier"][:35],
            f"[{status_color}]{r['status']}[/]",
            r["license_short"],
            use_tag,
            clip_disp,
            (r["error"] or ""),
        )
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold green]wrote {clips_manifest['stats']['ok']} clips →[/] "
        f"{clips_dir} (16 kHz mono PCM_16 WAV)"
    )
    console.print(f"[bold]manifest →[/] {clips_json}")
    console.print(
        f"[dim]elapsed {clips_manifest['stats']['elapsed_s']:.1f}s · "
        f"raw downloads stored in {raw_dir}[/]"
    )
    return 0 if clips_manifest["stats"]["ok"] == len(records) else 2


if __name__ == "__main__":
    sys.exit(main())
