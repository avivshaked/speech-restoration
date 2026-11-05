#!/usr/bin/env python
"""Discover CC-licensed podcast candidates on archive.org.

Produces a JSON manifest of candidate items for human review. Does NOT
download audio. After you review the manifest, a separate download script
(to be added) will fetch + segment the approved subset.

Usage:
    uv run python scripts/fetch_test_set_candidates.py
    uv run python scripts/fetch_test_set_candidates.py --num-candidates 50 \\
        --max-duration-min 15 --out data/test-sets/podcast-real/candidates.json
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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA_URL = "https://archive.org/metadata/"
USER_AGENT = "voice-cleanup/0.1 (test-set discovery)"

COMMERCIAL_SAFE_PREFIXES = (
    "https://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by/",
    "https://creativecommons.org/licenses/by-sa/",
    "http://creativecommons.org/licenses/by-sa/",
    "https://creativecommons.org/publicdomain/zero/",
    "http://creativecommons.org/publicdomain/zero/",
    "https://creativecommons.org/publicdomain/mark/",
    "http://creativecommons.org/publicdomain/mark/",
    # Deprecated CC Public Domain Dedication — still commercial-safe
    "https://creativecommons.org/licenses/publicdomain/",
    "http://creativecommons.org/licenses/publicdomain/",
)

# Non-commercial variants — usable for internal evaluation, NOT for redistribution
# (e.g., before/after demo samples in the Phase 4a writeup must come from
# commercial_safe candidates only).
NONCOMMERCIAL_PREFIXES = (
    "https://creativecommons.org/licenses/by-nc/",
    "http://creativecommons.org/licenses/by-nc/",
    "https://creativecommons.org/licenses/by-nc-sa/",
    "http://creativecommons.org/licenses/by-nc-sa/",
    "https://creativecommons.org/licenses/by-nc-nd/",
    "http://creativecommons.org/licenses/by-nc-nd/",
    "https://creativecommons.org/licenses/by-nd/",
    "http://creativecommons.org/licenses/by-nd/",
)

AUDIO_FORMATS_PREFERRED = ("VBR MP3", "MP3", "Flac", "WAVE", "Ogg Vorbis")


def http_get_json(
    url: str, console: Console | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    if console:
        console.log(f"[dim]HTTP GET[/] {url}")
    t0 = time.monotonic()
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        status = resp.status
        body = resp.read()
    elapsed_ms = (time.monotonic() - t0) * 1000
    if console:
        console.log(
            f"[dim]HTTP {status}[/] · {len(body) / 1024:.1f} KB · {elapsed_ms:.0f} ms"
        )
    return json.loads(body.decode("utf-8"))


def license_short(url: str | None) -> str:
    if not url:
        return "?"
    u = url.lower()
    if "publicdomain/zero" in u:
        return "CC0"
    if "publicdomain/mark" in u:
        return "PDM"
    if "licenses/publicdomain" in u:
        return "CC-PD"  # legacy CC public-domain dedication, predates CC0
    if "by-sa" in u:
        return "CC-BY-SA"
    if "by-nc-sa" in u:
        return "CC-BY-NC-SA"
    if "by-nc-nd" in u:
        return "CC-BY-NC-ND"
    if "by-nc" in u:
        return "CC-BY-NC"
    if "by-nd" in u:
        return "CC-BY-ND"
    if "/by/" in u:
        return "CC-BY"
    return url


def is_commercial_safe(url: str | None) -> bool:
    if not url:
        return False
    return any(url.lower().startswith(p) for p in COMMERCIAL_SAFE_PREFIXES)


def is_noncommercial(url: str | None) -> bool:
    if not url:
        return False
    return any(url.lower().startswith(p) for p in NONCOMMERCIAL_PREFIXES)


def license_is_allowed(url: str | None, allow_nc: bool) -> bool:
    if not url:
        return False
    return is_commercial_safe(url) or (allow_nc and is_noncommercial(url))


_TRAILING_NUMBERS_RE = re.compile(r"[_\-]?\d+[\w\-]*$")


def series_stem(identifier: str) -> str:
    """Strip trailing digits / episode markers to get a series identifier.

    Heuristic: `MahabharataEpisode12_TheKingdom` → `MahabharataEpisode`.
    Imperfect but catches archive.org's per-episode naming convention.
    """
    stem = _TRAILING_NUMBERS_RE.sub("", identifier).rstrip("_-")
    return stem.lower()


def dedupe_key(item: dict[str, Any]) -> tuple[str, str]:
    """Return (key, source) for series deduplication.

    Prefer the `creator` field (consistent across episodes); fall back to the
    series-stem heuristic when no creator is recorded. The `source` tag is
    just for verbose logging.
    """
    creator = item.get("creator")
    if isinstance(creator, list):
        creator = creator[0] if creator else None
    if creator and str(creator).strip():
        return (str(creator).strip().lower(), "creator")
    return (series_stem(item["identifier"]), "identifier-stem")


def is_english(metadata: dict[str, Any]) -> bool | None:
    """Return True/False if archive.org's `language` field is informative, else None."""
    md = metadata.get("metadata", {})
    lang = md.get("language")
    if lang is None:
        return None
    if isinstance(lang, list):
        lang = lang[0] if lang else None
    if not lang:
        return None
    norm = str(lang).strip().lower()
    english_synonyms = {"english", "eng", "en", "en-us", "en-gb"}
    return norm in english_synonyms


def search_archive(num_rows: int, console: Console) -> list[dict[str, Any]]:
    """Query archive.org advanced search for CC-licensed podcast audio."""
    query = (
        "mediatype:audio "
        "AND collection:podcasts "
        "AND licenseurl:*creativecommons*"
    )
    params = {
        "q": query,
        "fl[]": ["identifier", "title", "licenseurl", "creator", "date"],
        "rows": num_rows,
        "page": 1,
        "output": "json",
        "sort[]": "downloads desc",
    }
    url = f"{ARCHIVE_SEARCH_URL}?{urlencode(params, doseq=True)}"
    console.log(f"[bold]search query:[/] {query}")
    console.log(f"[bold]requesting up to {num_rows} hits, sorted by download count[/]")
    data = http_get_json(url, console=console)
    docs: list[dict[str, Any]] = data.get("response", {}).get("docs", [])
    total_found = data.get("response", {}).get("numFound", "?")
    console.log(
        f"archive.org reports [bold]{total_found}[/] total matching items; "
        f"returned [bold]{len(docs)}[/] descriptors"
    )
    return docs


def fetch_item_metadata(identifier: str, console: Console | None = None) -> dict[str, Any]:
    return http_get_json(f"{ARCHIVE_METADATA_URL}{identifier}", console=console)


def pick_audio_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best audio file from an archive.org item's file list."""
    by_format: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        fmt = f.get("format", "")
        by_format.setdefault(fmt, []).append(f)
    for preferred in AUDIO_FORMATS_PREFERRED:
        if preferred in by_format:
            return by_format[preferred][0]
    return None


def parse_length_seconds(length_field: Any) -> float | None:
    """archive.org file `length` is either seconds-as-string or HH:MM:SS string."""
    if length_field is None:
        return None
    s = str(length_field).strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            parts_f = [float(p) for p in parts]
        except ValueError:
            return None
        total = 0.0
        for p in parts_f:
            total = total * 60 + p
        return total
    try:
        return float(s)
    except ValueError:
        return None


def build_candidate(
    item: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any] | None:
    identifier = item["identifier"]
    license_url = item.get("licenseurl") or meta.get("metadata", {}).get("licenseurl")
    files = meta.get("files", [])
    audio = pick_audio_file(files)
    if audio is None:
        return None
    duration_s = parse_length_seconds(audio.get("length"))
    size_bytes = int(audio.get("size", 0)) if str(audio.get("size", "")).isdigit() else None
    return {
        "identifier": identifier,
        "title": item.get("title") or meta.get("metadata", {}).get("title") or identifier,
        "creator": item.get("creator") or meta.get("metadata", {}).get("creator"),
        "date": item.get("date") or meta.get("metadata", {}).get("date"),
        "license_url": license_url,
        "license_short": license_short(license_url),
        "commercial_safe": is_commercial_safe(license_url),
        "file_name": audio.get("name"),
        "file_format": audio.get("format"),
        "file_size_bytes": size_bytes,
        "duration_s": duration_s,
        "download_url": f"https://archive.org/download/{identifier}/{audio.get('name')}",
        "item_url": f"https://archive.org/details/{identifier}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/test-sets/podcast-real/candidates.json"),
        help="Where to write the JSON manifest",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=40,
        help="Target number of candidates that pass duration + license filters",
    )
    parser.add_argument(
        "--search-pool",
        type=int,
        default=200,
        help="How many archive.org hits to consider (over-fetch, then filter)",
    )
    parser.add_argument("--min-duration-min", type=float, default=2.0)
    parser.add_argument("--max-duration-min", type=float, default=30.0)
    parser.add_argument(
        "--allow-nc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Accept Creative-Commons NonCommercial variants (BY-NC, BY-NC-SA, BY-NC-ND, "
            "BY-ND). They are usable for internal eval; entries are tagged "
            "`commercial_safe: false` so the Phase 4a writeup can filter them out of "
            "any publicly redistributed before/after samples."
        ),
    )
    parser.add_argument(
        "--max-per-series",
        type=int,
        default=1,
        help="Cap per series stem to enforce diversity (default: 1).",
    )
    parser.add_argument(
        "--english-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject items whose archive.org `language` field is explicitly non-English. "
            "Items with no language metadata are kept (manual review required)."
        ),
    )
    parser.add_argument(
        "--api-delay-s",
        type=float,
        default=0.1,
        help="Polite delay between metadata fetches",
    )
    args = parser.parse_args(argv)

    console = Console()
    console.rule("[bold cyan]Voice Cleanup · test-set candidate discovery[/]")
    console.print(f"  search pool:       up to [bold]{args.search_pool}[/] items from archive.org")
    console.print(f"  target candidates: [bold]{args.num_candidates}[/]")
    console.print(
        f"  duration filter:   "
        f"[bold]{args.min_duration_min:.1f}–{args.max_duration_min:.1f}[/] minutes"
    )
    if args.allow_nc:
        console.print(
            "  license filter:    [bold]CC-BY / CC-BY-SA / CC0 / Public Domain[/] "
            "+ [yellow]NC variants (tagged commercial_safe=false)[/]"
        )
    else:
        console.print(
            "  license filter:    [bold]CC-BY / CC-BY-SA / CC0 / Public Domain[/] "
            "(strict commercial-safe only)"
        )
    console.print(f"  manifest out:      [bold]{args.out}[/]")
    console.print("  [yellow]no audio is downloaded — manifest only[/]\n")

    try:
        items = search_archive(args.search_pool, console)
    except (HTTPError, URLError) as e:
        console.print(f"[red]archive.org search failed:[/] {e}")
        return 2

    if not items:
        console.print("[red]no items returned; widen the search or try again later[/]")
        return 1

    candidates: list[dict[str, Any]] = []
    rejected = {
        "license": 0,
        "no_audio_file": 0,
        "duration": 0,
        "fetch_error": 0,
        "non_english": 0,
        "series_dedupe": 0,
    }
    series_counts: dict[str, int] = {}

    min_s = args.min_duration_min * 60
    max_s = args.max_duration_min * 60

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
        task = progress.add_task("inspecting items", total=len(items))
        for i, item in enumerate(items, start=1):
            if len(candidates) >= args.num_candidates:
                console.log(
                    f"[bold cyan]target reached[/] ({args.num_candidates} candidates); "
                    f"stopping early after inspecting {i - 1} items"
                )
                progress.update(task, completed=len(items))
                break
            ident = item["identifier"]
            console.log(
                f"[bold]inspecting[/] [{i}/{len(items)}] [cyan]{ident}[/] "
                f"· license=[dim]{license_short(item.get('licenseurl'))}[/]"
            )
            # license pre-check from search result
            search_license = item.get("licenseurl")
            if not license_is_allowed(search_license, args.allow_nc):
                console.log(
                    f"  [yellow]reject[/] license not in allowed set: "
                    f"{search_license!r}"
                )
                rejected["license"] += 1
                progress.update(task, advance=1)
                continue
            if is_noncommercial(search_license):
                console.log(
                    f"  [dim]note[/] non-commercial license "
                    f"({license_short(search_license)}) — accepted for eval, "
                    "will be tagged commercial_safe=false"
                )
            # series dedupe (cheap, before metadata fetch)
            dkey, dsource = dedupe_key(item)
            if args.max_per_series and series_counts.get(dkey, 0) >= args.max_per_series:
                console.log(
                    f"  [yellow]reject[/] series dedupe ({dsource}): key "
                    f"[cyan]{dkey[:60]}[/] already has {series_counts[dkey]} accepted "
                    f"(cap={args.max_per_series})"
                )
                rejected["series_dedupe"] += 1
                progress.update(task, advance=1)
                continue
            try:
                meta = fetch_item_metadata(ident, console=console)
            except (HTTPError, URLError) as e:
                console.log(f"  [red]reject[/] metadata fetch failed: {e}")
                rejected["fetch_error"] += 1
                progress.update(task, advance=1)
                continue
            # English-only filter — only rejects if archive.org has an explicit non-en label
            if args.english_only:
                lang_match = is_english(meta)
                lang_field = meta.get("metadata", {}).get("language")
                if lang_match is False:
                    console.log(
                        f"  [yellow]reject[/] non-English content: language="
                        f"{lang_field!r}"
                    )
                    rejected["non_english"] += 1
                    progress.update(task, advance=1)
                    continue
                console.log(
                    f"  language field: {lang_field!r} "
                    f"({'english' if lang_match else 'unknown — kept'})"
                )
            files = meta.get("files", [])
            audio_file = pick_audio_file(files)
            audio_formats = sorted({
                f.get("format", "?")
                for f in files
                if "audio" in f.get("format", "").lower()
                or f.get("format") in AUDIO_FORMATS_PREFERRED
            })
            console.log(
                f"  files in item: {len(files)} · audio formats seen: {audio_formats}"
            )
            if audio_file is None:
                console.log("  [yellow]reject[/] no usable audio file in any preferred format")
                rejected["no_audio_file"] += 1
                progress.update(task, advance=1)
                continue
            cand = build_candidate(item, meta)
            if cand is None:
                console.log("  [yellow]reject[/] candidate could not be built")
                rejected["no_audio_file"] += 1
                progress.update(task, advance=1)
                continue
            d = cand["duration_s"]
            console.log(
                f"  picked file: [cyan]{cand['file_name']}[/] "
                f"({cand['file_format']}, "
                f"{(cand['file_size_bytes'] or 0) / 1e6:.1f} MB, "
                f"duration={d}s)"
            )
            if d is None:
                console.log("  [yellow]reject[/] duration unknown")
                rejected["duration"] += 1
                progress.update(task, advance=1)
                continue
            if d < min_s or d > max_s:
                console.log(
                    f"  [yellow]reject[/] duration {d / 60:.1f}m outside "
                    f"[{args.min_duration_min:.1f}–{args.max_duration_min:.1f}]m"
                )
                rejected["duration"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"inspecting items · kept {len(candidates)}",
                )
                continue
            candidates.append(cand)
            series_counts[dkey] = series_counts.get(dkey, 0) + 1
            cand["dedupe_key"] = dkey
            cand["dedupe_source"] = dsource
            cs_tag = (
                "[green]commercial-safe[/]"
                if cand["commercial_safe"]
                else "[yellow]eval-only (NC)[/]"
            )
            console.log(
                f"  [bold green]ACCEPT[/] · title=[bold]{cand['title'][:70]}[/] "
                f"· license=[green]{cand['license_short']}[/] · {cs_tag} "
                f"· dedupe-key=[cyan]{dkey[:40]}[/] ({dsource}) "
                f"· {d / 60:.1f}m · kept={len(candidates)}/{args.num_candidates}"
            )
            progress.update(
                task,
                advance=1,
                description=f"inspecting items · kept {len(candidates)}",
            )
            time.sleep(args.api_delay_s)

    if not candidates:
        console.print("[red]no candidates passed the filters; loosen them and retry[/]")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_commercial = sum(1 for c in candidates if c["commercial_safe"])
    n_eval_only = len(candidates) - n_commercial
    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "tool": "scripts/fetch_test_set_candidates.py",
        "filters": {
            "min_duration_s": min_s,
            "max_duration_s": max_s,
            "allow_nc": args.allow_nc,
            "english_only": args.english_only,
            "max_per_series": args.max_per_series,
            "commercial_safe_prefixes": list(COMMERCIAL_SAFE_PREFIXES),
            "noncommercial_prefixes": list(NONCOMMERCIAL_PREFIXES),
        },
        "stats": {
            "search_pool_considered": len(items),
            "candidates_kept": len(candidates),
            "commercial_safe": n_commercial,
            "eval_only_nc": n_eval_only,
            "rejected": rejected,
        },
        "candidates": candidates,
    }
    args.out.write_text(json.dumps(manifest, indent=2))

    table = Table(
        title=(
            f"Candidates ({len(candidates)}: "
            f"{n_commercial} commercial-safe + {n_eval_only} eval-only NC) "
            "— review before downloading"
        ),
        show_lines=False,
        title_style="bold cyan",
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("identifier", style="cyan", overflow="fold")
    table.add_column("title", overflow="fold")
    table.add_column("license", style="green")
    table.add_column("use", justify="center")
    table.add_column("dur", justify="right")
    table.add_column("size", justify="right", style="dim")
    for i, c in enumerate(candidates):
        size_mb = (
            f"{c['file_size_bytes'] / 1e6:.1f} MB" if c["file_size_bytes"] else "?"
        )
        use_tag = (
            "[green]pub[/]" if c["commercial_safe"] else "[yellow]eval[/]"
        )
        table.add_row(
            str(i + 1),
            c["identifier"][:35],
            (c["title"] or "")[:60],
            c["license_short"],
            use_tag,
            f"{c['duration_s'] / 60:.1f}m",
            size_mb,
        )
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold green]wrote {len(candidates)} candidates →[/] {args.out}"
    )
    console.print(
        f"  rejected: [yellow]license={rejected['license']}[/], "
        f"[yellow]no_audio={rejected['no_audio_file']}[/], "
        f"[yellow]duration={rejected['duration']}[/], "
        f"[yellow]non_english={rejected['non_english']}[/], "
        f"[yellow]series_dedupe={rejected['series_dedupe']}[/], "
        f"[yellow]fetch_err={rejected['fetch_error']}[/]"
    )
    console.print()
    console.print(
        "[dim]next: review the manifest, drop or annotate entries you don't want, "
        "then a download script (TBD) will fetch + segment the approved subset.[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
