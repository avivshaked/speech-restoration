"""Eval panel runner — applies a degradation to each clip and runs the metric panel.

Phase 1 framing: we don't have enhancement models yet, so "evaluation" means running
the metric panel on a test set under a controlled degradation. This validates the
panel end-to-end and produces the same artifact shape we'll use for Phase 4 model
comparisons.

For `none`: only non-intrusive metrics (PESQ/ESTOI/SI-SDR on a clip vs itself are
trivial — PESQ ≈ 4.5, ESTOI = 1, SI-SDR → ∞ — so we skip them).
For `gaussian:SNR_DB`: degrades the clip, runs the full panel.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voice_cleanup.eval.enhancers import Enhancer
from voice_cleanup.eval.intrusive import estoi, pesq_wb, si_sdr
from voice_cleanup.eval.nisqa import nisqa_multidim
from voice_cleanup.eval.non_intrusive import dnsmos_p835, squim_mos
from voice_cleanup.eval.wer import whisper_wer

logger = logging.getLogger(__name__)


# --- degradation spec -------------------------------------------------------


@dc.dataclass(frozen=True)
class Degradation:
    """A reproducible degradation applied to each clip before scoring."""

    kind: str  # "none" | "gaussian"
    snr_db: float | None = None  # required for kind="gaussian"
    seed: int = 0

    @classmethod
    def parse(cls, spec: str) -> Degradation:
        """Parse a CLI string like 'none' or 'gaussian:-5'."""
        if spec == "none":
            return cls(kind="none")
        if spec.startswith("gaussian:"):
            snr = float(spec.split(":", 1)[1])
            return cls(kind="gaussian", snr_db=snr)
        raise ValueError(f"unknown degradation spec: {spec!r}")

    def label(self) -> str:
        if self.kind == "none":
            return "clean (identity)"
        if self.kind == "gaussian":
            return f"gaussian noise @ {self.snr_db:.1f} dB SNR"
        return self.kind

    def apply(self, clean: np.ndarray, seed_offset: int = 0) -> np.ndarray:
        if self.kind == "none":
            return clean.copy()
        if self.kind == "gaussian":
            assert self.snr_db is not None
            rng = np.random.default_rng(self.seed + seed_offset)
            noise = rng.standard_normal(clean.shape)
            sig_power = float(np.mean(clean**2))
            noise_power = float(np.mean(noise**2))
            target_noise_power = sig_power / (10 ** (self.snr_db / 10))
            noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-12))
            return (clean + noise).astype(np.float32)
        raise ValueError(f"unknown degradation kind: {self.kind!r}")


# --- clip discovery ---------------------------------------------------------


@dc.dataclass(frozen=True)
class ClipRecord:
    identifier: str
    path: Path
    sr: int
    license_short: str
    commercial_safe: bool


def load_test_set(test_set_dir: Path) -> list[ClipRecord]:
    """Load clip records from a test set directory's clips.json manifest."""
    manifest_path = test_set_dir / "clips.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no clips.json at {manifest_path} — run scripts/download_test_set.py first"
        )
    raw = json.loads(manifest_path.read_text())
    records: list[ClipRecord] = []
    for entry in raw.get("clips", []):
        if entry.get("status") != "ok":
            continue
        path = Path(entry["clip_path"])
        with sf.SoundFile(str(path)) as f:
            sr = int(f.samplerate)
        records.append(
            ClipRecord(
                identifier=entry["identifier"],
                path=path,
                sr=sr,
                license_short=entry.get("license_short", "?"),
                commercial_safe=bool(entry.get("commercial_safe", False)),
            )
        )
    return records


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


# --- panel run --------------------------------------------------------------


@dc.dataclass
class ClipResult:
    identifier: str
    license_short: str
    commercial_safe: bool
    duration_s: float
    pesq: float | None
    estoi: float | None
    si_sdr: float | None
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    dnsmos_ovrl: float | None
    squim_mos: float | None
    nisqa_mos: float | None
    nisqa_noi: float | None
    nisqa_dis: float | None
    nisqa_col: float | None
    nisqa_loud: float | None
    wer: float | None
    wer_ref_text: str | None
    wer_hyp_text: str | None
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


@dc.dataclass
class EvalReport:
    degradation: Degradation
    test_set_dir: str
    started_at: str
    elapsed_s: float
    clips: list[ClipResult]
    enhancer_name: str | None = None
    enhancer_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "degradation": {
                "kind": self.degradation.kind,
                "snr_db": self.degradation.snr_db,
                "seed": self.degradation.seed,
                "label": self.degradation.label(),
            },
            "enhancer": {
                "name": self.enhancer_name,
                "label": self.enhancer_label,
            },
            "test_set_dir": self.test_set_dir,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "clips": [c.to_dict() for c in self.clips],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Mean/std/n per metric across all clips with non-None scores."""
        keys = (
            "pesq", "estoi", "si_sdr",
            "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
            "squim_mos",
            "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
            "wer",
        )
        out: dict[str, dict[str, float | int]] = {}
        for k in keys:
            vals = [getattr(c, k) for c in self.clips if getattr(c, k) is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(len(arr)),
            }
        return out


def _pick_nmr(records: list[ClipRecord], current_idx: int) -> ClipRecord:
    """Choose a non-matching reference clip for SQUIM. Just the next clip cyclically."""
    return records[(current_idx + 1) % len(records)]


def run_panel_on_clip(
    clean_audio: np.ndarray,
    sr: int,
    degradation: Degradation,
    clip_idx: int,
    *,
    nmr_audio: np.ndarray | None,
    nmr_sr: int | None,
    run_squim: bool,
    run_nisqa: bool,
    run_wer: bool,
    enhancer: Enhancer | None = None,
) -> dict[str, Any]:
    """Run the metric panel on one clip with the given degradation.

    Returns a dict with metric keys plus wer_ref_text / wer_hyp_text strings.
    Values may be None when not applicable (intrusive metrics + WER under `none`).

    When `enhancer` is provided, the panel scores `enhancer.apply(degraded)`
    instead of `degraded` directly — that's how baseline / production model
    evaluation slots into the same pipeline.
    """
    degraded = degradation.apply(clean_audio, seed_offset=clip_idx)
    if enhancer is not None:
        logger.info("clip %d · running enhancer=%s on degraded audio", clip_idx + 1, enhancer.name)
        scored = enhancer.apply(degraded, sr)
        if len(scored) != len(degraded):
            logger.warning(
                "enhancer %s changed length: degraded=%d enhanced=%d — truncating to min",
                enhancer.name, len(degraded), len(scored),
            )
            n = min(len(scored), len(degraded), len(clean_audio))
            scored = scored[:n]
            # Truncate clean too for intrusive metrics — keep both refs aligned.
            clean_audio = clean_audio[:n]
    else:
        scored = degraded
    out: dict[str, Any] = {
        "pesq": None,
        "estoi": None,
        "si_sdr": None,
        "dnsmos_sig": None,
        "dnsmos_bak": None,
        "dnsmos_ovrl": None,
        "squim_mos": None,
        "nisqa_mos": None,
        "nisqa_noi": None,
        "nisqa_dis": None,
        "nisqa_col": None,
        "nisqa_loud": None,
        "wer": None,
        "wer_ref_text": None,
        "wer_hyp_text": None,
    }

    # All metrics from here on score `scored` (= enhanced output if an enhancer
    # was supplied, else == degraded). Intrusive metrics compare against clean.
    if degradation.kind != "none":
        out["pesq"] = pesq_wb(clean_audio, scored, sr=sr)
        out["estoi"] = estoi(clean_audio, scored, sr=sr)
        out["si_sdr"] = si_sdr(clean_audio, scored)

    dns = dnsmos_p835(scored, sr=sr)
    out["dnsmos_sig"] = dns["sig"]
    out["dnsmos_bak"] = dns["bak"]
    out["dnsmos_ovrl"] = dns["ovrl"]

    if run_squim and nmr_audio is not None and nmr_sr is not None:
        out["squim_mos"] = squim_mos(scored, nmr_audio, sr=sr)

    if run_nisqa:
        nisqa = nisqa_multidim(scored, sr=sr)
        out["nisqa_mos"] = nisqa["mos_pred"]
        out["nisqa_noi"] = nisqa["noi_pred"]
        out["nisqa_dis"] = nisqa["dis_pred"]
        out["nisqa_col"] = nisqa["col_pred"]
        out["nisqa_loud"] = nisqa["loud_pred"]

    if run_wer and degradation.kind != "none":
        wer_out = whisper_wer(clean_audio, scored, sr=sr)
        out["wer"] = wer_out["wer"]
        out["wer_ref_text"] = wer_out["ref_text"]
        out["wer_hyp_text"] = wer_out["hyp_text"]

    return out


def run_eval(
    test_set_dir: Path,
    degradation: Degradation,
    *,
    limit: int | None = None,
    run_squim: bool = True,
    run_nisqa: bool = True,
    run_wer: bool = True,
    enhancer: Enhancer | None = None,
    on_clip_done: Any = None,
) -> EvalReport:
    """Iterate clips in the test set, run the panel, return an EvalReport.

    `on_clip_done(idx, total, record, result)` is called after each clip — used by
    the CLI to refresh a live rich table without coupling runner to rich.
    """
    import datetime as dt

    records = load_test_set(test_set_dir)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"no usable clips in {test_set_dir}")
    logger.info(
        "eval · test-set=%s · clips=%d · degradation=%s · enhancer=%s · "
        "squim=%s · nisqa=%s · wer=%s",
        test_set_dir, len(records), degradation.label(),
        enhancer.label() if enhancer is not None else "none",
        run_squim, run_nisqa, run_wer,
    )

    t_start = time.monotonic()
    started_at = dt.datetime.now(dt.UTC).isoformat()
    results: list[ClipResult] = []
    for idx, rec in enumerate(records):
        logger.info(
            "[%d/%d] clip=%s · license=%s · commercial-safe=%s",
            idx + 1, len(records), rec.identifier, rec.license_short, rec.commercial_safe,
        )
        t0 = time.monotonic()
        clean, sr = _read_mono(rec.path)
        nmr_audio = nmr_sr = None
        if run_squim:
            nmr_rec = _pick_nmr(records, idx)
            nmr_audio, nmr_sr = _read_mono(nmr_rec.path)
            logger.info("  · SQUIM NMR ← %s", nmr_rec.identifier)

        scores = run_panel_on_clip(
            clean, sr, degradation, idx,
            nmr_audio=nmr_audio, nmr_sr=nmr_sr,
            run_squim=run_squim, run_nisqa=run_nisqa, run_wer=run_wer,
            enhancer=enhancer,
        )
        elapsed = time.monotonic() - t0
        result = ClipResult(
            identifier=rec.identifier,
            license_short=rec.license_short,
            commercial_safe=rec.commercial_safe,
            duration_s=len(clean) / sr,
            pesq=scores["pesq"],
            estoi=scores["estoi"],
            si_sdr=scores["si_sdr"],
            dnsmos_sig=scores["dnsmos_sig"],
            dnsmos_bak=scores["dnsmos_bak"],
            dnsmos_ovrl=scores["dnsmos_ovrl"],
            squim_mos=scores["squim_mos"],
            nisqa_mos=scores["nisqa_mos"],
            nisqa_noi=scores["nisqa_noi"],
            nisqa_dis=scores["nisqa_dis"],
            nisqa_col=scores["nisqa_col"],
            nisqa_loud=scores["nisqa_loud"],
            wer=scores["wer"],
            wer_ref_text=scores["wer_ref_text"],
            wer_hyp_text=scores["wer_hyp_text"],
            elapsed_s=elapsed,
        )
        results.append(result)
        if on_clip_done is not None:
            on_clip_done(idx, len(records), rec, result)

    total_elapsed = time.monotonic() - t_start
    logger.info("eval · done · %d clips in %.1fs", len(results), total_elapsed)
    return EvalReport(
        degradation=degradation,
        test_set_dir=str(test_set_dir),
        started_at=started_at,
        elapsed_s=total_elapsed,
        clips=results,
        enhancer_name=enhancer.name if enhancer is not None else None,
        enhancer_label=enhancer.label() if enhancer is not None else None,
    )


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    """Persist results.json (per-clip) and summary.json (aggregates)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    summary_path = out_dir / "summary.json"
    full = report.to_dict()
    results_path.write_text(json.dumps(full, indent=2))
    summary_path.write_text(json.dumps(
        {
            "degradation": full["degradation"],
            "test_set_dir": full["test_set_dir"],
            "started_at": full["started_at"],
            "elapsed_s": full["elapsed_s"],
            "n_clips": len(report.clips),
            "summary": full["summary"],
        },
        indent=2,
    ))
    return results_path, summary_path


__all__ = [
    "ClipRecord",
    "ClipResult",
    "Degradation",
    "Enhancer",
    "EvalReport",
    "load_test_set",
    "run_eval",
    "run_panel_on_clip",
    "write_report",
]


def iter_clip_paths(records: Iterable[ClipRecord]) -> Iterable[Path]:
    for r in records:
        yield r.path
