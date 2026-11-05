"""Two-track restoration eval — the studio-quality scoreboard.

Per the 2026-05-28 eval re-architecture (docs/decisions/log.md): the old 16 kHz /
real-MP3-as-reference panel measured *suppression*, not *restoration toward studio
quality*. This module is the restoration replacement, with two tracks:

- **Studio-reference track** (`run_studio_reference_eval`): take genuinely clean
  studio speech (EARS / DAPS), apply a known degradation recipe (data/degrade.py),
  restore it with each enhancer, and score the restoration against the *true studio
  reference*. Full-band (48 kHz default). Metrics: intrusive-vs-studio
  (PESQ/ESTOI/SI-SDR — diagnostics, they penalise resynthesis), non-intrusive MOS
  (DNSMOS-16k diagnostic + NISQA-48k headline), speaker-similarity vs the clean
  voice, and Whisper-WER vs the clean transcript.

- **Real-podcast no-reference track**: real degraded clips with no clean truth —
  non-intrusive MOS + WER + speaker-sim-vs-input only. (Reuses `score_panel` with
  `has_reference=False`; runner deferred until the silero-VAD re-cut lands.)

Every run can emit an **A/B listening pack** (clean → degraded → each enhancer, as
wavs + a markdown index) because Aviv QAs by ear (ex-sound-designer) and the
metric panel is explicitly not the sole verdict.

The metric functions auto-resample to their own native rates (PESQ→16k, DNSMOS→16k,
NISQA→48k, speaker-sim→16k), so this runner works at a single full-band eval rate
and lets each metric handle its own resampling — except PESQ, which *requires*
16 kHz, so we resample for it explicitly.
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from voice_cleanup.data.degrade import make_chain
from voice_cleanup.data.loudness import TARGET_LUFS, normalize_lufs
from voice_cleanup.eval.enhancers import Enhancer
from voice_cleanup.eval.intrusive import estoi, pesq_wb, si_sdr
from voice_cleanup.eval.nisqa import nisqa_multidim
from voice_cleanup.eval.non_intrusive import dnsmos_p835
from voice_cleanup.eval.speaker_sim import speaker_similarity
from voice_cleanup.eval.wer import transcribe, whisper_wer

logger = logging.getLogger(__name__)

PESQ_SR = 16000
DEFAULT_EVAL_SR = 48000

# Neutral, normal-voice read/conversational EARS material — the right "studio
# clean" reference. Excludes emotional/whispered/shouted/nonverbal recordings.
DEFAULT_EARS_PATTERNS = (
    "sentences_*_regular.wav",
    "rainbow_*_regular.wav",
    "freeform_speech_*.wav",
)


# --- studio clip discovery --------------------------------------------------


@dc.dataclass(frozen=True)
class StudioClip:
    identifier: str
    path: Path
    corpus: str  # "ears" | "daps"
    speaker: str


def discover_ears_clips(
    ears_dir: Path,
    *,
    patterns: tuple[str, ...] = DEFAULT_EARS_PATTERNS,
    per_speaker: int | None = 2,
) -> list[StudioClip]:
    """Find neutral-voice EARS clips, up to `per_speaker` per speaker dir."""
    clips: list[StudioClip] = []
    if not ears_dir.is_dir():
        return clips
    for spk_dir in sorted(p for p in ears_dir.iterdir() if p.is_dir()):
        matched: list[Path] = []
        for pat in patterns:
            matched.extend(sorted(spk_dir.glob(pat)))
        seen: set[Path] = set()
        ordered = [p for p in matched if not (p in seen or seen.add(p))]
        if per_speaker is not None:
            ordered = ordered[:per_speaker]
        for p in ordered:
            clips.append(
                StudioClip(
                    identifier=f"ears-{spk_dir.name}-{p.stem}",
                    path=p,
                    corpus="ears",
                    speaker=spk_dir.name,
                )
            )
    return clips


def discover_daps_clips(
    daps_dir: Path,
    *,
    condition: str = "clean",
    per_speaker: int | None = 1,
) -> list[StudioClip]:
    """Find DAPS studio clips (condition dir), up to `per_speaker` per speaker.

    DAPS filenames are like `f1_script1_clean.wav` — speaker = the `f1`/`m1` prefix.
    """
    clips: list[StudioClip] = []
    cond_dir = daps_dir / "daps" / condition
    if not cond_dir.is_dir():
        return clips
    by_speaker: dict[str, list[Path]] = {}
    for p in sorted(cond_dir.glob("*.wav")):
        speaker = p.stem.split("_", 1)[0]  # f1, m2, …
        by_speaker.setdefault(speaker, []).append(p)
    for speaker, paths in by_speaker.items():
        chosen = paths[:per_speaker] if per_speaker is not None else paths
        for p in chosen:
            clips.append(
                StudioClip(
                    identifier=f"daps-{condition}-{p.stem}",
                    path=p,
                    corpus="daps",
                    speaker=speaker,
                )
            )
    return clips


# --- audio helpers ----------------------------------------------------------


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    return torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy().astype(np.float32)


def _load_clip(
    clip: StudioClip,
    eval_sr: int,
    max_seconds: float | None,
    *,
    target_lufs: float | None = TARGET_LUFS,
) -> np.ndarray:
    """Load → mono → resample → (trim) → loudness-normalize the clean reference.

    EARS/DAPS clips aren't level-calibrated to each other (Aviv 2026-05-29: ~12 dB
    LUFS spread observed), so we normalize the *clean reference* to a common
    integrated loudness before degrading — otherwise the "studio target" set is
    incoherent and the A/B pack would compare level, not quality.
    """
    audio, sr = _read_mono(clip.path)
    audio = _resample(audio, sr, eval_sr)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * eval_sr)]
    if target_lufs is not None:
        audio = normalize_lufs(audio, eval_sr, target_lufs)
    return audio


# --- scoring panel (shared by both tracks) ----------------------------------


def score_panel(
    reference: np.ndarray,
    scored: np.ndarray,
    sr: int,
    *,
    has_reference: bool,
    run_dnsmos: bool = True,
    run_nisqa: bool = True,
    run_wer: bool = True,
    run_spk: bool = True,
    ref_text: str | None = None,
) -> dict[str, Any]:
    """Score `scored` against `reference`. `has_reference=False` skips intrusive metrics.

    On the studio-reference track `reference` is the true clean studio signal; on the
    real-podcast track it's the noisy input (so speaker-sim / WER measure drift from
    the source, and intrusive metrics are skipped).
    """
    out: dict[str, Any] = {
        "pesq": None, "estoi": None, "si_sdr": None,
        "dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovrl": None,
        "nisqa_mos": None, "nisqa_noi": None, "nisqa_dis": None,
        "nisqa_col": None, "nisqa_loud": None,
        "speaker_sim": None,
        "wer": None, "wer_ref_text": ref_text, "wer_hyp_text": None,
    }
    n = min(len(reference), len(scored))
    ref, hyp = reference[:n], scored[:n]

    if has_reference:
        ref16 = _resample(ref, sr, PESQ_SR)
        hyp16 = _resample(hyp, sr, PESQ_SR)
        out["pesq"] = pesq_wb(ref16, hyp16, sr=PESQ_SR)
        out["estoi"] = estoi(ref, hyp, sr=sr)
        out["si_sdr"] = si_sdr(ref, hyp)

    if run_dnsmos:
        dns = dnsmos_p835(hyp, sr=sr)
        out["dnsmos_sig"] = dns["sig"]
        out["dnsmos_bak"] = dns["bak"]
        out["dnsmos_ovrl"] = dns["ovrl"]

    if run_nisqa:
        nq = nisqa_multidim(hyp, sr=sr)
        out["nisqa_mos"] = nq["mos_pred"]
        out["nisqa_noi"] = nq["noi_pred"]
        out["nisqa_dis"] = nq["dis_pred"]
        out["nisqa_col"] = nq["col_pred"]
        out["nisqa_loud"] = nq["loud_pred"]

    if run_spk:
        out["speaker_sim"] = speaker_similarity(ref, hyp, sr=sr)

    if run_wer:
        wer_out = whisper_wer(ref, hyp, sr=sr, ref_text=ref_text)
        out["wer"] = wer_out["wer"]
        out["wer_ref_text"] = wer_out["ref_text"]
        out["wer_hyp_text"] = wer_out["hyp_text"]

    return out


# --- result + report types --------------------------------------------------


@dc.dataclass
class RestorationResult:
    identifier: str
    corpus: str
    speaker: str
    recipe: str
    enhancer: str
    duration_s: float
    pesq: float | None
    estoi: float | None
    si_sdr: float | None
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    dnsmos_ovrl: float | None
    nisqa_mos: float | None
    nisqa_noi: float | None
    nisqa_dis: float | None
    nisqa_col: float | None
    nisqa_loud: float | None
    speaker_sim: float | None
    wer: float | None
    wer_ref_text: str | None
    wer_hyp_text: str | None
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


_SUMMARY_KEYS = (
    "pesq", "estoi", "si_sdr",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
    "speaker_sim", "wer",
)


@dc.dataclass
class StudioReport:
    recipe: str
    eval_sr: int
    started_at: str
    elapsed_s: float
    results: list[RestorationResult]
    listening_pack_dir: str | None = None

    def summary_by_enhancer(self) -> dict[str, dict[str, dict[str, float | int]]]:
        """Per-enhancer mean/std/n over each metric across clips."""
        enhancers = sorted({r.enhancer for r in self.results})
        out: dict[str, dict[str, dict[str, float | int]]] = {}
        for enh in enhancers:
            rows = [r for r in self.results if r.enhancer == enh]
            metrics: dict[str, dict[str, float | int]] = {}
            for k in _SUMMARY_KEYS:
                vals = [getattr(r, k) for r in rows if getattr(r, k) is not None]
                vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                metrics[k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "n": int(len(arr)),
                }
            out[enh] = metrics
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": "studio-reference",
            "recipe": self.recipe,
            "eval_sr": self.eval_sr,
            "started_at": self.started_at,
            "elapsed_s": self.elapsed_s,
            "n_results": len(self.results),
            "listening_pack_dir": self.listening_pack_dir,
            "results": [r.to_dict() for r in self.results],
            "summary_by_enhancer": self.summary_by_enhancer(),
        }


# --- studio-reference runner ------------------------------------------------


def run_studio_reference_eval(
    clips: list[StudioClip],
    recipe: str,
    enhancers: list[Enhancer],
    *,
    eval_sr: int = DEFAULT_EVAL_SR,
    max_seconds: float | None = 20.0,
    seed: int = 0,
    run_dnsmos: bool = True,
    run_nisqa: bool = True,
    run_wer: bool = True,
    run_spk: bool = True,
    listening_pack_dir: Path | None = None,
    on_result: Any = None,
) -> StudioReport:
    """Degrade each clean studio clip, restore with each enhancer, score vs the clean ref.

    The clean reference's Whisper transcript is computed once per clip and reused as
    the WER pseudo-GT for all enhancers (saves N-1 Whisper passes per clip).
    """
    if not clips:
        raise ValueError("no studio clips to evaluate")
    if not enhancers:
        raise ValueError("no enhancers given")

    chain = make_chain(recipe)
    logger.info(
        "Restoration · studio-reference · clips=%d · recipe=%s [%s] · enhancers=%s · "
        "eval_sr=%d · dnsmos=%s nisqa=%s wer=%s spk=%s",
        len(clips), recipe, chain.describe(), [e.name for e in enhancers],
        eval_sr, run_dnsmos, run_nisqa, run_wer, run_spk,
    )
    if listening_pack_dir is not None:
        listening_pack_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    started_at = dt.datetime.now(dt.UTC).isoformat()
    results: list[RestorationResult] = []
    pack_entries: list[dict[str, Any]] = []

    for idx, clip in enumerate(clips):
        logger.info(
            "[%d/%d] clip=%s (%s/%s)",
            idx + 1, len(clips), clip.identifier, clip.corpus, clip.speaker,
        )
        clean = _load_clip(clip, eval_sr, max_seconds)
        degraded = chain.apply(clean, eval_sr, seed=seed + idx)

        ref_text: str | None = None
        if run_wer:
            ref_text = transcribe(clean, sr=eval_sr)

        if listening_pack_dir is not None:
            _write_pack_audio(listening_pack_dir, clip.identifier, "00_clean", clean, eval_sr)
            _write_pack_audio(listening_pack_dir, clip.identifier, "01_degraded", degraded, eval_sr)

        for enh in enhancers:
            t0 = time.monotonic()
            logger.info("  · enhancer=%s", enh.name)
            scored = enh.apply(degraded, eval_sr)
            panel = score_panel(
                clean, scored, eval_sr,
                has_reference=True,
                run_dnsmos=run_dnsmos, run_nisqa=run_nisqa, run_wer=run_wer, run_spk=run_spk,
                ref_text=ref_text,
            )
            result = RestorationResult(
                identifier=clip.identifier, corpus=clip.corpus, speaker=clip.speaker,
                recipe=recipe, enhancer=enh.name, duration_s=len(clean) / eval_sr,
                elapsed_s=time.monotonic() - t0, **{k: panel[k] for k in (
                    "pesq", "estoi", "si_sdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
                    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
                    "speaker_sim", "wer", "wer_ref_text", "wer_hyp_text",
                )},
            )
            results.append(result)
            if listening_pack_dir is not None:
                _write_pack_audio(
                    listening_pack_dir, clip.identifier, f"02_{enh.name}", scored, eval_sr
                )
            if on_result is not None:
                on_result(idx, len(clips), clip, result)

        if listening_pack_dir is not None:
            pack_entries.append(
                {"identifier": clip.identifier, "corpus": clip.corpus, "speaker": clip.speaker}
            )

    elapsed = time.monotonic() - t_start
    logger.info(
        "Restoration · studio-reference · done · %d results in %.1fs", len(results), elapsed
    )
    report = StudioReport(
        recipe=recipe, eval_sr=eval_sr, started_at=started_at, elapsed_s=elapsed,
        results=results, listening_pack_dir=str(listening_pack_dir) if listening_pack_dir else None,
    )
    if listening_pack_dir is not None:
        _write_pack_index(listening_pack_dir, report, [e.name for e in enhancers], pack_entries)
    return report


# --- listening pack + report writers ----------------------------------------


def _write_pack_audio(
    pack_dir: Path, clip_id: str, tag: str, audio: np.ndarray, sr: int,
    *, match_lufs: float | None = TARGET_LUFS,
) -> None:
    """Write one A/B-pack wav, loudness-matched so the ear judges quality not level.

    Every file (clean / degraded / each enhancer) is normalized to the same LUFS,
    so an enhancer can't "win" the listen by being louder. (Noted in INDEX.md so a
    genuine loudness change by an enhancer isn't mistaken for quality — that shows
    up in the metrics, not the pack.)
    """
    out = normalize_lufs(audio, sr, match_lufs) if match_lufs is not None else audio
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak * 0.98
    sf.write(str(pack_dir / f"{clip_id}__{tag}.wav"), out.astype(np.float32), sr)


def _write_pack_index(
    pack_dir: Path, report: StudioReport, enhancer_names: list[str], entries: list[dict[str, Any]]
) -> None:
    lines = [
        f"# A/B listening pack — {report.recipe} degradation @ {report.eval_sr} Hz",
        "",
        f"Generated {report.started_at}. {len(entries)} clips × "
        f"(clean → degraded → {' / '.join(enhancer_names)}).",
        "",
        "Listen in order per clip: **clean** (studio truth) → **degraded** (recipe applied) "
        "→ each **enhancer** output. The question is which enhancer gets closest back to clean.",
        "",
        f"All files are loudness-matched to {TARGET_LUFS:.0f} LUFS (ITU-R BS.1770) so you "
        "judge *quality*, not level. A real loudness change by an enhancer shows up in the "
        "metrics, not here.",
        "",
    ]
    for e in entries:
        cid = e["identifier"]
        lines.append(f"## {cid}  ({e['corpus']} / {e['speaker']})")
        lines.append(f"- `{cid}__00_clean.wav`")
        lines.append(f"- `{cid}__01_degraded.wav`")
        for name in enhancer_names:
            lines.append(f"- `{cid}__02_{name}.wav`")
        lines.append("")
    (pack_dir / "INDEX.md").write_text("\n".join(lines))


def write_studio_report(report: StudioReport, out_dir: Path) -> tuple[Path, Path]:
    """Persist results.json (per row) + REPORT.md (scoreboard)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    report_path = out_dir / "REPORT.md"
    results_path.write_text(json.dumps(report.to_dict(), indent=2))
    report_path.write_text(_render_scoreboard(report))
    return results_path, report_path


def _render_scoreboard(report: StudioReport) -> str:
    summary = report.summary_by_enhancer()
    cols = [
        ("nisqa_mos", "NISQA↑"), ("dnsmos_ovrl", "DNSMOS-OVRL↑"),
        ("speaker_sim", "SpkSim↑"), ("wer", "WER↓"),
        ("pesq", "PESQ↑"), ("estoi", "ESTOI↑"), ("si_sdr", "SI-SDR↑"),
    ]
    head = "| enhancer | " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [head, sep]
    for enh, metrics in summary.items():
        cells = []
        for key, _ in cols:
            m = metrics.get(key)
            cells.append(f"{m['mean']:.3f}" if m else "—")
        rows.append(f"| {enh} | " + " | ".join(cells) + " |")
    n_clips = len({r.identifier for r in report.results})
    return "\n".join([
        f"# Studio-reference restoration scoreboard — {report.recipe} @ {report.eval_sr} Hz",
        "",
        f"- **Track:** studio-reference (clean → degrade `{report.recipe}` → restore → score)",
        f"- **Clips:** {n_clips}  ·  **rows:** {len(report.results)}  ·  "
        f"**elapsed:** {report.elapsed_s:.0f}s  ·  **started:** {report.started_at}",
        f"- **Listening pack:** {report.listening_pack_dir or '(none)'}",
        "",
        "**Headline = NISQA + DNSMOS + speaker-sim + WER.** PESQ/ESTOI/SI-SDR are "
        "diagnostics only — they penalise generative resynthesis that doesn't "
        "waveform-align, so a generative restorer can score *worse* on them while "
        "sounding *better*. Read them as 'how much did the waveform move', not quality.",
        "",
        *rows,
        "",
    ])


__all__ = [
    "StudioClip", "StudioReport", "RestorationResult",
    "discover_ears_clips", "discover_daps_clips",
    "score_panel", "run_studio_reference_eval", "write_studio_report",
    "DEFAULT_EVAL_SR", "DEFAULT_EARS_PATTERNS",
]
