"""Non-intrusive (reference-free) speech quality metrics.

DNSMOS P.835 (Reddy et al. 2022) and SQUIM-MOS (Kumar et al. 2023).
NISQA is added in a follow-up via a vendored upstream snapshot.

DNSMOS preprocessing mirrors microsoft/DNS-Challenge:DNSMOS/dnsmos_local.py (MIT)
at https://github.com/microsoft/DNS-Challenge.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

logger = logging.getLogger(__name__)

# --- cache + model setup -----------------------------------------------------

CACHE_DIR = Path(
    os.environ.get("VOICE_CLEANUP_CACHE", str(Path.home() / ".cache" / "voice_cleanup"))
)
MODELS_DIR = CACHE_DIR / "models"

DNSMOS_P835_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_P835_PATH = MODELS_DIR / "dnsmos_p835.onnx"
DNSMOS_P835_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"

DNSMOS_SR = 16000
DNSMOS_INPUT_LENGTH_S = 9.01

_DNSMOS_SESSION: ort.InferenceSession | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dnsmos_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DNSMOS_P835_PATH.exists():
        size_mb = DNSMOS_P835_PATH.stat().st_size / 1024**2
        logger.info(
            "DNSMOS · model already cached at %s (%.2f MB) — verifying SHA-256",
            DNSMOS_P835_PATH, size_mb,
        )
        if _sha256(DNSMOS_P835_PATH) == DNSMOS_P835_SHA256:
            logger.info("DNSMOS · hash OK (%s)", DNSMOS_P835_SHA256[:12])
            return DNSMOS_P835_PATH
        logger.warning("DNSMOS · hash mismatch on disk — re-downloading")
    logger.info(
        "DNSMOS · model NOT cached — downloading (~1.2 MB) from %s to %s",
        DNSMOS_P835_URL, DNSMOS_P835_PATH,
    )
    t0 = time.monotonic()
    urllib.request.urlretrieve(DNSMOS_P835_URL, DNSMOS_P835_PATH)
    size_mb = DNSMOS_P835_PATH.stat().st_size / 1024**2
    logger.info(
        "DNSMOS · download complete in %.2fs · %.2f MB on disk",
        time.monotonic() - t0, size_mb,
    )
    actual = _sha256(DNSMOS_P835_PATH)
    if actual != DNSMOS_P835_SHA256:
        raise RuntimeError(
            f"DNSMOS model hash mismatch: got {actual}, expected {DNSMOS_P835_SHA256}"
        )
    logger.info("DNSMOS · downloaded hash OK (%s)", actual[:12])
    return DNSMOS_P835_PATH


def _get_dnsmos_session() -> ort.InferenceSession:
    global _DNSMOS_SESSION
    if _DNSMOS_SESSION is None:
        path = _ensure_dnsmos_model()
        logger.info("DNSMOS · loading ONNX session (CPU provider) from %s", path)
        t0 = time.monotonic()
        _DNSMOS_SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logger.info(
            "DNSMOS · session ready in %.0f ms · providers=%s",
            (time.monotonic() - t0) * 1000, _DNSMOS_SESSION.get_providers(),
        )
    return _DNSMOS_SESSION


# --- helpers -----------------------------------------------------------------


def _to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D signal, got shape {arr.shape}")
    return arr


def _resample_if_needed(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sr, target_sr)
    return resampled.squeeze(0).numpy()


# --- DNSMOS P.835 ------------------------------------------------------------


def _dnsmos_polyfit(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
    """Non-personalized polynomial post-fit from microsoft/DNS-Challenge."""
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))


def dnsmos_p835(signal: Any, sr: int = DNSMOS_SR) -> dict[str, float]:
    """Predict DNSMOS P.835 scores: SIG, BAK, OVRL (and raw variants).

    Args:
        signal: 1D waveform (numpy or torch). Any sample rate; resampled to 16 kHz.
        sr: Sample rate of `signal`.

    Returns:
        Dict with keys: sig, bak, ovrl, sig_raw, bak_raw, ovrl_raw. Each in MOS range [1, 5].
    """
    sess = _get_dnsmos_session()
    audio = _to_numpy_1d(signal)
    original_len = len(audio)
    logger.info(
        "DNSMOS · input sr=%d Hz, len=%.2fs (%d samples)",
        sr, original_len / sr, original_len,
    )
    if sr != DNSMOS_SR:
        logger.info("DNSMOS · resampling %d → %d Hz", sr, DNSMOS_SR)
        audio = _resample_if_needed(audio, sr, DNSMOS_SR)

    len_samples = int(DNSMOS_INPUT_LENGTH_S * DNSMOS_SR)
    if len(audio) < len_samples:
        logger.info(
            "DNSMOS · signal shorter than %.2fs window; repeat-padding from %.2fs",
            DNSMOS_INPUT_LENGTH_S, len(audio) / DNSMOS_SR,
        )
        while len(audio) < len_samples:
            audio = np.concatenate([audio, audio])
        logger.info("DNSMOS · padded to %.2fs", len(audio) / DNSMOS_SR)

    hop_samples = DNSMOS_SR  # 1-second hop
    num_hops = int(np.floor(len(audio) / DNSMOS_SR) - DNSMOS_INPUT_LENGTH_S) + 1
    logger.info("DNSMOS · running inference on %d × 9.01s windows (1s hop)", num_hops)

    sig_raws, bak_raws, ovr_raws = [], [], []
    sigs, baks, ovrs = [], [], []

    t0 = time.monotonic()
    for idx in range(num_hops):
        start = idx * hop_samples
        end = int(start + len_samples)
        seg = audio[start:end]
        if len(seg) < len_samples:
            continue
        x = seg.astype(np.float32)[np.newaxis, :]
        sig_raw, bak_raw, ovr_raw = sess.run(None, {"input_1": x})[0][0]
        sig, bak, ovr = _dnsmos_polyfit(sig_raw, bak_raw, ovr_raw)
        sig_raws.append(float(sig_raw))
        bak_raws.append(float(bak_raw))
        ovr_raws.append(float(ovr_raw))
        sigs.append(sig)
        baks.append(bak)
        ovrs.append(ovr)

    if not sigs:
        raise RuntimeError("DNSMOS produced no segments; signal may be malformed")

    result = {
        "sig": float(np.mean(sigs)),
        "bak": float(np.mean(baks)),
        "ovrl": float(np.mean(ovrs)),
        "sig_raw": float(np.mean(sig_raws)),
        "bak_raw": float(np.mean(bak_raws)),
        "ovrl_raw": float(np.mean(ovr_raws)),
        "num_hops": int(len(sigs)),
    }
    logger.info(
        "DNSMOS = SIG %.2f · BAK %.2f · OVRL %.2f (n=%d windows, %.0f ms total)",
        result["sig"], result["bak"], result["ovrl"], result["num_hops"],
        (time.monotonic() - t0) * 1000,
    )
    return result


# --- SQUIM-MOS ---------------------------------------------------------------

SQUIM_SR = 16000
_SQUIM_MODEL: torch.nn.Module | None = None


def _squim_cache_info() -> tuple[bool, Path, int]:
    """Return (cached_path_exists, expected_path, bytes_on_disk) for SQUIM weights.

    torchaudio writes bundle weights to <hub_dir>/torchaudio/models/<bundle._path>.
    The SQUIM_SUBJECTIVE bundle's `_path` is `squim_subjective_bvcc_daps.pth`.
    """
    from torchaudio.pipelines import SQUIM_SUBJECTIVE

    expected = (
        Path(torch.hub.get_dir())
        / "torchaudio"
        / "models"
        / getattr(SQUIM_SUBJECTIVE, "_path", "squim_subjective_bvcc_daps.pth")
    )
    if expected.is_file():
        return True, expected, expected.stat().st_size
    return False, expected, 0


def _get_squim_model() -> torch.nn.Module:
    global _SQUIM_MODEL
    if _SQUIM_MODEL is None:
        from torchaudio.pipelines import SQUIM_SUBJECTIVE

        cached, cache_path, cache_bytes = _squim_cache_info()
        if cached:
            logger.info(
                "SQUIM · weights already cached at %s (%.2f MB) — loading from disk · "
                "license=CC-BY-NC 4.0",
                cache_path, cache_bytes / 1024**2,
            )
        else:
            logger.info(
                "SQUIM · weights NOT cached — first run will download (~150 MB) to %s · "
                "license=CC-BY-NC 4.0",
                cache_path,
            )
        logger.info("SQUIM · initializing SQUIM_SUBJECTIVE model")
        t0 = time.monotonic()
        _SQUIM_MODEL = SQUIM_SUBJECTIVE.get_model().eval()
        elapsed = time.monotonic() - t0
        _, _, bytes_after = _squim_cache_info()
        logger.info(
            "SQUIM · model ready in %.2fs · on-disk size %.2f MB%s",
            elapsed, bytes_after / 1024**2,
            " (download completed during load)" if not cached else "",
        )
    return _SQUIM_MODEL


def squim_mos(signal: Any, reference: Any, sr: int = SQUIM_SR) -> float:
    """Predict subjective MOS (Kumar et al. 2023) via torchaudio SQUIM.

    Args:
        signal: 1D waveform to score.
        reference: 1D non-matching clean reference. Any clean speech; should NOT match
            `signal`'s content. The CLI is expected to fix this across runs for
            comparability. Weights are CC-BY-NC 4.0 — flag for any commercial use.
        sr: Sample rate for both inputs. Both are resampled to 16 kHz internally.

    Returns:
        MOS score, scalar float in approximately [1, 5].
    """
    model = _get_squim_model()
    sig = _to_numpy_1d(signal)
    ref = _to_numpy_1d(reference)
    logger.info(
        "SQUIM-MOS · sig=%.2fs · NMR-ref=%.2fs · sr=%d Hz",
        len(sig) / sr, len(ref) / sr, sr,
    )
    if sr != SQUIM_SR:
        logger.info("SQUIM-MOS · resampling both to %d Hz", SQUIM_SR)
    sig = _resample_if_needed(sig, sr, SQUIM_SR)
    ref = _resample_if_needed(ref, sr, SQUIM_SR)

    sig_t = torch.from_numpy(sig).unsqueeze(0)
    ref_t = torch.from_numpy(ref).unsqueeze(0)
    t0 = time.monotonic()
    with torch.inference_mode():
        score = model(sig_t, ref_t)
    val = float(score.squeeze().item())
    logger.info("SQUIM-MOS = %.3f (computed in %.0f ms)", val, (time.monotonic() - t0) * 1000)
    return val
