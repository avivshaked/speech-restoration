"""Enhancer wrappers for Phase-1 baseline comparisons.

An enhancer takes a degraded 1D waveform + its sample rate and returns the
enhanced 1D waveform at the same length and sample rate. The runner inserts
this step between `Degradation.apply` and metric scoring, so the panel scores
the enhanced output against the clean reference (intrusive) or in isolation
(non-intrusive).

Implementations land here as they're built. Order:
1. PassthroughEnhancer — sanity floor; output == input. Validates the slot
   end-to-end and gives a "what does the panel say about un-enhanced audio?"
   number to compare every other enhancer against.
2. DeepFilterNet3Enhancer — `pip install deepfilternet`.
3. ResembleEnhancer — heavier, vendor or pip.
4. MPSENetEnhancer — vendored research code, mirrors NISQA setup.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Enhancer(ABC):
    """Common interface for Phase-1 baseline enhancers."""

    #: Short, registry-stable identifier (used in CLI args, output paths).
    name: str = "enhancer"

    @abstractmethod
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance `audio`. Return same length, same sr."""

    def label(self) -> str:
        """Human-readable label for reports / CLI output."""
        return self.name


class PassthroughEnhancer(Enhancer):
    """Identity enhancer: output equals input.

    Serves as the panel's noisy-passthrough baseline — the floor that any real
    enhancer must beat. Also useful for end-to-end testing the enhancer slot.
    """

    name = "passthrough"

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        logger.info(
            "Passthrough enhancer · len=%.2fs (%d samples) · sr=%d Hz · no-op",
            len(audio) / sr, len(audio), sr,
        )
        t0 = time.monotonic()
        out = audio.copy()
        logger.info(
            "Passthrough enhancer · done in %.0f µs",
            (time.monotonic() - t0) * 1e6,
        )
        return out

    def label(self) -> str:
        return "passthrough (noisy floor)"


# --- registry --------------------------------------------------------------


_REGISTRY: dict[str, type[Enhancer]] = {
    "passthrough": PassthroughEnhancer,
}


def get_enhancer(name: str, **kwargs: Any) -> Enhancer:
    """Instantiate a registered enhancer by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown enhancer: {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def register(name: str, cls: type[Enhancer]) -> None:
    """Register an enhancer (used by phase-1 baselines as they come online)."""
    _REGISTRY[name] = cls


__all__ = [
    "Enhancer",
    "PassthroughEnhancer",
    "get_enhancer",
    "register",
    "registered_names",
]


# Import concrete enhancers for their registration side-effects, so the registry
# is complete whenever this module is imported (the CLI reads `registered_names()`
# at parse time). Placed at the bottom so `Enhancer` / `register` are already
# defined when these modules import them back.
from voice_cleanup.eval import mpsenet as _mpsenet  # noqa: E402,F401
from voice_cleanup.eval import resemble_enhance as _resemble_enhance  # noqa: E402,F401

# SGMSE+ wrapper imports lazily (the heavy sgmse deps load only on .apply()), so registering it
# is safe even though its isolated env isn't set up yet — it errors helpfully if actually run.
from voice_cleanup.eval import sgmse_enh as _sgmse  # noqa: E402,F401

# VoiceFixer is an optional heavy dep (pip `voicefixer`); register it only if importable so
# the rest of the panel still works without it.
try:
    # Taming presets compose VoiceFixer with a high-shelf cut — only useful if VF imported.
    from voice_cleanup.eval import tame as _tame  # noqa: E402,F401
    from voice_cleanup.eval import voicefixer_enh as _voicefixer  # noqa: E402,F401
except Exception as _exc:  # noqa: BLE001
    logger.info("VoiceFixer enhancer not available (%s) — skipping its registration", _exc)
