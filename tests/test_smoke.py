"""Smoke test: confirms the package imports cleanly and basic deps load."""

import voice_cleanup


def test_package_imports():
    assert voice_cleanup.__version__ == "0.1.0"


def test_torch_available():
    import torch

    assert torch.__version__


def test_torchaudio_available():
    import torchaudio

    assert torchaudio.__version__
