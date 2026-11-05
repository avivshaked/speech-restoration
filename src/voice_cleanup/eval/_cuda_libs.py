"""Make the CUDA-12 runtime libs visible to ctranslate2 (faster-whisper WER).

Torch in this project is a **CUDA-13** build and ships `libcublas.so.13`, but
ctranslate2 4.x links against the **CUDA-12** runtime (`libcublas.so.12`,
`libcudnn.so.9`). Those come from the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
pip packages, but their lib dirs aren't on `LD_LIBRARY_PATH` by default, so
`WhisperModel(..., device="cuda")` fails with "Library libcublas.so.12 is not
found". The two cublas majors coexist (different sonames: .12 vs .13), so adding
the cu12 dirs is safe for torch.

`LD_LIBRARY_PATH` is read by the dynamic linker at process start, so setting it
from inside Python isn't enough — we set it and `os.execv` once (guarded by an
env flag to avoid a re-exec loop). Call this **before importing torch** in any
entry point that runs the Whisper WER metric.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_GUARD = "_VC_CUDA_LIBPATH_SET"


def _nvidia_lib_dirs() -> list[str]:
    dirs: list[str] = []
    for mod in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            lib = os.path.join(list(spec.submodule_search_locations)[0], "lib")
            if os.path.isdir(lib):
                dirs.append(lib)
    return dirs


def ensure_cuda_libpath() -> None:
    """Prepend the cu12 cublas/cudnn lib dirs to LD_LIBRARY_PATH and re-exec once."""
    if os.environ.get(_GUARD):
        return
    dirs = _nvidia_lib_dirs()
    if not dirs:
        return  # nothing to add (CPU-only env, or libs absent) — let ct2 surface its own error
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([cur] if cur else []))
    os.environ[_GUARD] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])  # replaces this process (same PID)


__all__ = ["ensure_cuda_libpath"]
