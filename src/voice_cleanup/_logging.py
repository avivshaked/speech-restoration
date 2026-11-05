"""Centralized logging setup for voice_cleanup.

Default: verbose. Anything calling `setup_logging()` (or importing the package
top-level) gets a `RichHandler` at INFO level with timestamps and source paths.

Toggle with the `VOICE_CLEANUP_LOG_LEVEL` env var ("DEBUG", "INFO", "WARNING")
or pass `level=` explicitly. CLI scripts should call `setup_logging()` once on
startup; library modules just do `logger = logging.getLogger(__name__)`.
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False
_console: Console | None = None


def get_console() -> Console:
    """Return the shared Console instance used by RichHandler and direct output."""
    global _console
    if _console is None:
        _console = Console(stderr=False, log_time_format="%H:%M:%S")
    return _console


def setup_logging(level: str | int | None = None, force: bool = False) -> None:
    """Install a `RichHandler` on the root logger. Idempotent unless `force=True`.

    Args:
        level: Log level name or int. Defaults to env var `VOICE_CLEANUP_LOG_LEVEL`
            or "INFO".
        force: Re-install even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if level is None:
        level = os.environ.get("VOICE_CLEANUP_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    console = get_console()
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="%H:%M:%S",
    )

    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet a few noisy third-party loggers; promote ours.
    for noisy in ("urllib3", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("voice_cleanup").setLevel(level)

    _CONFIGURED = True
