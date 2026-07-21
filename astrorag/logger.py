"""
Centralised logging for AstroRAG.

All modules import their logger from this module rather than
calling logging.getLogger directly. This ensures consistent
formatting, file handlers, and log levels across the pipeline.
"""

from __future__ import annotations

import logging
import sys
from   pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from astrorag.paths import get_paths

# ── module-level state ────────────────────────────────────
_configured = False
_console    = Console(stderr=True)


def setup_logging(
    level:      str  = "INFO",
    log_file:   str  = "astrorag.log",
    use_rich:   bool = True,
) -> None:
    """
    Configure the root logger with console and file handlers.

    Idempotent — safe to call multiple times. Only configures once.

    Args:
        level:    Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Filename for the log file (created in logs/ directory).
        use_rich: If True, use rich formatted console output;
                  else plain stderr.
    """
    global _configured
    if _configured:
        return

    paths      = get_paths()
    log_path   = paths.log_path(Path(log_file).name)
    log_level  = getattr(logging, level.upper(), logging.INFO)

    # root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # remove any pre-existing handlers to prevent duplication
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # ── file handler ─────────────────────────────────────
    file_handler   = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # ── console handler ──────────────────────────────────
    if use_rich:
        rich_handler = RichHandler(
            console            = _console,
            show_time          = True,
            show_level         = True,
            show_path          = False,
            rich_tracebacks    = True,
            tracebacks_show_locals = False,
            markup             = True,
        )
        rich_handler.setLevel(log_level)
        root_logger.addHandler(rich_handler)
    else:
        stream_handler   = logging.StreamHandler(sys.stderr)
        stream_formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(log_level)
        root_logger.addHandler(stream_handler)

    # silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "openai", "httpcore", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True

    # log the setup completion
    setup_logger = logging.getLogger("astrorag.logger")
    setup_logger.info(
        f"Logging configured — level={level}, file={log_path}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the specified module name.

    Automatically configures logging on first call if not already
    configured, using default settings.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)