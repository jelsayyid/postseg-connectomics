"""Structured logging setup for the connectomics pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from connectomics_pipeline.utils.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure structured logging for the pipeline.

    Args:
        config: Logging configuration.

    Returns:
        Configured root logger for the pipeline.
    """
    logger = logging.getLogger("connectomics_pipeline")
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a pipeline module.

    Args:
        name: Module name (e.g. 'fragments.extraction').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"connectomics_pipeline.{name}")
