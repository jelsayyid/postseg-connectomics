"""Tests for connectomics_pipeline.utils.logging (setup_logging + get_logger)."""

from __future__ import annotations

import logging

from connectomics_pipeline.utils.config import LoggingConfig
from connectomics_pipeline.utils.logging import get_logger, setup_logging


class TestSetupLogging:
    def test_console_handler_added(self):
        """setup_logging with console=True attaches a StreamHandler."""
        config = LoggingConfig(level="INFO", file="", console=True)
        logger = setup_logging(config)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_file_handler_added(self, tmp_path):
        """setup_logging with a file path attaches a FileHandler."""
        log_file = str(tmp_path / "pipeline.log")
        config = LoggingConfig(level="DEBUG", file=log_file, console=False)
        logger = setup_logging(config)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert (tmp_path / "pipeline.log").exists()

    def test_both_handlers(self, tmp_path):
        """setup_logging can attach both console and file handlers."""
        log_file = str(tmp_path / "both.log")
        config = LoggingConfig(level="INFO", file=log_file, console=True)
        logger = setup_logging(config)
        types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in types
        assert logging.FileHandler in types

    def test_no_handlers_when_both_disabled(self):
        """setup_logging with console=False and no file produces no handlers."""
        config = LoggingConfig(level="WARNING", file="", console=False)
        logger = setup_logging(config)
        assert len(logger.handlers) == 0

    def test_level_applied(self):
        """Logger level is set to the value from config."""
        config = LoggingConfig(level="DEBUG", file="", console=False)
        logger = setup_logging(config)
        assert logger.level == logging.DEBUG

    def test_handlers_cleared_on_each_call(self):
        """Calling setup_logging twice does not accumulate duplicate handlers."""
        config = LoggingConfig(level="INFO", file="", console=True)
        setup_logging(config)
        logger = setup_logging(config)
        # Handlers are cleared before adding new ones
        assert len(logger.handlers) == 1


class TestGetLogger:
    def test_returns_child_logger(self):
        logger = get_logger("fragments.extraction")
        assert logger.name == "connectomics_pipeline.fragments.extraction"

    def test_is_logging_logger(self):
        assert isinstance(get_logger("test"), logging.Logger)
