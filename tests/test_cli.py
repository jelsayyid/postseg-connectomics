"""Tests for the CLI entry point (connectomics_pipeline.cli)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from connectomics_pipeline.cli import main


class TestCLI:
    def test_basic_invocation(self, tmp_path):
        """main() loads config, runs pipeline, and prints summary."""
        mock_config = MagicMock()
        mock_config.export.output_dir = str(tmp_path / "out")
        mock_structures = [MagicMock(), MagicMock()]

        with patch("connectomics_pipeline.cli.load_config", return_value=mock_config) as mock_load:
            with patch("connectomics_pipeline.cli.Pipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = mock_structures
                main(["--config", "test.yaml"])

        mock_load.assert_called_once_with("test.yaml")
        MockPipeline.return_value.run.assert_called_once()

    def test_output_dir_override(self, tmp_path):
        """--output-dir flag overrides config.export.output_dir."""
        mock_config = MagicMock()
        mock_config.export.output_dir = str(tmp_path / "orig")

        with patch("connectomics_pipeline.cli.load_config", return_value=mock_config):
            with patch("connectomics_pipeline.cli.Pipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = []
                main(["--config", "test.yaml", "--output-dir", "/new/out"])

        assert mock_config.export.output_dir == "/new/out"

    def test_verbose_flag_sets_debug_level(self):
        """--verbose sets config.logging.level to DEBUG."""
        mock_config = MagicMock()
        mock_config.export.output_dir = "/tmp/out"

        with patch("connectomics_pipeline.cli.load_config", return_value=mock_config):
            with patch("connectomics_pipeline.cli.Pipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = []
                main(["--config", "test.yaml", "--verbose"])

        assert mock_config.logging.level == "DEBUG"

    def test_no_output_dir_override_when_not_passed(self, tmp_path):
        """Without --output-dir, config.export.output_dir is not changed."""
        mock_config = MagicMock()
        original_dir = str(tmp_path / "original")
        mock_config.export.output_dir = original_dir
        # args.output_dir will be None; the conditional should not fire
        mock_config.export.output_dir = original_dir

        with patch("connectomics_pipeline.cli.load_config", return_value=mock_config):
            with patch("connectomics_pipeline.cli.Pipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = []
                main(["--config", "test.yaml"])

        # The attribute should not have been reassigned inside main
        assert mock_config.export.output_dir == original_dir

    def test_missing_config_arg_exits(self):
        """Omitting required --config argument raises SystemExit."""
        with pytest.raises(SystemExit):
            main([])
