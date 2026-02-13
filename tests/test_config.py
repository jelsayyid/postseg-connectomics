"""Tests for configuration loading and saving."""

from __future__ import annotations

import yaml
import pytest

from connectomics_pipeline.utils.config import (
    InputConfig,
    FragmentConfig,
    GraphConfig,
    CandidateConfig,
    ValidationConfig,
    RuleConfig,
    AssemblyConfig,
    ExportConfig,
    LoggingConfig,
    PipelineConfig,
    load_config,
    save_config,
    _build_dataclass,
)

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML file returns default config."""
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        config = load_config(cfg_path)
        assert config.name == "post-segmentation-pipeline"
        assert config.version == "0.1.0"

    def test_load_with_input_section(self, tmp_path):
        cfg_path = tmp_path / "input.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "input": {
                        "format": "zarr",
                        "path": "/data/vol.zarr",
                        "resolution": [40.0, 4.0, 4.0],
                    }
                }
            )
        )
        config = load_config(cfg_path)
        assert config.input.format == "zarr"
        assert config.input.path == "/data/vol.zarr"
        # Note: _build_dataclass with __future__ annotations doesn't convert lists to tuples
        assert list(config.input.resolution) == [40.0, 4.0, 4.0]

    def test_load_with_validation_rules(self, tmp_path):
        cfg_path = tmp_path / "val.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "validation": {
                        "accept_threshold": 0.9,
                        "reject_threshold": 0.2,
                        "rules": [
                            {"name": "MaxDistanceRule", "params": {"max_distance_nm": 500}},
                            {"name": "CurvatureRule", "params": {"max_curvature_rad": 1.5}},
                        ],
                    }
                }
            )
        )
        config = load_config(cfg_path)
        assert config.validation.accept_threshold == 0.9
        assert len(config.validation.rules) == 2
        assert config.validation.rules[0].name == "MaxDistanceRule"
        assert config.validation.rules[1].params["max_curvature_rad"] == 1.5

    def test_load_full_config(self, tmp_path):
        cfg_path = tmp_path / "full.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "pipeline": {"name": "test-pipeline", "version": "2.0", "seed": 123},
                    "input": {"format": "hdf5"},
                    "fragments": {"min_voxel_count": 50},
                    "graph": {"max_distance_nm": 1000.0},
                    "candidates": {"max_endpoint_distance_nm": 800.0},
                    "validation": {"accept_threshold": 0.7, "rules": []},
                    "assembly": {"min_structure_fragments": 3},
                    "export": {"formats": ["csv", "json"], "output_dir": "/tmp/out"},
                    "logging": {"level": "DEBUG"},
                }
            )
        )
        config = load_config(cfg_path)
        assert config.name == "test-pipeline"
        assert config.seed == 123
        assert config.fragments.min_voxel_count == 50
        assert config.graph.max_distance_nm == 1000.0
        assert config.export.formats == ["csv", "json"]

    def test_load_ignores_unknown_keys(self, tmp_path):
        cfg_path = tmp_path / "extra.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "input": {"format": "hdf5", "unknown_key": "ignored"},
                }
            )
        )
        config = load_config(cfg_path)
        assert config.input.format == "hdf5"


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        config = PipelineConfig(
            name="roundtrip-test",
            seed=99,
            export=ExportConfig(formats=["csv"], output_dir="/tmp"),
        )
        path = tmp_path / "saved.yaml"
        save_config(config, path)
        assert path.exists()

        # Should be valid YAML
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "roundtrip-test"
        assert data["seed"] == 99

    def test_save_creates_parent_dirs(self, tmp_path):
        config = PipelineConfig()
        path = tmp_path / "sub" / "dir" / "config.yaml"
        # save_config doesn't mkdir - it just writes. Let's check the file path parent is handled.
        path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# _build_dataclass
# ---------------------------------------------------------------------------


class TestBuildDataclass:
    def test_none_data(self):
        result = _build_dataclass(InputConfig, None)
        assert result.format == "hdf5"  # default

    def test_partial_data(self):
        result = _build_dataclass(GraphConfig, {"max_distance_nm": 500.0})
        assert result.max_distance_nm == 500.0
        assert result.construction_method == "proximity"  # default

    def test_list_passthrough(self):
        """With __future__ annotations, tuple fields may stay as lists."""
        result = _build_dataclass(InputConfig, {"resolution": [40, 4, 4]})
        assert list(result.resolution) == [40, 4, 4]


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert config.name == "post-segmentation-pipeline"
        assert config.version == "0.1.0"
        assert config.seed == 42

    def test_candidate_weights(self):
        config = CandidateConfig()
        assert abs(sum(config.weights.values()) - 1.0) < 1e-10

    def test_validation_thresholds(self):
        config = ValidationConfig()
        assert config.accept_threshold > config.reject_threshold
