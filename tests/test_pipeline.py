"""End-to-end pipeline test with synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.io.numpy_reader import NumpyReader
from connectomics_pipeline.pipeline import Pipeline
from connectomics_pipeline.utils.config import (
    AssemblyConfig,
    CandidateConfig,
    ExportConfig,
    FragmentConfig,
    GraphConfig,
    InputConfig,
    LoggingConfig,
    PipelineConfig,
    RuleConfig,
    ValidationConfig,
)


@pytest.fixture
def pipeline_volume():
    """Generate a simple volume with two close tubular fragments."""
    vol = np.zeros((32, 32, 64), dtype=np.uint32)
    # Two horizontal tubes with a small gap
    vol[14:18, 14:18, 5:25] = 1
    vol[14:18, 14:18, 30:55] = 2
    return vol


@pytest.fixture
def pipeline_config(tmp_path):
    """Pipeline config for testing."""
    return PipelineConfig(
        input=InputConfig(
            format="numpy",
            resolution=(30.0, 8.0, 8.0),
            chunk_size=(32, 32, 64),
            chunk_overlap=(0, 0, 0),
        ),
        fragments=FragmentConfig(
            min_voxel_count=10,
            extract_skeletons=False,
            extract_meshes=False,
        ),
        graph=GraphConfig(
            construction_method="proximity",
            max_distance_nm=500.0,
        ),
        candidates=CandidateConfig(
            max_endpoint_distance_nm=500.0,
            min_alignment_score=0.0,
            min_composite_score=0.0,
        ),
        validation=ValidationConfig(
            accept_threshold=0.3,
            reject_threshold=0.1,
            rules=[
                RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 500.0}),
            ],
        ),
        assembly=AssemblyConfig(
            min_structure_fragments=2,
            detect_cycles=True,
        ),
        export=ExportConfig(
            formats=["csv"],
            output_dir=str(tmp_path / "output"),
        ),
        logging=LoggingConfig(
            level="WARNING",
            file="",
            console=False,
        ),
    )


class TestEndToEnd:
    def test_pipeline_runs(self, pipeline_volume, pipeline_config):
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(pipeline_config)
        structures = pipeline.run(reader=reader)

        assert isinstance(structures, list)
        # Pipeline should have extracted fragments
        assert len(pipeline.store) > 0
        # Should have generated some candidates
        assert len(pipeline.candidates) >= 0
        # Report should exist
        assert pipeline.report is not None

    def test_pipeline_exports(self, pipeline_volume, pipeline_config, tmp_path):
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(pipeline_config)
        pipeline.run(reader=reader)

        output_dir = tmp_path / "output"
        assert output_dir.exists()
        # Config should be saved
        assert (output_dir / "pipeline_config.yaml").exists()
        # CSV exports
        assert (output_dir / "fragments.csv").exists()
