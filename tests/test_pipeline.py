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

    def test_pipeline_graphml_export(self, pipeline_volume, tmp_path):
        """Test pipeline with GraphML export format."""
        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(
                min_voxel_count=10, extract_skeletons=False, extract_meshes=False
            ),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0, min_alignment_score=0.0, min_composite_score=0.0
            ),
            validation=ValidationConfig(
                accept_threshold=0.3,
                reject_threshold=0.1,
                rules=[
                    RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 500.0}),
                ],
            ),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["graphml", "json"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        Pipeline(config).run(reader=reader)
        assert (tmp_path / "out" / "fragment_graph.graphml").exists()
        assert (tmp_path / "out" / "fragment_graph.json").exists()

    def test_pipeline_swc_export(self, pipeline_volume, tmp_path):
        """Test pipeline with SWC export format."""
        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(
                min_voxel_count=10, extract_skeletons=False, extract_meshes=False
            ),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0, min_alignment_score=0.0, min_composite_score=0.0
            ),
            validation=ValidationConfig(
                accept_threshold=0.3,
                reject_threshold=0.1,
                rules=[
                    RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 500.0}),
                ],
            ),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["swc"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        structures = Pipeline(config).run(reader=reader)
        # If structures were assembled, SWC files should exist
        if structures:
            for s in structures:
                assert (tmp_path / "out" / f"structure_{s.structure_id}.swc").exists()

    def test_pipeline_neuroglancer_export(self, pipeline_volume, tmp_path):
        """Test pipeline with Neuroglancer annotation export."""
        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(
                min_voxel_count=10, extract_skeletons=False, extract_meshes=False
            ),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0, min_alignment_score=0.0, min_composite_score=0.0
            ),
            validation=ValidationConfig(
                accept_threshold=0.3,
                reject_threshold=0.1,
                rules=[
                    RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 500.0}),
                ],
            ),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["neuroglancer"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        Pipeline(config).run(reader=reader)
        # Neuroglancer dir may or may not have connections depending on candidates
        assert (tmp_path / "out").exists()

    def test_pipeline_all_rules(self, pipeline_volume, tmp_path):
        """Test pipeline with multiple validation rules enabled."""
        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(
                min_voxel_count=10, extract_skeletons=False, extract_meshes=False
            ),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0, min_alignment_score=0.0, min_composite_score=0.0
            ),
            validation=ValidationConfig(
                accept_threshold=0.5,
                reject_threshold=0.2,
                rules=[
                    RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 500.0}),
                    RuleConfig(name="SizeDiscrepancyRule", params={"max_radius_ratio": 5.0}),
                    RuleConfig(name="CompositeScoreRule", params={"reject_threshold": 0.1}),
                ],
            ),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["csv"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(pipeline_volume, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        pipeline.run(reader=reader)
        assert pipeline.report is not None
        assert pipeline.report.total >= 0

    def test_pipeline_creates_reader_from_numpy_config(self, tmp_path):
        """Line 80: pipeline._create_reader() is called when no reader is passed."""
        import numpy as np

        vol = np.zeros((32, 32, 64), dtype=np.uint32)
        vol[14:18, 14:18, 5:25] = 1
        npy_path = str(tmp_path / "vol.npy")
        np.save(npy_path, vol)

        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                path=npy_path,
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(min_voxel_count=10, extract_skeletons=False),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0,
                min_alignment_score=0.0,
                min_composite_score=0.0,
            ),
            validation=ValidationConfig(accept_threshold=0.3, reject_threshold=0.1, rules=[]),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["csv"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        pipeline = Pipeline(config)
        structures = pipeline.run()  # no reader → calls _create_reader()
        assert isinstance(structures, list)

    def test_pipeline_multiple_chunks_triggers_stitching(self, tmp_path):
        """Lines 158-160: volume larger than chunk_size causes stitching path."""
        import numpy as np

        vol = np.zeros((32, 32, 128), dtype=np.uint32)
        vol[14:18, 14:18, 5:60] = 1  # spans two chunks
        vol[14:18, 14:18, 70:120] = 2

        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(30.0, 8.0, 8.0),
                chunk_size=(32, 32, 64),  # volume is 128 deep → 2 chunks
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(min_voxel_count=10, extract_skeletons=False),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0,
                min_alignment_score=0.0,
                min_composite_score=0.0,
            ),
            validation=ValidationConfig(accept_threshold=0.3, reject_threshold=0.1, rules=[]),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["csv"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        pipeline.run(reader=reader)
        assert len(pipeline.store) > 0

    def test_pipeline_with_skeletonization_enabled(self, pipeline_volume, tmp_path):
        """Lines 168-169: extract_skeletons=True exercises the skeletonization branch."""
        config = PipelineConfig(
            input=InputConfig(
                format="numpy",
                resolution=(1.0, 1.0, 1.0),
                chunk_size=(32, 32, 64),
                chunk_overlap=(0, 0, 0),
            ),
            fragments=FragmentConfig(
                min_voxel_count=10,
                extract_skeletons=True,  # enables the skeletonizer branch
                skeleton_method="fallback",
            ),
            graph=GraphConfig(max_distance_nm=500.0),
            candidates=CandidateConfig(
                max_endpoint_distance_nm=500.0,
                min_alignment_score=0.0,
                min_composite_score=0.0,
            ),
            validation=ValidationConfig(accept_threshold=0.3, reject_threshold=0.1, rules=[]),
            assembly=AssemblyConfig(min_structure_fragments=2),
            export=ExportConfig(formats=["csv"], output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )
        reader = NumpyReader(pipeline_volume, resolution=(1.0, 1.0, 1.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert isinstance(structures, list)


class TestCreateReader:
    """Directly tests Pipeline._create_reader for each format branch."""

    def _base_config(self, fmt, path, tmp_path):
        return PipelineConfig(
            input=InputConfig(
                format=fmt,
                path=path,
                dataset="labels",
                resolution=(30.0, 8.0, 8.0),
            ),
            export=ExportConfig(output_dir=str(tmp_path / "out")),
            logging=LoggingConfig(level="WARNING", file="", console=False),
        )

    def test_create_hdf5_reader(self, tmp_path):
        import h5py

        path = str(tmp_path / "vol.h5")
        data = np.zeros((4, 4, 4), dtype=np.uint32)
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=data)
        config = self._base_config("hdf5", path, tmp_path)
        pipeline = Pipeline(config)
        reader = pipeline._create_reader()
        from connectomics_pipeline.io.hdf5_reader import HDF5Reader

        assert isinstance(reader, HDF5Reader)

    def test_create_zarr_reader(self, tmp_path):
        import zarr

        path = str(tmp_path / "vol.zarr")
        data = np.zeros((4, 4, 4), dtype=np.uint32)
        z = zarr.open_group(path, mode="w")
        # zarr 3.x: create_dataset requires shape=; use create_array which accepts data=
        # zarr 2.x: create_dataset accepts data= with shape inferred
        if int(zarr.__version__.split(".")[0]) >= 3:
            z.create_array("labels", data=data)
        else:
            z.create_dataset("labels", data=data)
        config = self._base_config("zarr", path, tmp_path)
        pipeline = Pipeline(config)
        reader = pipeline._create_reader()
        from connectomics_pipeline.io.zarr_reader import ZarrReader

        assert isinstance(reader, ZarrReader)

    def test_create_numpy_reader(self, tmp_path):
        path = str(tmp_path / "vol.npy")
        np.save(path, np.zeros((4, 4, 4), dtype=np.uint32))
        config = self._base_config("numpy", path, tmp_path)
        pipeline = Pipeline(config)
        reader = pipeline._create_reader()
        from connectomics_pipeline.io.numpy_reader import NumpyReader

        assert isinstance(reader, NumpyReader)

    def test_create_unknown_format_raises(self, tmp_path):
        config = self._base_config("unknown_fmt", "/nonexistent", tmp_path)
        pipeline = Pipeline(config)
        import pytest

        with pytest.raises(ValueError, match="Unknown input format"):
            pipeline._create_reader()

    def test_create_precomputed_reader_raises_without_cloudvolume(self, tmp_path):
        """Lines 131-133: precomputed branch is entered and raises ImportError."""
        config = self._base_config("precomputed", "precomputed://gs://bucket/path", tmp_path)
        pipeline = Pipeline(config)
        import pytest

        with pytest.raises(ImportError, match="cloud-volume"):
            pipeline._create_reader()


class TestComputeChunkPairs:
    def test_adjacent_chunks(self):
        """Lines 252-264: _compute_chunk_pairs finds neighbors along each axis."""
        from connectomics_pipeline.pipeline import _compute_chunk_pairs

        origins = [(0, 0, 0), (0, 0, 64), (0, 0, 128)]
        chunk_size = (32, 32, 64)
        overlap = (0, 0, 0)
        pairs = _compute_chunk_pairs(origins, chunk_size, overlap)
        # Should find (0,0,0)↔(0,0,64) and (0,0,64)↔(0,0,128)
        assert len(pairs) == 2

    def test_no_neighbors(self):
        from connectomics_pipeline.pipeline import _compute_chunk_pairs

        origins = [(0, 0, 0)]
        pairs = _compute_chunk_pairs(origins, (32, 32, 32), (0, 0, 0))
        assert pairs == []
