"""Shared test fixtures for the connectomics pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.io.numpy_reader import NumpyReader
from connectomics_pipeline.utils.config import (
    CandidateConfig,
    FragmentConfig,
    GraphConfig,
    PipelineConfig,
    ValidationConfig,
)
from connectomics_pipeline.utils.types import (
    BoundingBox,
    CandidateConnection,
    Fragment,
    Skeleton,
)


@pytest.fixture
def resolution():
    """Standard test resolution in nm (z, y, x)."""
    return (30.0, 8.0, 8.0)


@pytest.fixture
def synthetic_volume():
    """Small synthetic labeled volume with 3 distinct objects."""
    vol = np.zeros((32, 64, 64), dtype=np.uint32)
    # Object 1: horizontal tube
    vol[14:18, 10:15, 5:30] = 1
    # Object 2: horizontal tube (nearby, same label to test stitching)
    vol[14:18, 10:15, 35:60] = 2
    # Object 3: vertical tube
    vol[5:28, 30:34, 30:34] = 3
    return vol


@pytest.fixture
def numpy_reader(synthetic_volume, resolution):
    """NumpyReader wrapping the synthetic volume."""
    return NumpyReader(synthetic_volume, resolution=resolution)


@pytest.fixture
def sample_fragments(resolution):
    """Pre-built fragments for testing."""
    res = np.array(resolution)

    frag_a = Fragment(
        fragment_id=0,
        label_id=1,
        voxel_count=500,
        bounding_box=BoundingBox(
            min_corner=np.array([0.0, 0.0, 0.0]),
            max_corner=np.array([100.0, 100.0, 200.0]),
        ),
        centroid=np.array([50.0, 50.0, 100.0]),
        endpoints=[np.array([50.0, 50.0, 0.0]), np.array([50.0, 50.0, 200.0])],
        skeleton=Skeleton(
            nodes=np.array([[50.0, 50.0, 0.0], [50.0, 50.0, 100.0], [50.0, 50.0, 200.0]]),
            edges=np.array([[0, 1], [1, 2]]),
            radii=np.array([5.0, 5.0, 5.0]),
        ),
    )

    frag_b = Fragment(
        fragment_id=1,
        label_id=1,
        voxel_count=400,
        bounding_box=BoundingBox(
            min_corner=np.array([0.0, 0.0, 250.0]),
            max_corner=np.array([100.0, 100.0, 450.0]),
        ),
        centroid=np.array([50.0, 50.0, 350.0]),
        endpoints=[np.array([50.0, 50.0, 250.0]), np.array([50.0, 50.0, 450.0])],
        skeleton=Skeleton(
            nodes=np.array([[50.0, 50.0, 250.0], [50.0, 50.0, 350.0], [50.0, 50.0, 450.0]]),
            edges=np.array([[0, 1], [1, 2]]),
            radii=np.array([5.0, 5.0, 5.0]),
        ),
    )

    frag_c = Fragment(
        fragment_id=2,
        label_id=2,
        voxel_count=300,
        bounding_box=BoundingBox(
            min_corner=np.array([0.0, 200.0, 0.0]),
            max_corner=np.array([100.0, 400.0, 100.0]),
        ),
        centroid=np.array([50.0, 300.0, 50.0]),
        endpoints=[np.array([50.0, 200.0, 50.0]), np.array([50.0, 400.0, 50.0])],
        skeleton=Skeleton(
            nodes=np.array([[50.0, 200.0, 50.0], [50.0, 300.0, 50.0], [50.0, 400.0, 50.0]]),
            edges=np.array([[0, 1], [1, 2]]),
            radii=np.array([4.0, 4.0, 4.0]),
        ),
    )

    return [frag_a, frag_b, frag_c]


@pytest.fixture
def fragment_store(sample_fragments):
    """FragmentStore populated with sample fragments."""
    store = FragmentStore()
    store.add_many(sample_fragments)
    return store


@pytest.fixture
def default_config():
    """Default pipeline configuration for testing."""
    return PipelineConfig()


@pytest.fixture
def fragment_config():
    return FragmentConfig(min_voxel_count=10)


@pytest.fixture
def graph_config():
    return GraphConfig(max_distance_nm=2000.0)


@pytest.fixture
def candidate_config():
    return CandidateConfig()


@pytest.fixture
def validation_config():
    return ValidationConfig(accept_threshold=0.8, reject_threshold=0.3)
