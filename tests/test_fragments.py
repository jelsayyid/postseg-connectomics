"""Tests for the fragments module."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.fragments.extraction import FragmentExtractor
from connectomics_pipeline.fragments.metadata import compute_endpoints, refine_centroid
from connectomics_pipeline.fragments.skeleton import Skeletonizer
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.config import FragmentConfig
from connectomics_pipeline.utils.types import BoundingBox, Fragment, Skeleton


class TestFragmentExtractor:
    def test_extract_basic(self, resolution):
        config = FragmentConfig(min_voxel_count=5)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        chunk[5:10, 5:10, 5:10] = 1  # 125 voxels

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 1
        assert fragments[0].label_id == 1
        assert fragments[0].voxel_count == 125

    def test_extract_multiple_labels(self, resolution):
        config = FragmentConfig(min_voxel_count=5)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        chunk[2:8, 2:8, 2:8] = 1
        chunk[12:18, 12:18, 12:18] = 2

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 2
        labels = {f.label_id for f in fragments}
        assert labels == {1, 2}

    def test_min_voxel_filter(self, resolution):
        config = FragmentConfig(min_voxel_count=200)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        chunk[5:10, 5:10, 5:10] = 1  # 125 voxels, should be filtered

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 0

    def test_boundary_detection(self, resolution):
        config = FragmentConfig(min_voxel_count=5)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        chunk[0:5, 5:10, 5:10] = 1  # Touches z=0 boundary

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 1
        assert fragments[0].is_boundary is True

    def test_non_boundary(self, resolution):
        config = FragmentConfig(min_voxel_count=5)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        chunk[5:10, 5:10, 5:10] = 1  # Interior

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 1
        assert fragments[0].is_boundary is False

    def test_connected_components(self, resolution):
        config = FragmentConfig(min_voxel_count=5)
        extractor = FragmentExtractor(config, resolution=resolution)

        chunk = np.zeros((20, 20, 20), dtype=np.uint32)
        # Same label, two disconnected regions
        chunk[2:5, 2:5, 2:5] = 1
        chunk[15:18, 15:18, 15:18] = 1

        fragments = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert len(fragments) == 2
        assert all(f.label_id == 1 for f in fragments)


class TestFragmentStore:
    def test_add_and_get(self, sample_fragments):
        store = FragmentStore()
        store.add(sample_fragments[0])
        assert len(store) == 1
        assert sample_fragments[0].fragment_id in store
        assert store[sample_fragments[0].fragment_id] is sample_fragments[0]

    def test_add_many(self, sample_fragments):
        store = FragmentStore()
        store.add_many(sample_fragments)
        assert len(store) == 3

    def test_query_radius(self, fragment_store, sample_fragments):
        # Query near fragment A's centroid
        results = fragment_store.query_radius(sample_fragments[0].centroid, 100.0)
        assert len(results) >= 1
        assert sample_fragments[0].fragment_id in [f.fragment_id for f in results]

    def test_query_bbox(self, fragment_store):
        bbox = BoundingBox(
            min_corner=np.array([0.0, 0.0, 0.0]),
            max_corner=np.array([100.0, 100.0, 500.0]),
        )
        results = fragment_store.query_bbox(bbox)
        assert len(results) >= 2  # fragments A and B

    def test_remove(self, fragment_store, sample_fragments):
        fragment_store.remove(sample_fragments[0].fragment_id)
        assert len(fragment_store) == 2
        assert sample_fragments[0].fragment_id not in fragment_store


class TestMetadata:
    def test_compute_endpoints_with_skeleton(self, sample_fragments):
        frag = sample_fragments[0]
        endpoints = compute_endpoints(frag)
        assert len(endpoints) == 2  # Two terminal nodes

    def test_compute_endpoints_without_skeleton(self):
        frag = Fragment(
            fragment_id=99,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 100),
            centroid=np.array([50.0, 50.0, 50.0]),
        )
        endpoints = compute_endpoints(frag)
        assert len(endpoints) == 1
        np.testing.assert_array_equal(endpoints[0], frag.centroid)

    def test_refine_centroid(self, sample_fragments):
        frag = sample_fragments[0]
        refined = refine_centroid(frag)
        expected = frag.skeleton.nodes.mean(axis=0)
        np.testing.assert_array_almost_equal(refined, expected)


class TestSkeletonizer:
    def test_fallback_skeletonize(self):
        skeletonizer = Skeletonizer(method="fallback", resolution=(1.0, 1.0, 1.0))
        # Create a small tube
        mask = np.zeros((10, 10, 30), dtype=np.uint8)
        mask[4:7, 4:7, 2:28] = 1

        skeleton = skeletonizer.skeletonize(mask)
        assert skeleton.num_nodes > 0
        assert skeleton.nodes.shape[1] == 3

    def test_empty_mask(self):
        skeletonizer = Skeletonizer(method="fallback", resolution=(1.0, 1.0, 1.0))
        mask = np.zeros((10, 10, 10), dtype=np.uint8)

        skeleton = skeletonizer.skeletonize(mask)
        assert skeleton.num_nodes == 0
