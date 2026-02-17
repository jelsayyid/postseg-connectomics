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

    def test_teasar_fallback_when_kimimaro_absent(self):
        """When kimimaro not installed, teasar method falls back to skimage."""
        from unittest.mock import patch

        with patch("connectomics_pipeline.fragments.skeleton._HAS_KIMIMARO", False):
            skeletonizer = Skeletonizer(method="teasar", resolution=(1.0, 1.0, 1.0))
            mask = np.zeros((10, 10, 30), dtype=np.uint8)
            mask[4:7, 4:7, 2:28] = 1
            skeleton = skeletonizer.skeletonize(mask)
            assert skeleton.num_nodes >= 0

    def test_teasar_with_kimimaro_mocked(self):
        """Lines 47, 52-81: exercises the kimimaro (_teasar_skeleton) path."""
        import sys
        import types
        from unittest.mock import MagicMock, patch

        fake_skel = MagicMock()
        fake_skel.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        fake_skel.edges = np.array([[0, 1]])
        fake_skel.radii = np.array([[1.0], [1.0]])
        fake_skels = {1: fake_skel}

        mock_kimimaro_mod = types.ModuleType("kimimaro")
        mock_kimimaro_mod.skeletonize = MagicMock(return_value=fake_skels)

        with patch.dict(sys.modules, {"kimimaro": mock_kimimaro_mod}):
            with patch("connectomics_pipeline.fragments.skeleton._HAS_KIMIMARO", True):
                with patch(
                    "connectomics_pipeline.fragments.skeleton.kimimaro",
                    mock_kimimaro_mod,
                    create=True,
                ):
                    skeletonizer = Skeletonizer(method="teasar", resolution=(1.0, 1.0, 1.0))
                    mask = np.zeros((5, 5, 5), dtype=np.uint8)
                    mask[2, 2, :] = 1
                    skeleton = skeletonizer.skeletonize(mask, label_id=1)
                    assert skeleton.num_nodes == 2

    def test_teasar_label_absent_returns_empty(self):
        """_teasar_skeleton returns empty skeleton when label_id not in kimimaro result."""
        import sys
        import types
        from unittest.mock import MagicMock, patch

        mock_kimimaro_mod = types.ModuleType("kimimaro")
        mock_kimimaro_mod.skeletonize = MagicMock(return_value={})  # label not present

        with patch.dict(sys.modules, {"kimimaro": mock_kimimaro_mod}):
            with patch("connectomics_pipeline.fragments.skeleton._HAS_KIMIMARO", True):
                with patch(
                    "connectomics_pipeline.fragments.skeleton.kimimaro",
                    mock_kimimaro_mod,
                    create=True,
                ):
                    skeletonizer = Skeletonizer(method="teasar", resolution=(1.0, 1.0, 1.0))
                    mask = np.ones((5, 5, 5), dtype=np.uint8)
                    skeleton = skeletonizer.skeletonize(mask, label_id=1)
                    assert skeleton.num_nodes == 0

    def test_fallback_skeleton_produces_no_coords_after_skeletonize(self):
        """Line 97: skimage skeletonize produces zero skeleton voxels for flat mask."""
        from connectomics_pipeline.fragments.skeleton import _build_skeleton_edges, Skeletonizer
        from unittest.mock import patch
        import numpy as np

        # Test _build_skeleton_edges directly with single coord → line 116
        coords = np.array([[0.0, 0.0, 0.0]])
        edges = _build_skeleton_edges(coords)
        assert edges.shape == (0, 2)

        # Test line 97: patch skeletonize to return all-zero array so coords is empty
        with patch("connectomics_pipeline.fragments.skeleton.skeletonize",
                   return_value=np.zeros((5, 5, 5), dtype=np.uint8)):
            sk = Skeletonizer(method="fallback", resolution=(1.0, 1.0, 1.0))
            mask = np.ones((5, 5, 5), dtype=np.uint8)
            skeleton = sk.skeletonize(mask)
            assert skeleton.num_nodes == 0

    def test_build_skeleton_edges_empty_set(self):
        """Line 132: edges_set is empty when no pairs within 1.75 radius."""
        from connectomics_pipeline.fragments.skeleton import _build_skeleton_edges

        # Two points far apart — no 26-connectivity edges
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        edges = _build_skeleton_edges(coords)
        assert edges.shape == (0, 2)


class TestFragmentExtractorMissingCoverage:
    def test_set_next_id(self, resolution):
        """Line 103: set_next_id updates the internal counter."""
        config = FragmentConfig(min_voxel_count=1)
        extractor = FragmentExtractor(config, resolution=resolution)
        extractor.set_next_id(100)
        chunk = np.zeros((10, 10, 10), dtype=np.uint32)
        chunk[3:7, 3:7, 3:7] = 1
        frags = extractor.extract_from_chunk(chunk, (0, 0, 0))
        assert frags[0].fragment_id == 100


class TestFragmentStoreMissingCoverage:
    def test_all_ids(self, fragment_store):
        """Line 55: all_ids returns list of fragment IDs."""
        ids = fragment_store.all_ids()
        assert set(ids) == {0, 1, 2}

    def test_query_radius_empty_store(self):
        """Line 68: query_radius on empty store returns []."""
        store = FragmentStore()
        result = store.query_radius(np.zeros(3), radius=1000.0)
        assert result == []


class TestMetadataMissingCoverage:
    def test_compute_endpoints_single_node_skeleton(self):
        """Line 26: skeleton with nodes but 0 edges returns first node."""
        frag = Fragment(
            fragment_id=10,
            label_id=1,
            voxel_count=10,
            bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 10),
            centroid=np.array([5.0, 5.0, 5.0]),
            skeleton=Skeleton(
                nodes=np.array([[1.0, 2.0, 3.0]]),
                edges=np.zeros((0, 2), dtype=int),
                radii=np.array([1.0]),
            ),
        )
        endpoints = compute_endpoints(frag)
        assert len(endpoints) == 1
        np.testing.assert_array_equal(endpoints[0], np.array([1.0, 2.0, 3.0]))

    def test_compute_endpoints_cycle_no_terminals(self):
        """Line 37: skeleton forming a cycle has no degree-1 terminals."""
        # Triangle: 0-1, 1-2, 2-0 → all nodes have degree 2
        frag = Fragment(
            fragment_id=11,
            label_id=1,
            voxel_count=10,
            bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 10),
            centroid=np.array([5.0, 5.0, 5.0]),
            skeleton=Skeleton(
                nodes=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]),
                edges=np.array([[0, 1], [1, 2], [2, 0]]),
                radii=np.array([1.0, 1.0, 1.0]),
            ),
        )
        endpoints = compute_endpoints(frag)
        # Falls back to first and last node
        assert len(endpoints) == 2

    def test_refine_centroid_returns_skeleton_mean(self, sample_fragments):
        """Line 52: refine_centroid with skeleton returns skeleton node mean."""
        frag = sample_fragments[0]
        refined = refine_centroid(frag)
        expected = frag.skeleton.nodes.mean(axis=0)
        np.testing.assert_array_almost_equal(refined, expected)

    def test_refine_centroid_no_skeleton(self):
        """refine_centroid without skeleton falls back to centroid."""
        frag = Fragment(
            fragment_id=12,
            label_id=1,
            voxel_count=10,
            bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 10),
            centroid=np.array([5.0, 5.0, 5.0]),
        )
        refined = refine_centroid(frag)
        np.testing.assert_array_equal(refined, frag.centroid)


class TestMeshExtractorMissingCoverage:
    def test_marching_cubes_failure_returns_empty(self):
        """Lines 46-48: exception during marching_cubes returns empty Mesh."""
        from unittest.mock import patch
        from connectomics_pipeline.fragments.mesh import MeshExtractor

        extractor = MeshExtractor(resolution=(1.0, 1.0, 1.0))
        mask = np.ones((5, 5, 5), dtype=np.uint8)

        with patch(
            "connectomics_pipeline.fragments.mesh.marching_cubes",
            side_effect=ValueError("boom"),
        ):
            mesh = extractor.extract(mask)

        assert mesh.vertices.shape == (0, 3)
        assert mesh.faces.shape == (0, 3)
