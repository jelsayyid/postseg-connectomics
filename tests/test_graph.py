"""Tests for the graph module."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.graph.spatial_index import EndpointIndex, SkeletonNodeIndex
from connectomics_pipeline.utils.config import GraphConfig


class TestFragmentGraph:
    def test_add_and_query(self, sample_fragments):
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        assert graph.num_nodes == 3

    def test_add_edge(self, sample_fragments):
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        graph.add_edge(0, 1, distance=50.0)
        assert graph.num_edges == 1
        assert graph.has_edge(0, 1)

    def test_get_neighbors(self, sample_fragments):
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        graph.add_edge(0, 1, distance=50.0)
        graph.add_edge(0, 2, distance=100.0)

        neighbors = graph.get_neighbors(0)
        assert set(neighbors) == {1, 2}

    def test_edge_data(self, sample_fragments):
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        ep_a = np.array([1.0, 2.0, 3.0])
        ep_b = np.array([4.0, 5.0, 6.0])
        graph.add_edge(0, 1, distance=50.0, endpoint_pair=(ep_a, ep_b))

        data = graph.get_edge_data(0, 1)
        assert data is not None
        assert data["distance"] == 50.0
        np.testing.assert_array_equal(data["endpoint_a"], ep_a)

    def test_subgraph(self, sample_fragments):
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        graph.add_edge(0, 1, distance=50.0)
        graph.add_edge(1, 2, distance=100.0)

        sub = graph.subgraph([0, 1])
        assert sub.num_nodes == 2
        assert sub.num_edges == 1


class TestGraphBuilder:
    def test_proximity_build(self, fragment_store):
        config = GraphConfig(construction_method="proximity", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        assert graph.num_nodes == 3
        # Fragments A and B are 250nm apart in centroid, should be connected
        assert graph.has_edge(0, 1)

    def test_endpoint_build(self, fragment_store):
        config = GraphConfig(construction_method="endpoint", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        assert graph.num_nodes == 3


class TestEndpointIndex:
    def test_query_radius(self, sample_fragments):
        index = EndpointIndex(sample_fragments)
        results = index.query_radius(np.array([50.0, 50.0, 200.0]), radius=100.0)
        assert len(results) > 0

    def test_query_nearest(self, sample_fragments):
        index = EndpointIndex(sample_fragments)
        results = index.query_nearest(np.array([50.0, 50.0, 225.0]), k=1)
        assert len(results) == 1
        # Should find the endpoint at [50, 50, 200] or [50, 50, 250]


class TestFragmentGraphMissingCoverage:
    def test_get_neighbors_missing_node_returns_empty(self):
        """get_neighbors for a node not in the graph returns []."""
        graph = FragmentGraph()
        assert graph.get_neighbors(999) == []

    def test_nodes_method(self, sample_fragments):
        """nodes() returns the set of all fragment IDs."""
        graph = FragmentGraph()
        for f in sample_fragments:
            graph.add_fragment(f)
        assert graph.nodes() == {0, 1, 2}

    def test_node_data_method(self, sample_fragments):
        """node_data() returns attribute dict for a given fragment ID."""
        graph = FragmentGraph()
        graph.add_fragment(sample_fragments[0])
        data = graph.node_data(0)
        assert "centroid" in data
        assert "voxel_count" in data


class TestGraphBuilderContactMethod:
    def test_contact_build(self):
        """Contact-based graph construction adds edges for fragments with overlapping bboxes."""
        from connectomics_pipeline.utils.types import BoundingBox, Fragment, Skeleton

        # Create two fragments whose bboxes genuinely overlap so query_bbox finds them
        f0 = Fragment(
            fragment_id=0,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(
                min_corner=np.array([0.0, 0.0, 0.0]),
                max_corner=np.array([100.0, 100.0, 150.0]),
            ),
            centroid=np.array([50.0, 50.0, 75.0]),
            endpoints=[np.array([50.0, 50.0, 0.0]), np.array([50.0, 50.0, 150.0])],
        )
        f1 = Fragment(
            fragment_id=1,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(
                min_corner=np.array([0.0, 0.0, 50.0]),
                max_corner=np.array([100.0, 100.0, 250.0]),
            ),
            centroid=np.array([50.0, 50.0, 150.0]),
            endpoints=[np.array([50.0, 50.0, 50.0]), np.array([50.0, 50.0, 250.0])],
        )
        from connectomics_pipeline.fragments.store import FragmentStore

        store = FragmentStore()
        store.add_many([f0, f1])

        config = GraphConfig(construction_method="contact", max_distance_nm=5000.0)
        builder = GraphBuilder(config)
        graph = builder.build(store)
        assert graph.num_nodes == 2
        assert graph.has_edge(0, 1)

    def test_unknown_method_raises(self, fragment_store):
        config = GraphConfig(construction_method="unknown_method", max_distance_nm=1000.0)
        builder = GraphBuilder(config)
        import pytest

        with pytest.raises(ValueError, match="Unknown graph construction method"):
            builder.build(fragment_store)


class TestEndpointIndexMissingCoverage:
    def test_empty_index_query_radius_returns_empty(self):
        """query_radius on an empty index returns []."""
        index = EndpointIndex([])
        assert index.query_radius(np.zeros(3), radius=100.0) == []

    def test_empty_index_query_nearest_returns_empty(self):
        """query_nearest on an empty index returns []."""
        index = EndpointIndex([])
        assert index.query_nearest(np.zeros(3), k=1) == []


class TestSkeletonNodeIndex:
    def test_indexes_all_skeleton_nodes(self, sample_fragments):
        """SkeletonNodeIndex contains entries for every skeleton node."""
        index = SkeletonNodeIndex(sample_fragments)
        # 3 fragments × 3 nodes each = 9 total
        assert len(index._frag_ids) == 9

    def test_query_radius_batch_finds_cross_fragment_node(self, sample_fragments):
        """Batch query finds frag_b's z=250 node from frag_a's z=200 node (50nm gap)."""
        index = SkeletonNodeIndex(sample_fragments)
        # frag_a's tip node at z=200; frag_b's nearest node is at z=250 → 50nm
        query = np.array([[50.0, 50.0, 200.0]])
        hits_list = index.query_radius_batch(query, radius=100.0)
        assert len(hits_list) == 1
        hit_frag_ids = {index.frag_id_at(i) for i in hits_list[0]}
        assert 1 in hit_frag_ids  # frag_b is within radius

    def test_query_radius_batch_excludes_same_fragment(self, sample_fragments):
        """All hits for a node are labelled with their correct fragment id."""
        index = SkeletonNodeIndex(sample_fragments)
        query = np.array([[50.0, 50.0, 100.0]])  # interior node of frag_a
        hits_list = index.query_radius_batch(query, radius=10.0)
        # Only very close nodes — all should belong to frag_a (id=0)
        for i in hits_list[0]:
            assert index.frag_id_at(i) == 0

    def test_empty_index_returns_empty_lists(self):
        """SkeletonNodeIndex with no fragments returns empty lists per query point."""
        index = SkeletonNodeIndex([])
        query = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = index.query_radius_batch(query, radius=1000.0)
        assert result == [[], []]

    def test_empty_query_returns_empty_list(self, sample_fragments):
        """Zero query points returns empty list (not a crash)."""
        index = SkeletonNodeIndex(sample_fragments)
        result = index.query_radius_batch(np.zeros((0, 3)), radius=500.0)
        assert result == []

    def test_fallback_to_endpoints_for_no_skeleton(self):
        """Fragments without a skeleton fall back to using endpoint positions."""
        from connectomics_pipeline.utils.types import BoundingBox, Fragment

        f = Fragment(
            fragment_id=99,
            label_id=99,
            voxel_count=50,
            bounding_box=BoundingBox(
                min_corner=np.zeros(3), max_corner=np.ones(3) * 100.0
            ),
            centroid=np.array([50.0, 50.0, 50.0]),
            endpoints=[np.array([0.0, 50.0, 50.0]), np.array([100.0, 50.0, 50.0])],
            skeleton=None,
        )
        index = SkeletonNodeIndex([f])
        assert len(index._frag_ids) == 2  # 2 endpoints used as proxy nodes
        assert all(fid == 99 for fid in index._frag_ids)


class TestSkeletonNodeBuilder:
    def test_skeleton_node_build_connects_adjacent_fragments(self, fragment_store):
        """skeleton_node method connects frag_a and frag_b (50nm skeleton-node gap)."""
        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        assert graph.num_nodes == 3
        assert graph.has_edge(0, 1)  # frag_a–frag_b; closest nodes 50nm apart

    def test_skeleton_node_edge_has_distance(self, fragment_store):
        """Edge added by skeleton_node method stores the node-pair distance."""
        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        data = graph.get_edge_data(0, 1)
        assert data is not None
        assert data["distance"] == pytest.approx(50.0)

    def test_skeleton_node_edge_has_endpoint_pair(self, fragment_store):
        """Edge added by skeleton_node method stores the contributing node coords."""
        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        data = graph.get_edge_data(0, 1)
        assert "endpoint_a" in data
        assert "endpoint_b" in data

    def test_skeleton_node_no_edge_beyond_radius(self, fragment_store):
        """skeleton_node method does not connect frag_c (orthogonal; nearest node > 250nm)."""
        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=100.0)
        builder = GraphBuilder(config)
        graph = builder.build(fragment_store)

        assert not graph.has_edge(0, 2)
        assert not graph.has_edge(1, 2)

    def test_skeleton_node_includes_pca_bbox_edges(self):
        """skeleton_node method adds bbox-overlap edges for PCA-only fragments."""
        from connectomics_pipeline.utils.types import BoundingBox, Fragment, Skeleton
        from connectomics_pipeline.fragments.store import FragmentStore

        # fa: small fragment with TEASAR skeleton, far from fb centroid
        fa = Fragment(
            fragment_id=0,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(np.zeros(3), np.array([200.0, 200.0, 200.0])),
            centroid=np.array([100.0, 100.0, 100.0]),
            endpoints=[np.array([5.0, 5.0, 5.0]), np.array([195.0, 195.0, 195.0])],
            skeleton=Skeleton(
                nodes=np.array([[5.0, 5.0, 5.0], [100.0, 100.0, 100.0], [195.0, 195.0, 195.0]]),
                edges=np.array([[0, 1], [1, 2]]),
                radii=np.array([3.0, 3.0, 3.0]),
            ),
        )
        # fb: large PCA-only fragment whose bbox overlaps fa but centroid is far (>500nm)
        fb = Fragment(
            fragment_id=1,
            label_id=2,
            voxel_count=100000,
            bounding_box=BoundingBox(np.array([150.0, 150.0, 150.0]), np.array([2000.0, 2000.0, 2000.0])),
            centroid=np.array([1000.0, 1000.0, 1000.0]),
            endpoints=[np.array([150.0, 150.0, 150.0]), np.array([2000.0, 2000.0, 2000.0])],
            skeleton=None,  # PCA-only
        )
        store = FragmentStore()
        store.add_many([fa, fb])

        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=500.0)
        builder = GraphBuilder(config)
        graph = builder.build(store)

        # fa and fb are > 500nm apart (centroid), skeleton nodes also > 500nm from fb endpoints
        # but their bboxes overlap — the PCA bbox pass should add an edge
        assert graph.has_edge(0, 1), "PCA bbox overlap edge should be added"

    def test_skeleton_node_best_edge_deduplication(self):
        """Only the closest skeleton-node pair is kept per fragment pair."""
        from connectomics_pipeline.utils.types import BoundingBox, Fragment, Skeleton
        from connectomics_pipeline.fragments.store import FragmentStore

        # Two fragments whose skeleton nodes form two close pairs:
        # frag_a nodes: z=0, z=90; frag_b nodes: z=100, z=200
        # Closest pair is (z=90, z=100) → 10nm; next is (z=0, z=100) → 100nm
        fa = Fragment(
            fragment_id=0,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(np.zeros(3), np.array([10.0, 10.0, 90.0])),
            centroid=np.array([5.0, 5.0, 45.0]),
            endpoints=[np.array([5.0, 5.0, 0.0]), np.array([5.0, 5.0, 90.0])],
            skeleton=Skeleton(
                nodes=np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 90.0]]),
                edges=np.array([[0, 1]]),
                radii=np.array([3.0, 3.0]),
            ),
        )
        fb = Fragment(
            fragment_id=1,
            label_id=2,
            voxel_count=100,
            bounding_box=BoundingBox(np.array([0.0, 0.0, 100.0]), np.array([10.0, 10.0, 200.0])),
            centroid=np.array([5.0, 5.0, 150.0]),
            endpoints=[np.array([5.0, 5.0, 100.0]), np.array([5.0, 5.0, 200.0])],
            skeleton=Skeleton(
                nodes=np.array([[5.0, 5.0, 100.0], [5.0, 5.0, 200.0]]),
                edges=np.array([[0, 1]]),
                radii=np.array([3.0, 3.0]),
            ),
        )
        store = FragmentStore()
        store.add_many([fa, fb])

        config = GraphConfig(construction_method="skeleton_node", max_distance_nm=200.0)
        builder = GraphBuilder(config)
        graph = builder.build(store)

        assert graph.num_edges == 1  # only one edge per fragment pair
        data = graph.get_edge_data(0, 1)
        assert data["distance"] == pytest.approx(10.0)  # closest pair selected
