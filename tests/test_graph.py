"""Tests for the graph module."""

from __future__ import annotations

import numpy as np

from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.graph.spatial_index import EndpointIndex
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
