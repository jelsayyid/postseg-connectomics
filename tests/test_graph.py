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
