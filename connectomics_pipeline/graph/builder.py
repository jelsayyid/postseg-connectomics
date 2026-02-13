"""Graph construction from fragment spatial relationships."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.graph.spatial_index import EndpointIndex
from connectomics_pipeline.utils.config import GraphConfig
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import Fragment

logger = get_logger("graph.builder")


class GraphBuilder:
    """Build a fragment adjacency graph using various strategies."""

    def __init__(self, config: GraphConfig):
        self.config = config

    def build(self, store: FragmentStore) -> FragmentGraph:
        """Build a fragment graph from all fragments in the store.

        Args:
            store: Fragment store with spatial index.

        Returns:
            FragmentGraph with nodes and edges.
        """
        graph = FragmentGraph()
        fragments = store.all_fragments()

        for f in fragments:
            graph.add_fragment(f)

        method = self.config.construction_method
        if method == "contact":
            self._add_contact_edges(graph, fragments, store)
        elif method == "proximity":
            self._add_proximity_edges(graph, fragments, store)
        elif method == "endpoint":
            self._add_endpoint_edges(graph, fragments)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")

        logger.info(
            "Built graph with %d nodes, %d edges (method=%s)",
            graph.num_nodes,
            graph.num_edges,
            method,
        )
        return graph

    def _add_contact_edges(
        self, graph: FragmentGraph, fragments: List[Fragment], store: FragmentStore
    ) -> None:
        """Add edges where fragment bounding boxes overlap (contact-based)."""
        for f in fragments:
            expanded = f.bounding_box.expand(1.0)
            neighbors = store.query_bbox(expanded)
            for n in neighbors:
                if n.fragment_id != f.fragment_id and not graph.has_edge(
                    f.fragment_id, n.fragment_id
                ):
                    dist = float(np.linalg.norm(f.centroid - n.centroid))
                    if dist <= self.config.max_distance_nm:
                        graph.add_edge(f.fragment_id, n.fragment_id, distance=dist)

    def _add_proximity_edges(
        self, graph: FragmentGraph, fragments: List[Fragment], store: FragmentStore
    ) -> None:
        """Add edges between fragments within max_distance_nm."""
        for f in fragments:
            neighbors = store.query_radius(f.centroid, self.config.max_distance_nm)
            for n in neighbors:
                if n.fragment_id != f.fragment_id and not graph.has_edge(
                    f.fragment_id, n.fragment_id
                ):
                    dist = float(np.linalg.norm(f.centroid - n.centroid))
                    graph.add_edge(f.fragment_id, n.fragment_id, distance=dist)

    def _add_endpoint_edges(self, graph: FragmentGraph, fragments: List[Fragment]) -> None:
        """Add edges connecting nearest endpoint pairs."""
        endpoint_index = EndpointIndex(fragments)
        seen: set[Tuple[int, int]] = set()

        for f in fragments:
            for ep in f.endpoints:
                matches = endpoint_index.query_radius(ep, self.config.max_distance_nm)
                for frag_id, other_ep, dist in matches:
                    if frag_id == f.fragment_id:
                        continue
                    edge_key = (min(f.fragment_id, frag_id), max(f.fragment_id, frag_id))
                    if edge_key in seen:
                        # Update if this pair is closer
                        existing = graph.get_edge_data(edge_key[0], edge_key[1])
                        if existing and dist < existing["distance"]:
                            graph.add_edge(
                                f.fragment_id,
                                frag_id,
                                distance=dist,
                                endpoint_pair=(ep, other_ep),
                            )
                    else:
                        seen.add(edge_key)
                        graph.add_edge(
                            f.fragment_id,
                            frag_id,
                            distance=dist,
                            endpoint_pair=(ep, other_ep),
                        )
