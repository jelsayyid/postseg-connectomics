"""Graph construction from fragment spatial relationships."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.graph.spatial_index import EndpointIndex, SkeletonNodeIndex
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
        elif method == "skeleton_node":
            self._add_skeleton_node_edges(graph, fragments)
            self._add_pca_bbox_edges(graph, fragments)
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

    def _add_skeleton_node_edges(self, graph: FragmentGraph, fragments: List[Fragment]) -> None:
        """Add edges using all skeleton nodes, not just TEASAR endpoints.

        Indexes every skeleton node from every fragment in a single KD-tree.
        For each fragment, batch-queries its own skeleton nodes against that
        tree and keeps the single closest cross-fragment node pair per
        fragment pair.  This exposes splits that occur in the interior of
        long axons — the main class of errors missed by endpoint-only graphs.

        Fragments without a skeleton (processed via PCA endpoints) fall back
        to their endpoint positions as proxy query nodes.
        """
        node_index = SkeletonNodeIndex(fragments)
        # best_edge: edge_key -> (dist, node_from_frag, node_from_other)
        best_edge: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]] = {}

        for f in fragments:
            if f.skeleton is not None and f.skeleton.num_nodes > 0:
                query_pts = f.skeleton.nodes  # (N, 3)
            elif f.endpoints:
                query_pts = np.array(f.endpoints)
            else:
                query_pts = f.centroid.reshape(1, 3)

            # Batch KD-tree query: one call for all nodes of this fragment
            all_hit_lists = node_index.query_radius_batch(query_pts, self.config.max_distance_nm)

            for qi, hit_indices in enumerate(all_hit_lists):
                query_node = query_pts[qi]
                for idx in hit_indices:
                    other_frag_id = node_index.frag_id_at(idx)
                    if other_frag_id == f.fragment_id:
                        continue
                    other_node = node_index.coord_at(idx)
                    dist = float(np.linalg.norm(other_node - query_node))
                    edge_key = (
                        min(f.fragment_id, other_frag_id),
                        max(f.fragment_id, other_frag_id),
                    )
                    if edge_key not in best_edge or dist < best_edge[edge_key][0]:
                        best_edge[edge_key] = (dist, query_node.copy(), other_node.copy())

        for (frag_a_id, frag_b_id), (dist, node_a, node_b) in best_edge.items():
            graph.add_edge(
                frag_a_id,
                frag_b_id,
                distance=dist,
                endpoint_pair=(node_a, node_b),
            )

    def _add_pca_bbox_edges(self, graph: FragmentGraph, fragments: List[Fragment]) -> None:
        """Add edges for fragments that use PCA endpoints (no TEASAR skeleton)
        by testing bounding-box overlap with every other fragment.

        Large fragments fall back to PCA endpoints — just two points at the
        axon tips — which may be far from an interior split boundary.  Two
        fragments that should be merged (split from the same axon) will have
        overlapping bounding boxes even when their centroids and PCA tips are
        hundreds of nm apart.  This pass adds those edges so the candidate
        generator and validation rules can evaluate them.

        The centroid-to-centroid distance is stored as the edge distance;
        the alignment/curvature rules will reject crossing-axon false positives.
        """
        pca_frags = [f for f in fragments if f.skeleton is None or f.skeleton.num_nodes == 0]
        all_frags = fragments  # check PCA vs all, not just PCA vs PCA

        added = 0
        for fa in pca_frags:
            for fb in all_frags:
                if fb.fragment_id == fa.fragment_id:
                    continue
                if graph.has_edge(fa.fragment_id, fb.fragment_id):
                    continue
                if not fa.bounding_box.overlaps(fb.bounding_box):
                    continue
                dist = float(np.linalg.norm(fa.centroid - fb.centroid))
                graph.add_edge(fa.fragment_id, fb.fragment_id, distance=dist)
                added += 1

        if added:
            logger.debug("PCA bbox pass added %d edges", added)
