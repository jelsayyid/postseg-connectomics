"""Candidate connection generation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from connectomics_pipeline.candidates.alignment import compute_alignment_score
from connectomics_pipeline.candidates.continuity import compute_continuity_score
from connectomics_pipeline.candidates.proximity import compute_proximity_score
from connectomics_pipeline.candidates.scoring import compute_composite_score, compute_size_score
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.config import CandidateConfig
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import CandidateConnection

logger = get_logger("candidates.generator")


class CandidateGenerator:
    """Generate and score candidate connections from graph edges."""

    def __init__(self, config: CandidateConfig, store: FragmentStore):
        self.config = config
        self.store = store
        self._next_id = 0

    def generate(self, graph: FragmentGraph) -> List[CandidateConnection]:
        """Generate candidate connections for all graph edges.

        Args:
            graph: Fragment adjacency graph.

        Returns:
            List of scored CandidateConnection objects, filtered by thresholds.
        """
        candidates = []

        for frag_a_id, frag_b_id in graph.edges():
            frag_a = self.store.get(frag_a_id)
            frag_b = self.store.get(frag_b_id)
            if frag_a is None or frag_b is None:
                continue

            edge_data = graph.get_edge_data(frag_a_id, frag_b_id) or {}

            # Determine endpoint pair
            ep_a = edge_data.get("endpoint_a")
            ep_b = edge_data.get("endpoint_b")
            if ep_a is None or ep_b is None:
                ep_a, ep_b = _find_best_endpoint_pair(frag_a, frag_b)

            candidate = self._score_candidate(frag_a, frag_b, ep_a, ep_b)
            if candidate is not None:
                candidates.append(candidate)

        logger.info(
            "Generated %d candidates from %d edges",
            len(candidates),
            graph.num_edges,
        )
        return candidates

    def _score_candidate(self, frag_a, frag_b, ep_a, ep_b) -> CandidateConnection | None:
        """Score a single candidate connection."""
        distance = float(np.linalg.norm(ep_b - ep_a))

        prox = compute_proximity_score(distance, self.config.max_endpoint_distance_nm)
        align = compute_alignment_score(frag_a, frag_b, ep_a, ep_b)
        cont = compute_continuity_score(frag_a, frag_b, ep_a, ep_b)

        # Size score from skeleton radii
        radius_a = _estimate_radius(frag_a)
        radius_b = _estimate_radius(frag_b)
        size = compute_size_score(radius_a, radius_b)

        composite = compute_composite_score(prox, align, cont, size, self.config.weights)

        # Filter by thresholds
        if composite < self.config.min_composite_score:
            return None
        if align < self.config.min_alignment_score:
            return None

        candidate_id = self._next_id
        self._next_id += 1

        return CandidateConnection(
            candidate_id=candidate_id,
            fragment_a=frag_a.fragment_id,
            fragment_b=frag_b.fragment_id,
            endpoint_a=ep_a,
            endpoint_b=ep_b,
            proximity_score=prox,
            alignment_score=align,
            continuity_score=cont,
            size_score=size,
            composite_score=composite,
        )


def _find_best_endpoint_pair(frag_a, frag_b) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest endpoint pair between two fragments."""
    best_dist = float("inf")
    best_a = frag_a.centroid
    best_b = frag_b.centroid

    eps_a = frag_a.endpoints if frag_a.endpoints else [frag_a.centroid]
    eps_b = frag_b.endpoints if frag_b.endpoints else [frag_b.centroid]

    for ea in eps_a:
        for eb in eps_b:
            d = float(np.linalg.norm(ea - eb))
            if d < best_dist:
                best_dist = d
                best_a = ea
                best_b = eb

    return best_a, best_b


def _estimate_radius(fragment) -> float:
    """Estimate fragment radius from skeleton or voxel count."""
    if fragment.skeleton is not None and len(fragment.skeleton.radii) > 0:
        return float(np.median(fragment.skeleton.radii))
    # Rough estimate from voxel count assuming spherical shape
    return float((3.0 * fragment.voxel_count / (4.0 * np.pi)) ** (1.0 / 3.0))
