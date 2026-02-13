"""Validation rules for candidate connections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ConnectionStatus,
    ValidationResult,
)


class ValidationRule(ABC):
    """Base class for validation rules."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult: ...


class MaxDistanceRule(ValidationRule):
    """Reject if gap distance exceeds threshold."""

    def __init__(self, max_distance_nm: float = 1500.0):
        self.max_distance_nm = max_distance_nm

    @property
    def name(self) -> str:
        return "MaxDistanceRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        dist = candidate.gap_distance
        if dist > self.max_distance_nm:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=1.0,
                reason=f"Gap distance {dist:.1f} nm exceeds max {self.max_distance_nm:.1f} nm",
            )
        confidence = 1.0 - (dist / self.max_distance_nm)
        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=confidence,
            reason=f"Gap distance {dist:.1f} nm within limit",
        )


class CurvatureRule(ValidationRule):
    """Reject if connecting curvature exceeds threshold."""

    def __init__(self, max_curvature_deg: float = 90.0):
        self.max_curvature_rad = np.radians(max_curvature_deg)

    @property
    def name(self) -> str:
        return "CurvatureRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        frag_a = store.get(candidate.fragment_a)
        frag_b = store.get(candidate.fragment_b)
        if frag_a is None or frag_b is None:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.AMBIGUOUS,
                confidence=0.0,
                reason="Fragment not found in store",
            )

        # Estimate curvature at connection junction
        curvature = self._estimate_junction_curvature(
            frag_a, frag_b, candidate.endpoint_a, candidate.endpoint_b
        )

        if curvature > self.max_curvature_rad:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=min(1.0, curvature / np.pi),
                reason=f"Curvature {np.degrees(curvature):.1f} deg exceeds max "
                f"{np.degrees(self.max_curvature_rad):.1f} deg",
            )

        confidence = 1.0 - (curvature / self.max_curvature_rad)
        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=confidence,
            reason=f"Curvature {np.degrees(curvature):.1f} deg within limit",
        )

    def _estimate_junction_curvature(self, frag_a, frag_b, ep_a, ep_b) -> float:
        """Estimate curvature at the junction between two fragments."""
        dir_a = self._get_direction(frag_a, ep_a)
        dir_b = self._get_direction(frag_b, ep_b)

        connection = ep_b - ep_a
        conn_norm = np.linalg.norm(connection)
        if conn_norm < 1e-12:
            return 0.0
        connection = connection / conn_norm

        # Angle between fragment A direction and connection
        cos_angle = np.clip(np.dot(dir_a, connection), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _get_direction(self, fragment, endpoint) -> np.ndarray:
        direction = endpoint - fragment.centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm


class DirectionReversalRule(ValidationRule):
    """Reject if fragments point away from each other."""

    def __init__(self, min_alignment: float = 0.0):
        self.min_alignment = min_alignment

    @property
    def name(self) -> str:
        return "DirectionReversalRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        if candidate.alignment_score < self.min_alignment:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=1.0 - candidate.alignment_score,
                reason=f"Alignment score {candidate.alignment_score:.3f} below "
                f"min {self.min_alignment:.3f}",
            )
        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=candidate.alignment_score,
            reason=f"Alignment score {candidate.alignment_score:.3f} acceptable",
        )


class SizeDiscrepancyRule(ValidationRule):
    """Reject if fragment size ratio is too large."""

    def __init__(self, max_radius_ratio: float = 3.0):
        self.max_radius_ratio = max_radius_ratio

    @property
    def name(self) -> str:
        return "SizeDiscrepancyRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        frag_a = store.get(candidate.fragment_a)
        frag_b = store.get(candidate.fragment_b)
        if frag_a is None or frag_b is None:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.AMBIGUOUS,
                confidence=0.0,
                reason="Fragment not found",
            )

        va = frag_a.voxel_count
        vb = frag_b.voxel_count
        ratio = max(va, vb) / max(min(va, vb), 1)

        # Convert voxel ratio to approximate radius ratio (cube root)
        radius_ratio = ratio ** (1.0 / 3.0)

        if radius_ratio > self.max_radius_ratio:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=min(1.0, radius_ratio / (2 * self.max_radius_ratio)),
                reason=f"Radius ratio {radius_ratio:.2f} exceeds max {self.max_radius_ratio:.1f}",
            )

        confidence = 1.0 - (radius_ratio - 1.0) / (self.max_radius_ratio - 1.0)
        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=max(0.0, confidence),
            reason=f"Radius ratio {radius_ratio:.2f} within limit",
        )


class BranchingLimitRule(ValidationRule):
    """Reject if accepting would exceed branch limit."""

    def __init__(self, max_branches: int = 10):
        self.max_branches = max_branches

    @property
    def name(self) -> str:
        return "BranchingLimitRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        if graph is None:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.ACCEPTED,
                confidence=0.5,
                reason="No graph available for branch checking",
            )

        degree_a = len(graph.get_neighbors(candidate.fragment_a))
        degree_b = len(graph.get_neighbors(candidate.fragment_b))
        max_degree = max(degree_a, degree_b)

        if max_degree >= self.max_branches:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=0.8,
                reason=f"Max degree {max_degree} would exceed branch limit {self.max_branches}",
            )

        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=1.0 - (max_degree / self.max_branches),
            reason=f"Max degree {max_degree} within branch limit",
        )


class CompositeScoreRule(ValidationRule):
    """Reject if composite score is below reject threshold."""

    def __init__(self, reject_threshold: float = 0.3):
        self.reject_threshold = reject_threshold

    @property
    def name(self) -> str:
        return "CompositeScoreRule"

    def evaluate(
        self,
        candidate: CandidateConnection,
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationResult:
        if candidate.composite_score < self.reject_threshold:
            return ValidationResult(
                rule_name=self.name,
                decision=ConnectionStatus.REJECTED,
                confidence=1.0 - candidate.composite_score,
                reason=f"Composite score {candidate.composite_score:.3f} below "
                f"reject threshold {self.reject_threshold:.3f}",
            )
        return ValidationResult(
            rule_name=self.name,
            decision=ConnectionStatus.ACCEPTED,
            confidence=candidate.composite_score,
            reason=f"Composite score {candidate.composite_score:.3f} above threshold",
        )


# Registry for rule creation from config
RULE_REGISTRY: Dict[str, type] = {
    "MaxDistanceRule": MaxDistanceRule,
    "CurvatureRule": CurvatureRule,
    "DirectionReversalRule": DirectionReversalRule,
    "SizeDiscrepancyRule": SizeDiscrepancyRule,
    "BranchingLimitRule": BranchingLimitRule,
    "CompositeScoreRule": CompositeScoreRule,
}


def create_rule(name: str, params: Dict[str, Any]) -> ValidationRule:
    """Create a validation rule from a name and parameters.

    Args:
        name: Rule class name.
        params: Keyword arguments for the rule constructor.

    Returns:
        Instantiated ValidationRule.
    """
    if name not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule: {name}. Available: {list(RULE_REGISTRY.keys())}")
    return RULE_REGISTRY[name](**params)
