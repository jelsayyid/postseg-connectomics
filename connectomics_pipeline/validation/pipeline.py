"""Validation pipeline orchestrating multiple rules."""

from __future__ import annotations

from typing import Dict, List, Optional

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.config import ValidationConfig
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ValidationReport,
    ValidationResult,
)
from connectomics_pipeline.validation.report import build_report
from connectomics_pipeline.validation.rules import ValidationRule, create_rule

logger = get_logger("validation.pipeline")


class ValidationPipeline:
    """Run a sequence of validation rules on candidate connections."""

    def __init__(self, config: ValidationConfig, rules: Optional[List[ValidationRule]] = None):
        self.config = config
        if rules is not None:
            self.rules = rules
        else:
            self.rules = [create_rule(rc.name, rc.params) for rc in config.rules]

    def validate(
        self,
        candidates: List[CandidateConnection],
        store: FragmentStore,
        graph: Optional[FragmentGraph] = None,
    ) -> ValidationReport:
        """Run all rules on all candidates.

        Args:
            candidates: List of candidate connections to validate.
            store: Fragment store for rule evaluation.
            graph: Optional fragment graph for branching checks.

        Returns:
            ValidationReport with accept/reject/ambiguous decisions.
        """
        all_results: Dict[int, List[ValidationResult]] = {}

        for candidate in candidates:
            results = []
            for rule in self.rules:
                result = rule.evaluate(candidate, store, graph)
                results.append(result)
            all_results[candidate.candidate_id] = results

        report = build_report(
            all_results,
            candidates,
            accept_threshold=self.config.accept_threshold,
            reject_threshold=self.config.reject_threshold,
        )

        logger.info(
            "Validation complete: %d accepted, %d rejected, %d ambiguous",
            len(report.accepted),
            len(report.rejected),
            len(report.ambiguous),
        )
        return report
