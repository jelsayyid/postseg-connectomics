"""ML-based false-positive filter applied after rule-based validation.

After rule-based validation, ~355K candidates are accepted but only ~1.25K are
true positives (Precision≈0.004).  The root cause is that long-range cross-axon
pairs (gap 500–2000 nm) cannot be distinguished from genuine splits by any single
rule threshold.  A gradient-boosted classifier trained on the 6 existing candidate
features can learn a non-linear decision boundary in this feature space.

Usage
-----
Training (offline, requires ground-truth oracle):

    python scripts/train_ml_filter.py \\
        --connections output/xpress_training/connections.csv \\
        --fragments  output/xpress_training/fragments.csv \\
        --skeletons  data/xpress/XPRESS_training_skels.npz \\
        --seg        /tmp/xpress_full.h5 \\
        --out        models/xpress_ml_filter.pkl \\
        --recall-target 0.99

Inference (inside the pipeline, configured via yaml):

    ml_filter:
      enabled: true
      model_path: "models/xpress_ml_filter.pkl"
      threshold: 0.0    # 0 = use threshold saved in model file

The filter moves candidates from ``report.accepted`` → ``report.rejected`` for
candidates whose predicted TP-probability falls below ``threshold``.  Candidates
remain in ``report.results`` with their original rule results unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from connectomics_pipeline.utils.types import CandidateConnection, ConnectionStatus, ValidationReport

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Features extracted from CandidateConnection for the classifier.
# Order must match the training script.
_FEATURES = [
    "gap_distance",
    "proximity_score",
    "alignment_score",
    "continuity_score",
    "size_score",
    "composite_score",
]


def _extract_features(candidates: List[CandidateConnection]) -> np.ndarray:
    """Extract a (N, 6) float32 feature matrix from candidate objects."""
    rows = []
    for c in candidates:
        rows.append([
            c.gap_distance,
            c.proximity_score,
            c.alignment_score,
            c.continuity_score,
            c.size_score,
            c.composite_score,
        ])
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, len(_FEATURES)), dtype=np.float32)


class MLFilter:
    """Post-validation ML filter that rejects low-probability candidates.

    Loads a pre-trained scikit-learn classifier (saved via ``joblib``) and
    applies it to the accepted candidates in a ``ValidationReport``.  Candidates
    whose predicted TP-probability is below ``threshold`` are moved from
    ``report.accepted`` to ``report.rejected``, and their ``status`` field on
    the ``CandidateConnection`` object is updated to ``REJECTED``.

    The filter is intentionally a thin wrapper: all scoring logic lives in the
    training script (``scripts/train_ml_filter.py``).  The model artifact stores
    the recommended threshold alongside the classifier so that ``threshold=0``
    (the default) uses the threshold chosen during training.

    Args:
        model_path: Path to the joblib model file produced by the training
            script.  The file must contain a dict with keys ``"model"`` (a
            fitted sklearn estimator with ``predict_proba``) and optionally
            ``"threshold"`` (float in [0, 1]) and ``"features"`` (list of str).
        threshold: Decision threshold for the positive class probability.
            Values ≥ threshold are classified as TPs (kept in accepted).
            Pass 0.0 (default) to use the threshold stored in the model file.
    """

    def __init__(self, model_path: str, threshold: float = 0.0):
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required for MLFilter.  Install scikit-learn: "
                "pip install scikit-learn"
            ) from exc

        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(
                f"Model file {model_path!r} must contain a dict with key 'model'. "
                "Re-run scripts/train_ml_filter.py to regenerate."
            )
        self._model = artifact["model"]
        self._saved_threshold: float = float(artifact.get("threshold", 0.5))
        self._features: List[str] = artifact.get("features", _FEATURES)

        # threshold=0.0 sentinel → use the threshold stored in the model file
        self.threshold = self._saved_threshold if threshold == 0.0 else threshold

        logger.info(
            "MLFilter loaded: model=%s  threshold=%.4f  features=%s",
            model_path,
            self.threshold,
            self._features,
        )

    def filter_report(
        self,
        candidates: List[CandidateConnection],
        report: ValidationReport,
    ) -> ValidationReport:
        """Apply the ML filter and return an updated ValidationReport.

        Moves low-probability candidates from ``report.accepted`` to
        ``report.rejected`` in-place, then returns the (mutated) report.

        Args:
            candidates: All candidate connections (accepted + rejected + ambiguous).
            report: ValidationReport produced by the rule-based pipeline.

        Returns:
            The same ``report`` object with updated ``accepted`` / ``rejected``
            lists and candidate ``status`` fields.
        """
        accepted_set = set(report.accepted)
        if not accepted_set:
            return report

        # Collect accepted candidates in list order for stable feature extraction
        accepted_cands = [c for c in candidates if c.candidate_id in accepted_set]
        if not accepted_cands:
            return report

        X = _extract_features(accepted_cands)

        try:
            probas: np.ndarray = self._model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.warning("MLFilter.predict_proba failed (%s); skipping filter.", exc)
            return report

        new_accepted: List[int] = []
        ml_rejected: List[int] = []

        for cand, prob in zip(accepted_cands, probas):
            if prob >= self.threshold:
                new_accepted.append(cand.candidate_id)
            else:
                ml_rejected.append(cand.candidate_id)
                cand.status = ConnectionStatus.REJECTED

        report.accepted = new_accepted
        report.rejected = report.rejected + ml_rejected

        n_removed = len(ml_rejected)
        logger.info(
            "MLFilter: removed %d / %d accepted candidates (threshold=%.4f); "
            "%d remain accepted",
            n_removed,
            len(accepted_cands),
            self.threshold,
            len(new_accepted),
        )
        return report

    @property
    def saved_threshold(self) -> float:
        """Threshold that was stored in the model file during training."""
        return self._saved_threshold
