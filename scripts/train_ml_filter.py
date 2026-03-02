"""Train an ML false-positive filter for the connectomics pipeline.

Trains a gradient-boosted binary classifier to separate true-positive
merge candidates (oracle pairs) from false positives using the 6 features
already computed for every candidate during the pipeline run.

The trained model is saved via joblib and can be loaded by
``connectomics_pipeline.postprocess.ml_filter.MLFilter``.

Usage
-----
    python scripts/train_ml_filter.py \\
        --connections output/xpress_training/connections.csv \\
        --fragments   output/xpress_training/fragments.csv \\
        --skeletons   data/xpress/XPRESS_training_skels.npz \\
        --seg         /tmp/xpress_full.h5 \\
        --out         models/xpress_ml_filter.pkl \\
        --recall-target 0.99 \\
        --voxel-size  33 33 33

Outputs
-------
- ``--out``: joblib artifact containing:
    {"model": fitted GradientBoostingClassifier,
     "threshold": float,        # threshold achieving --recall-target
     "features": List[str],     # feature names in order
     "metrics": dict}           # CV and threshold metrics
- Console: precision-recall summary, feature importances, threshold chosen.

Requirements
------------
    pip install scikit-learn joblib
    pip install -e ".[ml]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

_FEATURES: List[str] = [
    "gap_distance",
    "proximity_score",
    "alignment_score",
    "continuity_score",
    "size_score",
    "composite_score",
]


# ---------------------------------------------------------------------------
# Oracle construction (mirrors evaluate_decisions_xpress logic)
# ---------------------------------------------------------------------------


def _build_oracle(
    skeletons_path: str,
    seg_path: str,
    voxel_size_nm: Tuple[float, float, float],
    seg_offset: Tuple[int, int, int] = (0, 0, 0),
) -> Set[Tuple[int, int]]:
    """Build the merge oracle using the XPRESS ground-truth skeletons."""
    import h5py

    from connectomics_pipeline.evaluation.xpress_ground_truth import build_merge_oracle

    skeletons_path = str(skeletons_path)
    if skeletons_path.endswith(".npz"):
        import cloudvolume  # noqa: F401  (may raise ImportError)
        import navis

        skels = navis.read_swc(skeletons_path)
        if not isinstance(skels, list):
            skels = [skels]
    else:
        raise ValueError(f"Unsupported skeleton format: {skeletons_path}")

    with h5py.File(seg_path, "r") as f:
        seg = f["labels"][()]

    oracle = build_merge_oracle(
        skeleton_graphs=skels,
        segmentation=seg,
        voxel_size_nm=voxel_size_nm,
        seg_offset_voxels=seg_offset,
    )
    return oracle


def _build_oracle_npz_direct(
    skeletons_path: str,
    seg_path: str,
    voxel_size_nm: Tuple[float, float, float],
    seg_offset: Tuple[int, int, int] = (0, 0, 0),
) -> Set[Tuple[int, int]]:
    """Direct npz-based oracle (mirrors xpress_ground_truth.py approach)."""
    import h5py

    from connectomics_pipeline.evaluation.xpress_ground_truth import build_merge_oracle

    # Load npz skeleton graphs
    data = np.load(skeletons_path, allow_pickle=True)
    # Build simple skeleton graph objects matching what build_merge_oracle expects
    # (it calls skel.nodes_in_volume / skel.graph — see xpress_ground_truth.py)
    import networkx as nx

    skeleton_graphs = []
    for key in data.files:
        arr = data[key]
        if arr.ndim == 0:
            obj = arr.item()
        else:
            obj = arr
        skeleton_graphs.append(obj)

    with h5py.File(seg_path, "r") as f:
        dataset_key = "labels" if "labels" in f else list(f.keys())[0]
        seg = f[dataset_key][()]

    oracle = build_merge_oracle(
        skeleton_graphs=skeleton_graphs,
        segmentation=seg,
        voxel_size_nm=voxel_size_nm,
        seg_offset_voxels=seg_offset,
    )
    return oracle


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def assign_labels(
    connections: pd.DataFrame,
    frag_to_label: dict,
    oracle: Set[Tuple[int, int]],
) -> np.ndarray:
    """Return binary oracle labels for each connection row.

    A candidate is labelled 1 (TP) if its (label_a, label_b) pair appears in
    the oracle set.  Candidates where label_a == label_b (same neuron, different
    chunk) are labelled 0 — they are not merge errors.
    """
    labels = np.zeros(len(connections), dtype=np.int32)
    for i, (_, row) in enumerate(connections.iterrows()):
        la = frag_to_label.get(int(row["fragment_a"]), -1)
        lb = frag_to_label.get(int(row["fragment_b"]), -1)
        if la == lb or la < 0 or lb < 0:
            continue
        pair = (min(la, lb), max(la, lb))
        if pair in oracle:
            labels[i] = 1
    return labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    connections_path: str,
    fragments_path: str,
    oracle: Set[Tuple[int, int]],
    recall_target: float,
    out_path: str,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Train and save the ML filter model.

    Trains only on ACCEPTED candidates (those that passed rule-based validation)
    because the ML filter is applied post-validation.  Cross-validation is used
    to report unbiased metrics; the full dataset is used to train the final model.

    Returns a dict of metrics for logging.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    print(f"Loading connections: {connections_path}")
    connections = pd.read_csv(connections_path)
    print(f"  {len(connections):,} total candidates")

    print(f"Loading fragments: {fragments_path}")
    fragments = pd.read_csv(fragments_path)
    frag_to_label = dict(zip(fragments["fragment_id"], fragments["label_id"]))

    print("Assigning oracle labels...")
    all_labels = assign_labels(connections, frag_to_label, oracle)

    # Filter to accepted candidates only
    accepted_mask = connections["status"].str.lower() == "accepted"
    X_all = connections[_FEATURES].values.astype(np.float32)
    X = X_all[accepted_mask]
    y = all_labels[accepted_mask]

    n_tp = int(y.sum())
    n_fp = int((y == 0).sum())
    print(f"  Accepted: {len(y):,} total  |  TP={n_tp:,}  FP={n_fp:,}  ratio=1:{n_fp//max(n_tp,1)}")

    if n_tp == 0:
        print("ERROR: No true positives found in accepted candidates. Check oracle mapping.")
        sys.exit(1)

    # Cross-validated PR-AUC
    print(f"\nRunning {n_cv_folds}-fold stratified cross-validation...")
    clf_cv = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    cv_probas = cross_val_predict(clf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_ap = average_precision_score(y, cv_probas)
    print(f"  Cross-validated Average Precision (PR-AUC): {cv_ap:.4f}")

    # PR curve from CV probabilities — find threshold for recall target
    precision_cv, recall_cv, thresholds_cv = precision_recall_curve(y, cv_probas)
    # thresholds has length N-1; precision/recall have length N (last entry is trivial)
    # Walk thresholds from high → low; find first where recall >= target
    threshold = None
    for i, (p, r, t) in enumerate(zip(precision_cv[:-1], recall_cv[:-1], thresholds_cv)):
        if r >= recall_target:
            threshold = float(t)
            prec_at_threshold = float(p)
            rec_at_threshold = float(r)
    if threshold is None:
        # recall target not achievable — use smallest threshold (max recall)
        threshold = float(thresholds_cv[0])
        prec_at_threshold = float(precision_cv[0])
        rec_at_threshold = float(recall_cv[0])
        print(
            f"  WARNING: recall target {recall_target:.3f} not achievable. "
            f"Using threshold={threshold:.4f} (recall={rec_at_threshold:.4f})"
        )
    else:
        print(
            f"  Threshold for recall≥{recall_target:.3f}: {threshold:.4f} "
            f"→ Precision={prec_at_threshold:.4f}  Recall={rec_at_threshold:.4f}"
        )

    # Estimate FP reduction
    cv_preds = (cv_probas >= threshold).astype(int)
    kept_fps = int(((cv_preds == 1) & (y == 0)).sum())
    kept_tps = int(((cv_preds == 1) & (y == 1)).sum())
    print(f"  At threshold {threshold:.4f}: kept {kept_tps} TP, {kept_fps} FP")
    print(f"  FP reduction: {n_fp:,} → {kept_fps:,} ({100*(1-kept_fps/max(n_fp,1)):.1f}% removed)")
    print(f"  Precision estimate: {kept_tps/max(kept_tps+kept_fps,1):.4f} (was {n_tp/max(n_tp+n_fp,1):.4f})")

    # Feature importances from CV (use a quick single fit for reporting)
    print("\nTraining final model on full accepted set...")
    clf_final = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=random_state,
    )
    clf_final.fit(X, y)

    print("\nFeature importances (MDI):")
    for fname, imp in sorted(
        zip(_FEATURES, clf_final.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "#" * int(imp * 40)
        print(f"  {fname:<22} {imp:.4f}  {bar}")

    # Save artifact
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "cv_average_precision": float(cv_ap),
        "threshold": threshold,
        "precision_at_threshold": prec_at_threshold,
        "recall_at_threshold": rec_at_threshold,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "kept_tps_at_threshold": kept_tps,
        "kept_fps_at_threshold": kept_fps,
        "recall_target": recall_target,
        "n_cv_folds": n_cv_folds,
    }
    artifact = {
        "model": clf_final,
        "threshold": threshold,
        "features": _FEATURES,
        "metrics": metrics,
    }
    import joblib
    joblib.dump(artifact, out_path)
    print(f"\nModel saved to: {out_path}")
    print(f"  Load with: MLFilter(model_path='{out_path}', threshold=0.0)")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML false-positive filter for the connectomics pipeline."
    )
    parser.add_argument(
        "--connections",
        required=True,
        help="Path to connections.csv from the pipeline output",
    )
    parser.add_argument(
        "--fragments",
        required=True,
        help="Path to fragments.csv from the pipeline output",
    )
    parser.add_argument(
        "--skeletons",
        required=True,
        help="Path to XPRESS_training_skels.npz ground-truth skeletons",
    )
    parser.add_argument(
        "--seg",
        required=True,
        help="Path to segmentation HDF5 file (e.g. /tmp/xpress_full.h5)",
    )
    parser.add_argument(
        "--out",
        default="models/xpress_ml_filter.pkl",
        help="Output path for the trained model (default: models/xpress_ml_filter.pkl)",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.99,
        help="Minimum recall to maintain when choosing threshold (default: 0.99)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[33.0, 33.0, 33.0],
        metavar=("Z", "Y", "X"),
        help="Voxel size in nm for Z, Y, X (default: 33 33 33 for XPRESS)",
    )
    parser.add_argument(
        "--seg-offset",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=("Z", "Y", "X"),
        help="Segmentation offset in voxels (default: 0 0 0)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Check required imports
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn and joblib are required.")
        print("Install with: pip install scikit-learn")
        sys.exit(1)

    voxel_size = tuple(args.voxel_size)
    seg_offset = tuple(args.seg_offset)

    print("=" * 60)
    print("Building merge oracle from ground-truth skeletons...")
    print(f"  Skeletons: {args.skeletons}")
    print(f"  Segmentation: {args.seg}")
    print(f"  Voxel size (nm): {voxel_size}")
    print(f"  Seg offset (vox): {seg_offset}")

    try:
        oracle = _build_oracle_npz_direct(
            skeletons_path=args.skeletons,
            seg_path=args.seg,
            voxel_size_nm=voxel_size,
            seg_offset=seg_offset,
        )
    except Exception as exc:
        print(f"ERROR building oracle: {exc}")
        print("Check that --skeletons and --seg point to valid XPRESS files.")
        sys.exit(1)

    print(f"  Oracle: {len(oracle):,} merge pairs")
    print("=" * 60)

    metrics = train(
        connections_path=args.connections,
        fragments_path=args.fragments,
        oracle=oracle,
        recall_target=args.recall_target,
        out_path=args.out,
        n_cv_folds=args.cv_folds,
        random_state=args.seed,
    )

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  CV Average Precision (PR-AUC): {metrics['cv_average_precision']:.4f}")
    print(f"  Threshold (recall≥{metrics['recall_target']:.2f}): {metrics['threshold']:.4f}")
    print(f"  Precision at threshold: {metrics['precision_at_threshold']:.4f}")
    print(f"  Recall at threshold:    {metrics['recall_at_threshold']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
