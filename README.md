# Connectomics Post-Segmentation Pipeline

**CPSC 4900 Senior Project — Dr. Aaron T. Kuan Lab, Yale School of Medicine**

A modular, conservative post-segmentation pipeline for large-scale connectomics data. Takes the output of automated segmentation workflows and produces graph-based connectivity representations suitable for downstream analysis, proofreading, and visualization in Neuroglancer.

**Design philosophy:** Conservative correctness over aggressive merging. Uncertainty is explicitly preserved, never discarded. Three-outcome validation (ACCEPT / REJECT / AMBIGUOUS) — ambiguous connections are flagged for human review rather than forced to a binary decision.

## Pipeline Overview

```
Segmentation Volume → Fragment Extraction → Graph Construction → Candidate Generation
    → Conservative Validation → Assembly → Export
```

1. **Fragment extraction** — connected components from segmentation volume, with TEASAR skeletonization to capture interior structure
2. **Graph construction** — skeleton-node KD-tree graph exposing interior splits along long axons; optional long-range endpoint pass for genuine segmentation gaps
3. **Candidate generation** — per-edge composite scoring (proximity, alignment, continuity, size) with distance-conditioned weight switching for long-range pairs
4. **Conservative validation** — seven configurable rules; hard-rejects are explicit and auditable
5. **Assembly** — merge accepted connections, detect topology issues (cycles, branching, ambiguity)
6. **Export** — GraphML, CSV, SWC, Neuroglancer annotations, corrected precomputed segmentation

## Validation Results

### CREMI Sample A (Drosophila EM)

| Metric | Value |
|--------|-------|
| Precision | **1.000** |
| Recall | **0.909** |
| F1 | **0.952** |

64×256×256 voxel crop, 40×4×4 nm resolution, ~800 neuron labels. Evaluation uses a label-ID oracle (pairs with the same GT label that appear as separate fragments). Best understood as a pipeline correctness check rather than a realistic proofreading benchmark, since the input is human-annotated labels rather than automated segmentation output.

### XPRESS Challenge (Mouse White Matter XNH)

| Metric | Value |
|--------|-------|
| Oracle coverage | **83.5%** (1,252 / 1,499 true merge pairs reached candidate stage) |
| Recall | **0.994** (of covered oracle pairs, 99.4% correctly accepted) |
| Precision | 0.004 |

Full 699³ voxel training volume, 33 nm isotropic resolution, myelinated cortical axons. Evaluation uses skeleton-based ground truth (XPRESS challenge oracle). This is the primary domain-appropriate benchmark — automated (imperfect) segmentation input, with true split errors along axon interiors that the pipeline must detect and propose to merge. Low precision reflects the open challenge of discriminating same-axon from different-axon long-range pairs.

## Key Features

### Graph Construction (`skeleton_node` method)

Rather than indexing only TEASAR degree-1 endpoints, the `skeleton_node` method indexes **every skeleton node** from every fragment in a single KD-tree and batch-queries them. This exposes splits that occur in the interior of long axons — the primary error class in XPRESS, which an endpoint-only graph cannot represent by design.

An optional **long-range endpoint pass** adds a supplemental graph construction step that queries degree-1 endpoints at a larger search radius (`max_endpoint_search_nm`), targeting genuine segmentation gaps wider than the standard skeleton-node radius.

### Distance-Conditioned Scoring

The standard composite score (`0.35×proximity + 0.30×alignment + 0.25×continuity + 0.10×size`) is unreliable for long-range pairs where gap distance > ~1000 nm: proximity decays to near-zero, suppressing the composite score even when alignment and continuity are strong. The candidate generator supports a `long_range_weights` config that switches to a proximity-free weight vector (`0.45×alignment + 0.40×continuity + 0.15×size`) for pairs above a configurable distance threshold.

### Conservative Validation Rules

Seven built-in rules, each returning ACCEPT / REJECT / AMBIGUOUS with a confidence score:

| Rule | Description |
|------|-------------|
| `MaxDistanceRule` | Hard-reject if gap exceeds physical distance limit |
| `CurvatureRule` | Reject if junction angle exceeds threshold; optionally skips check for long-range pairs where the endpoint-centroid direction estimate is unreliable (`skip_distance_nm`) |
| `DirectionReversalRule` | Reject if fragments point away from each other |
| `SizeDiscrepancyRule` | Reject if radius ratio is implausible |
| `BranchingLimitRule` | Reject if merge would create excessive branching |
| `OverlapRule` | Reject if fragments significantly overlap (likely a merge error) |
| `CompositeScoreRule` | Hard-reject if composite score falls below a minimum |

All rules are configurable via YAML and composable; the validation pipeline short-circuits on any hard REJECT.

## Installation

```bash
# Core installation
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development tools (pytest, black, mypy)
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, NetworkX, h5py, PyYAML, scikit-learn. Optional: kimimaro (TEASAR skeletonization), zarr, cloud-volume (Neuroglancer precomputed I/O).

## Quick Start

```bash
# Run on CREMI Sample A (requires data/cremi_crop.hdf)
connectomics-pipeline --config configs/cremi_sample_a.yaml

# Run on XPRESS training volume (requires data/xpress/xpress_full.h5)
connectomics-pipeline --config configs/xpress_sample.yaml

# Run with a custom config
connectomics-pipeline --config configs/default.yaml
```

## Configuration

All pipeline parameters are controlled via YAML config files. `configs/default.yaml` documents the full option set. Domain-specific configs:

- `configs/cremi_sample_a.yaml` — CREMI Drosophila EM (anisotropic, endpoint graph)
- `configs/xpress_sample.yaml` — XPRESS mouse white matter XNH (isotropic 33 nm, skeleton-node graph, long-range pass)

Key config sections:

```yaml
graph:
  construction_method: "skeleton_node"   # endpoint | skeleton_node
  max_distance_nm: 500                   # standard search radius
  max_endpoint_search_nm: 2000           # long-range pass radius (0 = disabled)

candidates:
  max_endpoint_distance_nm: 600          # proximity decay reference distance
  weights: {proximity: 0.35, alignment: 0.30, continuity: 0.25, size: 0.10}
  long_range_threshold_nm: 1000          # switch to proximity-free weights above this
  long_range_weights: {proximity: 0.00, alignment: 0.45, continuity: 0.40, size: 0.15}

validation:
  accept_threshold: 0.25
  reject_threshold: 0.15
  long_range_distance_nm: 1000           # distance-conditioned accept threshold
  long_range_accept_threshold: 0.20
  rules:
    - name: "CurvatureRule"
      params:
        max_curvature_deg: 150
        skip_distance_nm: 1000           # skip unreliable check for long-range pairs
```

## Project Structure

```
connectomics_pipeline/
├── io/              # Volume readers: HDF5, Zarr, Neuroglancer precomputed, NumPy
├── fragments/       # Fragment extraction, TEASAR skeletonization, meshing, stitching
├── graph/           # Skeleton-node and endpoint graph construction, KD-tree indexing
├── candidates/      # Composite scoring: proximity, alignment, continuity, size
├── validation/      # Seven configurable validation rules + report builder
├── assembly/        # Structure assembly, cycle detection, ambiguity flagging
├── export/          # GraphML, CSV, SWC, Neuroglancer annotations, precomputed seg
├── evaluation/      # Ground truth evaluation: label-ID oracle, XPRESS skeleton oracle
├── visualization/   # Diagnostic plots and Neuroglancer annotation layers
└── utils/           # Config loading, types, spatial math, logging
```

## Output Formats

| Format | Config key | Description |
|--------|------------|-------------|
| GraphML | `"graphml"` | Fragment adjacency graph for network analysis |
| CSV | `"csv"` | Fragment metadata, per-connection decisions, structure summaries |
| SWC | `"swc"` | Neuron morphology in standard traced-neuron format |
| Neuroglancer annotations | `"neuroglancer"` | Line annotations (green/red/yellow per decision) as an annotation layer |
| Corrected precomputed seg | `"precomputed_seg"` | Segmentation volume with accepted merges applied, in Neuroglancer precomputed format |

## Testing

```bash
pytest tests/                                                        # run all tests
pytest tests/ --cov=connectomics_pipeline --cov-report=term-missing  # with coverage
```

**391 tests**, passing on Python 3.10, 3.11, and 3.12. CI runs automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`): tests, black formatting check, and mypy type checking.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Full system design, data flow, and module interfaces
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — All pipeline runs with quantitative results (Experiments 1–16)
- [`docs/TESTING.md`](docs/TESTING.md) — Test structure, fixtures, and how to add tests
- [`docs/TESTING_PLAN.md`](docs/TESTING_PLAN.md) — Phase-by-phase validation strategy and status

## References

- XPRESS Challenge — [github.com/htem/xpress-challenge](https://github.com/htem/xpress-challenge)
- Lin et al., 2021 — PyTorch Connectomics (arXiv:2112.05754)
- Dorkenwald et al., 2025 — CAVE (Nature Methods)
- MICrONS Consortium, 2025 — Functional Connectomics (Nature)
- Plaza et al., 2018 — Analyzing Image Segmentation for Connectomics (BMC Medical Imaging)
