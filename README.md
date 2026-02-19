# Connectomics Post-Segmentation Pipeline

**CPSC 4900 Senior Project**

A modular, conservative post-segmentation pipeline for large-scale connectomics data. Takes the output of automated segmentation workflows (e.g., PyTorch Connectomics) and produces graph-based connectivity representations and corrected segmentation volumes suitable for downstream analysis, proofreading, and visualization in Neuroglancer.

Built for the Dr. Aaron T. Kuan Lab at Yale School of Medicine.

## Pipeline Overview

```
Segmentation Volume → Fragment Extraction → Graph Construction → Candidate Generation
    → Conservative Validation → Assembly → Export (graphs, SWC, metadata,
                                                    Neuroglancer annotations,
                                                    corrected precomputed segmentation)
```

**Design philosophy:** Conservative correctness over aggressive merging. Uncertainty is explicitly preserved, never discarded. Three-outcome validation (ACCEPT / REJECT / AMBIGUOUS) — ambiguous connections are flagged for human review rather than forced to a binary decision.

## Validation Status

The pipeline has been validated on:
- **Synthetic data** — controlled tubular geometries with known ground truth split/merge scenarios
- **CREMI Sample A** — real Drosophila brain EM (64×256×256 crop, 40×4×4 nm resolution, 800 neuron labels)
  - Precision: **1.000** | Recall: **0.909** | F1: **0.952** (label-ID oracle evaluation)
  - Corrected precomputed segmentation output confirmed loadable in Neuroglancer

## Installation

```bash
# Core installation
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development tools (pytest, black, mypy)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the pipeline with a config file
connectomics-pipeline --config configs/default.yaml

# Run on CREMI Sample A (requires data/cremi_crop.hdf)
connectomics-pipeline --config configs/cremi_sample_a.yaml --verbose
```

## Project Structure

```
connectomics_pipeline/
├── io/              # Volume readers: HDF5, Zarr, Neuroglancer precomputed, NumPy
├── fragments/       # Fragment extraction, skeletonization, meshing, stitching
├── graph/           # Fragment adjacency graph construction
├── candidates/      # Candidate connection generation with composite scoring
├── validation/      # Conservative validation rules (7 built-in rules)
├── assembly/        # Structure assembly from validated connections
├── export/          # GraphML, SWC, CSV, Neuroglancer annotations,
│                    #   corrected precomputed segmentation
├── evaluation/      # Ground truth evaluation (precision/recall/F1)
├── visualization/   # Diagnostic plots and Neuroglancer annotation layers
└── utils/           # Configuration, types, spatial math, logging
```

## Output Formats

| Format | Config key | Description |
|--------|------------|-------------|
| GraphML | `"graphml"` | Fragment adjacency graph for network analysis |
| JSON | `"json"` | Same graph in JSON-serializable form |
| CSV | `"csv"` | Fragment metadata, connection decisions, structure summaries |
| SWC | `"swc"` | Neuron morphology (standard format for traced neurons) |
| Neuroglancer annotations | `"neuroglancer"` | Line annotations (green/red/yellow per decision) loadable as an annotation layer |
| Corrected precomputed segmentation | `"precomputed_seg"` | Corrected segmentation volume in Neuroglancer precomputed format with accepted merges applied |

## Configuration

All pipeline parameters are controlled via YAML config files. See `configs/default.yaml` for the full set of options with documentation.

Key sections: `input`, `fragments`, `graph`, `candidates`, `validation`, `assembly`, `export` (including `evaluate_ground_truth`), `logging`.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=connectomics_pipeline --cov-report=term-missing
```

349 tests, 100% coverage on all non-optional-dependency modules. CI runs on every push via GitHub Actions.

## Documentation

- [Architecture Document](ARCHITECTURE.md) — Full system design, data flow, and module interfaces
- [Testing Documentation](docs/TESTING.md) — Test structure, fixtures, and how to add tests
- [Testing Plan](docs/TESTING_PLAN.md) — Phase-by-phase validation strategy and status
- [Experiment Log](docs/EXPERIMENT_LOG.md) — All pipeline runs with quantitative results

## References

- Lin et al., 2021 — PyTorch Connectomics (arXiv:2112.05754)
- Dorkenwald et al., 2025 — CAVE (Nature Methods)
- MICrONS Consortium, 2025 — Functional Connectomics (Nature)
- Plaza et al., 2018 — Analyzing Image Segmentation for Connectomics (BMC Medical Imaging)
