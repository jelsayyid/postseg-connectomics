# Testing Documentation

## Overview

The connectomics pipeline uses **pytest** as its testing framework. Tests are organized by module and cover unit tests, integration tests, edge cases, and end-to-end pipeline tests. All standard tests use synthetic in-memory data — no external volumes are required to run the suite.

**Current status: 349 tests, 100% line coverage on all non-optional-dependency modules. CI runs on every push via GitHub Actions.**

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific module
pytest tests/test_validation.py

# Run a specific test
pytest tests/test_candidates.py::TestProximityScore::test_zero_distance

# Run with coverage report
pytest tests/ --cov=connectomics_pipeline --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures (volumes, fragments, configs)
│
│   # Core pipeline modules
├── test_io.py                     # Volume reader tests
├── test_fragments.py              # Fragment extraction, metadata, store, skeletonization
├── test_graph.py                  # Graph construction and spatial index
├── test_candidates.py             # Candidate generation and all scoring functions
├── test_validation.py             # All 7 validation rules + pipeline orchestration
├── test_assembly.py               # Assembly, topology detection, confidence
│
│   # Export and I/O formats
├── test_export.py                 # GraphML, JSON, SWC, CSV, Neuroglancer export
├── test_precomputed_segmentation.py  # Corrected precomputed segmentation writer
├── test_precomputed_reader.py     # Neuroglancer precomputed volume reader
├── test_zarr_reader.py            # Zarr volume reader
│
│   # Supporting modules
├── test_stitching.py              # Cross-chunk fragment boundary stitching
├── test_mesh.py                   # Marching cubes mesh extraction
├── test_spatial.py                # Spatial math utilities
├── test_report.py                 # ValidationReport construction
├── test_config.py                 # Config loading, saving, defaults
├── test_config_nested.py          # Nested YAML config edge cases
├── test_logging_setup.py          # Logging configuration
├── test_cli.py                    # CLI argument parsing
│
│   # Evaluation
├── test_ground_truth.py           # Ground truth precision/recall evaluation
│
│   # Visualization
├── test_visualization.py          # 3D plots, Neuroglancer annotation JSON
│
│   # Robustness
├── test_edge_cases.py             # 51 edge case tests across all modules
└── test_pipeline.py               # End-to-end integration tests
```

**Total: 349 tests**

## Test Categories

### Unit Tests — Core Pipeline

| Module | File | What's Covered |
|--------|------|----------------|
| I/O | `test_io.py` | NumpyReader, HDF5 roundtrip, chunk reading and iteration |
| Fragments | `test_fragments.py` | Extraction (single/multi-label, non-contiguous), min-voxel filtering, boundary detection, store CRUD, spatial queries, centroid/endpoint metadata, skeletonization |
| Graph | `test_graph.py` | Graph add/query, edge operations, neighbors, subgraph extraction, proximity graph building, spatial index queries |
| Candidates | `test_candidates.py` | Proximity scoring (zero/max/mid/monotonic), alignment, continuity, size matching, composite score, full candidate generation |
| Validation | `test_validation.py` | All 7 rules (MaxDistance, Curvature, DirectionReversal, SizeDiscrepancy, BranchingLimit, Overlap, CompositeScore), rule factory, full pipeline |
| Assembly | `test_assembly.py` | Topology (no cycles, cycle detection, branch points), confidence (single/weakest-link/empty), assembler |

### Unit Tests — Supporting Modules

| Module | File | What's Covered |
|--------|------|----------------|
| Export | `test_export.py` | GraphML, JSON, SWC, CSV, Neuroglancer annotation JSON |
| Precomputed segmentation | `test_precomputed_segmentation.py` | Component relabeling, union-find merging, info file, chunk byte layout, roundtrip |
| Stitching | `test_stitching.py` | Cross-chunk fragment merging, boundary agreement |
| Spatial utilities | `test_spatial.py` | Distance, direction, angle, curvature math |
| Config | `test_config.py`, `test_config_nested.py` | YAML loading, saving, defaults, nested structures |
| Evaluation | `test_ground_truth.py` | TP/FP/TN/FN counting, precision/recall/F1, edge cases (missing fragments, all-ambiguous) |
| Visualization | `test_visualization.py` | 3D connection plots, Neuroglancer annotation layer JSON, color generation |

### Robustness Tests

| File | Tests | What's Covered |
|------|-------|----------------|
| `test_edge_cases.py` | 51 | Empty/degenerate volumes, extreme fragment counts, scoring edge cases (zero/negative/NaN inputs), validation with missing fragments, assembly with no accepted connections, full pipeline edge cases |

### Integration Tests

| File | What's Covered |
|------|----------------|
| `test_pipeline.py` | Full end-to-end runs with synthetic data; GraphML, SWC, Neuroglancer, CSV export paths; all validation rules enabled; config saved alongside outputs |

## Fixtures (conftest.py)

Shared fixtures provide consistent synthetic test data across all modules:

- **`resolution`** — Standard anisotropic resolution: `(30.0, 8.0, 8.0)` nm (z, y, x)
- **`synthetic_volume`** — 32×64×64 array with 3 distinct tubular label regions
- **`numpy_reader`** — NumpyReader wrapping the synthetic volume
- **`sample_fragments`** — 3 pre-built Fragment objects (A, B, C) with skeletons, endpoints, and bounding boxes modeling a split-correction scenario (A↔B) and a false-merge scenario (A↔C)
- **`fragment_store`** — FragmentStore populated with the 3 sample fragments
- **`default_config`** / **`fragment_config`** / **`graph_config`** / **`candidate_config`** / **`validation_config`** — Config fixtures with test-appropriate parameters

## Coverage Notes

Coverage is measured with `pytest-cov`. Modules excluded from the 100% target are those gated by optional dependencies that are not installed in CI:

| Module | Status | Reason |
|--------|--------|--------|
| `io/precomputed_reader.py` | Excluded | Requires `cloud-volume` |
| `io/zarr_reader.py` | Tested via mocks | Zarr test fixtures mock the format |
| `fragments/skeleton.py` | Partial | TEASAR path requires `kimimaro` |
| `cli.py` | Tested | CLI argument parsing covered |
| `visualization/plot_connections.py` | Tested via mocks | Matplotlib mocked in CI |

All other modules have 100% line coverage.

## Adding New Tests

1. Place test files in `tests/` with the `test_` prefix
2. Use existing fixtures from `conftest.py` where possible
3. Follow the `TestClassName::test_method` pattern
4. For new fixtures shared across modules, add them to `conftest.py`
5. Use `tmp_path` (pytest built-in) for any tests that write to disk
6. Run `black tests/ --line-length 100` before committing — CI enforces formatting
