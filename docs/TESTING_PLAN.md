# Testing Plan

## Goal

Validate the post-segmentation connectomics pipeline for correctness, robustness, and performance before use on real lab data. Testing follows a phased approach — each phase builds on the previous one.

## Phase 1: Unit Test Baseline — COMPLETE

**Status:** Done (349 tests passing, up from initial 72)

All individual modules have unit tests covering their core logic:
- I/O readers, fragment extraction, graph construction, candidate scoring, validation rules, assembly, topology detection, export formats, evaluation, visualization

**Deliverable:** Green test suite. Documented in Experiment Log entries 1–4.

## Phase 2: Coverage Analysis & Gap Filling — COMPLETE

**Status:** Done (68% → 100% on non-optional-dependency modules, 72 → 316 core tests)

- Identified and filled coverage gaps in export, stitching, spatial utilities, config, and visualization modules
- Added `pytest-cov` to CI
- Achieved 100% line coverage on all modules not gated by optional dependencies (`cloud-volume`, `kimimaro`)

**Deliverable:** Coverage report in Experiment Log entries 2–3.

## Phase 3: Edge Case & Robustness Testing — COMPLETE

**Status:** Done (51 edge case tests added, 1 real bug found and fixed)

- Empty/degenerate volumes, single-voxel fragments, extreme label counts
- Boundary conditions: fragments at volume edges, spanning full volume
- Extreme parameters: zero thresholds, very large distances, all rules enabled
- Scoring edge cases: zero/negative distances, coincident endpoints, zero radii

**Bug found:** `connection_statistics()` crashed with `KeyError` on empty candidate lists — would have surfaced on real sparse data. Fixed and covered by new tests.

**Deliverable:** `tests/test_edge_cases.py`, Experiment Log entry 4.

## Phase 4: Parameter Sensitivity Analysis — PARTIAL

**Status:** Real-data threshold tuning done (Experiment 7); synthetic sweep not yet done

Completed:
- Analyzed TP vs FP score distributions on CREMI Sample A real data
- Identified clean score gap between true positives (composite=1.000) and false positives (composite=0.627–0.634)
- Added `CompositeScoreRule` at threshold 0.65: precision 0.556 → 1.000, F1 0.702 → 0.952

Remaining:
- Systematic parameter sweep script varying `accept_threshold`, `reject_threshold`, scoring weights, `max_distance_nm`
- Sensitivity curves (threshold vs. precision/recall) on synthetic data with known ground truth

## Phase 5: Scalability & Performance Testing — Planned

**Tasks:**
1. Generate synthetic volumes at increasing sizes: 64³, 128³, 256³, 512³
2. Measure per-phase execution time and peak memory usage
3. Identify bottlenecks (likely: spatial indexing, graph construction, or skeleton extraction)
4. Establish baseline performance numbers for the lab's expected data scale

**Why this matters:** Real connectomics volumes are large (multi-GB). We need to know where the pipeline will slow down.

## Phase 6: Integration Testing with Real Data — COMPLETE (CREMI), Ongoing (Lab Data)

**Status:** CREMI Sample A complete. Lab data pending.

Completed:
- **CREMI Sample A** (Drosophila brain EM, 64×256×256 crop, 40×4×4 nm resolution)
  - Full end-to-end run: extraction → stitching → graph → candidates → validation → assembly → export
  - Ground truth evaluation using label-ID oracle: precision 1.000, recall 0.909, F1 0.952
  - Corrected precomputed segmentation output confirmed in Neuroglancer format
  - Documented in Experiment Log entries 5–7

Remaining:
- Run on CREMI Sample B and C for generalization check
- Run on lab data (Kuan Lab) with before/after proofreading corrections for benchmark against real ground truth
- Compare pipeline output against CAVE correction annotations

## Phase 7: Regression Suite & CI — COMPLETE

**Status:** Done

- GitHub Actions CI runs on every push: pytest (all 349 tests), black formatting check, mypy type checking
- All checks enforced before merge
- Config is saved alongside every pipeline output run for full reproducibility

## Summary

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Unit test baseline | **Complete** — 349 tests |
| 2 | Coverage analysis & gap filling | **Complete** — 100% coverage |
| 3 | Edge cases & robustness | **Complete** — 51 edge case tests, 1 bug fixed |
| 4 | Parameter sensitivity | **Partial** — real-data tuning done, synthetic sweep pending |
| 5 | Scalability & performance | Planned |
| 6 | Real-data integration testing | **CREMI complete** — lab data pending |
| 7 | Regression suite & CI | **Complete** — GitHub Actions |
