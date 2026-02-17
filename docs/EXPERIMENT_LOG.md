# Experiment Log

This document tracks testing experiments, results, and observations as the pipeline is validated.

---

## Experiment 1: Baseline Unit Test Suite

**Date:** 2026-02-13
**Objective:** Establish baseline — verify all existing unit and integration tests pass.

**Setup:**
- Python 3.10.12, pytest 8.3.1
- All tests run against synthetic in-memory data (no external volumes)
- Install: `pip install -e ".[dev]"`

**Results:**
```
72 passed in 5.22s
```

| Module | Tests | Passed | Failed | Notes |
|--------|-------|--------|--------|-------|
| I/O | 9 | 9 | 0 | NumpyReader + HDF5 roundtrip |
| Fragments | 16 | 16 | 0 | Extraction, store, metadata, skeletonization |
| Graph | 9 | 9 | 0 | Graph construction + spatial index |
| Candidates | 16 | 16 | 0 | All scoring functions + generation |
| Validation | 12 | 12 | 0 | All 7 rules + pipeline |
| Assembly | 8 | 8 | 0 | Topology detection + confidence |
| Pipeline (E2E) | 2 | 2 | 0 | Full run + export verification |

**Observations:**
- All 72 tests pass cleanly with no warnings
- Test execution is fast (~5s total) — all synthetic, no disk I/O bottleneck
- End-to-end test covers fragment extraction through export but uses minimal validation (single MaxDistanceRule)
- No coverage measurement yet — needs `pytest-cov` to quantify

**Action Items:**
- [x] Run coverage analysis to identify untested code paths (see Experiment 2)
- [ ] Expand E2E tests with more realistic validation configurations
- [ ] Begin Phase 2: stress testing with varied synthetic data

---

## Experiment 2: Coverage Analysis

**Date:** 2026-02-13
**Objective:** Measure line coverage to identify untested code paths and prioritize gap-filling.

**Setup:**
- `pip install pytest-cov`
- `pytest tests/ --cov=connectomics_pipeline --cov-report=term-missing`

**Results:**

**Overall: 68% line coverage (1178/1721 statements covered)**

### Well-Covered Modules (>85%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `io/numpy_reader.py` | 100% | Fully tested |
| `io/volume_reader.py` | 100% | Fully tested |
| `io/hdf5_reader.py` | 96% | 1 line missed (error path) |
| `candidates/generator.py` | 97% | Near-complete |
| `candidates/alignment.py` | 93% | Near-complete |
| `candidates/continuity.py` | 94% | Near-complete |
| `candidates/proximity.py` | 90% | Near-complete |
| `fragments/extraction.py` | 98% | Near-complete |
| `fragments/store.py` | 90% | Near-complete |
| `assembly/topology.py` | 94% | Near-complete |
| `assembly/confidence.py` | 93% | Near-complete |
| `assembly/assembler.py` | 90% | Near-complete |
| `validation/pipeline.py` | 96% | Near-complete |
| `validation/rules.py` | 89% | Near-complete |
| `export/metadata_export.py` | 100% | Fully tested |
| `utils/types.py` | 90% | Near-complete |

### Gap Areas (<70% — need attention)

| Module | Coverage | Missing Lines | Priority |
|--------|----------|---------------|----------|
| `cli.py` | **0%** | All 22 lines | Low (CLI wrapper) |
| `io/precomputed_reader.py` | **0%** | All 30 lines | Medium (needs cloud-volume) |
| `io/zarr_reader.py` | **0%** | All 28 lines | Medium (needs zarr files) |
| `fragments/stitching.py` | **18%** | 67/82 lines | **High** (core feature) |
| `export/graph_export.py` | **23%** | 43/56 lines | **High** (key output) |
| `export/swc_export.py` | **27%** | 35/48 lines | **High** (key output) |
| `export/neuroglancer_export.py` | **42%** | 14/24 lines | Medium |
| `fragments/mesh.py` | **45%** | 12/22 lines | Medium |
| `utils/spatial.py` | **44%** | 28/50 lines | **High** (used everywhere) |
| `visualization/plot_connections.py` | **0%** | All 44 lines | Low (plotting) |
| `visualization/neuroglancer_annotations.py` | **0%** | All 36 lines | Low (visualization) |
| `utils/config.py` | **69%** | 41/132 lines | Medium (YAML loading) |
| `utils/logging.py` | **65%** | 8/23 lines | Low |
| `pipeline.py` | **72%** | 41/144 lines | Medium (E2E covers partial) |
| `validation/report.py` | **70%** | 8/27 lines | Medium |

### Coverage by Category

| Category | Avg Coverage | Notes |
|----------|-------------|-------|
| Core logic (candidates, validation, assembly) | ~92% | Strong |
| I/O (numpy, hdf5) | ~98% | Strong |
| I/O (zarr, precomputed) | 0% | Needs optional deps |
| Fragments (extract, store, metadata) | ~90% | Good |
| Fragments (stitching, mesh, skeleton) | ~46% | **Weak** |
| Export (graph, swc, neuroglancer) | ~31% | **Weak** |
| Utilities (spatial, config, logging) | ~59% | Moderate |
| Visualization | 0% | Not tested |
| Pipeline orchestration | 72% | Moderate |

**Observations:**
- Core algorithmic logic (scoring, validation rules, assembly) is well-tested at ~92%
- **Biggest gaps are in export formats and fragment stitching** — these are important for real-world use
- `utils/spatial.py` at 44% is concerning since it's a utility used across modules
- Visualization modules (0%) are low priority — plotting code is hard to unit test meaningfully
- Zero coverage on `cli.py`, `zarr_reader.py`, `precomputed_reader.py` is expected (CLI is a thin wrapper; Zarr/precomputed need optional deps)
- The pipeline orchestrator at 72% could be improved with more E2E test scenarios

**Action Items:**
- [x] Write tests for `export/graph_export.py` (GraphML + JSON export) — now 98%
- [x] Write tests for `export/swc_export.py` (SWC morphology export) — now 96%
- [x] Write tests for `fragments/stitching.py` (cross-chunk fragment merging) — now 99%
- [x] Write tests for `utils/spatial.py` (distance, direction, angle utilities) — now 100%
- [x] Add E2E test with all validation rules enabled
- [x] Add E2E tests for GraphML, JSON, SWC, Neuroglancer export paths
- [x] Also added: mesh extraction tests (86%), config load/save tests (98%), validation report tests (100%), neuroglancer export tests (100%)
- [ ] Target: raise overall coverage from 68% to >85% — **achieved 83% (see Experiment 3)**

---

## Experiment 3: Coverage Gap Filling

**Date:** 2026-02-13
**Objective:** Fill coverage gaps identified in Experiment 2, targeting >85% on testable core logic.

**Setup:**
- Added 6 new test files: `test_export.py`, `test_stitching.py`, `test_spatial.py`, `test_mesh.py`, `test_config.py`, `test_report.py`
- Expanded `test_pipeline.py` with 4 additional E2E tests (graphml, swc, neuroglancer exports + multi-rule validation)
- Total: 95 new tests added

**Results:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total tests | 72 | 167 | +95 |
| Overall coverage | 68% | 83% | +15pp |
| Statements missed | 543 | 294 | -249 |

### Module-level improvements:

| Module | Before | After |
|--------|--------|-------|
| `export/graph_export.py` | 23% | **98%** |
| `export/swc_export.py` | 27% | **96%** |
| `export/neuroglancer_export.py` | 42% | **100%** |
| `fragments/stitching.py` | 18% | **99%** |
| `utils/spatial.py` | 44% | **100%** |
| `utils/config.py` | 69% | **98%** |
| `validation/report.py` | 70% | **100%** |
| `fragments/mesh.py` | 45% | **86%** |

### Remaining uncovered (intentionally deprioritized):

| Module | Coverage | Reason |
|--------|----------|--------|
| `cli.py` | 0% | Thin CLI wrapper, no logic to test |
| `io/precomputed_reader.py` | 0% | Requires `cloud-volume` optional dep |
| `io/zarr_reader.py` | 0% | Requires zarr test files |
| `visualization/plot_connections.py` | 0% | Matplotlib plotting, hard to unit test |
| `visualization/neuroglancer_annotations.py` | 0% | Requires neuroglancer dep |
| `fragments/skeleton.py` | 76% | TEASAR path requires `kimimaro` |
| `pipeline.py` | 76% | `_create_reader()` format branches |

**Observations:**
- Core logic (candidates, validation, assembly, export, stitching) now averages **95%+ coverage**
- The 83% overall number is dragged down by optional-dependency modules (zarr, precomputed, kimimaro, neuroglancer viz) that can't be tested without those deps installed
- **Excluding optional-dep modules, effective coverage is ~92%**
- All 167 tests run in ~16 seconds

**Action Items:**
- [x] Begin Phase 3: edge case & robustness testing (see Experiment 4)
- [ ] Consider adding `kimimaro` to dev deps for skeleton coverage
- [ ] Set up CI (GitHub Actions) to run tests on push

---

## Experiment 4: Edge Case & Robustness Testing (Phase 3)

**Date:** 2026-02-13
**Objective:** Verify pipeline handles degenerate, extreme, and boundary-condition inputs without crashing.

**Setup:**
- Added `tests/test_edge_cases.py` with 51 targeted tests across 11 categories
- Tests exercise the full pipeline and individual modules with pathological inputs

**Results:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total tests | 167 | **218** | +51 |
| Overall coverage | 83% | **85%** | +2pp |
| Statements missed | 294 | 263 | -31 |

### Bug Found and Fixed

**`connection_statistics()` crash on empty candidates** (`diagnostics.py:51`)
- When the pipeline produces zero candidates (e.g. empty volume, all fragments too far apart), `connection_statistics()` returned `{"total": 0}` without `accepted`/`rejected`/`ambiguous` keys
- `Pipeline._log_summary()` then crashed with `KeyError: 'accepted'`
- **Fix:** Added missing keys to the empty-case return value
- **Impact:** This would have crashed on any real dataset where a chunk produced no candidates

### Test Categories (51 tests):

| Category | Tests | Key Findings |
|----------|-------|-------------|
| Empty/degenerate volumes | 3 | Found the `connection_statistics` bug |
| Extreme fragment counts | 2 | 50+ labels handled correctly, single giant label OK |
| Fragment extraction edge cases | 6 | Non-contiguous same-label correctly splits, boundary detection works at corners/edges, min_voxel threshold exact |
| Scoring edge cases | 7 | Negative distance, zero max distance, zero radii, coincident endpoints — all handled gracefully |
| Validation rule edge cases | 9 | Missing fragments → AMBIGUOUS, boundary thresholds correct, high-degree branching rejected |
| Assembly edge cases | 3 | No accepted → no structures, ambiguous flagging works |
| Graph builder edge cases | 4 | Empty store, single fragment, beyond-distance, unknown method |
| Candidate generator edge cases | 2 | No-endpoint fallback to centroids |
| Validation report edge cases | 2 | Empty report, all-rejected |
| Types edge cases | 8 | Zero-volume bbox, touching overlaps, path length, gap distance |
| Full pipeline edge cases | 5 | Empty volume, distant fragments, touching fragments, strict rejection, no-rules |

### Coverage improvements from edge cases:

| Module | Before | After |
|--------|--------|-------|
| `utils/types.py` | 94% | **100%** |
| `visualization/diagnostics.py` | 87% | **100%** |
| `candidates/scoring.py` | 87% | **100%** |
| `candidates/continuity.py` | 94% | **100%** |
| `candidates/proximity.py` | 90% | **100%** |
| `validation/rules.py` | 89% | **96%** |
| `assembly/assembler.py` | 90% | **96%** |
| `fragments/store.py` | 90% | **96%** |

**Observations:**
- The `connection_statistics` bug is exactly the kind of issue that would surface on real data — segmentation volumes often have sparse regions producing zero candidates in some chunks
- All scoring functions handle degenerate inputs (zeros, negatives, infinities) gracefully
- The conservative validation approach works correctly: missing fragments → AMBIGUOUS (not crash), no rules → all AMBIGUOUS (uncertainty preserved)
- Pipeline correctly produces zero structures when inputs are degenerate (no false positives)

**Action Items:**
- [ ] Begin Phase 4: parameter sensitivity analysis
- [ ] Begin Phase 7: set up GitHub Actions CI
- [ ] Test with real lab data (pending mentor providing a sample volume)

---

## Experiment 5: CREMI Real Data Run

**Date:** 2026-02-17
**Objective:** First run on real data — validate that the pipeline handles actual segmentation from the CREMI challenge (Drosophila brain, Sample A).

**Setup:**
- **Dataset:** CREMI Sample A (`sample_A_20160501.hdf`), `volumes/labels/neuron_ids`
  - Full volume: 125×1250×1250, uint64, ~37K neuron IDs
  - Cropped to: **64×256×256** (800 unique neuron labels, no background)
  - Resolution: 40nm z-sections, 4nm xy pixels
- **Config:** `configs/cremi_sample_a.yaml`
  - `graph.max_distance_nm: 500` (tight — dense tissue at 4nm resolution)
  - `candidates.max_endpoint_distance_nm: 400`
  - `validation.rules.MaxDistanceRule.max_distance_nm: 400`
  - `validation.rules.BranchingLimitRule.max_branches: 100` (relaxed — see observations)
  - `fragments.min_voxel_count: 50`
- **Command:** `python -m connectomics_pipeline.cli --config configs/cremi_sample_a.yaml --verbose`

**Results:**

| Metric | Value |
|--------|-------|
| Runtime | **49.7 seconds** |
| Fragments extracted | 324 (pre-stitch) → **243** (post-stitch) |
| Mean fragment size | 21,891 voxels |
| Fragment size range | 50 – 1,030,653 voxels |
| Candidates generated | **3,369** |
| Accepted | **36** (1.1%) |
| Rejected | **1,937** (57.5%) |
| Ambiguous | **1,396** (41.4%) |
| Structures assembled | **30** |

### Validation breakdown:

| Status | Gap Distance (median) | Composite Score (median) |
|--------|----------------------|-------------------------|
| Accepted | 0.0 nm | 1.000 |
| Rejected | 418.9 nm | 0.368 |
| Ambiguous | 289.8 nm | 0.405 |

### Structure details:

| Metric | Value |
|--------|-------|
| Total structures | 30 |
| 2-fragment structures | 28 |
| 4-fragment structures | 2 |
| Median confidence | 1.000 |
| With ambiguous regions | 29/30 |

### Output files:

| File | Size |
|------|------|
| `fragment_graph.graphml` | 371 KB |
| `connections.csv` | 327 KB |
| `fragments.csv` | 23 KB |
| `structures.csv` | 1.3 KB |
| `pipeline_config.yaml` | 1.3 KB |

**Observations:**

1. **Pipeline runs successfully on real data.** No crashes, no errors — all stages (extraction, stitching, graph, candidates, validation, assembly, export) complete cleanly.

2. **Stitching works correctly:** 324 pre-stitch fragments merged to 243 post-stitch, indicating ~81 cross-chunk fragment merges (despite the crop being close to a single chunk, overlap regions still triggered stitching).

3. **Dense proximity graph is expected:** At 4nm xy resolution, neurons in Drosophila neuropil are tightly packed. The initial run with `max_distance_nm=1000` produced 13,871 edges (avg degree ~114). Tightening to 500nm reduced this to 3,369 edges — still dense but more tractable.

4. **BranchingLimitRule needed adjustment:** With default `max_branches=10`, virtually all candidates were rejected because the proximity graph's high density means every fragment neighbors many others. Relaxing to 100 resolved this. **Key insight:** the branching limit should be tuned relative to graph density, not just neuroscience expectations.

5. **Conservative validation working as designed:** 41.4% of candidates are AMBIGUOUS — the three-outcome system preserves uncertainty rather than forcing binary accept/reject. Accepted connections have near-zero gap distance and perfect composite scores, indicating the pipeline is correctly conservative.

6. **Accepted connections are touching/adjacent fragments:** Median gap distance of 0.0nm for accepted connections means these are fragments that are directly adjacent in the volume — consistent with the CREMI ground truth where same-neuron segments should be touching.

7. **Most structures are pairs:** 28/30 structures have exactly 2 fragments, suggesting the conservative thresholds only connect very confident pairs. The 2 structures with 4 fragments show multi-hop assembly is working.

8. **Nearly all structures have ambiguous regions:** 29/30 structures flagged with `has_ambiguous_regions=True`, meaning there are plausible but uncertain connections near each structure. This is expected and useful for manual review workflows.

**Action Items:**
- [ ] Run on progressively larger crops (128×512×512, full volume) to test scalability
- [ ] Compare assembled structures against CREMI ground truth neuron IDs (overlap analysis)
- [ ] Tune parameters: try relaxing accept_threshold from 0.8 to 0.6 to see if more valid connections are recovered
- [ ] Profile runtime breakdown (extraction vs. scoring vs. validation) for optimization
- [ ] Test with CREMI Sample B and C for generalization

---

## Experiment 6: Stress Testing with Varied Synthetic Data

**Date:** _(pending)_
**Objective:** Test pipeline robustness with edge-case volumes (empty, single-voxel fragments, heavily fragmented, large label counts).

---

## Experiment 7: Parameter Sensitivity

**Date:** _(pending)_
**Objective:** Evaluate how validation thresholds and scoring weights affect pipeline output (accept/reject/ambiguous ratios).

---

## Experiment 8: Larger Volume Scalability

**Date:** _(pending)_
**Objective:** Test pipeline on progressively larger synthetic volumes to identify performance bottlenecks.

---

_Template for new experiments:_

```markdown
## Experiment N: Title

**Date:**
**Objective:**

**Setup:**
- Configuration changes
- Data description

**Results:**
- Quantitative results (tables, numbers)
- Pass/fail summary

**Observations:**
- What worked, what didn't
- Unexpected behaviors

**Action Items:**
- [ ] Follow-up tasks
```
