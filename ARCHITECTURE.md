# Post-Segmentation Systems Pipeline — Architecture Document

**Project:** CPSC 4900 Senior Project  
**Advisor:** Dr. Xiuye (Sue) Chen  
**Collaborating Lab:** Dr. Aaron T. Kuan Lab (Yale School of Medicine)  

---

## 1. System Overview

This pipeline consumes voxel-wise segmentation volumes (produced by frameworks such as PyTorch Connectomics) and transforms them into graph-based connectivity representations suitable for downstream analysis, proofreading, and visualization.

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     POST-SEGMENTATION PIPELINE                         │
│                                                                         │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────────────┐  │
│  │  INPUT    │──▶│  FRAGMENT   │──▶│   GRAPH   │──▶│    CANDIDATE     │  │
│  │  LOADER   │   │  EXTRACTOR │   │  BUILDER  │   │    GENERATOR     │  │
│  └──────────┘   └────────────┘   └───────────┘   └──────────────────┘  │
│                                                           │             │
│                                                           ▼             │
│  ┌──────────┐   ┌────────────┐   ┌───────────────────────────────────┐ │
│  │  EXPORT  │◀──│  ASSEMBLER │◀──│   CONSERVATIVE VALIDATOR          │ │
│  └──────────┘   └────────────┘   └───────────────────────────────────┘ │
│       │                                                                 │
│       ▼                                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  OUTPUTS: connectivity graphs, metadata, confidence scores,       │  │
│  │           visualization exports, diagnostic reports               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Architecture

### 2.1 `io` — Input/Output Handling

**Responsibility:** Load segmentation volumes from various formats; abstract storage details from the rest of the pipeline.

**Supported formats:**
- HDF5 (`.h5` / `.hdf5`)
- Zarr (local and cloud-compatible)
- Precomputed (Neuroglancer-style)
- Raw NumPy arrays (for testing)

**Key design decisions:**
- All loaders implement a common `VolumeReader` interface returning chunks as `numpy.ndarray`
- Chunked/block-wise reading is mandatory — full volumes may exceed memory
- Lazy loading with configurable chunk overlap (needed for cross-boundary fragment detection)

**Interface:**

```python
class VolumeReader(Protocol):
    """Abstract interface for reading segmentation volumes."""
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Full volume shape (Z, Y, X)."""
        ...
    
    @property
    def dtype(self) -> np.dtype:
        """Label dtype (typically uint32 or uint64)."""
        ...
    
    @property
    def resolution(self) -> Tuple[float, float, float]:
        """Voxel resolution in physical units (nm), (Z, Y, X)."""
        ...
    
    def read_chunk(self, slices: Tuple[slice, slice, slice]) -> np.ndarray:
        """Read a subvolume defined by slices."""
        ...
    
    def chunk_iterator(self, chunk_size: Tuple[int, int, int],
                       overlap: Tuple[int, int, int] = (0, 0, 0)) -> Iterator:
        """Iterate over the volume in chunks with optional overlap."""
        ...
```

**Files:**
- `io/volume_reader.py` — Protocol + base utilities
- `io/hdf5_reader.py` — HDF5 loader
- `io/zarr_reader.py` — Zarr loader
- `io/precomputed_reader.py` — Neuroglancer precomputed loader
- `io/numpy_reader.py` — In-memory array wrapper (testing)

---

### 2.2 `fragments` — Fragment Extraction & Representation

**Responsibility:** Extract connected components from segmentation labels, compute per-fragment metadata, and optionally generate skeletons and meshes.

**Pipeline within this module:**

```
Segmentation labels
  → Connected component extraction (per-label, handles chunk boundaries)
  → Fragment metadata computation (volume, bounding box, centroid, endpoints)
  → [Optional] Skeletonization (via TEASAR or similar)
  → [Optional] Mesh extraction (marching cubes)
  → FragmentStore (indexed collection of Fragment objects)
```

**Key data structures:**

```python
@dataclass
class Fragment:
    """A single connected component from the segmentation."""
    fragment_id: int              # Unique ID
    label_id: int                 # Original segmentation label
    voxel_count: int              # Size in voxels
    bounding_box: BoundingBox     # (min_z, min_y, min_x, max_z, max_y, max_x)
    centroid: np.ndarray          # (z, y, x) in physical coordinates
    endpoints: List[np.ndarray]   # Terminal points (from skeleton or boundary analysis)
    
    # Optional representations
    skeleton: Optional[Skeleton]         # Node-edge skeleton graph
    mesh: Optional[Mesh]                 # Surface mesh (vertices, faces)
    
    # Metadata
    is_boundary: bool             # Touches chunk boundary
    chunk_origin: Tuple[int, ...]  # Which chunk this was extracted from

@dataclass
class BoundingBox:
    min_corner: np.ndarray  # (z, y, x)
    max_corner: np.ndarray  # (z, y, x)

@dataclass  
class Skeleton:
    nodes: np.ndarray       # (N, 3) coordinates
    edges: np.ndarray       # (E, 2) indices into nodes
    radii: np.ndarray       # (N,) estimated radius at each node

@dataclass
class Mesh:
    vertices: np.ndarray    # (V, 3) coordinates
    faces: np.ndarray       # (F, 3) triangle indices

class FragmentStore:
    """Indexed collection of fragments with spatial queries."""
    
    def add(self, fragment: Fragment) -> None: ...
    def get(self, fragment_id: int) -> Fragment: ...
    def query_bbox(self, bbox: BoundingBox) -> List[Fragment]: ...
    def query_radius(self, point: np.ndarray, radius: float) -> List[Fragment]: ...
    def get_boundary_fragments(self) -> List[Fragment]: ...
    def __len__(self) -> int: ...
```

**Chunk boundary handling:**
- Fragments touching chunk boundaries are flagged (`is_boundary=True`)
- Cross-boundary fragment merging uses label agreement in overlap zones
- A `ChunkStitcher` class handles the global relabeling

**Files:**
- `fragments/extraction.py` — Connected component extraction
- `fragments/metadata.py` — Bounding box, centroid, endpoint computation
- `fragments/skeleton.py` — Skeletonization (TEASAR wrapper)
- `fragments/mesh.py` — Marching cubes mesh extraction
- `fragments/store.py` — FragmentStore with spatial indexing (KD-tree backed)
- `fragments/stitching.py` — Cross-chunk boundary stitching

---

### 2.3 `graph` — Graph Construction

**Responsibility:** Build the fragment adjacency graph. Nodes represent fragments (or fragment endpoints). Edges represent spatial adjacency or proximity.

**Graph structure:**

```python
@dataclass
class FragmentGraph:
    """Graph where nodes are fragments and edges represent spatial relationships."""
    
    graph: nx.Graph  # NetworkX graph
    
    # Node attributes (stored on graph nodes):
    #   - fragment_id: int
    #   - centroid: np.ndarray
    #   - endpoints: List[np.ndarray]
    #   - voxel_count: int
    #   - label_id: int
    
    # Edge attributes (stored on graph edges):
    #   - distance: float          (minimum distance between fragments)
    #   - contact_area: float      (shared boundary area, if adjacent)
    #   - endpoint_pair: Tuple     (which endpoints are closest)
    
    def from_fragment_store(self, store: FragmentStore, 
                            max_distance: float) -> None:
        """Build graph by connecting fragments within max_distance."""
        ...
    
    def get_neighbors(self, fragment_id: int) -> List[int]: ...
    def get_edge_data(self, fid_a: int, fid_b: int) -> dict: ...
    def subgraph(self, fragment_ids: Set[int]) -> 'FragmentGraph': ...
```

**Graph construction strategies:**
1. **Contact-based:** Fragments sharing a voxel face boundary get an edge (contact_area > 0)
2. **Proximity-based:** Fragments within a configurable distance threshold get an edge
3. **Endpoint-based:** Edges connect fragment endpoints that are spatially close

All three can be combined. The graph builder is configurable.

**Files:**
- `graph/builder.py` — Graph construction from FragmentStore
- `graph/fragment_graph.py` — FragmentGraph class and utilities
- `graph/spatial_index.py` — Spatial indexing utilities for neighbor queries

---

### 2.4 `candidates` — Candidate Connection Generation

**Responsibility:** Propose fragment-to-fragment connections that *might* represent the same neuron. High recall, low commitment.

**Candidate generation criteria:**

| Criterion | Description | Implementation |
|-----------|-------------|----------------|
| Proximity | Endpoints within distance threshold | KD-tree nearest-neighbor query |
| Orientation | Tangent vectors at endpoints roughly aligned | Dot product of tangent vectors |
| Continuity | Smooth path would connect the fragments | Curvature estimation |
| Size compatibility | Fragments have compatible radii | Radius ratio check |

**Key data structures:**

```python
@dataclass
class CandidateConnection:
    """A proposed connection between two fragments."""
    candidate_id: int
    fragment_a: int                # Source fragment ID
    fragment_b: int                # Target fragment ID
    endpoint_a: np.ndarray         # Connection point on fragment A
    endpoint_b: np.ndarray         # Connection point on fragment B
    
    # Scores (all in [0, 1], higher = more plausible)
    proximity_score: float         # Based on distance
    alignment_score: float         # Based on tangent alignment
    continuity_score: float        # Based on curvature smoothness
    size_score: float              # Based on radius compatibility
    
    # Composite
    composite_score: float         # Weighted combination
    
    # Status
    status: ConnectionStatus       # PROPOSED, ACCEPTED, REJECTED, AMBIGUOUS

class ConnectionStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"  # Explicitly unresolved
```

**Design philosophy:** This stage should be *generous* — it is better to propose a connection that later gets rejected than to miss a real connection. The conservative filtering happens in the next stage.

**Files:**
- `candidates/generator.py` — Main candidate generation logic
- `candidates/proximity.py` — Distance-based candidate search
- `candidates/alignment.py` — Tangent/orientation scoring
- `candidates/continuity.py` — Curvature and smoothness scoring
- `candidates/scoring.py` — Composite score computation

---

### 2.5 `validation` — Conservative Validation

**Responsibility:** Filter candidate connections. Accept only high-confidence connections. Reject clearly implausible ones. Explicitly flag ambiguous cases.

**Three-outcome model:**

```
ACCEPT    — High confidence this is a real connection
REJECT    — High confidence this is NOT a real connection
AMBIGUOUS — Cannot determine; preserve for human review
```

**Validation checks (applied in sequence):**

```python
class ValidationRule(Protocol):
    """Interface for a single validation check."""
    
    def validate(self, candidate: CandidateConnection,
                 graph: FragmentGraph,
                 store: FragmentStore) -> ValidationResult: ...

@dataclass
class ValidationResult:
    rule_name: str
    decision: ConnectionStatus  # ACCEPT, REJECT, or AMBIGUOUS
    confidence: float           # How confident the rule is
    reason: str                 # Human-readable explanation
```

**Built-in validation rules:**

| Rule | Rejects when... |
|------|-----------------|
| `MaxDistanceRule` | Gap between endpoints exceeds threshold |
| `CurvatureRule` | Connecting path would require excessive bending |
| `DirectionReversalRule` | Fragments point away from each other |
| `SizeDiscrepancyRule` | Fragment radii are wildly different |
| `BranchingLimitRule` | Accepting would create implausible branching patterns |
| `OverlapRule` | Fragments' bounding boxes overlap excessively (merge error) |
| `CompositeScoreRule` | Composite candidate score below threshold |

**Validation pipeline:**

```python
class ValidationPipeline:
    """Runs candidates through a sequence of validation rules."""
    
    def __init__(self, rules: List[ValidationRule],
                 accept_threshold: float = 0.8,
                 reject_threshold: float = 0.3):
        self.rules = rules
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
    
    def validate_all(self, candidates: List[CandidateConnection],
                     graph: FragmentGraph,
                     store: FragmentStore) -> ValidationReport:
        """Run all candidates through validation rules.
        
        Decision logic:
        - If ANY rule gives a hard REJECT → REJECT
        - If all rules pass AND composite confidence >= accept_threshold → ACCEPT
        - Otherwise → AMBIGUOUS
        """
        ...
```

**Files:**
- `validation/pipeline.py` — ValidationPipeline orchestration
- `validation/rules.py` — Built-in validation rules
- `validation/report.py` — ValidationReport data structure

---

### 2.6 `assembly` — Structure Assembly

**Responsibility:** Merge accepted connections into coherent structures (reconstructed neurons/axons). Preserve uncertainty labels.

**Assembly strategy:**

```
1. Start with FragmentGraph
2. Add edges for ACCEPTED connections
3. Find connected components → each is a reconstructed structure
4. Attach AMBIGUOUS connections as annotations (not merged)
5. Compute per-structure confidence (based on weakest link)
6. Detect and flag problematic topologies (unexpected cycles, excessive branching)
```

**Key data structures:**

```python
@dataclass
class AssembledStructure:
    """A reconstructed neuron or axon assembled from fragments."""
    structure_id: int
    fragment_ids: List[int]            # Constituent fragments
    accepted_connections: List[int]     # CandidateConnection IDs that were accepted
    ambiguous_connections: List[int]    # Nearby ambiguous connections (not merged)
    
    # Quality metrics
    confidence: float                  # Min confidence across all accepted connections
    total_path_length: float           # Physical length of assembled structure
    num_branches: int                  # Branch points
    
    # Flags
    has_ambiguous_regions: bool        # Any ambiguous connections touch this structure
    topology_warnings: List[str]       # E.g., "unexpected cycle detected"

class Assembler:
    def assemble(self, graph: FragmentGraph,
                 validation_report: ValidationReport) -> List[AssembledStructure]:
        """Build assembled structures from validated connections."""
        ...
```

**Files:**
- `assembly/assembler.py` — Main assembly logic
- `assembly/topology.py` — Topology checks (cycle detection, branch analysis)
- `assembly/confidence.py` — Per-structure confidence computation

---

### 2.7 `export` — Output Generation

**Responsibility:** Export assembled structures and metadata in formats consumable by downstream tools.

**Export formats:**

| Format | Use case |
|--------|----------|
| NetworkX GraphML / JSON | Graph analysis |
| SWC | Neuron morphology (standard format) |
| Neuroglancer precomputed | Web-based visualization |
| CSV / Parquet | Tabular metadata and statistics |
| HDF5 | Compact binary storage |

**Exported data:**

```python
class Exporter:
    def export_graph(self, structures: List[AssembledStructure],
                     format: str, path: str) -> None: ...
    
    def export_metadata(self, structures: List[AssembledStructure],
                        path: str) -> None: ...
    
    def export_validation_report(self, report: ValidationReport,
                                 path: str) -> None: ...
    
    def export_swc(self, structure: AssembledStructure,
                   store: FragmentStore, path: str) -> None: ...
```

**Files:**
- `export/graph_export.py` — Graph format exports
- `export/swc_export.py` — SWC neuron format
- `export/metadata_export.py` — CSV/Parquet tables
- `export/neuroglancer_export.py` — Precomputed visualization volumes

---

### 2.8 `visualization` — Diagnostics & Visualization

**Responsibility:** Tools for inspecting pipeline outputs, debugging connections, and generating diagnostic reports.

**Capabilities:**
- Render accepted vs. rejected vs. ambiguous connections (color-coded)
- 3D scatter/line plots of fragments and connections
- Per-structure detail views
- Summary statistics dashboards
- Neuroglancer-compatible annotation layers

**Files:**
- `visualization/plot_connections.py` — 3D connection visualization
- `visualization/diagnostics.py` — Summary statistics and reports
- `visualization/neuroglancer_annotations.py` — Annotation layer generation

---

### 2.9 `utils` — Shared Utilities

- `utils/config.py` — Pipeline configuration (YAML-based)
- `utils/logging.py` — Structured logging
- `utils/spatial.py` — Spatial math utilities (distances, tangents, curvature)
- `utils/types.py` — Shared type definitions

---

## 3. Configuration System

All pipeline parameters are controlled via a single YAML config:

```yaml
pipeline:
  name: "post-segmentation-pipeline"
  version: "0.1.0"

input:
  format: "hdf5"              # hdf5 | zarr | precomputed | numpy
  path: "/data/segmentation.h5"
  dataset: "labels"
  resolution: [30, 8, 8]      # nm (z, y, x)
  chunk_size: [128, 256, 256]
  chunk_overlap: [8, 16, 16]

fragments:
  min_voxel_count: 100         # Discard tiny fragments
  extract_skeletons: true
  extract_meshes: false
  skeleton_method: "teasar"
  skeleton_invalidation_d0: 10  # TEASAR parameter

graph:
  construction_method: "proximity"  # contact | proximity | endpoint
  max_distance_nm: 2000            # Max gap to consider

candidates:
  max_endpoint_distance_nm: 1500
  min_alignment_score: 0.3
  weights:
    proximity: 0.35
    alignment: 0.30
    continuity: 0.25
    size: 0.10

validation:
  accept_threshold: 0.8
  reject_threshold: 0.3
  rules:
    - name: "MaxDistanceRule"
      max_distance_nm: 1500
    - name: "CurvatureRule"
      max_curvature_deg: 90
    - name: "DirectionReversalRule"
      min_alignment: 0.0
    - name: "SizeDiscrepancyRule"
      max_radius_ratio: 3.0
    - name: "BranchingLimitRule"
      max_branches: 10

assembly:
  min_structure_fragments: 2
  flag_ambiguous: true

export:
  formats: ["graphml", "csv", "swc"]
  output_dir: "/output/results"

logging:
  level: "INFO"
  file: "pipeline.log"
```

---

## 4. Data Flow Summary

```
                         ┌─────────────┐
                         │ Config YAML │
                         └──────┬──────┘
                                │
                                ▼
  Segmentation Volume ──▶ [VolumeReader] 
                                │
                        numpy chunks (uint64)
                                │
                                ▼
                     [FragmentExtractor]
                                │
                     FragmentStore (indexed)
                                │
                                ▼
                      [GraphBuilder]
                                │
                       FragmentGraph (nx.Graph)
                                │
                                ▼
                   [CandidateGenerator]
                                │
                  List[CandidateConnection]
                                │
                                ▼
                  [ValidationPipeline]
                                │
                     ValidationReport
                     (accepted/rejected/ambiguous)
                                │
                                ▼
                       [Assembler]
                                │
                List[AssembledStructure]
                                │
                                ▼
                       [Exporter]
                                │
              GraphML, SWC, CSV, precomputed
```

---

## 5. Directory Structure

```
connectomics-pipeline/
├── connectomics_pipeline/
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── io/
│   │   ├── __init__.py
│   │   ├── volume_reader.py     # Protocol + base
│   │   ├── hdf5_reader.py
│   │   ├── zarr_reader.py
│   │   ├── precomputed_reader.py
│   │   └── numpy_reader.py
│   ├── fragments/
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   ├── metadata.py
│   │   ├── skeleton.py
│   │   ├── mesh.py
│   │   ├── store.py
│   │   └── stitching.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── fragment_graph.py
│   │   └── spatial_index.py
│   ├── candidates/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── proximity.py
│   │   ├── alignment.py
│   │   ├── continuity.py
│   │   └── scoring.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── rules.py
│   │   └── report.py
│   ├── assembly/
│   │   ├── __init__.py
│   │   ├── assembler.py
│   │   ├── topology.py
│   │   └── confidence.py
│   ├── export/
│   │   ├── __init__.py
│   │   ├── graph_export.py
│   │   ├── swc_export.py
│   │   ├── metadata_export.py
│   │   └── neuroglancer_export.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_connections.py
│   │   ├── diagnostics.py
│   │   └── neuroglancer_annotations.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── spatial.py
│       └── types.py
├── tests/
│   ├── __init__.py
│   ├── test_io.py
│   ├── test_fragments.py
│   ├── test_graph.py
│   ├── test_candidates.py
│   ├── test_validation.py
│   ├── test_assembly.py
│   └── test_pipeline.py
├── scripts/
│   ├── run_pipeline.py          # CLI entry point
│   ├── inspect_connections.py   # Diagnostic script
│   └── generate_test_data.py    # Synthetic data generator
├── configs/
│   ├── default.yaml
│   └── example_small.yaml
├── docs/
│   ├── ARCHITECTURE.md          # This file
│   └── USER_GUIDE.md
├── examples/
│   └── demo_notebook.ipynb
├── pyproject.toml
└── README.md
```

---

## 6. Design Principles

1. **Conservative by default.** Never force a merge. When in doubt, label as AMBIGUOUS.
2. **Modular.** Each stage can be run independently. Intermediate results can be saved and loaded.
3. **Reproducible.** Full config is logged. Random seeds are controllable. Results are deterministic for the same inputs and config.
4. **Format-agnostic.** Input loaders are pluggable. The pipeline doesn't care where segmentation came from.
5. **Uncertainty-preserving.** Ambiguous connections are first-class citizens, not discarded.
6. **Scalable.** Chunked processing, spatial indexing, and lazy loading support large volumes.

---

## 7. Dependencies

### Core
- `numpy` — Array operations
- `scipy` — Spatial algorithms (KD-tree, connected components, distance transforms)
- `scikit-image` — Image processing (marching cubes, skeletonization)
- `networkx` — Graph data structures and algorithms
- `h5py` — HDF5 I/O
- `zarr` — Zarr I/O
- `pyyaml` — Configuration

### Optional
- `kimimaro` — TEASAR skeletonization (preferred over scikit-image for neuron skeletons)
- `cloud-volume` — Neuroglancer precomputed format
- `navis` — Neuron analysis and SWC handling
- `matplotlib` — Basic plotting
- `pandas` — Tabular metadata export
- `pyarrow` — Parquet export

### Development
- `pytest` — Testing
- `black` — Code formatting
- `mypy` — Type checking
