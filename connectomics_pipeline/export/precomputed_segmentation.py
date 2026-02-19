"""Export a corrected segmentation volume in Neuroglancer precomputed format.

The corrected volume is built by:
1. Re-labeling the input volume so every connected component gets a unique
   integer ID (the "fragment-level" segmentation).
2. Applying a union-find merge for every ACCEPTED candidate connection, so
   the two component IDs become one.
3. Writing the result as a single-scale Neuroglancer precomputed raw volume.

The output directory can be served via any HTTP server and loaded in
Neuroglancer as a segmentation layer, making accepted merges directly visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from scipy import ndimage

from connectomics_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from connectomics_pipeline.fragments.store import FragmentStore
    from connectomics_pipeline.utils.types import CandidateConnection

logger = get_logger("export.precomputed_segmentation")


def build_corrected_volume(
    volume: np.ndarray,
    candidates: List[CandidateConnection],
    store: FragmentStore,
    resolution: Tuple[float, ...],
) -> np.ndarray:
    """Return a relabeled volume with accepted merges applied.

    Each connected component in *volume* receives a unique integer ID.
    Pairs of components connected by an ACCEPTED candidate are then unified
    under the same ID via union-find.

    Args:
        volume: Original segmentation, shape (Z, Y, X), any integer dtype.
        candidates: All pipeline candidates (only accepted ones are used).
        store: Fragment store for centroid lookup.
        resolution: (z, y, x) voxel size in nm, used to convert physical
            centroids back to voxel indices.

    Returns:
        uint64 array of shape (Z, Y, X) with corrected labels.
    """
    res = np.array(resolution, dtype=float)

    # ------------------------------------------------------------------
    # Step 1: assign a unique component ID to every connected component
    # ------------------------------------------------------------------
    comp_vol = np.zeros(volume.shape, dtype=np.uint64)
    next_comp: int = 1

    for lbl in np.unique(volume):
        if lbl == 0:
            continue
        mask = volume == lbl
        labeled, n_comps = ndimage.label(mask)
        for cid in range(1, n_comps + 1):
            comp_vol[labeled == cid] = next_comp
            next_comp += 1

    n_total = next_comp - 1
    logger.info("Re-labeled volume into %d connected components", n_total)

    # ------------------------------------------------------------------
    # Step 2: map pipeline fragment_ids → component IDs via centroid
    # ------------------------------------------------------------------
    vol_shape = np.array(volume.shape, dtype=int)  # (Z, Y, X)

    frag_to_comp: dict[int, int] = {}
    missed = 0
    for frag in store.all_fragments():
        voxel = np.round(frag.centroid / res).astype(int)
        voxel = np.clip(voxel, 0, vol_shape - 1)
        comp_id = int(comp_vol[voxel[0], voxel[1], voxel[2]])
        if comp_id > 0:
            frag_to_comp[frag.fragment_id] = comp_id
        else:
            missed += 1

    if missed:
        logger.warning("%d fragments could not be matched to a component via centroid", missed)

    # ------------------------------------------------------------------
    # Step 3: union-find over component IDs for accepted connections
    # ------------------------------------------------------------------
    parent = list(range(next_comp))  # index = comp_id; parent[i] = i initially

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return
        # Always root under the lower ID for determinism
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    accepted_count = 0
    for cand in candidates:
        if cand.status.value != "accepted":
            continue
        comp_a = frag_to_comp.get(cand.fragment_a)
        comp_b = frag_to_comp.get(cand.fragment_b)
        if comp_a is not None and comp_b is not None:
            _union(comp_a, comp_b)
            accepted_count += 1

    logger.info("Applied %d accepted merge(s) via union-find", accepted_count)

    # ------------------------------------------------------------------
    # Step 4: relabel every voxel to its canonical root component ID
    # ------------------------------------------------------------------
    corrected = np.zeros(volume.shape, dtype=np.uint64)
    for comp_id in range(1, next_comp):
        mask = comp_vol == comp_id
        if mask.any():
            corrected[mask] = np.uint64(_find(comp_id))

    return corrected


def write_precomputed(
    volume: np.ndarray,
    output_dir: Path,
    resolution: Tuple[float, ...],
) -> None:
    """Write a uint64 volume as a single-scale Neuroglancer precomputed layer.

    Produces:
        <output_dir>/info           — JSON metadata file
        <output_dir>/1_1_1/<chunk>  — raw little-endian uint64 chunk

    The output directory can be served with any static HTTP server and loaded
    in Neuroglancer as a segmentation source.

    Args:
        volume: uint64 array of shape (Z, Y, X).
        output_dir: Destination directory (created if absent).
        resolution: (z, y, x) voxel size in nm. Written to info as (x, y, z).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, X = volume.shape
    rz, ry, rx = float(resolution[0]), float(resolution[1]), float(resolution[2])

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "key": "1_1_1",
                "size": [X, Y, Z],
                "resolution": [rx, ry, rz],
                "chunk_sizes": [[X, Y, Z]],
                "encoding": "raw",
                "voxel_offset": [0, 0, 0],
            }
        ],
    }

    with open(output_dir / "info", "w") as f:
        json.dump(info, f, indent=2)

    scale_dir = output_dir / "1_1_1"
    scale_dir.mkdir(exist_ok=True)

    # Neuroglancer raw encoding: little-endian uint64, x varies fastest.
    # Our array is (Z, Y, X) C-order → X is already the fastest axis. ✓
    chunk_name = f"0-{X}_0-{Y}_0-{Z}"
    chunk_bytes = volume.astype("<u8").tobytes()
    with open(scale_dir / chunk_name, "wb") as f:
        f.write(chunk_bytes)

    logger.info(
        "Wrote precomputed segmentation to %s (%dx%dx%d voxels, %d bytes)",
        output_dir,
        X,
        Y,
        Z,
        len(chunk_bytes),
    )
