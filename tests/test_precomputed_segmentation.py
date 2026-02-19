"""Tests for connectomics_pipeline.export.precomputed_segmentation."""

import json

import numpy as np
import pytest

from connectomics_pipeline.export.precomputed_segmentation import (
    build_corrected_volume,
    write_precomputed,
)
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.types import (
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
)


RESOLUTION = (40.0, 4.0, 4.0)  # z, y, x nm/voxel


def _frag(fid: int, label_id: int, centroid_vox) -> Fragment:
    c = np.array(centroid_vox, dtype=float) * np.array(RESOLUTION)
    bb = BoundingBox(min_corner=c - 20, max_corner=c + 20)
    return Fragment(
        fragment_id=fid,
        label_id=label_id,
        voxel_count=8,
        bounding_box=bb,
        centroid=c,
    )


def _cand(cid, fa, fb, status) -> CandidateConnection:
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=fa,
        fragment_b=fb,
        endpoint_a=np.zeros(3),
        endpoint_b=np.ones(3) * 10.0,
        proximity_score=0.9,
        alignment_score=0.8,
        continuity_score=0.8,
        size_score=0.9,
        composite_score=0.85,
        status=status,
    )


@pytest.fixture
def simple_volume():
    """4×8×8 volume (Z,Y,X) with two separate label-1 blobs and one label-2 blob.

    A gap column at x=4 separates blob A (x=0-3) from blob B (x=5-7) so
    ndimage.label treats them as distinct connected components.

    Label 1 blob A:  z=0-1, y=0-3, x=0-3   centroid vox ≈ (0,1,1)
    Label 1 blob B:  z=0-1, y=0-3, x=5-7   centroid vox ≈ (0,1,6)
    Label 2 blob C:  z=2-3, y=4-7, x=4-7   centroid vox ≈ (2,5,5)
    """
    vol = np.zeros((4, 8, 8), dtype=np.uint64)
    vol[0:2, 0:4, 0:4] = 1  # blob A (x=0-3)
    vol[0:2, 0:4, 5:8] = 1  # blob B (x=5-7, gap at x=4)
    vol[2:4, 4:8, 4:8] = 2  # blob C
    return vol


@pytest.fixture
def store_for_volume():
    """Fragment store matching the blobs in simple_volume."""
    store = FragmentStore()
    # frag 0 → blob A, label 1, centroid vox (0,1,1)
    store.add(_frag(0, 1, [0, 1, 1]))
    # frag 1 → blob B, label 1, centroid vox (0,1,6)
    store.add(_frag(1, 1, [0, 1, 6]))
    # frag 2 → blob C, label 2, centroid vox (2,5,5)
    store.add(_frag(2, 2, [2, 5, 5]))
    return store


# ------------------------------------------------------------------
# build_corrected_volume tests
# ------------------------------------------------------------------

class TestBuildCorrectedVolume:
    def test_no_accepted_keeps_components_separate(self, simple_volume, store_for_volume):
        """With no accepted connections, every component keeps its own ID."""
        candidates = [
            _cand(0, 0, 1, ConnectionStatus.REJECTED),
            _cand(1, 0, 2, ConnectionStatus.REJECTED),
        ]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        # Three components → three distinct non-zero labels
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 3

    def test_accepted_same_label_merges_components(self, simple_volume, store_for_volume):
        """Accepting frag 0 + frag 1 (both label 1) unifies their component IDs."""
        candidates = [_cand(0, 0, 1, ConnectionStatus.ACCEPTED)]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        # Blob A and blob B now share a label; blob C is different
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 2

        # All voxels that were label-1 blobs share the same corrected label
        blob_a_label = corrected[0, 1, 1]
        blob_b_label = corrected[0, 1, 5]
        assert blob_a_label == blob_b_label
        assert blob_a_label != 0

    def test_cross_label_merge_unifies_different_labels(self, simple_volume, store_for_volume):
        """Accepting frag 0 (label 1) + frag 2 (label 2) unifies them."""
        candidates = [_cand(0, 0, 2, ConnectionStatus.ACCEPTED)]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        labels = set(np.unique(corrected)) - {0}
        # frag1 (blob B) is unmerged, frag0 and frag2 share a label → 2 distinct IDs
        assert len(labels) == 2

    def test_output_dtype_is_uint64(self, simple_volume, store_for_volume):
        corrected = build_corrected_volume(simple_volume, [], store_for_volume, RESOLUTION)
        assert corrected.dtype == np.uint64

    def test_output_shape_matches_input(self, simple_volume, store_for_volume):
        corrected = build_corrected_volume(simple_volume, [], store_for_volume, RESOLUTION)
        assert corrected.shape == simple_volume.shape

    def test_background_remains_zero(self, simple_volume, store_for_volume):
        corrected = build_corrected_volume(simple_volume, [], store_for_volume, RESOLUTION)
        assert corrected[simple_volume == 0].max() == 0

    def test_empty_candidates(self, simple_volume, store_for_volume):
        """Empty candidate list produces a valid component-labeled volume."""
        corrected = build_corrected_volume(
            simple_volume, [], store_for_volume, RESOLUTION
        )
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 3  # three distinct components, no merges

    def test_ambiguous_candidates_ignored(self, simple_volume, store_for_volume):
        """Ambiguous candidates do not trigger a merge."""
        candidates = [_cand(0, 0, 1, ConnectionStatus.AMBIGUOUS)]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 3  # no merge applied

    def test_all_merged_into_one(self, simple_volume, store_for_volume):
        """Accepting all pairs results in a single foreground label."""
        candidates = [
            _cand(0, 0, 1, ConnectionStatus.ACCEPTED),
            _cand(1, 1, 2, ConnectionStatus.ACCEPTED),
        ]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 1

    def test_missing_fragment_in_store(self, simple_volume, store_for_volume):
        """Accepted candidate referencing unknown fragment_id is skipped."""
        candidates = [_cand(0, 0, 99, ConnectionStatus.ACCEPTED)]
        corrected = build_corrected_volume(
            simple_volume, candidates, store_for_volume, RESOLUTION
        )
        # Should not crash; 3 components remain separate
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 3

    def test_single_label_volume(self):
        """Volume with a single label produces one component ID."""
        vol = np.ones((4, 4, 4), dtype=np.uint64)
        store = FragmentStore()
        store.add(_frag(0, 1, [2, 2, 2]))
        corrected = build_corrected_volume(vol, [], store, RESOLUTION)
        labels = set(np.unique(corrected)) - {0}
        assert len(labels) == 1

    def test_empty_volume(self):
        """All-background volume returns all-zero corrected volume."""
        vol = np.zeros((4, 4, 4), dtype=np.uint64)
        store = FragmentStore()
        corrected = build_corrected_volume(vol, [], store, RESOLUTION)
        assert corrected.max() == 0


# ------------------------------------------------------------------
# write_precomputed tests
# ------------------------------------------------------------------

class TestWritePrecomputed:
    def test_creates_info_file(self, tmp_path, simple_volume):
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        assert (tmp_path / "info").exists()

    def test_info_file_is_valid_json(self, tmp_path, simple_volume):
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        info = json.loads((tmp_path / "info").read_text())
        assert info["@type"] == "neuroglancer_multiscale_volume"
        assert info["type"] == "segmentation"
        assert info["data_type"] == "uint64"

    def test_info_size_is_xyz(self, tmp_path, simple_volume):
        """size in info must be [X, Y, Z], not [Z, Y, X]."""
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        info = json.loads((tmp_path / "info").read_text())
        Z, Y, X = simple_volume.shape
        assert info["scales"][0]["size"] == [X, Y, Z]

    def test_info_resolution_is_xyz(self, tmp_path, simple_volume):
        """resolution in info must be [rx, ry, rz] (x, y, z order)."""
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        info = json.loads((tmp_path / "info").read_text())
        rz, ry, rx = RESOLUTION
        assert info["scales"][0]["resolution"] == [rx, ry, rz]

    def test_creates_scale_directory(self, tmp_path, simple_volume):
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        assert (tmp_path / "1_1_1").is_dir()

    def test_creates_chunk_file(self, tmp_path, simple_volume):
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        Z, Y, X = simple_volume.shape
        chunk_name = f"0-{X}_0-{Y}_0-{Z}"
        assert (tmp_path / "1_1_1" / chunk_name).exists()

    def test_chunk_byte_size(self, tmp_path, simple_volume):
        """Chunk file must contain exactly Z*Y*X uint64 values (8 bytes each)."""
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        Z, Y, X = simple_volume.shape
        chunk_name = f"0-{X}_0-{Y}_0-{Z}"
        chunk_bytes = (tmp_path / "1_1_1" / chunk_name).read_bytes()
        assert len(chunk_bytes) == Z * Y * X * 8

    def test_chunk_roundtrip(self, tmp_path, simple_volume):
        """Data written and read back matches the original volume."""
        write_precomputed(simple_volume, tmp_path, RESOLUTION)
        Z, Y, X = simple_volume.shape
        chunk_name = f"0-{X}_0-{Y}_0-{Z}"
        raw = (tmp_path / "1_1_1" / chunk_name).read_bytes()
        recovered = np.frombuffer(raw, dtype="<u8").reshape(Z, Y, X)
        np.testing.assert_array_equal(recovered, simple_volume.astype(np.uint64))

    def test_output_dir_created_if_absent(self, tmp_path, simple_volume):
        nested = tmp_path / "deep" / "nested"
        write_precomputed(simple_volume, nested, RESOLUTION)
        assert (nested / "info").exists()

    def test_single_voxel_volume(self, tmp_path):
        vol = np.array([[[42]]], dtype=np.uint64)
        write_precomputed(vol, tmp_path, RESOLUTION)
        chunk_name = "0-1_0-1_0-1"
        raw = (tmp_path / "1_1_1" / chunk_name).read_bytes()
        assert len(raw) == 8
        assert np.frombuffer(raw, dtype="<u8")[0] == 42
