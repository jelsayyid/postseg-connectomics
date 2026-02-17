"""Tests for ZarrReader and _open_zarr_array."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from connectomics_pipeline.io.zarr_reader import ZarrReader, _open_zarr_array


@pytest.fixture
def zarr_group_path(tmp_path):
    """Zarr store with a 'labels' dataset inside a group."""
    data = np.arange(27, dtype=np.uint32).reshape(3, 3, 3)
    store_path = str(tmp_path / "test.zarr")
    z = zarr.open_group(store_path, mode="w")
    z.create_dataset("labels", data=data)
    return store_path, data


@pytest.fixture
def zarr_direct_path(tmp_path):
    """Zarr array stored directly (no enclosing group)."""
    data = np.arange(8, dtype=np.uint32).reshape(2, 2, 2)
    store_path = str(tmp_path / "direct.zarr")
    zarr.save(store_path, data)
    return store_path, data


class TestOpenZarrArray:
    def test_with_dataset_name(self, zarr_group_path):
        path, data = zarr_group_path
        arr = _open_zarr_array(path, "labels")
        assert isinstance(arr, zarr.Array)
        assert arr.shape == data.shape

    def test_without_dataset_name(self, zarr_direct_path):
        path, data = zarr_direct_path
        arr = _open_zarr_array(path, "")
        assert isinstance(arr, zarr.Array)
        assert arr.shape == data.shape

    def test_type_error_when_group_returned(self, tmp_path):
        """Navigating to a sub-group (not an array) raises TypeError."""
        store_path = str(tmp_path / "grouped.zarr")
        z = zarr.open_group(store_path, mode="w")
        z.require_group("subgroup")
        with pytest.raises(TypeError, match="Expected zarr.Array"):
            _open_zarr_array(store_path, "subgroup")


class TestZarrReader:
    def test_shape(self, zarr_group_path):
        path, data = zarr_group_path
        reader = ZarrReader(path, dataset="labels")
        assert reader.shape == data.shape

    def test_dtype(self, zarr_group_path):
        path, data = zarr_group_path
        reader = ZarrReader(path, dataset="labels")
        assert reader.dtype == data.dtype

    def test_resolution(self, zarr_group_path):
        path, _ = zarr_group_path
        res = (10.0, 4.0, 4.0)
        reader = ZarrReader(path, dataset="labels", resolution=res)
        assert reader.resolution == res

    def test_read_chunk_full(self, zarr_group_path):
        path, data = zarr_group_path
        reader = ZarrReader(path, dataset="labels")
        chunk = reader.read_chunk((0, 0, 0), (3, 3, 3))
        np.testing.assert_array_equal(chunk, data)

    def test_read_chunk_partial(self, zarr_group_path):
        path, data = zarr_group_path
        reader = ZarrReader(path, dataset="labels")
        chunk = reader.read_chunk((1, 0, 0), (1, 3, 3))
        np.testing.assert_array_equal(chunk, data[1:2, :, :])

    def test_without_dataset_no_group(self, zarr_direct_path):
        path, data = zarr_direct_path
        reader = ZarrReader(path, dataset="")
        assert reader.shape == data.shape
        chunk = reader.read_chunk((0, 0, 0), (2, 2, 2))
        np.testing.assert_array_equal(chunk, data)
