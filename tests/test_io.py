"""Tests for the IO module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from connectomics_pipeline.io.numpy_reader import NumpyReader


class TestNumpyReader:
    def test_shape(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        assert reader.shape == synthetic_volume.shape

    def test_dtype(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        assert reader.dtype == synthetic_volume.dtype

    def test_resolution(self, resolution):
        data = np.zeros((10, 10, 10), dtype=np.uint32)
        reader = NumpyReader(data, resolution=resolution)
        assert reader.resolution == resolution

    def test_read_chunk(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        chunk = reader.read_chunk((0, 0, 0), (10, 10, 10))
        assert chunk.shape == (10, 10, 10)
        np.testing.assert_array_equal(chunk, synthetic_volume[:10, :10, :10])

    def test_read_chunk_returns_copy(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        chunk = reader.read_chunk((0, 0, 0), (10, 10, 10))
        chunk[:] = 999
        chunk2 = reader.read_chunk((0, 0, 0), (10, 10, 10))
        assert not np.all(chunk2 == 999)

    def test_chunk_iterator(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        chunks = list(reader.chunk_iterator((16, 32, 32), (0, 0, 0)))
        assert len(chunks) > 0
        for offset, chunk in chunks:
            assert len(offset) == 3
            assert chunk.ndim == 3

    def test_chunk_iterator_with_overlap(self, synthetic_volume, resolution):
        reader = NumpyReader(synthetic_volume, resolution=resolution)
        chunks_no_overlap = list(reader.chunk_iterator((16, 32, 32), (0, 0, 0)))
        chunks_with_overlap = list(reader.chunk_iterator((16, 32, 32), (4, 8, 8)))
        # With overlap, there should be more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="3D"):
            NumpyReader(np.zeros((10, 10)))


class TestHDF5Reader:
    def test_read_write_roundtrip(self, synthetic_volume, resolution):
        from connectomics_pipeline.io.hdf5_reader import HDF5Reader

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=synthetic_volume)

        reader = HDF5Reader(path, dataset="labels", resolution=resolution)
        assert reader.shape == synthetic_volume.shape
        assert reader.dtype == synthetic_volume.dtype

        chunk = reader.read_chunk((0, 0, 0), (10, 10, 10))
        np.testing.assert_array_equal(chunk, synthetic_volume[:10, :10, :10])

        Path(path).unlink()
