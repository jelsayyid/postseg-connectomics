"""Tests for PrecomputedReader (cloud-volume dependency mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from connectomics_pipeline.io.precomputed_reader import PrecomputedReader


class TestPrecomputedReaderImportError:
    def test_raises_when_cloudvolume_missing(self):
        with patch("connectomics_pipeline.io.precomputed_reader._HAS_CLOUDVOLUME", False):
            with pytest.raises(ImportError, match="cloud-volume"):
                PrecomputedReader("precomputed://gs://bucket/path")


class TestPrecomputedReaderMocked:
    """Tests using a fully mocked CloudVolume instance.

    Since cloud-volume is not installed, CloudVolume doesn't exist as a module
    attribute. We inject it via sys.modules patching so that the import inside
    __init__ resolves to our mock.
    """

    def _make_reader(self, zyx_shape=(10, 20, 30), resolution=(30.0, 8.0, 8.0)):
        import sys
        import types

        z, y, x = zyx_shape
        mock_vol = MagicMock()
        mock_vol.shape = (x, y, z, 1)
        mock_vol.dtype = np.dtype("uint32")

        mock_cv_module = types.ModuleType("cloudvolume")
        mock_cv_class = MagicMock(return_value=mock_vol)
        mock_cv_module.CloudVolume = mock_cv_class

        # Inject the fake cloudvolume module and mark the reader as available
        with patch.dict(sys.modules, {"cloudvolume": mock_cv_module}):
            with patch("connectomics_pipeline.io.precomputed_reader._HAS_CLOUDVOLUME", True):
                with patch(
                    "connectomics_pipeline.io.precomputed_reader.CloudVolume",
                    mock_cv_class,
                    create=True,
                ):
                    reader = PrecomputedReader(
                        "precomputed://gs://bucket/path", resolution=resolution
                    )
        reader._vol = mock_vol
        return reader, mock_vol

    def test_shape_transposed_zyx(self):
        """shape property converts CloudVolume (x,y,z) to pipeline (z,y,x)."""
        reader, _ = self._make_reader(zyx_shape=(10, 20, 30))
        assert reader.shape == (10, 20, 30)

    def test_dtype(self):
        reader, _ = self._make_reader()
        assert reader.dtype == np.dtype("uint32")

    def test_resolution(self):
        reader, _ = self._make_reader(resolution=(15.0, 4.0, 4.0))
        assert reader.resolution == (15.0, 4.0, 4.0)

    def test_read_chunk_shape_and_transpose(self):
        """read_chunk returns (z,y,x) shaped array after squeeze+transpose."""
        reader, mock_vol = self._make_reader(zyx_shape=(10, 20, 30))
        # CloudVolume returns (dx, dy, dz, 1)
        fake_data = np.arange(2 * 4 * 6, dtype=np.uint32).reshape(6, 4, 2, 1)
        mock_vol.__getitem__ = MagicMock(return_value=fake_data)
        chunk = reader.read_chunk((0, 0, 0), (2, 4, 6))
        # After squeeze: (6,4,2); after transpose(2,1,0): (2,4,6) = (dz,dy,dx)
        assert chunk.shape == (2, 4, 6)
