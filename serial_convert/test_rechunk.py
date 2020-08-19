import pytest
import numpy as np
import xarray as xr
from rechunk import get_target_chunk, rechunk_dataset

# tests for get_target_chunk
# create ndarray with 1024 * 1024 base (1 MB)
# test


@pytest.fixture
def data(scope="module"):
    # defined to be 1 MB per last two dims
    return np.random.randn(5, 10, 1024, 128).astype(np.float)


def test_get_target_chunk_leading_ones(data):

    # 10 mb
    chunks = get_target_chunk(data, 10)
    assert chunks == [1, 10, 1024, 128]


def test_get_target_chunk_entire_array_chunk(data):
    chunks = get_target_chunk(data, 50)
    assert chunks == list(data.shape)


def test_get_target_chunk_partial_dim(data):
    chunks = get_target_chunk(data, 2)
    assert chunks == [1, 2, 1024, 128]


def test_get_target_chunk_empty():
    dat = np.array([]).astype(np.float)
    assert get_target_chunk(dat, 1) == [0]


@pytest.fixture
def zarr_path(tmpdir, data):
    da = xr.DataArray(data=data, dims=("time", "lev", "x", "y"))
    ds = da.to_dataset(name="var1")
    ds["var2"] = da.copy()

    zarr_path = str(tmpdir.join("orig.zarr"))
    ds.to_zarr(zarr_path)
    return zarr_path


def test_rechunk_zarr_dataset(tmpdir, zarr_path):
    rechunked = str(tmpdir.join("rechunked.zarr"))
    rechunk_dataset(zarr_path, rechunked, 5)

    orig = xr.open_zarr(zarr_path)
    new = xr.open_zarr(rechunked)
    xr.testing.assert_allclose(orig, new)
