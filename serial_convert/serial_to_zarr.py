import zarr
import sys
import yaml
import dask
import argparse
import os
import dask.array as da
import netCDF4 as ncf
import numpy as np
from typing import Iterable, Mapping, Optional, MutableMapping, Sequence
from numpy import ndarray
from dask.diagnostics import ProgressBar

SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

DEFAULT_FILLVALS = {np.dtype(key): val for key, val in ncf.default_fillvals.items()}
DEFAULT_FILLVALS[np.dtype("bool")] = DEFAULT_FILLVALS[np.dtype("int64")]


def _filter_savepoints(
    savepoints: Iterable[ser.Savepoint],
    filter_key: str
) -> Iterable[ser.Savepoint]:
    new_savepoints = [
        savepoint 
        for savepoint in savepoints 
        if filter_key in str(savepoint)
    ]

    return new_savepoints


def _xarray_compatible_dims(
    attrs: MutableMapping, 
    prepend_dims: Sequence[str] = ("savepoint", "rank")
) -> Mapping:
    dims = attrs.pop("dimensions", [])
    
    # length 1 dimension (scalars) aren't specified in CCPP, add here
    if not dims:
        dims = ["scalar"]
    dims = list(prepend_dims) + dims
    
    # attribute necessary for xarray compatibility
    attrs["_ARRAY_DIMENSIONS"] = dims

    return attrs


def _array_to_dask_array(source_array: ndarray, name: str) -> da.Array:

    # Preserve chunking of zarr if it is one
    if hasattr(source_array, "chunks"):
        chunks = source_array.chunks
    else:
        chunks = None

    return da.from_array(source_array, chunks=chunks, name=name)

def _lazy_serialized_read(
    varname: str,
    output: np.ndarray,
    savepoints: Iterable[ser.Savepoint],
    serializers: Iterable[ser.Serializer]
) -> da.Array:
    """Compile a dask array from savepoint read operations using delayed objects"""

    tmp_dask_out = _array_to_dask_array(output, varname)
    by_savepoint = []
    
    for i, savepoint in enumerate(savepoints):
        
        by_rank = []
        
        for rank, serializer in enumerate(serializers):
            source = dask.delayed(serializer.read)(varname, savepoint)
            dest = tmp_dask_out[i, rank]
            delayed_da = da.from_delayed(source, dest.shape, dtype=dest.dtype)
            by_rank.append(delayed_da)
        
        by_rank = da.stack(by_rank, axis=0)
        by_savepoint.append(by_rank)

    full_var_output = da.stack(by_savepoint, axis=0)

    return full_var_output


class SerializedPhysicsConverter():

    """
    Load serialized data and convert to another format.
    """

    RANKS = 6
    TEMPLATE = "Generator_rank{:d}"
    
    def __init__(
        self,
        prefix: str,
        savepoint_filter: Optional[str] = None,
        var_attrs: Optional[Mapping[str, Mapping]] = None
    ):
        """
        Args:
            prefix: path to SerialBox savepoint data
            savepoint_filter: filter to savepoints whose string representation includes this key
            var_attrs:  mapping of variable names to their metadata.  Stored as attributes in the
                        new format
        """

        self.serializers = [
            ser.Serializer(ser.OpenModeKind.Read, prefix, self.TEMPLATE.format(i)) 
            for i in range(self.RANKS)
        ]
        
        init_serializer = self.serializers[0]
        self.savepoints = init_serializer.savepoint_list()
        if savepoint_filter is not None:
            self.savepoints = _filter_savepoints(self.savepoints, savepoint_filter)

        self.var_names = init_serializer.fields_at_savepoint(self.savepoints[0])
        self.var_attrs = {} if var_attrs is None else var_attrs
        self.num_savepoints = len(self.savepoints)

        self.var_info = self._process_var_metadata(init_serializer)

    def _process_var_metadata(self, init_serializer: ser.Serializer) -> dict:
        """Add attributes if given and determine dataset properties for conversion"""

        var_info = {}
        for var in self.var_names:
            data = init_serializer.read(var, self.savepoints[0])

            # TODO: Should store constants in single element array?
            var_info[var] = {
                "dataset_kwargs": {
                    "shape": (self.num_savepoints, self.RANKS, *data.shape),
                    "chunks": (1, 1, *data.shape),
                    "dtype": str(data.dtype),
                },
                "_FillValue": DEFAULT_FILLVALS[data.dtype]
            }

            if var in self.var_attrs:
                attrs = _xarray_compatible_dims(dict(self.var_attrs[var]))
                var_info[var].update(attrs)
            
        
        return var_info

    def _ser_to_backend(self, backend: Mapping[str, ndarray]):
        """Use dask to translate serialized data to ndarray-access-like backend"""

        da_by_var = []
        for var in self.var_names:
            
            output = backend[var]
            var_to_store = _lazy_serialized_read(
                var, output, self.savepoints, self.serializers
            )
            da_by_var.append(var_to_store)

        output_arrays = [backend[var] for var in self.var_names]
        with ProgressBar():
            da.store(da_by_var, output_arrays)

    def _init_zarr(self, path: str):

        """Initialize zarr storage with attributes for all variables"""

        out = zarr.open_group(path, mode="w-")

        for var in self.var_names:
            info = dict(self.var_info[var])
            ds_kwargs = info.pop("dataset_kwargs", {})
            var_group = out.create_dataset(var, **ds_kwargs)
            var_group.attrs.update(info)

        return out

    def save_zarr(self, path: str) -> zarr.Group:
        """
        Save serialized data to a xarray-compatible zarr dataset

        Args:
            path: Path for zarr store to be written to
        """

        out_zarr = self._init_zarr(path)
        self._ser_to_backend(out_zarr)

        return out_zarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("serial_data_dir")
    parser.add_argument("metadata_file")
    parser.add_argument("outdir")

    args = parser.parse_args()
    prefix = args.serial_data_dir
    with open(args.metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    outdir = args.outdir
    save = SerializedPhysicsConverter(prefix, savepoint_filter="-in-", var_attrs=metadata)
    save.save_zarr(os.path.join(outdir, "phys_in.zarr"))
    save = SerializedPhysicsConverter(prefix, savepoint_filter="-out-", var_attrs=metadata)
    save.save_zarr(os.path.join(outdir, "phys_out.zarr"))

