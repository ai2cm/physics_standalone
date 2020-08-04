import zarr
import sys
import yaml
from typing import Iterable, Mapping
from numpy import ndarray

SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser


def _filter_savepoints(savepoints: Iterable[ser.Savepoint], filter_key: str):
    new_savepoints = [
        savepoint 
        for savepoint in savepoints 
        if filter_key in str(savepoint)
    ]

    return new_savepoints


class SerializedPhysicsConverter():

    RANKS = 6
    TEMPLATE = "Generator_rank{:d}"
    
    def __init__(self, prefix: str, savepoint_filter: str = None, var_attrs: Mapping = None) -> None:

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
        var_info = {}
        for var in self.var_names:
            data = init_serializer.read(var, self.savepoints[0])

            # TODO: Should store constants in single element array
            var_info[var] = {
                "dataset_kwargs": {
                    "shape": (self.num_savepoints, self.RANKS, *data.shape),
                    "chunk": (1, 1, *data.shape),
                    "dtype": data.dtype,
                },
                "is_constant": bool(data.shape)
            }

            if var in self.var_attrs:
                attrs = self.var_attrs[var]
                dims = attrs.pop("dimensions")
                
                # Switch to expected attr name for xarray if not a constant
                if dims:
                    dims = ["savepoint", "rank"] + dims
                    attrs["_ARRAY_DIMENSIONS"] = dims
            
        
        return var_info

    def _ser_to_backend(self, backend: Mapping[str, ndarray]):
        
        for rank, serializer in enumerate(self.serializers):
            for var in self.var_names:
                for k, savepoint in enumerate(self.savepoints):

                    var_out = backend[var]
                    data = serializer.read(var, savepoint)
                    if data.size == 1:
                        data = data.item()
                    var_out[k, rank] = data

    def _init_zarr(self, path: str):

        out = zarr.open_group(path, mode="w")

        for var in self.var_names:
            info = self.var_info[var]
            out.create_dataset(var, **info["dataset_kwargs"])
            out.attrs

        return out

    def save_zarr(self, path: str) -> zarr.Group:

        out_zarr = self._init_zarr(path)
        self._ser_to_backend(out_zarr)

        return out_zarr

if __name__ == "__main__":

    prefix = "./turb/data"
    metadata = yaml.safe_load("./turb/parameter_metadata.yaml")
    save = SerializedPhysicsConverter(prefix, savepoint_filter="-in-", var_attrs=metadata)
    save.save_zarr("/home/user/turb_in.zarr")

