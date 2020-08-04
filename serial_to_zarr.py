import zarr
import sys
import yaml
from typing import Iterable, Mapping
from numpy import ndarray

SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser


def _filter_savepoints(savepoints: Iterable[ser.Savepoint], filter: str):
    new_savepoints = [
        savepoint 
        for savepoint in savepoints 
        if filter in str(savepoint)
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
        self.var_attrs = var_attrs
        self.num_savepoints = len(self.savepoints)

        self.var_info = self._read_var_info(init_serializer)

    def _read_var_info(self, init_serializer: ser.Serializer) -> dict:
        var_info = {}
        for var in self.var_names:
            data = init_serializer.read(var, self.savepoints[0])
            var_info[var] = {
                "shape": (self.num_savepoints, self.RANKS, *data.shape),
                "chunk": (1, 1, *data.shape),
                "dtype": data.dtype,
            }
            
        
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
            out.create_dataset(var, **info)

        return out

    def save_zarr(self, path: str) -> zarr.Group:

        out_zarr = self._init_zarr(path)
        self._ser_to_backend(out_zarr)

        return out_zarr

if __name__ == "__main__":

    prefix = "./data"
    metadata = yaml.safe_load("./parameter_metadata.yaml")
    save = SerializedPhysicsConverter(prefix, savepoint_filter="-in-", var_attrs=metadata)
    save.save_zarr("/home/user/turb_in.zarr")

