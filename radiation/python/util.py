import gt4py
import numpy as np
import xarray as xr
from config import *

BACKEND = "gtc:gt:cpu_ifirst"
default_origin = (0, 0, 0)

# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        if BACKEND == "gtcuda":
            data.synchronize()

        if var != "tauaer" and var != "taucld":
            np_dict[var] = np.squeeze(data.view(np.ndarray))
        else:
            tmp = np.squeeze(data.view(np.ndarray))
            np_dict[var] = np.transpose(tmp, (0, 2, 1))

    return np_dict


def compare_data(data, ref_data, explicit=True, blocking=True):

    wrong = []
    flag = True

    for var in data:

        if not np.allclose(
            data[var], ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
        ):

            wrong.append(var)
            flag = False

        else:

            if explicit:
                print(f"Successfully validated {var}!")

    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag:
            print(f"Output data does not match reference data for field {wrong}!")


def create_storage_from_array(
    var, backend, shape, dtype, default_origin=default_origin
):
    out = gt4py.storage.from_array(
        var, backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def create_storage_zeros(backend, shape, dtype):
    out = gt4py.storage.zeros(
        backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def create_storage_ones(backend, shape, dtype):
    out = gt4py.storage.ones(
        backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def loadlookupdata(name):
    """
    Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables
    """
    ds = xr.open_dataset("../lookupdata/radlw_" + name + "_data.nc")

    lookupdict = dict()
    lookupdict_gt4py = dict()

    for var in ds.data_vars.keys():
        # print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :], (npts, 1, nlp1, 1)
            )
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1)
            )
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :, :], (npts, 1, nlp1, 1, 1, 1)
            )

        lookupdict_gt4py[var] = create_storage_from_array(
            lookupdict[var], BACKEND, shape_nlp1, (DTYPE_FLT, ds[var].shape)
        )

    ds2 = xr.open_dataset("../lookupdata/radlw_ref_data.nc")
    tmp = np.tile(ds2["chi_mls"].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1))

    lookupdict_gt4py["chi_mls"] = create_storage_from_array(
        tmp, BACKEND, shape_nlp1, (DTYPE_FLT, ds2["chi_mls"].shape)
    )

    return lookupdict_gt4py
