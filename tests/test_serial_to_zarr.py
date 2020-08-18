#!/usr/bin/env python3

"""
Python script that reads in serialbox data and compares with the converted zarr
storage object.
"""

import os
import sys
import numpy as np
import xarray as xr
import argparse

SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

IN_VARS = ["ix", "im", "km", "ntrac", "ntcw", "ntiw", "u1", "v1", "t1", "q1", "swh", "hlw", \
    "xmu", "garea", "psk", "rbsoil", "zorl", "u10m", "v10m", "fm", "fh", "tsea", "heat", "evap", \
    "stress", "spd1", "prsi", "del", "prsl", "prslk", "phii", "phil", "delt", "dspheat", "kinver", \
    "xkzm_s", "dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "xkzm_m", \
    "xkzm_h", "hpbl"]

OUT_VARS = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]


SELECT_SP = None

parser = argparse.ArgumentParser()
parser.add_argument("serial_data", help="Path to serialized data directory")
parser.add_argument("zarr_data", help="Path to converted zarr file directory")

args = parser.parse_args()


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
        d[var] = data
    return d


def compare_data(savepoint_data, xr_dataset):

    for var, data in savepoint_data.items():
        np.testing.assert_allclose(data, xr_dataset[var].values)


converted_in = xr.open_zarr(os.path.join(args.zarr_data, "phys_in.zarr"), mask_and_scale=False)
converted_out = xr.open_zarr(os.path.join(args.zarr_data, "phys_out.zarr"), mask_and_scale=False)


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(ser.OpenModeKind.Read, args.serial_data, "Generator_rank" + str(tile))

    savepoints = serializer.savepoint_list()

    for i, sp in enumerate(savepoints):

        spt_idx = i // 2

        if SELECT_SP is not None:
            if sp.name != SELECT_SP["savepoint"]:
                continue

        if "-in-" in sp.name:

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            sp_in = sp
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp_in)
            compare_data(in_data, converted_in.isel(savepoint=spt_idx, rank=tile))

            # read serialized output data
            sp_out = serializer.savepoint[sp.name.replace("-in-", "-out-")]
            out_data = data_dict_from_var_list(OUT_VARS, serializer, sp_out)
            compare_data(out_data, converted_out.isel(savepoint=spt_idx, rank=tile))
