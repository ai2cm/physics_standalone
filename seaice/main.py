#!/usr/bin/env python3

import os
import sys
import numpy as np
import sea_ice as si

SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
#SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

IN_VARS = ["im", "km", "ps", "t1", "q1", "delt", "sfcemis", "dlwflx", \
           "sfcnsw", "sfcdsw", "srflag", "cm", "ch", "prsl1", "prslki", \
           "islimsk", "wind", "flag_iter", "lprnt", "ipr", "cimin", \
           "hice", "fice", "tice", "weasd", "tskin", "tprcp", "stc", \
           "ep", "snwdph", "qsurf", "snowmt", "gflux", "cmm", "chh", \
           "evap", "hflx"]

OUT_VARS = ["hice", "fice", "tice", "weasd", "tskin", "tprcp", "stc", \
            "ep", "snwdph", "qsurf", "snowmt", "gflux", "cmm", "chh", \
            "evap", "hflx"]


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        d[var] = serializer.read(var, savepoint)
    return d

def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(ref_data.keys()), \
        "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), \
            "Data does not match for field " + key

for tile in range(6):

    serializer = ser.Serializer(ser.OpenModeKind.Read, "./data", "Generator_rank" + str(tile))

    savepoints = serializer.savepoint_list()

    isready = False
    for sp in savepoints:

        if sp.name.startswith("sfc_sice-in"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp)
            
            # TODO: remove once we validate
            # attach meta-info for debugging purposes
            ser_inside = ser.Serializer(ser.OpenModeKind.Read, "./data", "Serialized_rank" + str(tile))
            sp_inside = ser_inside.savepoint[sp.name.replace("-in-", "-inside-")]
            in_data["serializer"] = ser_inside
            in_data["savepoint"] = sp_inside

            # run Python version
            out_data = si.run(in_data)
            isready = True

        if sp.name.startswith("sfc_sice-out"):

            if not isready:
                raise Exception("out-of-order data encountered: " + sp.name)

            print("> validating ", f"tile-{tile}", sp)

            # read serialized output data
            ref_data = data_dict_from_var_list(OUT_VARS, serializer, sp)

            # check result
            compare_data(out_data, ref_data)

            isready = False
            del in_data, out_data, ref_data
