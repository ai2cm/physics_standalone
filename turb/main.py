#!/usr/bin/env python3

import os
import sys
import numpy as np

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
#SELECT_SP = {"tile": 2, "savepoint": "satmedmfvdif-in-000000"}


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
        d[var] = data
    return d


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(ser.OpenModeKind.Read, "./data", "Generator_rank" + str(tile))

    savepoints = serializer.savepoint_list()

    for sp in savepoints:

        if SELECT_SP is not None:
            if sp.name != SELECT_SP["savepoint"]:
                continue

        if "-in-" in sp.name:

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            sp_in = sp
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp_in)

            # read serialized output data
            sp_out = serializer.savepoint[sp.name.replace("-in-", "-out-")]
            out_data = data_dict_from_var_list(OUT_VARS, serializer, sp_out)


