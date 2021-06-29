#!/usr/bin/env python3

import os
import sys
import numpy as np
import get_prs_fv3

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser


IN_VARS = ["ix", "levs", "ntrac", "phii", "prsi", "tgrs", "qgrs", "del", "del_gz"]

# IN_VARS2 = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]
OUT_VARS = ["del", "del_gz"]

SELECT_SP = None
# SELECT_SP = {"tile": 2, "savepoint": "satmedmfvdif-in-iter2-000000"}


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
        d[var] = data
    return d


def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(
        ref_data.keys()
    ), "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        ind = np.array(
            np.nonzero(~np.isclose(exp_data[key], ref_data[key], equal_nan=True))
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i, exp_data[key][i], ref_data[key][i])
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), (
            "Data does not match for field " + key
        )


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(
        ser.OpenModeKind.Read, "../data", "Generator_rank" + str(tile)
    )

    # serializer_custom = ser.Serializer(
    #     ser.OpenModeKind.Read, "./dump", "Serialized_rank" + str(tile)
    # )

    savepoints = serializer.savepoint_list()

    # savepoints_custom = serializer_custom.savepoint_list()

    isready = False
    for sp in savepoints:

        if SELECT_SP is not None:
            if sp.name != SELECT_SP["savepoint"] and sp.name != SELECT_SP[
                "savepoint"
            ].replace("-in-", "-out-"):
                continue

        if sp.name.startswith("prsfv3-in"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp)

            # in_data_custom = data_dict_from_var_list(IN_VARS2, serializer_custom, savepoints_custom[0])

            # run Python version
            out_data = get_prs_fv3.run(in_data)

            isready = True

        if sp.name.startswith("prsfv3-out"):

            if not isready:
                raise Exception("out-of-order data encountered: " + sp.name)

            print("> validating ", f"tile-{tile}", sp)

            # read serialized output data
            ref_data = data_dict_from_var_list(OUT_VARS, serializer, sp)

            # check result
            compare_data(out_data, ref_data)

            isready = False
