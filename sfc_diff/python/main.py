#!/usr/bin/env python3

import os
import sys
import numpy as np
import sfc_diff as sd

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser


IN_VARS = [
    "tgrs_sfc",
    "qgrs_sfc",
    "zlvl",
    "wind",
    "prsl_sfc",
    "work3",
    "sigmaf",
    "vegtype",
    "shdmax",
    "ivegsrc",
    "z01d",
    "zt1d",
    "flag_iter",
    "redrag",
    "u10m",
    "v10m",
    "sfc_z0_type",
    "wet",
    "dry",
    "icy",
    "tsfc3",
    "tsurf3",
    "snowd3",
    "zorl3",
    "uustar3",
    "cd3",
    "cdq3",
    "rb3",
    "stress3",
    "ffmm3",
    "ffhh3",
    "fm103",
    "fh23",
]

# IN_VARS2 = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]
OUT_VARS = [
    "zorl3",
    "uustar3",
    "cd3",
    "cdq3",
    "rb3",
    "stress3",
    "ffmm3",
    "ffhh3",
    "fm103",
    "fh23",
]

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
        ser.OpenModeKind.Read, "./sfc_diff/data", "Generator_rank" + str(tile)
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

        if sp.name.startswith("SfcDiff2-In"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp)

            z0rl3, uustar3, cd3, cdq3, rb3, stress3, ffmm3, ffhh3, fm103, fh23 = sd.sfc_diff(144, 
                                                                            in_data["tgrs_sfc"], 
                                                                            in_data["qgrs_sfc"],
                                                                            in_data["zlvl"],
                                                                            in_data["wind"],
                                                                            in_data["prsl_sfc"],
                                                                            in_data["work3"],
                                                                            in_data["sigmaf"],
                                                                            in_data["vegtype"],
                                                                            in_data["shdmax"],
                                                                            in_data["ivegsrc"],
                                                                            in_data["z01d"],
                                                                            in_data["zt1d"],
                                                                            in_data["flag_iter"],
                                                                            in_data["redrag"],
                                                                            in_data["u10m"],
                                                                            in_data["v10m"],
                                                                            in_data["sfc_z0_type"],
                                                                            in_data["wet"],
                                                                            in_data["dry"],
                                                                            in_data["icy"],
                                                                            in_data["tsfc3"],
                                                                            in_data["tsurf3"],
                                                                            in_data["snowd3"],
                                                                            in_data["zorl3"],
                                                                            in_data["uustar3"],
                                                                            in_data["cd3"],
                                                                            in_data["cdq3"],
                                                                            in_data["rb3"],
                                                                            in_data["stress3"],
                                                                            in_data["ffmm3"],
                                                                            in_data["ffhh3"],
                                                                            in_data["fm103"],
                                                                            in_data["fh23"],
                                                                            )

            out_data = {}
            out_data["zorl3"] = z0rl3
            out_data["uustar3"] = uustar3
            out_data["cd3"] = cd3
            out_data["cdq3"] = cdq3
            out_data["rb3"] = rb3
            out_data["stress3"] = stress3
            out_data["ffmm3"] = ffmm3
            out_data["ffhh3"] = ffhh3
            out_data["fm103"] = fm103
            out_data["fh23"] = fh23

            isready = True

        if sp.name.startswith("SfcDiff2-Out"):

            if not isready:
                raise Exception("out-of-order data encountered: " + sp.name)

            print("> validating ", f"tile-{tile}", sp)

            # read serialized output data
            ref_data = data_dict_from_var_list(OUT_VARS, serializer, sp)

            # check result
            compare_data(out_data, ref_data)

            isready = False
