#!/usr/bin/env python3

import os
import sys
import numpy as np
from numpy.lib.npyio import save
import noah_lsm

#SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
SERIALBOX_DIR = "/usr/local/serialbox/"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

IN_VARS = ["im", "km", "ps", "t1", "q1", "soiltyp", "vegtype", "sigmaf", \
           "sfcemis", "dlwflx", "dswsfc", "snet", "delt", "tg3", "cm", \
           "ch", "prsl1", "prslki", "zf", "land", "wind", \
           "slopetyp", "shdmin", "shdmax", "snoalb", "sfalb", "flag_iter", "flag_guess", \
           "lheatstrg", "isot", "ivegsrc", "bexppert", "xlaipert", "vegfpert", "pertvegf", \
           "weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy", \
           "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx", \
           "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf", \
           "smcwlt2", "smcref2", "wet1"]

IN_VARS2 = ["zsoil_noah_ref", "canopy_ref", "q0_ref", "theta1_ref", "rho_ref", "qs1_ref", "zsoil_ref", "flag_test_ref", \
            "sldpth_ref"]

OUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy", \
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx", \
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf", \
            "smcwlt2", "smcref2", "wet1"]

SELECT_SP = None
#SELECT_SP = {"tile": 2, "savepoint": "sfc_drv-in-iter2-000000"}


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
    assert set(exp_data.keys()) == set(ref_data.keys()), \
             "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        ind = np.array(np.nonzero(~np.isclose(exp_data[key], ref_data[key], equal_nan=True)))
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i, exp_data[key][i], ref_data[key][i])
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), \
            "Data does not match for field " + key


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(ser.OpenModeKind.Read, "./data", "Generator_rank" + str(tile))

    serializer2 = ser.Serializer(ser.OpenModeKind.Read, "./dump", "Serialized_rank" + str(tile))

    savepoints = serializer.savepoint_list()

    savepoint2 = serializer2.savepoint_list()

    isready = False
    for sp in savepoints:

        if SELECT_SP is not None:
            if sp.name != SELECT_SP["savepoint"] and \
               sp.name != SELECT_SP["savepoint"].replace("-in-", "-out-"):
                continue

        if sp.name.startswith("sfc_drv-in"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data = data_dict_from_var_list(IN_VARS, serializer, sp)

            in_data_test = data_dict_from_var_list(IN_VARS2, serializer2, savepoint2[0])

            # run Python version
            out_data = noah_lsm.run(in_data, in_data_test)
            
            isready = True

        if sp.name.startswith("sfc_drv-out"):

            if not isready:
                raise Exception("out-of-order data encountered: " + sp.name)

            print("> validating ", f"tile-{tile}", sp)

            # read serialized output data
            ref_data = data_dict_from_var_list(OUT_VARS, serializer, sp)

            # check result
            compare_data(out_data, ref_data)

            isready = False

