#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit

import gt4py as gt
import numpy as np
from gt4py import gtscript

DT_F = gtscript.Field[np.float64]
DT_I = gtscript.Field[np.int32]

OUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy",
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx",
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf",
            "smcwlt2", "smcref2", "wet1"]

IN_VARS = ["ps", "t1", "q1", "soiltyp", "vegtype", "sigmaf", \
            "sfcemis", "dlwflx", "dswsfc", "snet", "tg3", "cm", \
            "ch", "prsl1", "prslki", "zf", "land", "wind", \
            "slopetyp", "shdmin", "shdmax", "snoalb", "sfalb", "flag_iter", "flag_guess", \
            "bexppert", "xlaipert", "vegfpert", "pertvegf", \
            "weasd", "snwdph", "tskin", "tprcp", "srflag", "smc0", "smc1", "smc2", "smc3", \
            "stc0", "stc1", "stc2", "stc3", "slc0", "slc1", "slc2", "slc3", "canopy", \
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx", \
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf", \
            "smcwlt2", "smcref2", "wet1", "tbpvs"]  

SCALAR_VARS = ["c1xpvs", "c2xpvs", "delt", "lheatstrg", "ivegsrc"]     

def numpy_to_gt4py_storage(arr, backend):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, 1))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    return gt.storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def gt4py_to_numpy_storage(arr, backend):
    """convert gt4py storage to numpy storage"""
    if backend == "gtcuda":
        arr.synchronize()
    data = arr.view(np.ndarray)
    return np.reshape(data, (data.shape[0]))

def run(in_dict, in_dict2, backend):
    """run function"""

    # special handling of two dimensional arrays
    stc = in_dict.pop("stc")
    in_dict["stc0"] = stc[:, 0]
    in_dict["stc1"] = stc[:, 1]
    in_dict["stc2"] = stc[:, 2]
    in_dict["stc3"] = stc[:, 3]
    smc = in_dict.pop("smc")
    in_dict["smc0"] = smc[:, 0]
    in_dict["smc1"] = smc[:, 1]
    in_dict["smc2"] = smc[:, 2]
    in_dict["smc3"] = smc[:, 3]
    slc = in_dict.pop("slc")
    in_dict["slc0"] = slc[:, 0]
    in_dict["slc1"] = slc[:, 1]
    in_dict["slc2"] = slc[:, 2]
    in_dict["slc3"] = slc[:, 3]

    # setup storages
    scalar_dict = {k: in_dict[k] for k in SCALAR_VARS}
    out_dict = {k: numpy_to_gt4py_storage(in_dict[k].copy(), backend=backend) for k in OUT_VARS}
    in_dict = {k: numpy_to_gt4py_storage(in_dict[k], backend=backend) for k in IN_VARS}

    # compile stencil
    sfc_drv = gtscript.stencil(definition=sfc_drv_defs, backend=backend, externals={})

    # set timer
    tic = timeit.default_timer()

    # call sea-ice parametrization
    sfc_drv(**in_dict, **out_dict, **scalar_dict)

    # set timer
    toc = timeit.default_timer()

    # calculate elapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {k: gt4py_to_numpy_storage(out_dict[k], backend=backend) for k in OUT_VARS}

    # special handling of of two dimensional arrays
    stc[:, 0] = out_dict.pop("stc0")[:]
    stc[:, 1] = out_dict.pop("stc1")[:]
    stc[:, 2] = out_dict.pop("stc2")[:]
    stc[:, 3] = out_dict.pop("stc3")[:]
    out_dict["stc"] = stc
    smc[:, 0] = out_dict.pop("smc0")[:]
    smc[:, 1] = out_dict.pop("smc1")[:]
    smc[:, 2] = out_dict.pop("smc2")[:]
    smc[:, 3] = out_dict.pop("smc3")[:]
    out_dict["smc"] = smc
    slc[:, 0] = out_dict.pop("slc0")[:]
    slc[:, 1] = out_dict.pop("slc1")[:]
    slc[:, 2] = out_dict.pop("slc2")[:]
    slc[:, 3] = out_dict.pop("slc3")[:]
    out_dict["slc"] = slc

    return out_dict

def sfc_drv_defs(
    ps: DT_F, t1: DT_F, q1: DT_F, soiltyp: DT_I, vegtype: DT_I, sigmaf: DT_F,
    sfcemis: DT_F, dlwflx: DT_F, dswsfc: DT_F, snet: DT_F, tg3: DT_F, cm: DT_F, ch: DT_F,
    prsl1: DT_F, prslki: DT_F, zf: DT_F, land: DT_I, wind: DT_F, slopetyp: DT_I,
    shdmin: DT_F, shdmax: DT_F, snoalb: DT_F, sfalb: DT_F, flag_iter: DT_I, flag_guess: DT_I,
    isot: DT_F, bexppert: DT_F, xlaipert: DT_F, vegfpert: DT_F, pertvegf: DT_F,
    weasd: DT_F, snwdph: DT_F, tskin: DT_F, tprcp: DT_F, srflag: DT_F, smc0: DT_F, smc1: DT_F, smc2: DT_F, smc3: DT_F, 
    stc0: DT_F, stc1: DT_F, stc2: DT_F, stc3: DT_F, slc0: DT_F, slc1: DT_F, slc2: DT_F, slc3: DT_F,
    canopy: DT_F, trans: DT_F, tsurf: DT_F, zorl: DT_F,
    sncovr1: DT_F, qsurf: DT_F, gflux: DT_F, drain: DT_F, evap: DT_F, hflx: DT_F, ep: DT_F, runoff: DT_F,
    cmm: DT_F, chh: DT_F, evbs: DT_F, evcw: DT_F, sbsno: DT_F, snowc: DT_F, stm: DT_F, snohf: DT_F,
    smcwlt2: DT_F, smcref2: DT_F, wet1: DT_F
):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
    
        return