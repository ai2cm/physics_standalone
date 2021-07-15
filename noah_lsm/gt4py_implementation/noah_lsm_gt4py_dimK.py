#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit

import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *
from stencils_dimK import *

DT_F2 = gtscript.Field[np.float64]
DT_I2 = gtscript.Field[np.int32]
DT_F = gtscript.Field[gtscript.IJ, np.float64]
DT_I = gtscript.Field[gtscript.IJ, np.int32]

BACKEND = "gtx86"

INOUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "canopy",
              "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx",
              "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf",
              "smcwlt2", "smcref2", "wet1"]
INOUT_MULTI_VARS = ["smc", "stc", "slc"]

IN_VARS = ["ps", "t1", "q1", "soiltyp", "vegtype", "sigmaf",
           "sfcemis", "dlwflx", "dswsfc", "snet", "tg3", "cm",
           "ch", "prsl1", "prslki", "zf", "land", "wind",
           "slopetyp", "shdmin", "shdmax", "snoalb", "sfalb", "flag_iter", "flag_guess",
           "bexppert", "xlaipert", "vegfpert", "pertvegf"]

SCALAR_VARS = ["km", "delt", "lheatstrg", "ivegsrc"]

SOIL_VARS = [bb, satdk, satdw, f11, satpsi,
             qtz, drysmc, maxsmc, refsmc, wltsmc]
VEG_VARS = [nroot_data, snupx, rsmtbl, rgltbl, hstbl, lai_data]

SOIL_VARS_NAMES = ["bexp", "dksat", "dwsat", "f1", "psisat",
                   "quartz", "smcdry", "smcmax", "smcref", "smcwlt"]
VEG_VARS_NAMES = ["nroot", "snup", "rsmin", "rgl", "hs", "xlai"]

TEMP_VARS_1D = ["zsoil_root", "q0", "cmc", "th2", "rho", "qs1", # prepare_sflx
                "prcp", "dqsdt2", "snowh", "sneqv", "chx", "cmx", "z0",
                "shdfac", "kdt", "frzx", "sndens", "prcp1",
                "sncovr", "df1", "ssoil", "t2v", "fdown", "cpx1",
                "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old",
                "canopy_old", "etp", "t24", "rch", "epsca", "rr", "flx2",
                "rc", "pc", "rcs", "rct", "rcq", "rcsoil",  # canres
                "edir1", "ec1", "sgx", "edir", "ec",  # nopac / snopac
                "ett1", "eta1", "ett", "eta", "denom", "sicemax",
                "dd", "dice", "pcpdrp", "rhsct", "drip", "runoff1", "runoff2",
                "runoff3", "dew", "yy", "zz1", "beta", "flx1", "flx2", "flx3",
                "esnow", "csoil_loc", "soilm", "snomlt", "tsea", "sneqv_new"
                ]

TEMP_VARS_1D_INT = ["count", "snowng", "ice"]


TEMP_VARS_2D = ["sldpth", "rtdis", "smc_old", "stc_old", "slc_old",
                "et1", "et", "rhstt", "rhsts", "ai", "bi", "ci", "dsmdz", "ddz", "gx",
                "wdf", "wcnd", "sice", "p", "delta", "stsoil", "hcpct", "tbk", "df1k", "dtsdz", "ddz"
                ]

PREPARE_VARS = ["zsoil", "km", "zsoil_root", "q0", "cmc", "th2", "rho", "qs1", "ice",
                "prcp", "dqsdt2", "snowh", "sneqv", "chx", "cmx", "z0",
                "shdfac", "kdt", "frzx", "sndens", "snowng", "prcp1",
                "sncovr", "df1", "ssoil", "t2v", "fdown", "cpx1",
                "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old",
                "canopy_old", "etp", "t24", "rch", "epsca", "rr", "flx2", "tsea",
                "sldpth", "rtdis", "smc_old", "stc_old", "slc_old",
                "cmm", "chh", "tsurf", "t1", "q1", "vegtype", "sigmaf", "sfcemis", "dlwflx", "snet", "cm",
                "ch", "prsl1", "prslki", "land", "wind", "snoalb", "sfalb",
                "flag_iter", "flag_guess", "bexppert", "xlaipert", "fpvs",
                "weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc",
                "canopy", "zorl", "delt", "ivegsrc", "bexp", "dksat", "rsmin",
                "quartz", "smcdry", "smcmax", "smcref", "smcwlt", "nroot", "snup", "xlai"]


CANRES_VARS = ["dswsfc", "nroot", "count", "chx", "q2", "q2sat", "dqsdt2", "sfctmp",
               "cpx1", "sfcprs", "sfcemis", "sh2o", "smcwlt", "smcref", "zsoil", "rsmin",
               "rgl", "hs", "xlai", "flag_iter", "land", "zsoil_root", "shdfac",
               "rc", "pc", "rcs", "rct", "rcq", "rcsoil"]


NOPAC_VARS1 = ["etp", "nroot", "smcmax", "smcwlt", "smcref", "smcdry", "dt", "shdfac",
               "cmc", "sh2o", "flag_iter", "land", "sneqv", "count",
               "sgx", "gx", "edir1", "ec1", "edir", "ec"]  # output

NOPAC_VARS2 = ["shdfac", "etp", "flag_iter", "land", "sneqv", "count", "rtdis", "sgx",
               "denom", "gx"]  # output

NOPAC_VARS3 = ["shdfac", "etp", "flag_iter", "land", "sneqv", "count", "gx", "cmc", "pc",
               "denom", "edir1", "ec1",
               "et1", "ett1", "eta1", "et", "ett", "eta"]  # output

NOPAC_VARS4 = ["smcmax", "dt", "smcwlt", "prcp", "prcp1", "zsoil", "shdfac", "ec1",
               "cmc", "sh2o", "smc", "flag_iter", "land",
               "etp", "sneqv", "sicemax", "dd", "dice", "pcpdrp", "rhsct", "drip", "sice", "dew"]


NOPAC_VARS5 = ["smcmax", "etp", "dt", "bexp", "kdt", "frzx", "dksat", "dwsat", "slope",
               "zsoil", "sh2o", "flag_iter", "land", "sneqv", "sicemax", "dd", "dice", "pcpdrp", "edir1", "et1",
               "ai", "bi", "rhstt", "ci", "runoff1", "runoff2", "dsmdz", "ddz", "wdf", "wcnd", "p", "delta"]

NOPAC_VARS6 = ["dt", "smcmax", "flag_iter", "land", "sice", "zsoil", "rhsct", "ci", "sneqv",
               "sh2o", "smc", "cmc", "runoff3"]

NOPAC_VARS7 = ["land", "flag_iter", "sneqv", "etp", "eta", "smc", "quartz", "smcmax", "ivegsrc",
               "vegtype", "shdfac", "fdown", "sfcemis", "t24", "sfctmp", "rch", "th2",
               "epsca", "rr", "zsoil", "dt", "psisat", "bexp", "ice", "stc", "hcpct", "tbk", "df1k", "dtsdz", "ddz", "csoil_loc",
               "sh2o", "stsoil", "rhsts", "ai", "bi", "ci", "p", "delta", "tbot", "yy", "zz1", "df1", "beta"]

NOPAC_VARS8 = ["land", "flag_iter", "sneqv", "zsoil", "stsoil", "p", "delta",
               "yy", "zz1", "df1", "ssoil", "tsea", "stc"]

SNOPAC_VARS1 = ["smcmax", "dt", "smcwlt", "prcp", "prcp1", "zsoil", "shdfac", "ec1",
                "cmc", "sh2o", "smc", "flag_iter", "land", "etp",
                       "sncovr", "ice", "snowng", "ffrozp", "sfctmp", "eta", "snowh",
                       "df1", "rr", "rch", "fdown", "flx2", "sfcemis", "t24", "th2", "stc",
                       "sicemax", "dd", "dice", "pcpdrp", "rhsct", "drip", "sice", "dew",
                       "tsea", "sneqv", "sneqv_new", "flx1", "flx3", "esnow", "ssoil", "snomlt"]

CLEANUP_VARS = ["land", "flag_iter", "eta", "ssoil", "edir", "ec", "ett", "esnow", "sncovr", "soilm",
        "flx1", "flx2", "flx3", "etp", "runoff1", "runoff2", "cmc", "snowh", "sneqv", "z0",
            "rho", "chx", "wind", "q1", "flag_guess", "weasd_old", "snwdph_old", "tskin_old",
            "canopy_old", "tprcp_old", "srflag_old", "smc_old", "slc_old", "stc_old", "th2",
            "tsea", "et", "ice", "runoff3", "dt", "sfcprs", "t2v", "snomlt",
            "zsoil", "smcmax", "smcwlt", "smcref", "ch"]


def numpy_to_gt4py_storage(arr, backend):
    """convert numpy storage to gt4py storage"""
    if arr.ndim > 1:
        data = np.reshape(arr, (arr.shape[0], 1,  arr.shape[1]))
        if data.dtype == "bool":
            data = data.astype(np.int32)
        return gt.storage.from_array(data, backend=backend, shape=(arr.shape[0], 1,  arr.shape[1]), default_origin=(0, 0, 0))
    else:
        data = np.reshape(arr, (arr.shape[0], 1))
        if data.dtype == "bool":
            data = data.astype(np.int32)
        return gt.storage.from_array(data, backend=backend, shape=(arr.shape[0], 1), mask=(True, True, False), default_origin=(0, 0, 0))


def gt4py_to_numpy_storage_onelayer(arr, backend):
    """convert gt4py storage to numpy storage"""
    if backend == "gtcuda":
        arr.synchronize()
    data = arr.view(np.ndarray)
    return np.reshape(data, (data.shape[0]))


def gt4py_to_numpy_storage_multilayer(arr, backend):
    """convert gt4py storage to numpy storage"""
    if backend == "gtcuda":
        arr.synchronize()
    data = arr.view(np.ndarray)
    return np.reshape(data, (data.shape[0], data.shape[2]))


def numpy_table_to_gt4py_storage(arr, index_arr, backend):
    new_arr = np.empty(index_arr.size, dtype=arr.dtype)
    for i in range(arr.size):
        new_arr[np.where(index_arr == i)] = arr[i]

    data = np.reshape(new_arr, (new_arr.shape[0], 1))
    if data.dtype == "bool" or data.dtype == "int64":
        data = data.astype(np.int32)
    return gt.storage.from_array(data, backend=backend, shape=(new_arr.shape[0], 1), mask=(True, True, False), default_origin=(0, 0, 0))


def initialize_gt4py_storage(dim1, dim2, type):
    if dim2 != 1:
        return gt.storage.zeros(backend=BACKEND, dtype=type, shape=(dim1, 1, dim2), default_origin=(0, 0, 0))
    else:
        return gt.storage.zeros(backend=BACKEND, dtype=type, shape=(dim1, 3), mask=(True, True, False), default_origin=(0, 0, 0))


def fpvs_fn(c1xpvs, c2xpvs, tbpvs, t):
    nxpvs = 7501.
    xj = np.minimum(np.maximum(c1xpvs+c2xpvs*t, 1.), nxpvs)
    jx = np.minimum(xj, nxpvs - 1.).astype(int)
    fpvs = tbpvs[jx-1]+(xj-jx)*(tbpvs[jx]-tbpvs[jx-1])

    return fpvs


def run(in_dict, in_dict2, backend):
    """run function"""

    km = in_dict["km"]
    im = in_dict["im"]

    # Fortran starts with one, python with zero
    in_dict["vegtype"] -= 1
    in_dict["soiltyp"] -= 1
    in_dict["slopetyp"] -= 1

    # prepare some constant vars
    fpvs = fpvs_fn(in_dict2["c1xpvs"], in_dict2["c2xpvs"],
                   in_dict2["tbpvs"], in_dict["t1"])
    zsoil = np.repeat([[-0.1, -0.4, -1.0, -2.0]], in_dict["im"], axis=0)
    
    # setup storages
    table_dict = {**{SOIL_VARS_NAMES[k]: numpy_table_to_gt4py_storage(
        SOIL_VARS[k], in_dict["soiltyp"], backend=backend) for k in range(len(SOIL_VARS))}, **{VEG_VARS_NAMES[k]: numpy_table_to_gt4py_storage(
            VEG_VARS[k], in_dict["vegtype"], backend=backend) for k in range(len(VEG_VARS))}, **{"slope": numpy_table_to_gt4py_storage(slope_data, in_dict["slopetyp"], backend=backend)}}

    scalar_dict = {**{k: in_dict[k] for k in SCALAR_VARS}}
    out_dict = {k: numpy_to_gt4py_storage(
        in_dict[k].copy(), backend=backend) for k in (INOUT_VARS + INOUT_MULTI_VARS)}
    in_dict = {**{k: numpy_to_gt4py_storage(in_dict[k], backend=backend) for k in IN_VARS},
               **{"fpvs": numpy_to_gt4py_storage(fpvs, backend=backend), "zsoil": numpy_to_gt4py_storage(zsoil, backend=backend)}}
    #    **{"zsoil": numpy_to_gt4py_storage(zsoil, backend=backend)}}

    # compile stencil
    # sfc_drv = gtscript.stencil(
    # definition=sfc_drv_defs, backend=backend, externals={})
    # set timer
    tic = timeit.default_timer()

    general_dict = {"zsoil": zsoil, **in_dict,
                    **scalar_dict, **table_dict, **out_dict}

    # prepare for sflx
    general_dict = {**general_dict, **{k: initialize_gt4py_storage(im, 1, np.float64) for k in TEMP_VARS_1D}, 
                    **{k: initialize_gt4py_storage(im, km, np.float64) for k in TEMP_VARS_2D},
                    **{k: initialize_gt4py_storage(im, 1, np.int32) for k in TEMP_VARS_1D_INT}}

    
    prepare_sflx(**{k: general_dict[k] for k in PREPARE_VARS})

    general_dict["cm"] = general_dict["cmx"]
    general_dict["dt"] = general_dict["delt"]
    general_dict["sfcprs"] = general_dict["prsl1"]
    general_dict["q2sat"] = general_dict["qs1"]
    general_dict["sh2o"] = general_dict["slc"]
    general_dict["sfctmp"] = general_dict["t1"]
    general_dict["q2"] = general_dict["q0"]
    general_dict["tbot"] = general_dict["tg3"]
    general_dict["ffrozp"] = general_dict["srflag"]

    canres(**{k: general_dict[k] for k in CANRES_VARS})

    nopac_evapo_first(**{k: general_dict[k] for k in NOPAC_VARS1})

    nopac_evapo_second(**{k: general_dict[k] for k in NOPAC_VARS2})

    nopac_evapo_third(**{k: general_dict[k] for k in NOPAC_VARS3})

    nopac_smflx_first(**{k: general_dict[k] for k in NOPAC_VARS4})

    nopac_smflx_second(**{k: general_dict[k] for k in NOPAC_VARS5})

    rosr12_second(general_dict["delta"], general_dict["p"],
                  general_dict["flag_iter"], general_dict["land"], general_dict["ci"])

    nopac_smflx_third(**{k: general_dict[k] for k in NOPAC_VARS6})

    nopac_shflx_first(**{k: general_dict[k] for k in NOPAC_VARS7})


    nopac_shflx_second(**{k: general_dict[k] for k in NOPAC_VARS8})

    snopac_evapo_first(general_dict["ice"], general_dict["sncovr"],
                       **{k: general_dict[k] for k in NOPAC_VARS1})

    snopac_evapo_second(general_dict["ice"], general_dict["sncovr"],
                        **{k: general_dict[k] for k in NOPAC_VARS2})

    snopac_evapo_third(general_dict["ice"], general_dict["sncovr"], general_dict["edir"], general_dict["ec"],
                       **{k: general_dict[k] for k in NOPAC_VARS3})

    snopac_smflx_first(**{k: general_dict[k] for k in SNOPAC_VARS1})

    snopac_smflx_second(general_dict["ice"], **
                        {k: general_dict[k] for k in NOPAC_VARS5})

    rosr12_second(general_dict["delta"], general_dict["p"],
                  general_dict["flag_iter"], general_dict["land"], general_dict["ci"])

    snopac_smflx_third(general_dict["ice"], **
                       {k: general_dict[k] for k in NOPAC_VARS6})

    snopac_shflx_first(general_dict["ssoil"], 
                       **{k: general_dict[k] for k in NOPAC_VARS7})

    snopac_shflx_second(general_dict["ice"], general_dict["sneqv_new"], general_dict["sncovr"], general_dict["dt"], general_dict["snowh"], general_dict["sndens"],
                        **{k: general_dict[k] for k in NOPAC_VARS8})

    general_dict["cmx"] = general_dict["cm"]
    general_dict["delt"] = general_dict["dt"]
    general_dict["prsl1"] = general_dict["sfcprs"]
    general_dict["qs1"] = general_dict["q2sat"]
    general_dict["sigmaf"] = general_dict["shdfac"]
    general_dict["slc"] = general_dict["sh2o"]
    general_dict["q0"] = general_dict["q2"]
    general_dict["tg3"] = general_dict["tbot"]
    general_dict["srflag"] = general_dict["ffrozp"]

    # post sflx data handling
    out_dict = {k: general_dict[k] for k in (INOUT_VARS + INOUT_MULTI_VARS)}
    cleanup_sflx(**{k: general_dict[k] for k in CLEANUP_VARS}, **out_dict)
    # set timer
    toc = timeit.default_timer()

    # calculate elapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {**{k: gt4py_to_numpy_storage_onelayer(
        out_dict[k], backend=backend) for k in INOUT_VARS}, **{k: gt4py_to_numpy_storage_multilayer(
            out_dict[k], backend=backend) for k in INOUT_MULTI_VARS}}

    return elapsed_time, out_dict
