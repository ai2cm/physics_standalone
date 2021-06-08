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
DT_FK = gtscript.Field[gtscript.IK, np.float64]

BACKEND = "numpy"

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

SCALAR_VARS = ["delt", "lheatstrg", "ivegsrc"]

SOIL_VARS = [bb, satdk, satdw, f11, satpsi,
             qtz, drysmc, maxsmc, refsmc, wltsmc]
VEG_VARS = [nroot_data, snupx, rsmtbl, rgltbl, hstbl, lai_data]
SOIL_VARS_NAMES = ["bexp", "dksat", "dwsat", "f1", "psisat",
                   "quartz", "smcdry", "smcmax", "smcref", "smcwlt"]
VEG_VARS_NAMES = ["nroot", "snup", "rsmin", "rgl", "hs", "xlai"]

PREPARE_VARS_1D = ["zsoil_root", "q0", "cmc", "th2", "rho", "qs1", "ice",
                   "prcp", "dqsdt2", "snowh", "sneqv", "chx", "cmx", "z0",
                   "shdfac", "kdt", "frzx", "sndens", "snowng", "prcp1",
                   "sncovr", "df1", "ssoil", "t2v", "fdown", "cpx1",
                   "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old", 
                   "canopy_old", "etp", "t24", "rch", "epsca", "rr", "flx2"]

PREPARE_VARS_2D = ["sldpth", "rtdis", "smc_old", "stc_old", "slc_old"]
CANRES_VARS = ["rc", "pc", "rcs", "rct", "rcq", "rcsoil"]


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

    # zsoil = np.repeat([[-0.1], [-0.4], [-1.0], [-2.0]], in_dict["im"], axis=1)
    zsoil = np.array([[-0.1, -0.4, -1.0, -2.0]])
    zsoil = gt.storage.from_array(zsoil, backend=backend, shape=(
        1, 4), mask=(True, False, True), default_origin=(0, 0, 0))

    # setup storages
    table_dict = {**{SOIL_VARS_NAMES[k]: numpy_table_to_gt4py_storage(
        SOIL_VARS[k], in_dict["soiltyp"], backend=backend) for k in range(len(SOIL_VARS))}, **{VEG_VARS_NAMES[k]: numpy_table_to_gt4py_storage(
            VEG_VARS[k], in_dict["vegtype"], backend=backend) for k in range(len(VEG_VARS))}, **{"slope": numpy_table_to_gt4py_storage(slope_data, in_dict["slopetyp"], backend=backend)}}

    scalar_dict = {**{k: in_dict[k] for k in SCALAR_VARS}}
    out_dict = {k: numpy_to_gt4py_storage(
        in_dict[k].copy(), backend=backend) for k in (INOUT_VARS + INOUT_MULTI_VARS)}
    in_dict = {**{k: numpy_to_gt4py_storage(in_dict[k], backend=backend) for k in IN_VARS},
               **{"fpvs": numpy_to_gt4py_storage(fpvs, backend=backend)}}
    #    **{"zsoil": numpy_to_gt4py_storage(zsoil, backend=backend)}}

    # compile stencil
    # sfc_drv = gtscript.stencil(
    # definition=sfc_drv_defs, backend=backend, externals={})

    # set timer
    tic = timeit.default_timer()

    general_dict = {**in_dict, **scalar_dict, **table_dict, **out_dict}

    # prepare for sflx
    prepare_dict1 = {k: initialize_gt4py_storage(
        im, 1, np.float64) for k in PREPARE_VARS_1D}
    prepare_dict2 = {k: initialize_gt4py_storage(
        im, km, np.float64) for k in PREPARE_VARS_2D}

    general_dict = {**general_dict, **prepare_dict1, **prepare_dict2}

    prepare_sflx(zsoil, km, **prepare_dict1, **prepare_dict2, **
                 in_dict, **out_dict, **scalar_dict, **table_dict)

    # prepare_sflx(in_dict["t1"], in_dict["q1"], in_dict["cm"], in_dict["ch"], in_dict["prsl1"],
    #              in_dict["prslki"], in_dict["zf"], in_dict["land"], in_dict["wind"], in_dict["flag_iter"],
    #              in_dict["flag_guess"], in_dict["fpvs"], zsoil, in_dict["sigmaf"],
    #              in_dict["vegtype"], table_dict["nroot"], in_dict["bexppert"], in_dict["xlaipert"],
    #              in_dict["sfalb"], table_dict["snup"], in_dict["snoalb"], in_dict["snet"], in_dict["dlwflx"],
    #              out_dict["weasd"], out_dict["snwdph"],
    #              out_dict["tskin"], out_dict["tprcp"], out_dict["srflag"], out_dict["canopy"], out_dict["zorl"],
    #              out_dict["smc"], out_dict["stc"], out_dict["slc"], scalar_dict["delt"], scalar_dict["ivegsrc"], km,
    #              table_dict["dksat"], table_dict["smcmax"], table_dict["smcref"], table_dict["smcwlt"], table_dict["smcdry"],table_dict["bexp"], table_dict["xlai"], table_dict["quartz"], zsoil_root, q0)

    # run sflx algorithm
    canres_dict = {k: initialize_gt4py_storage(
        im, 1, np.float64) for k in CANRES_VARS}

    general_dict = {**general_dict, **canres_dict}
    general_dict["shdfac"] = general_dict["sigmaf"]
    general_dict["dt"] = general_dict["delt"]
    general_dict["sfcprs"] = general_dict["prsl1"]
    general_dict["q2sat"] = general_dict["qs1"]
    general_dict["shdfac"] = general_dict["sigmaf"]
    general_dict["ch"] = general_dict["chx"]
    general_dict["sh2o"] = general_dict["slc"]
    general_dict["sfctmp"] = general_dict["t1"]
    general_dict["q2"] = general_dict["q0"]
    general_dict["tbot"] = general_dict["tg3"]

    canres(table_dict["nroot"], in_dict["dswsfc"], prepare_dict1["chx"], prepare_dict1["q0"], prepare_dict1["qs1"], prepare_dict1["dqsdt2"], in_dict["t1"],
           prepare_dict1["cpx1"], in_dict["prsl1"], in_dict["sfcemis"], out_dict[
               "slc"], table_dict["smcwlt"], table_dict["smcref"], zsoil, table_dict["rsmin"],
           table_dict["rgl"], table_dict["hs"], table_dict["xlai"], in_dict[
               "flag_iter"], in_dict["land"], prepare_dict1["zsoil_root"], in_dict["sigmaf"],
           **canres_dict)

    general_dict["count"] = initialize_gt4py_storage(im, 1, np.int32)
    general_dict["gx"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["edir1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["ec1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["sgx"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["edir"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["ec"] = initialize_gt4py_storage(im, 1, np.float64)

    nopac_evapo_first(table_dict["nroot"], prepare_dict1["etp"], table_dict["smcmax"], table_dict["smcwlt"], table_dict["smcref"],
                      table_dict["smcdry"], scalar_dict["delt"], in_dict["sigmaf"],
                      general_dict["cmc"], general_dict["slc"], general_dict["flag_iter"], general_dict[
                          "land"], general_dict["sneqv"], general_dict["count"],
                      # output
                      general_dict["sgx"], general_dict["gx"], general_dict["edir1"], general_dict["ec1"], general_dict["edir"], general_dict["ec"])

    general_dict["denom"] = initialize_gt4py_storage(im, 1, np.float64)
    nopac_evapo_second(general_dict["etp"], general_dict["sigmaf"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                       general_dict["sneqv"], general_dict["count"], general_dict["rtdis"], general_dict["sgx"],
                       # output
                       general_dict["denom"], general_dict["gx"])

    general_dict["et1"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ett1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["eta1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["et"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ett"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["eta"] = initialize_gt4py_storage(im, 1, np.float64)

    nopac_evapo_third(general_dict["etp"], general_dict["sigmaf"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                      general_dict["sneqv"], general_dict["count"], general_dict["gx"], general_dict["cmc"], general_dict["pc"],
                      general_dict["denom"], general_dict["edir1"], general_dict["ec1"],
                      # output
                      general_dict["et1"], general_dict["ett1"], general_dict["eta1"], general_dict["et"], general_dict["ett"], general_dict["eta"])

    general_dict["sicemax"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["dd"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["dice"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["pcpdrp"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["rhstt"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ai"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["bi"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ci"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["dsmdz"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ddz"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["wdf"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["wcnd"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["rhsct"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["drip"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["sice"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["runoff1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["runoff2"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["p"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["delta"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["dew"] = initialize_gt4py_storage(im, 1, np.float64)

    nopac_smflx_first(general_dict["dt"], general_dict["smcmax"], general_dict["smcwlt"], general_dict["prcp"], general_dict["prcp1"], zsoil, general_dict["shdfac"], general_dict["ec1"],
                      general_dict["cmc"], general_dict["sh2o"], general_dict["smc"], general_dict["flag_iter"], general_dict["land"], general_dict[
                          "etp"], general_dict["sneqv"], general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["rhsct"], general_dict["drip"], general_dict["sice"], general_dict["dew"])

    nopac_smflx_second(general_dict["etp"], general_dict["smcmax"], general_dict["dt"], general_dict["bexp"], general_dict["kdt"], general_dict["frzx"], general_dict["dksat"], general_dict["dwsat"], general_dict["slope"],
                       zsoil, general_dict["sh2o"], general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["edir1"], general_dict["et1"],
                       # outpus
                       general_dict["rhstt"], general_dict["ci"], general_dict["runoff1"], general_dict["runoff2"], general_dict["dsmdz"], general_dict["ddz"], general_dict["wdf"], general_dict["wcnd"], general_dict["p"], general_dict["delta"])

    general_dict["stsoil"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["yy"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["zz1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["beta"] = initialize_gt4py_storage(im, 1, np.float64)
    nopac_shflx_first(general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], general_dict["etp"], general_dict["eta"], general_dict["smc"], general_dict["quartz"], general_dict["smcmax"], general_dict["ivegsrc"],
                      general_dict["vegtype"], general_dict["shdfac"], general_dict["fdown"], general_dict["sfcemis"], general_dict["t24"], general_dict["sfctmp"], general_dict["rch"], general_dict["th2"], 
                      general_dict["epsca"], general_dict["rr"], zsoil, general_dict["dt"], general_dict["psisat"], general_dict["bexp"], general_dict["ice"], general_dict["stc"], 
                      general_dict["sh2o"], general_dict["stsoil"], general_dict["p"], general_dict["delta"], general_dict["tbot"], general_dict["yy"], general_dict["zz1"], general_dict["df1"], general_dict["beta"])

    nopac_shflx_second(general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], zsoil, general_dict["stsoil"], general_dict["p"], general_dict["delta"], 
                        general_dict["yy"], general_dict["zz1"], general_dict["df1"],
                        general_dict["ssoil"], general_dict["t1"], general_dict["stc"])
    # post sflx data handling
    # TODO: handling of smc0
    # cleanup_sflx(in_dict["flag_iter"], in_dict["land"], eta, sheat, ssoil, edir,
    #              ec, ett, esnow, sncovr, soilm, flx1,
    #              flx2, flx3, smcwlt, smcref, etp, smc0,
    #              smcmax, runoff1, runoff2, cmc, snowh, sneqv,
    #              z0, rho, ch, wind, q1, elocp, cpinv,
    #              hvapi, in_dict["flag_guess"], weasd_old, snwdph_old, tskin_old,
    #              canopy_old, tprcp_old, srflag_old,smc_old, slc_old,
    #              stc_old, tsurf)

    # set timer
    toc = timeit.default_timer()

    # calculate elapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {**{k: gt4py_to_numpy_storage_onelayer(
        general_dict[k], backend=backend) for k in INOUT_VARS}, **{k: gt4py_to_numpy_storage_multilayer(
            general_dict[k], backend=backend) for k in INOUT_MULTI_VARS}}

    return out_dict

