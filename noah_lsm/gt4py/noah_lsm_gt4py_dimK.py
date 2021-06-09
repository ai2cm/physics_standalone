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

SCALAR_VARS = ["km", "delt", "lheatstrg", "ivegsrc"]

SOIL_VARS = [bb, satdk, satdw, f11, satpsi,
             qtz, drysmc, maxsmc, refsmc, wltsmc]
VEG_VARS = [nroot_data, snupx, rsmtbl, rgltbl, hstbl, lai_data]

SOIL_VARS_NAMES = ["bexp", "dksat", "dwsat", "f1", "psisat",
                   "quartz", "smcdry", "smcmax", "smcref", "smcwlt"]
VEG_VARS_NAMES = ["nroot", "snup", "rsmin", "rgl", "hs", "xlai"]

TEMP_VARS_1D = ["zsoil_root", "q0", "cmc", "th2", "rho", "qs1", "ice",  # prepare_sflx
                "prcp", "dqsdt2", "snowh", "sneqv", "chx", "cmx", "z0",
                "shdfac", "kdt", "frzx", "sndens", "snowng", "prcp1",
                "sncovr", "df1", "ssoil", "t2v", "fdown", "cpx1",
                "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old",
                "canopy_old", "etp", "t24", "rch", "epsca", "rr", "flx2",
                "rc", "pc", "rcs", "rct", "rcq", "rcsoil",  # canres
                "edir1", "ec1", "sgx", "edir", "ec",  # nopac / snopac
                "ett1", "eta1", "ett", "eta", "etns1", "etns", "denom", "sicemax",
                "dd", "dice", "pcpdrp", "rhsct", "drip", "runoff1", "runoff2",
                "runoff3", "dew", "yy", "zz1", "beta", "flx1", "flx2", "flx3",
                "esnow",
                ]


TEMP_VARS_2D = ["sldpth", "rtdis", "smc_old", "stc_old", "slc_old",
                "et1", "et", "rhstt", "ai", "bi", "ci", "dsmdz", "ddz", "gx", 
                "wdf", "wcnd", "sice", "p", "delta", "stsoil",

                ]

PREPARE_VARS = ["zsoil", "km", "zsoil_root", "q0", "cmc", "th2", "rho", "qs1", "ice",
                "prcp", "dqsdt2", "snowh", "sneqv", "chx", "cmx", "z0",
                "shdfac", "kdt", "frzx", "sndens", "snowng", "prcp1",
                "sncovr", "df1", "ssoil", "t2v", "fdown", "cpx1",
                "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old",
                "canopy_old", "etp", "t24", "rch", "epsca", "rr", "flx2",
                "sldpth", "rtdis", "smc_old", "stc_old", "slc_old",
                "t1", "q1", "vegtype", "sigmaf", "sfcemis", "dlwflx", "snet", "cm",
                "ch", "prsl1", "prslki", "land", "wind", "snoalb", "sfalb",
                "flag_iter", "flag_guess", "bexppert", "xlaipert", "fpvs",
                "weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc",
                "canopy", "zorl", "delt", "ivegsrc", "bexp", "dksat",
                "quartz", "smcdry", "smcmax", "smcref", "smcwlt", "nroot", "snup", "xlai"]


CANRES_VARS = ["dswsfc", "nroot", "ch", "q2", "q2sat", "dqsdt2", "sfctmp",
               "cpx1", "sfcprs", "sfcemis", "sh2o", "smcwlt", "smcref", "zsoil", "rsmin",
               "rgl", "hs", "xlai", "flag_iter", "land", "zsoil_root", "shdfac",
               "rc", "pc", "rcs", "rct", "rcq", "rcsoil"]


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

    general_dict = {"zsoil": zsoil, **in_dict,
                    **scalar_dict, **table_dict, **out_dict}
        # counter
    general_dict["count"] = initialize_gt4py_storage(im, 1, np.int32)


    # prepare for sflx
    general_dict = {**general_dict, **{k: initialize_gt4py_storage(
        im, 1, np.float64) for k in TEMP_VARS_1D}}
    general_dict = {**general_dict, **{k: initialize_gt4py_storage(
        im, km, np.float64) for k in TEMP_VARS_2D}}

    prepare_sflx(**{k: general_dict[k] for k in PREPARE_VARS})

    general_dict["cm"] = general_dict["cmx"]
    general_dict["dt"] = general_dict["delt"]
    general_dict["sfcprs"] = general_dict["prsl1"]
    general_dict["q2sat"] = general_dict["qs1"]
    general_dict["shdfac"] = general_dict["sigmaf"]
    general_dict["ch"] = general_dict["chx"]
    general_dict["sh2o"] = general_dict["slc"]
    general_dict["sfctmp"] = general_dict["t1"]
    general_dict["q2"] = general_dict["q0"]
    general_dict["tbot"] = general_dict["tg3"]
    general_dict["ffrozp"] = general_dict["srflag"]

    canres(**{k: general_dict[k] for k in CANRES_VARS})

    nopac_evapo_first(general_dict["etp"], table_dict["nroot"], table_dict["smcmax"], table_dict["smcwlt"], table_dict["smcref"],
                      table_dict["smcdry"], scalar_dict["delt"], in_dict["sigmaf"],
                      general_dict["cmc"], general_dict["slc"], general_dict["flag_iter"], general_dict[
                          "land"], general_dict["sneqv"], general_dict["count"],
                      # output
                      general_dict["sgx"], general_dict["gx"], general_dict["edir1"], general_dict["ec1"], general_dict["edir"], general_dict["ec"])

    nopac_evapo_second(general_dict["sigmaf"], general_dict["etp"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                       general_dict["sneqv"], general_dict["count"], general_dict["rtdis"], general_dict["sgx"],
                       # output
                       general_dict["denom"], general_dict["gx"])

    nopac_evapo_third(general_dict["sigmaf"], general_dict["etp"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                      general_dict["sneqv"], general_dict["count"], general_dict["gx"], general_dict["cmc"], general_dict["pc"],
                      general_dict["denom"], general_dict["edir1"], general_dict["ec1"],
                      # output
                      general_dict["et1"], general_dict["ett1"], general_dict["eta1"], general_dict["et"], general_dict["ett"], general_dict["eta"])

    nopac_smflx_first(general_dict["smcmax"], general_dict["dt"], general_dict["smcwlt"], general_dict["prcp"], general_dict["prcp1"], zsoil, general_dict["shdfac"], general_dict["ec1"],
                      general_dict["cmc"], general_dict["sh2o"], general_dict["smc"], general_dict["flag_iter"], general_dict["land"], general_dict[
                          "etp"], general_dict["sneqv"], general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["rhsct"], general_dict["drip"], general_dict["sice"], general_dict["dew"])

    nopac_smflx_second(general_dict["smcmax"], general_dict["etp"], general_dict["dt"], general_dict["bexp"], general_dict["kdt"], general_dict["frzx"], general_dict["dksat"], general_dict["dwsat"], general_dict["slope"],
                       zsoil, general_dict["sh2o"], general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], general_dict[
                           "sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["edir1"], general_dict["et1"],
                       # outpus
                       general_dict["rhstt"], general_dict["ci"], general_dict["runoff1"], general_dict["runoff2"], general_dict["dsmdz"], general_dict["ddz"], general_dict["wdf"], general_dict["wcnd"], general_dict["p"], general_dict["delta"])

    rosr12_second(general_dict["delta"], general_dict["p"],
                  general_dict["flag_iter"], general_dict["land"], general_dict["ci"])

    nopac_smflx_third(general_dict["dt"], general_dict["smcmax"], general_dict["flag_iter"], general_dict["land"], general_dict["sice"], general_dict["sldpth"], general_dict["rhsct"], general_dict["ci"],
                      general_dict["sneqv"],
                      # outpus
                      general_dict["sh2o"], general_dict["smc"], general_dict["cmc"], general_dict["runoff3"])

    nopac_shflx_first(general_dict["land"], general_dict["flag_iter"], general_dict["sneqv"], general_dict["etp"], general_dict["eta"], general_dict["smc"], general_dict["quartz"], general_dict["smcmax"], general_dict["ivegsrc"],
                      general_dict["vegtype"], general_dict["shdfac"], general_dict["fdown"], general_dict[
                          "sfcemis"], general_dict["t24"], general_dict["sfctmp"], general_dict["rch"], general_dict["th2"],
                      general_dict["epsca"], general_dict["rr"], zsoil, general_dict["dt"], general_dict[
                          "psisat"], general_dict["bexp"], general_dict["ice"], general_dict["stc"],
                      general_dict["sh2o"], general_dict["stsoil"], general_dict["p"], general_dict["delta"], general_dict["tbot"], general_dict["yy"], general_dict["zz1"], general_dict["df1"], general_dict["beta"])

    nopac_shflx_second(general_dict["land"], general_dict["flag_iter"], general_dict["sneqv"], zsoil, general_dict["stsoil"], general_dict["p"], general_dict["delta"],
                       general_dict["yy"], general_dict["zz1"], general_dict["df1"],
                       general_dict["ssoil"], general_dict["t1"], general_dict["stc"])

    snopac_evapo_first(general_dict["etp"], general_dict["nroot"], general_dict["smcmax"], general_dict["smcwlt"], general_dict["smcref"],
                       general_dict["smcdry"], general_dict["dt"], general_dict["shdfac"],
                       general_dict["cmc"], general_dict["sh2o"], general_dict["ice"], general_dict["sncovr"], general_dict["flag_iter"], general_dict[
        "land"], general_dict["sneqv"], general_dict["count"],
        # output
        general_dict["sgx"], general_dict["gx"], general_dict["edir1"], general_dict["ec1"], general_dict["edir"], general_dict["ec"])

    snopac_evapo_second(general_dict["shdfac"], general_dict["etp"], general_dict["flag_iter"], general_dict["land"],
                        general_dict["sneqv"], general_dict["count"], general_dict[
                            "rtdis"], general_dict["sgx"], general_dict["ice"], general_dict["sncovr"],
                        general_dict["denom"], general_dict["gx"])

    snopac_evapo_third(general_dict["shdfac"], general_dict["etp"], general_dict["flag_iter"], general_dict["land"],
                       general_dict["sneqv"], general_dict["count"], general_dict["gx"], general_dict["cmc"], general_dict["pc"],
                       general_dict["denom"], general_dict["edir1"], general_dict["ec1"], general_dict["ice"], general_dict["sncovr"],
                       # output
                       general_dict["et1"], general_dict["ett1"], general_dict["etns1"], general_dict["et"], general_dict["ett"], general_dict["etns"])

    snopac_smflx_first(general_dict["smcmax"], general_dict["dt"], general_dict["smcwlt"], general_dict["prcp"], general_dict["prcp1"], general_dict["zsoil"], general_dict["shdfac"], general_dict["ec1"],
                       general_dict["cmc"], general_dict["sh2o"], general_dict["smc"], general_dict[
                           "flag_iter"], general_dict["land"], general_dict["etp"],
                       general_dict["sncovr"], general_dict["ice"], general_dict["snowng"], general_dict[
                           "ffrozp"], general_dict["sfctmp"], general_dict["etns"], general_dict["snowh"],
                       general_dict["df1"], general_dict["rr"], general_dict["rch"], general_dict["fdown"], general_dict[
                           "flx2"], general_dict["sfcemis"], general_dict["t24"], general_dict["th2"], general_dict["stc"],
                       # output
                       general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict[
                           "pcpdrp"], general_dict["rhsct"], general_dict["drip"], general_dict["sice"], general_dict["dew"],
                       general_dict["t1"], general_dict["sneqv"], general_dict["flx1"], general_dict["flx3"], general_dict["esnow"], general_dict["ssoil"])

    snopac_smflx_second(general_dict["smcmax"], general_dict["etp"], general_dict["dt"], general_dict["bexp"], general_dict["kdt"], general_dict["frzx"], general_dict["dksat"], general_dict["dwsat"], general_dict["slope"],
                        zsoil, general_dict["sh2o"], general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], general_dict[
                            "sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["edir1"], general_dict["et1"],
                        # outpus
                        general_dict["rhstt"], general_dict["ci"], general_dict["runoff1"], general_dict["runoff2"], general_dict["dsmdz"], general_dict["ddz"], general_dict["wdf"], general_dict["wcnd"], general_dict["p"], general_dict["delta"])

    rosr12_second(general_dict["delta"], general_dict["p"],
                  general_dict["flag_iter"], general_dict["land"], general_dict["ci"])

    snopac_smflx_third(general_dict["dt"], general_dict["smcmax"], general_dict["flag_iter"], general_dict["land"], general_dict["sice"], general_dict["sldpth"], general_dict["rhsct"], general_dict["ci"],
                       general_dict["sneqv"],
                       # outpus
                       general_dict["sh2o"], general_dict["smc"], general_dict["cmc"], general_dict["runoff3"])

    snopac_shflx_first(general_dict["land"], general_dict["flag_iter"], general_dict["sneqv"], general_dict["etp"], general_dict["etns"], general_dict["smc"], general_dict["quartz"], general_dict["smcmax"], general_dict["ivegsrc"],
                       general_dict["vegtype"], general_dict["shdfac"], general_dict["fdown"], general_dict[
                           "sfcemis"], general_dict["t24"], general_dict["sfctmp"], general_dict["rch"], general_dict["th2"],
                       general_dict["epsca"], general_dict["rr"], zsoil, general_dict["dt"], general_dict[
                           "psisat"], general_dict["bexp"], general_dict["ice"], general_dict["stc"],
                       general_dict["sh2o"], general_dict["stsoil"], general_dict["p"], general_dict["delta"], general_dict["tbot"], general_dict["yy"], general_dict["zz1"], general_dict["df1"], general_dict["beta"])

    snopac_shflx_second(general_dict["land"], general_dict["flag_iter"], general_dict["sneqv"], zsoil, general_dict["stsoil"], general_dict["p"], general_dict["delta"],
                        general_dict["yy"], general_dict["zz1"], general_dict["df1"],
                        general_dict["ssoil"], general_dict["t1"], general_dict["stc"])

    general_dict["cmx"] = general_dict["cm"]
    general_dict["delt"] = general_dict["dt"]
    general_dict["prsl1"] = general_dict["sfcprs"]
    general_dict["qs1"] = general_dict["q2sat"]
    general_dict["sigmaf"] = general_dict["shdfac"]
    general_dict["chx"] = general_dict["ch"]
    general_dict["slc"] = general_dict["sh2o"]
    general_dict["t1"] = general_dict["sfctmp"]
    general_dict["q0"] = general_dict["q2"]
    general_dict["tg3"] = general_dict["tbot"]
    general_dict["srflag"] = general_dict["ffrozp"]

    # post sflx data handling
    out_dict = {k: general_dict[k] for k in (INOUT_VARS + INOUT_MULTI_VARS)}
    # cleanup_sflx(
    #     general_dict["flag_iter"], general_dict["land"], general_dict["eta"], general_dict["ssoil"], general_dict["edir"], general_dict["ec"], general_dict["ett"], general_dict["esnow"], general_dict["sncovr"], general_dict["soilm"],
    #     general_dict["flx1"], general_dict["flx2"], general_dict["flx3"], general_dict["etp"], general_dict["runoff1"], general_dict["runoff2"],
    #     general_dict["cmc"], general_dict["snowh"], general_dict["sneqv"], general_dict["z0"], general_dict["rho"], general_dict["ch"], general_dict["wind"], general_dict["q1"], general_dict["elocp"], general_dict["cpinv"], general_dict["hvapi"], general_dict["flag_guess"],
    #     general_dict["weasd_old"], general_dict["snwdph_old"], general_dict["tskin_old"], general_dict["canopy_old"], general_dict["tprcp_old"], general_dict["srflag_old"],
    #     general_dict["smc_old"], general_dict["slc_old"], general_dict["stc_old"], general_dict["th2"], general_dict["t1"], general_dict["et"], general_dict["ice"], general_dict["runoff3"], general_dict["dt"],
    #     general_dict["sfcprs"], general_dict["t2v"], general_dict["snomlt"], general_dict["zsoil"],
    #     # TODO: output
    #     **out_dict)

    # set timer
    toc = timeit.default_timer()

    # calculate elapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {**{k: gt4py_to_numpy_storage_onelayer(
        out_dict[k], backend=backend) for k in INOUT_VARS}, **{k: gt4py_to_numpy_storage_multilayer(
            out_dict[k], backend=backend) for k in INOUT_MULTI_VARS}}

    return out_dict
