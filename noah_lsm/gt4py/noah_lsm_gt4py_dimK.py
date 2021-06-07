#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit

import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *

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
                   "weasd_old", "snwdph_old", "tskin_old", "tprcp_old", "srflag_old", "canopy_old", "etp"]

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

    canres(table_dict["nroot"], in_dict["dswsfc"], prepare_dict1["chx"], prepare_dict1["q0"], prepare_dict1["qs1"], prepare_dict1["dqsdt2"], in_dict["t1"],
           prepare_dict1["cpx1"], in_dict["prsl1"], in_dict["sfcemis"], out_dict[
               "slc"], table_dict["smcwlt"], table_dict["smcref"], zsoil, table_dict["rsmin"],
           table_dict["rgl"], table_dict["hs"], table_dict["xlai"], in_dict[
               "flag_iter"], in_dict["land"], prepare_dict1["zsoil_root"], in_dict["sigmaf"],
           **canres_dict)

    count = initialize_gt4py_storage(im, 1, np.int32)
    gx = initialize_gt4py_storage(im, km, np.float64)
    edir1 = initialize_gt4py_storage(im, 1, np.float64)
    ec1 = initialize_gt4py_storage(im, 1, np.float64)
    sgx = initialize_gt4py_storage(im, 1, np.float64)
    general_dict = {"count": count, "edir1": edir1,
                    "ec1": ec1, "gx": gx, "sgx": sgx, **general_dict}

    nopac_evapo_first(table_dict["nroot"], prepare_dict1["etp"], table_dict["smcmax"], table_dict["smcwlt"], table_dict["smcref"],
                      table_dict["smcdry"], scalar_dict["delt"], in_dict["sigmaf"],
                      general_dict["cmc"], general_dict["slc"], general_dict["flag_iter"], general_dict[
                          "land"], general_dict["sneqv"], general_dict["count"],
                      # output
                      general_dict["sgx"], general_dict["gx"], general_dict["edir1"], general_dict["ec1"])

    general_dict["denom"] = initialize_gt4py_storage(im, 1, np.float64)
    nopac_evapo_second(general_dict["etp"], general_dict["sigmaf"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                       general_dict["sneqv"], general_dict["count"], general_dict["rtdis"], general_dict["sgx"],
                       # output
                       general_dict["denom"], general_dict["gx"])

    general_dict["et1"] = initialize_gt4py_storage(im, km, np.float64)
    general_dict["ett1"] = initialize_gt4py_storage(im, 1, np.float64)
    general_dict["eta1"] = initialize_gt4py_storage(im, 1, np.float64)
    nopac_evapo_third(general_dict["etp"], general_dict["sigmaf"], general_dict["slc"], general_dict["flag_iter"], general_dict["land"],
                      general_dict["sneqv"], general_dict["count"], general_dict["gx"], general_dict["cmc"], general_dict["pc"],
                      general_dict["denom"], general_dict["edir1"], general_dict["ec1"],
                      # output
                      general_dict["et1"], general_dict["ett1"], general_dict["eta1"])

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

    nopac_smflx_first(general_dict["dt"], general_dict["smcmax"], general_dict["smcwlt"], general_dict["prcp1"], zsoil, general_dict["shdfac"], general_dict["ec1"],
                      general_dict["cmc"], general_dict["sh2o"], general_dict["smc"], general_dict["flag_iter"], general_dict["land"], general_dict[
                          "etp"], general_dict["sneqv"], general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["rhsct"], general_dict["drip"], general_dict["sice"])

    nopac_smflx_second(general_dict["etp"], general_dict["smcmax"], general_dict["dt"], general_dict["bexp"], general_dict["kdt"], general_dict["frzx"], general_dict["dksat"], general_dict["dwsat"], general_dict["slope"],
                       zsoil, general_dict["sh2o"], general_dict["flag_iter"], general_dict["land"], general_dict["sneqv"], general_dict["sicemax"], general_dict["dd"], general_dict["dice"], general_dict["pcpdrp"], general_dict["edir1"], general_dict["et1"],
                       # outpus
                       general_dict["rhstt"], general_dict["ci"], general_dict["runoff1"], general_dict["runoff2"], general_dict["dsmdz"], general_dict["ddz"], general_dict["wdf"], general_dict["wcnd"], general_dict["p"], general_dict["delta"])

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
        out_dict[k], backend=backend) for k in INOUT_VARS}, **{k: gt4py_to_numpy_storage_multilayer(
            out_dict[k], backend=backend) for k in INOUT_MULTI_VARS}}

    return out_dict


@gtscript.function
def csnow_fn(sndens):
    # --- ... subprograms called: none

    unit = 0.11631

    c = 0.328 * 10**(2.25*sndens)
    sncond = unit * c

    return sncond


@gtscript.function
def snow_new_fn(sfctmp, sn_new, snowh, sndens):
    # --- ... subprograms called: none

    # conversion into simulation units
    snowhc = snowh * 100.0
    newsnc = sn_new * 100.0
    tempc = sfctmp - tfreez

    # calculating new snowfall density
    if tempc <= -15.0:
        dsnew = 0.05
    else:
        dsnew = 0.05 + 0.0017*(tempc + 15.0)**1.5

    # adjustment of snow density depending on new snowfall
    hnewc = newsnc / dsnew
    sndens = (snowhc*sndens + hnewc*dsnew) / (snowhc + hnewc)
    snowhc = snowhc + hnewc
    snowh = snowhc * 0.01

    return snowh, sndens


@gtscript.function
def snfrac_fn(sneqv, snup, salp):
    # --- ... subprograms called: none

    # determine snow fraction cover.
    if sneqv < snup:
        rsnow = sneqv / snup
        sncovr = 1.0 - (exp(-salp*rsnow) - rsnow*exp(-salp))
    else:
        sncovr = 1.0

    return sncovr


@gtscript.function
def alcalc_fn(alb, snoalb, sncovr):
    # --- ... subprograms called: none

    # calculates albedo using snow effect
    # snoalb: max albedo over deep snow
    albedo = alb + sncovr * (snoalb - alb)
    if (albedo > snoalb):
        albedo = snoalb

    return albedo


@gtscript.function
def tdfcnd_fn(smc, qz, smcmax, sh2o):
    # --- ... subprograms called: none

    # calculates thermal diffusivity and conductivity of the soil for a given point and time

    # saturation ratio
    satratio = smc / smcmax

    thkice = 2.2
    thkw = 0.57
    thko = 2.0
    thkqtz = 7.7

    # solids` conductivity
    thks = (thkqtz**qz) * (thko**(1.0-qz))

    # unfrozen fraction
    xunfroz = (sh2o + 1.e-9) / (smc + 1.e-9)

    # unfrozen volume for saturation (porosity*xunfroz)
    xu = xunfroz*smcmax

    # saturated thermal conductivity
    thksat = thks**(1.-smcmax) * thkice**(smcmax-xu) * thkw**(xu)

    # dry density in kg/m3
    gammd = (1.0 - smcmax) * 2700.0

    # dry thermal conductivity in w.m-1.k-1
    thkdry = (0.135*gammd + 64.7) / (2700.0 - 0.947*gammd)

    if sh2o+0.0005 < smc:    # frozen
        ake = satratio
    elif satratio > 0.1:
        # kersten number
        ake = log(satratio)/log(10) + 1.0   # log10 from ln
    else:
        ake = 0.0

    # thermal conductivity
    df = ake * (thksat - thkdry) + thkdry

    return df


@gtscript.function
def penman_fn(sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
              cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra, ffrozp):
    # --- ... subprograms called: none

    flx2 = 0.0
    # # prepare partial quantities for penman equation.
    delta = elcp * cpfac * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = t24 * 6.48e-8 / (sfcprs*ch) + 1.0
    rho = sfcprs / (rd1*t2v)
    rch = rho * cpx * ch

    # adjust the partial sums / products with the latent heat
    # effects caused by falling precipitation.
    if not snowng:
        if prcp > 0.:
            rr += cph2o1*prcp/rch
    else:
        # fractional snowfall/rainfall
        rr += (cpice*ffrozp+cph2o1*(1.-ffrozp))*prcp/rch

    # ssoil = 13.753581783277639
    fnet = fdown - sfcems*sigma1*t24 - ssoil

    # include the latent heat effects of frzng rain converting to ice
    # on impact in the calculation of flx2 and fnet.
    if frzgra:
        flx2 = -lsubf * prcp
        fnet = fnet - flx2

    # finish penman equation calculations.

    rad = fnet/rch + th2 - sfctmp
    a = elcp * cpfac * (q2sat - q2)

    epsca = (a*rr + rad*delta) / (delta + rr)
    etp = epsca * rch / lsubc

    return t24, etp, rch, epsca, rr, flx2


@gtscript.stencil(backend=BACKEND)
def prepare_sflx(
    zsoil: DT_FK, km: int,
    # output 1D
    zsoil_root: DT_F, q0: DT_F, cmc: DT_F, th2: DT_F, rho: DT_F, qs1: DT_F, ice: DT_F,
    prcp: DT_F, dqsdt2: DT_F, snowh: DT_F, sneqv: DT_F, chx: DT_F, cmx: DT_F, z0: DT_F,
    shdfac: DT_F, kdt: DT_F, frzx: DT_F, sndens: DT_F, snowng: DT_F, prcp1: DT_F,
    sncovr: DT_F, df1: DT_F, ssoil: DT_F, t2v: DT_F, fdown: DT_F, cpx1: DT_F,
    weasd_old: DT_F, snwdph_old: DT_F, tskin_old: DT_F, tprcp_old: DT_F, srflag_old: DT_F,
    canopy_old: DT_F, etp: DT_F,
    # output 2D
    sldpth: DT_F2, rtdis: DT_F2, smc_old: DT_F2, stc_old: DT_F2, slc_old: DT_F2,
    # input output
    ps: DT_F, t1: DT_F, q1: DT_F, soiltyp: DT_I, vegtype: DT_I, sigmaf: DT_F,
    sfcemis: DT_F, dlwflx: DT_F, dswsfc: DT_F, snet: DT_F, tg3: DT_F, cm: DT_F, ch: DT_F,
    prsl1: DT_F, prslki: DT_F, zf: DT_F, land: DT_I, wind: DT_F, slopetyp: DT_I,
    shdmin: DT_F, shdmax: DT_F, snoalb: DT_F, sfalb: DT_F, flag_iter: DT_I, flag_guess: DT_I,
    bexppert: DT_F, xlaipert: DT_F, vegfpert: DT_F, pertvegf: DT_F, fpvs: DT_F,
    weasd: DT_F, snwdph: DT_F, tskin: DT_F, tprcp: DT_F, srflag: DT_F, smc: DT_F2, stc: DT_F2, slc: DT_F2,
    canopy: DT_F, trans: DT_F, tsurf: DT_F, zorl: DT_F,
    sncovr1: DT_F, qsurf: DT_F, gflux: DT_F, drain: DT_F, evap: DT_F, hflx: DT_F, ep: DT_F, runoff: DT_F,
    cmm: DT_F, chh: DT_F, evbs: DT_F, evcw: DT_F, sbsno: DT_F, snowc: DT_F, stm: DT_F, snohf: DT_F,
    smcwlt2: DT_F, smcref2: DT_F, wet1: DT_F, delt: float, lheatstrg: int, ivegsrc: int,
    bexp: DT_F, dksat: DT_F, dwsat: DT_F, f1: DT_F, psisat: DT_F, quartz: DT_F, smcdry: DT_F, smcmax: DT_F, smcref: DT_F, smcwlt: DT_F,
    nroot: DT_I, snup: DT_F, rsmin: DT_F, rgl: DT_F, hs: DT_F, xlai: DT_F, slope: DT_F,
):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD):
        # calculate sldpth and rtdis from redprm
        with interval(-1, None):

            ### ------- PREPARATION ------- ###

            # set constant parameters
            rhoh2o = 1000.
            a2 = 17.2693882
            a3 = 273.16
            a4 = 35.86
            a23m4 = a2*(a3-a4)

            # save land-related prognostic fields for guess run
            if land and flag_guess:
                weasd_old = weasd
                snwdph_old = snwdph
                tskin_old = tskin
                canopy_old = canopy
                tprcp_old = tprcp
                srflag_old = srflag

            if flag_iter and land:
                # initialization block
                canopy = max(canopy, 0.0)

                q0 = max(q1, 1.e-8)
                theta1 = t1 * prslki
                rho = prsl1 / (rd * t1 * (1.0 + rvrdm1 * q0))
                qs1 = fpvs
                qs1 = max(eps * qs1 / (prsl1 + epsm1 * qs1), 1.e-8)

                q0 = min(qs1, q0)

                # noah: prepare variables to run noah lsm
                # configuration information
                ice = 0

                # forcing data
                prcp = rhoh2o * tprcp / delt
                dqsdt2 = qs1 * a23m4 / (t1 - a4)**2

                # history variables
                cmc = canopy * 0.001
                snowh = snwdph * 0.001
                sneqv = weasd * 0.001

                if((sneqv != 0.) and (snowh == 0.)):
                    snowh = 10.0 * sneqv

                chx = ch * wind
                cmx = cm * wind
                chh = chx * rho
                cmm = cmx

                z0 = zorl / 100.

            if land and flag_guess:
                smc_old = smc
                stc_old = stc
                slc_old = slc

            ### ------- END PREPARATION ------- ###

            if flag_iter and land:
                # root distribution
                zsoil_root = zsoil
                count = km - 1

                # thickness of each soil layer
                sldpth = zsoil[0, -1] - zsoil[0, 0]
                # root distribution
                if nroot <= count:
                    rtdis = 0.0
                    zsoil_root = zsoil[0, -1]
                else:
                    rtdis = - sldpth / zsoil_root

                # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
                # as a flag for non-soil medium
                sh2o = slc
                if ice != 0:
                    smc = 1.
                    sh2o = 1.

        with interval(1, -1):
            ### ------- PREPARATION ------- ###
            # save land-related prognostic fields for guess run
            if land and flag_guess:
                smc_old = smc
                stc_old = stc
                slc_old = slc
            ### ------- END PREPARATION ------- ###

            if flag_iter and land:
                # thickness of each soil layer
                sldpth = zsoil[0, -1] - zsoil[0, 0]
                count = count[0, 0, +1] - 1
                # root distribution
                if nroot <= count:
                    rtdis = 0.0
                    zsoil_root = zsoil[0, -1]
                else:
                    rtdis = - sldpth / zsoil_root

                # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
                # as a flag for non-soil medium
                sh2o = slc
                if ice != 0:
                    smc = 1.
                    sh2o = 1.

        with interval(0, 1):
            if flag_iter and land:
                # thickness of first soil layer
                sldpth = - zsoil[0, 0]
                # root distribution
                count = count[0, 0, +1] - 1
                if nroot <= count:
                    rtdis = 0.0
                else:
                    rtdis = - sldpth / zsoil_root

                ffrozp = srflag
                dt = delt
                sfctmp = t1
                q2 = q0
                sfcems = sfcemis
                sfcprs = prsl1
                th2 = theta1
                q2sat = qs1
                shdfac = sigmaf

                shdfac0 = shdfac

                if ivegsrc == 2 and vegtype == 12:
                    ice = -1
                    shdfac = 0.0

                if ivegsrc == 1 and vegtype == 14:
                    ice = -1
                    shdfac = 0.0

                kdt = refkdt * dksat / refdk

                frzfact = (smcmax / smcref) * (0.412 / 0.468)

                # to adjust frzk parameter to actual soil type: frzk * frzfact
                frzx = frzk * frzfact

                if vegtype + 1 == bare:
                    shdfac = 0.0

                if ivegsrc == 1 and vegtype == 12:
                    rsmin = 400.0*(1-shdfac0)+40.0*shdfac0
                    shdfac = shdfac0
                    smcmax = 0.45*(1-shdfac0)+smcmax*shdfac0
                    smcref = 0.42*(1-shdfac0)+smcref*shdfac0
                    smcwlt = 0.40*(1-shdfac0)+smcwlt*shdfac0
                    smcdry = 0.40*(1-shdfac0)+smcdry*shdfac0

                bexp = bexp * min(1. + bexppert, 2.)

                xlai = xlai * (1.+xlaipert)
                xlai = max(xlai, .75)

                # over sea-ice or glacial-ice, if s.w.e. (sneqv) below threshold
                # lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then
                # set at lower bound and store the source increment in subsurface
                # runoff/baseflow (runoff2).
                if (ice == -1) and (sneqv < 0.10):
                    # TODO: check if it is called
                    sneqv = 0.10
                    snowh = 1.00

                # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
                # as a flag for non-soil medium
                sh2o = slc
                if ice != 0:
                    smc = 1.
                    sh2o = 1.

                # if input snowpack is nonzero, then compute snow density "sndens"
                # and snow thermal conductivity "sncond" (note that csnow is a function subroutine)
                if sneqv == 0.:
                    sndens = 0.
                    snowh = 0.
                    sncond = 1.
                else:
                    sndens = sneqv / snowh
                    sndens = max(0.0, min(1.0, sndens))
                    # TODO: sncond is that necessary? is it later overwritten without using before?
                    sncond = csnow_fn(sndens)

                # determine if it's precipitating and what kind of precip it is.
                # if it's prcping and the air temp is colder than 0 c, it's snowing!
                # if it's prcping and the air temp is warmer than 0 c, but the grnd
                # temp is colder than 0 c, freezing rain is presumed to be falling.
                snowng = (prcp > 0.) and (ffrozp > 0.)
                frzgra = (prcp > 0.) and (ffrozp <= 0.) and (t1 <= tfreez)

                # if either prcp flag is set, determine new snowfall (converting
                # prcp rate from kg m-2 s-1 to a liquid equiv snow depth in meters)
                # and add it to the existing snowpack.

                # snowfall
                if snowng:
                    sn_new = ffrozp*prcp * dt * 0.001
                    sneqv = sneqv + sn_new
                    prcp1 = (1.-ffrozp)*prcp

                # freezing rain
                if frzgra:
                    sn_new = prcp * dt * 0.001
                    sneqv = sneqv + sn_new
                    prcp1 = 0.0

                if snowng or frzgra:

                    # update snow density based on new snowfall, using old and new
                    # snow.  update snow thermal conductivity
                    snowh, sndens = snow_new_fn(sfctmp, sn_new, snowh, sndens)
                    sncond = csnow_fn(sndens)

                else:
                    # precip is liquid (rain), hence save in the precip variable
                    # that later can wholely or partially infiltrate the soil (along
                    # with any canopy "drip" added to this later)
                    prcp1 = prcp

                # determine snowcover fraction and albedo fraction over land.
                if ice != 0:
                    sncovr = 1.0
                    albedo = 0.65    # albedo over sea-ice, glacial- ice

                else:
                    # non-glacial land
                    # if snow depth=0, set snowcover fraction=0, albedo=snow free albedo.
                    if sneqv == 0.:
                        sncovr = 0.
                        albedo = sfalb

                    else:
                        # determine snow fraction cover.
                        # determine surface albedo modification due to snowdepth state.
                        sncovr = snfrac_fn(sneqv, snup, salp)
                        albedo = alcalc_fn(sfalb, snoalb, sncovr)

                        # thermal conductivity for sea-ice case, glacial-ice case
                if ice != 0:
                    df1 = 2.2

                else:
                    # calculate the subsurface heat flux, which first requires calculation
                    # of the thermal diffusivity.
                    df1 = tdfcnd_fn(smc, quartz, smcmax, sh2o)
                    if ivegsrc == 1 and vegtype == 12:
                        df1 = 3.24*(1.-shdfac) + shdfac*df1*exp(sbeta*shdfac)
                    else:
                        df1 = df1 * exp(sbeta*shdfac)

                dsoil = -0.5 * zsoil

                if sneqv == 0.:
                    ssoil = df1 * (t1 - stc) / dsoil
                else:
                    dtot = snowh + dsoil
                    frcsno = snowh / dtot
                    frcsoi = dsoil / dtot

                    # arithmetic mean (parallel flow)
                    df1a = frcsno*sncond + frcsoi*df1

                    # geometric mean (intermediate between harmonic and arithmetic mean)
                    df1 = df1a*sncovr + df1 * (1.0-sncovr)

                    # calculate subsurface heat flux
                    ssoil = df1 * (t1 - stc) / dtot

                # calc virtual temps and virtual potential temps needed by
                # subroutines sfcdif and penman.
                t2v = sfctmp * (1.0 + 0.61*q2)

                # surface exchange coefficients computed externally and passed in,
                # hence subroutine sfcdif not called.
                fdown = snet + dlwflx

                # # enhance cp as a function of z0 to mimic heat storage
                cpx = cp
                cpx1 = cp1
                cpfac = 1.0

                t24, etp, rch, epsca, rr, flx2 = penman_fn(sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
                                                           cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra, ffrozp)


@gtscript.function
def devap_fn(etp1, smc, shdfac, smcmax, smcdry, fxexp):
    # --- ... subprograms called: none

    # calculates direct soil evaporation
    sratio = (smc - smcdry) / (smcmax - smcdry)

    if sratio > 0.0:
        fx = sratio**fxexp
        fx = max(min(fx, 1.0), 0.0)
    else:
        fx = 0.0

    # allow for the direct-evap-reducing effect of shade
    edir1 = fx * (1.0 - shdfac) * etp1
    return edir1


@gtscript.function
def transp_first_fn(nroot, smc, smcwlt, smcref, count, sgx):
    # initialize plant transp to zero for all soil layers.
    if nroot > count:
        gx = max(0.0, min(1.0, (smc-smcwlt)/(smcref - smcwlt)))
    else:
        gx = 0.

    sgx += gx / nroot

    return sgx, gx


@gtscript.function
def transp_second_fn(rtdis, gx, sgx, denom):
    # initialize plant transp to zero for all soil layers.
    rtx = rtdis + gx - sgx

    gx *= max(rtx, 0.0)

    denom += gx

    if denom <= 0.0:
        denom = 1.0

    return denom, gx


@gtscript.function
def transp_third_fn(etp1, cmc, cmcmax, shdfac, pc, cfactr, gx, denom):

    if cmc != 0.0:
        etp1a = shdfac * pc * etp1 * (1.0 - (cmc / cmcmax) ** cfactr)
    else:
        etp1a = shdfac * pc * etp1

    et1 = etp1a * gx / denom

    return et1


@gtscript.function
def evapo_boundarylayer_fn(nroot, cmc, cmcmax, etp1, dt,
                           sh2o, smcmax, smcwlt, smcref, smcdry,
                           shdfac, cfactr, fxexp, count, sgx):
    # --- ... subprograms called: devap, transp

    ec1 = 0.0
    edir1 = 0.0

    if etp1 > 0.0:
        # retrieve direct evaporation from soil surface.
        if shdfac < 1.0:
            edir1 = devap_fn(etp1, sh2o, shdfac, smcmax, smcdry, fxexp)

        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.:
            # calculates transpiration for the veg class

            # calculate canopy evaporation.
            if cmc > 0.0:
                ec1 = shdfac * ((cmc/cmcmax)**cfactr) * etp1
            else:
                ec1 = 0.0

            # ec should be limited by the total amount of available water on the canopy
            cmc2ms = cmc / dt
            ec1 = min(cmc2ms, ec1)

            sgx, gx = transp_first_fn(nroot, sh2o, smcwlt, smcref, count, sgx)

    return edir1, ec1, sgx, gx


@gtscript.function
def evapo_first_fn(nroot, etp1, sh2o, smcwlt, smcref, shdfac, count, sgx):
    # --- ... subprograms called: devap, transp

    if etp1 > 0.0:
        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.:
            # calculates transpiration for the veg class

            sgx, gx = transp_first_fn(nroot, sh2o, smcwlt, smcref, count, sgx)

    return sgx, gx


@gtscript.function
def evapo_second_fn(etp1, shdfac, rtdis, gx, sgx, denom):
    # --- ... subprograms called: devap, transp

    if etp1 > 0.0:
        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.:
            # calculates transpiration for the veg class

            denom, gx = transp_second_fn(rtdis, gx, sgx, denom)

    return denom, gx


@gtscript.function
def evapo_third_fn(etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1):
    # --- ... subprograms called: devap, transp
    et1 = 0.0

    if etp1 > 0.0:
        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.:
            # calculates transpiration for the veg class

            et1 = transp_third_fn(
                etp1, cmc, cmcmax, shdfac, pc, cfactr, gx, denom)

            ett1 += et1

    return et1, ett1


@gtscript.stencil(backend=BACKEND)
def canres(nroot: DT_I, dswsfc: DT_F, chx: DT_F, q0: DT_F, qs1: DT_F, dqsdt2: DT_F, t1: DT_F,
           cpx1: DT_F, prsl1: DT_F, sfcemis: DT_F, slc: DT_F2, smcwlt: DT_F, smcref: DT_F, zsoil: DT_FK, rsmin: DT_F,
           rgl: DT_F, hs: DT_F, xlai: DT_F, flag_iter: DT_I, land: DT_I, zsoil_root: DT_F, sigmaf: DT_F,
           # output
           rc: DT_F, pc: DT_F, rcs: DT_F, rct: DT_F, rcq: DT_F, rcsoil: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                sfcprs = prsl1
                q2sat = qs1
                shdfac = sigmaf
                ch = chx
                sh2o = slc
                sfctmp = t1
                q2 = q0

                rc = 0.0
                rcs = 0.0
                rct = 0.0
                rcq = 0.0
                rcsoil = 0.0
                pc = 0.0

                if shdfac > 0.0:
                    count = 0

                    if nroot > count:
                        gx = max(0.0, min(1.0, (sh2o-smcwlt)/(smcref - smcwlt)))
                        zsoil_root = zsoil

                    sum = zsoil[0, 0] * gx / zsoil_root

        with interval(1, None):
            if flag_iter and land:
                if shdfac > 0.0:
                    sh2o = slc
                    count = count[0, 0, -1] + 1

                    if nroot > count:
                        gx = max(0.0, min(1.0, (sh2o-smcwlt)/(smcref - smcwlt)))

                    sum = sum[0, 0, -1] + \
                        (zsoil[0, 0] - zsoil[0, -1]) * gx / zsoil_root

        with interval(-1, None):
            if flag_iter and land:
                if shdfac > 0.0:
                    rcsoil = max(sum, 0.001)

                    # contribution due to incoming solar radiation
                    ff = 0.55 * 2.0 * dswsfc / (rgl*xlai)
                    rcs = (ff + rsmin/rsmax) / (1.0 + ff)
                    rcs = max(rcs, 0.0001)

                    # contribution due to air temperature at first model level above ground
                    rct = 1.0 - 0.0016 * (topt - sfctmp)**2.0
                    rct = max(rct, 0.0001)

                    # contribution due to vapor pressure deficit at first model level.
                    rcq = 1.0 / (1.0 + hs*(q2sat-q2))
                    rcq = max(rcq, 0.01)

                    # determine canopy resistance due to all factors
                    rc = rsmin / (xlai*rcs*rct*rcq*rcsoil)
                    rr = (4.0*sfcemis*sigma1*rd1/cpx1) * \
                        (sfctmp**4.0)/(sfcprs*ch) + 1.0
                    delta = (lsubc/cpx1) * dqsdt2

                    pc = (rr + delta) / (rr*(1.0 + rc*ch) + delta)


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_first(nroot: DT_I, etp: DT_F, smcmax: DT_F, smcwlt: DT_F, smcref: DT_F,
                      smcdry: DT_F, delt: float, sigmaf: DT_F,
                      cmc: DT_F, slc: DT_F2,
                      flag_iter: DT_I, land: DT_I, sneqv: DT_F, count: DT_I,
                      # output
                      sgx: DT_F, gx: DT_F2, edir1: DT_F, ec1: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            shdfac = sigmaf
            sh2o = slc
            dt = delt

            if flag_iter and land:
                count = 0

                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    edir1 = 0.0
                    ec1 = 0.0

                    if etp > 0.0:
                        edir1, ec1, sgx, gx = evapo_boundarylayer_fn(nroot, cmc, cmcmax, etp1, dt, sh2o, smcmax, smcwlt,
                                                                     smcref, smcdry, shdfac, cfactr, fxexp, count, sgx)
        with interval(1, None):
            if flag_iter and land:
                shdfac = sigmaf
                sh2o = slc
                count += 1
                if sneqv == 0.0:
                    if etp > 0.0:
                        sgx, gx = evapo_first_fn(
                            nroot, etp1, sh2o, smcwlt, smcref, shdfac, count, sgx)


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_second(etp: DT_F, sigmaf: DT_F, slc: DT_F2, flag_iter: DT_I, land: DT_I,
                       sneqv: DT_F, count: DT_I, rtdis: DT_F2, sgx: DT_F,
                       # output
                       denom: DT_F, gx: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0.0
        with interval(...):
            if flag_iter and land:
                sh2o = slc
                shdfac = sigmaf
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)
                count += 1


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_third(etp: DT_F, sigmaf: DT_F, slc: DT_F2, flag_iter: DT_I, land: DT_I,
                      sneqv: DT_F, count: DT_I, gx: DT_F2, cmc: DT_F, pc: DT_F,
                      denom: DT_F, edir1: DT_F, ec1: DT_F,
                      # output
                      et1: DT_F2, ett1: DT_F, eta1: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0.0
        with interval(...):
            if flag_iter and land:
                shdfac = sigmaf
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                count += 1

        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    eta1 = edir1 + ett1 + ec1


@gtscript.function
def wdfcnd_fn(smc, smcmax, bexp, dksat, dwsat, sicemax):
    # calc the ratio of the actual to the max psbl soil h2o content of each layer
    factr = min(1.0, max(0.0, 0.2/smcmax))
    factr0 = min(1.0, max(0.0, smc/smcmax))

    # prep an expntl coef and calc the soil water diffusivity
    expon = bexp + 2.0
    wdf = dwsat * factr0 ** expon

    # frozen soil hydraulic diffusivity.
    if sicemax > 0.:
        vkwgt = 1.0 / (1.0 + (500.0*sicemax)**3.0)
        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr**expon

    # reset the expntl coef and calc the hydraulic conductivity
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * factr0 ** expon

    return wdf, wcnd


@gtscript.function
def srt_first_upperboundary_fn(sh2o, pcpdrp, zsoil, smcmax, smcwlt, sice, sicemax, dd, dice):

    sicemax = max(sice, 0.0)

    if pcpdrp != 0:
        # frozen ground version
        smcav = smcmax - smcwlt
        dd = -zsoil * (smcav - (sh2o + sice - smcwlt))
        dice = -zsoil * sice

    return sicemax, dd, dice


@gtscript.function
def srt_first_fn(sh2o, pcpdrp, zsoil, smcmax, smcwlt, sice, sicemax, dd, dice):

    sicemax = max(sice, sicemax)

    if pcpdrp != 0:
        # frozen ground version
        smcav = smcmax - smcwlt
        dd += (zsoil[0, -1] - zsoil[0, 0]) * (smcav - (sh2o + sice - smcwlt))
        dice += (zsoil[0, -1] - zsoil[0, 0]) * sice

    return sicemax, dd, dice


@gtscript.function
def srt_second_upperboundary_fn(edir, et, sh2o, pcpdrp, zsoil, dwsat,
                                dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice,
                                # output
                                rhstt, runoff1, ci):
    # determine rainfall infiltration rate and runoff
    cvfrz = 3

    pddum = pcpdrp
    runoff1 = 0.0

    if pcpdrp != 0:
        # frozen ground version
        dt1 = dt/86400.

        val = 1.0 - exp(-kdt*dt1)
        ddt = dd * val

        px = pcpdrp * dt

        if px < 0.0:
            px = 0.0

        infmax = (px*(ddt/(px+ddt)))/dt

        # reduction of infiltration based on frozen ground parameters
        fcr = 1.

        if dice > 1.e-2:
            acrt = cvfrz * frzx / dice
            ialp1 = cvfrz - 1  # = 2

            # Hardcode for ialp1 = 2
            sum = 1. + acrt**2. / 2. + acrt

            fcr = 1. - exp(-acrt) * sum

        infmax *= fcr

        wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

        infmax = max(infmax, wcnd)
        infmax = min(infmax, px)

        if pcpdrp > infmax:
            runoff1 = pcpdrp - infmax
            pddum = infmax

    wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-.5*zsoil[0, 1])
    ai = 0.0
    bi = wdf * ddz / (-zsoil)
    ci = -bi

    # calc rhstt for the top layer
    dsmdz = (sh2o[0, 0, 0] - sh2o[0, 0, 1]) / (-0.5*zsoil)
    rhstt = (wdf*dsmdz + wcnd - pddum + edir + et) / zsoil

    return rhstt, runoff1, ai, bi, ci, dsmdz, ddz, wdf, wcnd


@gtscript.function
def srt_second_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, sicemax,
                  # output
                  rhstt, ci, dsmdz, ddz, wdf, wcnd):

    wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

    # 2. Layer
    denom2 = zsoil[0, -1] - zsoil[0, 0]
    denom = zsoil[0, -1] - zsoil[0, +1]
    dsmdz = (sh2o[0, 0, 0] - sh2o[0, 0, +1])/(denom * 0.5)
    ddz = 2.0 / denom
    ci = - wdf * ddz / denom2
    slopx = 1.
    numer = wdf*dsmdz + slopx*wcnd - \
        wdf[0, 0, -1]*dsmdz[0, 0, -1] - wcnd[0, 0, -1] + et
    rhstt = -numer / denom2

    # calc matrix coefs
    ai = - wdf[0, 0, -1]*ddz[0, 0, -1] / denom2
    bi = -(ai + ci)

    return rhstt, ai, bi, ci, dsmdz, ddz, wdf, wcnd


@gtscript.function
def srt_second_lowerboundary_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, sicemax, slope,
                                # output
                                rhstt, ci, dsmdz, ddz, wdf, wcnd):

    wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

    # 2. Layer
    denom2 = zsoil[0, -1] - zsoil[0, 0]
    dsmdz = 0.0
    ci = 0.0
    slopx = slope
    numer = wdf*dsmdz + slopx*wcnd - \
        wdf[0, 0, -1]*dsmdz[0, 0, -1] - wcnd[0, 0, -1] + et
    rhstt = -numer / denom2

    # calc matrix coefs
    ai = - wdf[0, 0, -1]*ddz[0, 0, -1] / denom2
    bi = -(ai + ci)

    runoff2 = slope * wcnd

    return rhstt, ai, bi, ci, runoff2


@gtscript.function
def rosr12_first_upperboundary_fn(ai, bi, ci, d):
    p = -ci/bi
    delta = d/bi
    return p, delta


@gtscript.function
def rosr12_first_fn(ai, bi, ci, d, p, delta):
    p = - ci / (bi + ai * p[0, 0, -1])
    delta = (d - ai*delta[0, 0, -1])/(bi + ai*p[0, 0, -1])
    return p, delta


@gtscript.function
def sstep_upperboundary_fn(sh2o, smc, smcmax, sice, ci, sldpth):
    
    wplus = 0.
    
    sh2o = sh2o + ci + wplus/sldpth
    stot = sh2o + sice

    if stot > smcmax:
        wplus = (stot-smcmax)*sldpth
    else:
        wplus = 0.

    smc = max(min(stot, smcmax), 0.02)
    sh2o = max(smc-sice, 0.0)

    return wplus, smc, sh2o

@gtscript.function
def sstep_fn(sh2o, smc, smcmax, sice, ci, sldpth, wplus):

    sh2o = sh2o + ci + wplus[0,0,-1]/sldpth
    stot = sh2o + sice

    if stot > smcmax:
        wplus = (stot-smcmax)*sldpth
    else:
        wplus = 0.

    smc = max(min(stot, smcmax), 0.02)
    sh2o = max(smc-sice, 0.0)

    return wplus, smc, sh2o


@gtscript.function
def smflx_first_upperboundary_fn(dt, smcmax, smcwlt, cmcmax, prcp1, zsoil, ec1, cmc,
                                 sh2o, smc, shdfac, sicemax, dd, dice, pcpdrp, rhsct):
    # compute the right hand side of the canopy eqn term
    rhsct = shdfac*prcp1 - ec1

    drip = 0.0
    trhsct = dt * rhsct
    excess = cmc + trhsct

    if excess > cmcmax:
        drip = excess - cmcmax

    # pcpdrp is the combined prcp1 and drip (from cmc) that goes into the soil
    pcpdrp = (1.0 - shdfac)*prcp1 + drip/dt

    # store ice content at each soil layer before calling srt and sstep
    sice = smc-sh2o

    sicemax, dd, dice = srt_first_upperboundary_fn(
        sh2o, pcpdrp, zsoil, smcmax, smcwlt, sice, sicemax, dd, dice)

    return rhsct, drip, sice, sicemax, dd, dice, pcpdrp, sice


@gtscript.function
def smflx_first_fn(smcmax, smcwlt, zsoil, sh2o, smc, pcpdrp, sicemax,
                   dd, dice):
    # store ice content at each soil layer before calling srt and sstep
    sice = smc-sh2o

    sicemax, dd, dice = srt_first_fn(
        sh2o, pcpdrp, zsoil, smcmax, smcwlt, sice, sicemax, dd, dice)

    return sicemax, dd, dice, sice


@gtscript.function
def smflx_second_upperboundary_fn(edir, et, sh2o, pcpdrp, zsoil, dwsat,
                                  dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice,
                                  # output
                                  rhstt, runoff1, ci, dsmdz, ddz, wdf, wcnd):
    rhstt, runoff1, ai, bi, ci, dsmdz, ddz, wdf, wcnd = srt_second_upperboundary_fn(edir, et, sh2o, pcpdrp, zsoil, dwsat,
                                                                                    dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice,
                                                                                    rhstt, runoff1, ci)
    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhstt *= dt

    p, delta = rosr12_first_upperboundary_fn(ai, bi, ci, rhstt)

    return rhstt, ci, runoff1, dsmdz, ddz, wdf, wcnd, p, delta


@gtscript.function
def smflx_second_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax,
                    # output
                    rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta):
    rhstt, ai, bi, ci, dsmdz, ddz, wdf, wcnd = srt_second_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, sicemax,
                                                             rhstt, ci, dsmdz, ddz, wdf, wcnd)
    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhstt *= dt
    p, delta = rosr12_first_fn(ai, bi, ci, rhstt, p, delta)

    return rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta


@gtscript.function
def smflx_second_lowerboundary_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, slope,
                                  # output
                                  rhstt, ci, runoff2, dsmdz, ddz, wdf, wcnd, p, delta):
    rhstt, ai, bi, ci, runoff2 = srt_second_lowerboundary_fn(et, sh2o, zsoil, dwsat, dksat, smcmax, bexp, sicemax, slope,
                                                             rhstt, ci, dsmdz, ddz, wdf, wcnd)
    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhstt *= dt
    p, delta = rosr12_first_fn(ai, bi, ci, rhstt, p, delta)

    return rhstt, ci, runoff2, p, delta


@gtscript.stencil(backend=BACKEND)
def nopac_smflx_first(dt: float, smcmax: DT_F, smcwlt: DT_F, prcp1: DT_F, zsoil: DT_FK, shdfac: DT_F, ec1: DT_F,
                      cmc: DT_F, sh2o: DT_F2, smc: DT_F2, flag_iter: DT_I, land: DT_I, etp: DT_F, sneqv: DT_F,
                      sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, rhsct: DT_F, drip: DT_F, sice: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        rhsct, drip, sice, sicemax, dd, dice, pcpdrp, sice = smflx_first_upperboundary_fn(
                            dt, smcmax, smcwlt, cmcmax, prcp1, zsoil, ec1, cmc,
                            sh2o, smc, shdfac, sicemax, dd, dice, pcpdrp, rhsct)

        with interval(1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        sicemax, dd, dice, sice = smflx_first_fn(smcmax, smcwlt, zsoil, sh2o, smc, pcpdrp, sicemax,
                                                                 dd, dice)


@gtscript.stencil(backend=BACKEND)
def nopac_smflx_second(etp: DT_F, smcmax: DT_F, dt: float, bexp: DT_F, kdt: DT_F, frzx: DT_F, dksat: DT_F, dwsat: DT_F, slope: DT_F,
                       zsoil: DT_FK, sh2o: DT_F2, flag_iter: DT_I, land: DT_I, sneqv: DT_F, sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, edir1: DT_F, et1: DT_F2,
                       # outpus
                       rhstt: DT_F2, ci: DT_F2, runoff1: DT_F, runoff2: DT_F, dsmdz: DT_F2, ddz: DT_F2, wdf: DT_F2, wcnd: DT_F2, p: DT_F2, delta: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        rhstt, ci, runoff1, dsmdz, ddz, wdf, wcnd, p, delta = smflx_second_upperboundary_fn(edir1, et1, sh2o, pcpdrp, zsoil, dwsat,
                                                                                                            dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice,
                                                                                                            rhstt, runoff1, ci, dsmdz, ddz, wdf, wcnd)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta = smflx_second_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax,
                                                                                     rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta)

        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        rhstt, ci, runoff2, p, delta = smflx_second_lowerboundary_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, slope,
                                  rhstt, ci, runoff2, dsmdz, ddz, wdf, wcnd, p, delta)


@gtscript.stencil(backend=BACKEND)
def rosr12_second(p: DT_F2, delta: DT_F2, flag_iter: DT_I, land: DT_I, ci: DT_F2):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD):
        with interval(-1, None):
            if flag_iter and land:
                ci = delta

    with computation(BACKWARD):
        with interval(0, -1):
            if flag_iter and land:
                ci = p[0, 0, 0] * ci[0, 0, +1] + delta
        

@gtscript.stencil(backend=BACKEND)
def nopac_smflx_third(smcmax: DT_F, dt: float, flag_iter: DT_I, land: DT_I, sice: DT_F2, sldpth: DT_F2, rhsct: DT_F, ci: DT_F,
                       # outpus
                       sh2o: DT_F2, smc: DT_F2, cmc: DT_F, runoff3: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                wplus, smc, sh2o = sstep_upperboundary_fn(sh2o, smc, smcmax, sice, ci, sldpth)
        
        with interval(1, -1):
            if flag_iter and land:
                wplus, smc, sh2o = sstep_fn(sh2o, smc, smcmax, sice, ci, sldpth, wplus)
        
        with interval(-1, None):
            if flag_iter and land:
                runoff3 = wplus

                # update canopy water content/interception
                cmc += dt * rhsct
                if cmc < 1.e-20:
                    cmc = 0.0
                cmc = min(cmc, cmcmax)

@gtscript.stencil(backend=BACKEND)
def cleanup_sflx(
    flag_iter: DT_I, land: DT_I, eta: DT_F, sheat: DT_F, ssoil: DT_F, edir: DT_F, ec: DT_F, ett: DT_F, esnow: DT_F, sncovr: DT_F, soilm: DT_F,
    flx1: DT_F, flx2: DT_F, flx3: DT_F, smcwlt: DT_F, smcref: DT_F, etp: DT_F, smc0: DT_F, smcmax: DT_F, runoff1: DT_F, runoff2: DT_F,
    cmc: DT_F, snowh: DT_F, sneqv: DT_F, z0: DT_F, rho: DT_F, ch: DT_F, wind: DT_F, q1: DT_F, elocp: DT_F, cpinv: DT_F, hvapi: DT_F, flag_guess: DT_I,
    weasd_old: DT_F, snwdph_old: DT_F, tskin_old: DT_F, canopy_old: DT_F, tprcp_old: DT_F, srflag_old: DT_F,
    smc_old: DT_F2, slc_old: DT_F2, stc_old: DT_F2, tsurf: DT_F
    # TODO: Output
):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):

        if flag_iter and land:
            # output
            evap = eta
            hflx = sheat
            gflux = ssoil

            evbs = edir
            evcw = ec
            trans = ett
            sbsno = esnow
            snowc = sncovr
            stm = soilm * 1000.0
            snohf = flx1 + flx2 + flx3

            smcwlt2 = smcwlt
            smcref2 = smcref

            ep = etp
            wet1 = smc0 / smcmax

            runoff = runoff1 * 1000.0
            drain = runoff2 * 1000.0

            canopy = cmc * 1000.0
            snwdph = snowh * 1000.0
            weasd = sneqv * 1000.0
            sncovr1 = sncovr
            zorl = z0 * 100.

            # compute qsurf
            rch = rho * cp * ch * wind
            qsurf = q1 + evap / (hvap/cp * rch)
            tem = 1.0 / rho
            hflx = hflx * tem / cp
            evap = evap * tem / hvap

        # restore land-related prognostic fields for guess run
        if land and flag_guess:
            weasd = weasd_old
            snwdph = snwdph_old
            tskin = tskin_old
            canopy = canopy_old
            tprcp = tprcp_old
            srflag = srflag_old
            smc = smc_old
            stc = stc_old
            slc = slc_old
        elif land:
            tskin = tsurf
