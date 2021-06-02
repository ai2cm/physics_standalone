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

SCALAR_VARS = ["delt", "lheatstrg", "ivegsrc"]

SOIL_VARS = [bb, satdk, satdw, f11, satpsi,
             qtz, drysmc, maxsmc, refsmc, wltsmc]
VEG_VARS = [nroot_data, snupx, rsmtbl, rgltbl, hstbl, lai_data]
SOIL_VARS_NAMES = ["bexp", "dksat", "dwsat", "f1", "psisat",
                   "quartz", "smcdry", "smcmax", "smcref", "smcwlt"]
VEG_VARS_NAMES = ["nroot", "snup", "rsmin", "rgl", "hs", "xlai"]


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
        return gt.storage.from_array(data, backend=backend, shape=(arr.shape[0], 1), mask=(True,True,False), default_origin=(0, 0, 0))

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
    return gt.storage.from_array(data, backend=backend, shape=(new_arr.shape[0], 1), mask=(True,True,False), default_origin=(0, 0, 0))


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
    zsoil = gt.storage.from_array(zsoil, backend=backend, shape=(1, 4), mask=(True,False,True), default_origin=(0, 0, 0))

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

    # prepare for sflx
    zsoil_root = gt.storage.zeros(backend=BACKEND, dtype=np.float64, shape=(im,1), mask=(True,True,False), default_origin=(0,0,0))

    prepare_sflx(in_dict["t1"], in_dict["q1"], in_dict["cm"], in_dict["ch"], in_dict["prsl1"],
                 in_dict["prslki"], in_dict["zf"], in_dict["land"], in_dict["wind"], in_dict["flag_iter"], 
                 in_dict["flag_guess"], in_dict["fpvs"], zsoil, in_dict["sigmaf"], 
                 in_dict["vegtype"], table_dict["nroot"], out_dict["weasd"], out_dict["snwdph"], 
                 out_dict["tskin"], out_dict["tprcp"], out_dict["srflag"], out_dict["canopy"], out_dict["zorl"], 
                 out_dict["smc"], out_dict["stc"], out_dict["slc"], scalar_dict["delt"], scalar_dict["ivegsrc"], km)

    # run sflx algorithm 
    # TODO

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

@gtscript.stencil(backend=BACKEND)
def prepare_sflx(
    t1: DT_F, q1: DT_F, cm: DT_F, ch: DT_F, prsl1: DT_F, prslki: DT_F, 
    zf: DT_F, land: DT_I, wind: DT_F, flag_iter: DT_I, flag_guess: DT_I,
    fpvs: DT_F, zsoil: DT_FK, shdfac: DT_F, vegtyp: DT_I, nroot: DT_I,
    weasd: DT_F, snwdph: DT_F, tskin: DT_F, tprcp: DT_F, srflag: DT_F, 
    canopy: DT_F, zorl: DT_F, smc: DT_F2, stc: DT_F2, slc: DT_F2, delt: float, ivegsrc: int,
    km: int 
    # weasd_old: DT_F, snwdph_old: DT_F, tskin_old: DT_F, tprcp_old: DT_F, srflag_old: DT_F, 
    # canopy_old: DT_F, smc_old: DT_F2, stc_old: DT_F2, slc_old: DT_F2
):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD): 
        # calculate sldpth and rtdis from redprm
        with interval(-1, None):
            if flag_iter and land: 
                # root distribution
                zsoil_root = zsoil[0, 0]
                count = km - 1
        
        with interval(1, None):
            if flag_iter and land: 
                # thickness of each soil layer
                sldpth = zsoil[0,-1] - zsoil[0, 0]                   
                # root distribution
                if nroot <= count:
                    rtdis = 0.0
                    zsoil_root = zsoil[0, -1]
                else: 
                    rtdis = - sldpth / zsoil_root
                count -= 1

        with interval(...):
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
                smc_old = smc
                stc_old = stc
                slc_old = slc

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

                # initialization
                runoff1 = 0.
                runoff2 = 0.
                runoff3 = 0.
                snomlt = 0.

                pc = 0.

                shdfac0 = shdfac

                # is not called
                if ivegsrc == 2 and vegtyp == 12:
                    ice = -1
                    shdfac = 0.0

                if ivegsrc == 1 and vegtyp == 14:
                    ice = -1
                    shdfac = 0.0

        with interval(0, 1):
            if flag_iter and land: 
                # thickness of first soil layer
                sldpth = zsoil[0, -1] - zsoil[0, 0]
                # root distribution 
                if nroot <= count:
                    rtdis = 0.0
                else: 
                    rtdis = - sldpth / zsoil_root




@gtscript.stencil(backend=BACKEND)
def cleanup_sflx(
    flag_iter:DT_I, land:DT_I, eta: DT_F, sheat: DT_F, ssoil: DT_F, edir: DT_F, ec: DT_F, ett: DT_F, esnow: DT_F, sncovr: DT_F, soilm: DT_F, 
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

