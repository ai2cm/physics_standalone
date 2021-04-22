#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit

import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *

DT_F = gtscript.Field[np.float64]
DT_I = gtscript.Field[np.int32]

OUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy",
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx",
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf",
            "smcwlt2", "smcref2", "wet1"]

IN_VARS = ["ps", "t1", "q1", "soiltyp", "vegtype", "sigmaf",
           "sfcemis", "dlwflx", "dswsfc", "snet", "tg3", "cm",
           "ch", "prsl1", "prslki", "zf", "land", "wind",
           "slopetyp", "shdmin", "shdmax", "snoalb", "sfalb", "flag_iter", "flag_guess",
           "bexppert", "xlaipert", "vegfpert", "pertvegf",
           "weasd", "snwdph", "tskin", "tprcp", "srflag", "smc0", "smc1", "smc2", "smc3",
           "stc0", "stc1", "stc2", "stc3", "slc0", "slc1", "slc2", "slc3", "canopy",
           "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx",
           "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf",
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
    scalar_dict = {**{k: in_dict[k] for k in SCALAR_VARS}, **{k: in_dict2[k] for k in SCALAR_VARS}}
    out_dict = {k: numpy_to_gt4py_storage(
        in_dict[k].copy(), backend=backend) for k in OUT_VARS}
    in_dict = {**{k: numpy_to_gt4py_storage(
        in_dict[k], backend=backend) for k in IN_VARS}, **{k: numpy_to_gt4py_storage(
        in_dict2[k], backend=backend) for k in IN_VARS}}

    # compile stencil
    sfc_drv = gtscript.stencil(
        definition=sfc_drv_defs, backend=backend, externals={})

    # set timer
    tic = timeit.default_timer()

    # call sea-ice parametrization
    sfc_drv(**in_dict, **out_dict, **scalar_dict)

    # set timer
    toc = timeit.default_timer()

    # calculate elapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {k: gt4py_to_numpy_storage(
        out_dict[k], backend=backend) for k in OUT_VARS}

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
    smcwlt2: DT_F, smcref2: DT_F, wet1: DT_F, tbpvs: DT_F,
    c1xpvs: float, c2xpvs: float, delt: float, lheatstrg: float, ivegsrc: float
):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):

        # set constant parameters
        cpinv = 1. / cp
        hvapi = 1./hvap
        elocp = hvap/cp
        rhoh2o = 1000.
        a2 = 17.2693882
        a3 = 273.16
        a4 = 35.86
        a23m4 = a2*(a3-a4)
        # TODO: find alternative for using 4 scalars
        zsoil_noah0 = -0.1
        zsoil_noah1 = -0.4
        zsoil_noah2 = -1.0
        zsoil_noah3 = -2.0

        # Fortran starts with one, python with zero
        vegtype -= 1
        soiltyp -= 1

        # save land-related prognostic fields for guess run
        if land & flag_guess:
            weasd_old = weasd
            snwdph_old = snwdph
            tskin_old = tskin
            canopy_old = canopy
            tprcp_old = tprcp
            srflag_old = srflag
            smc_old0 = smc0
            smc_old1 = smc1
            smc_old2 = smc2
            smc_old3 = smc3
            stc_old0 = stc0
            stc_old1 = stc1
            stc_old2 = stc2
            stc_old3 = stc3
            slc_old0 = slc0
            slc_old1 = slc1
            slc_old2 = slc2
            slc_old3 = slc3

        if flag_iter & land:
            # initialization block
            ep = 0.0
            evap = 0.0
            hflx = 0.0
            gflux = 0.0
            drain = 0.0
            canopy = max(canopy, 0.0)

            evbs = 0.0
            evcw = 0.0
            trans = 0.0
            sbsno = 0.0
            snowc = 0.0
            snohf = 0.0

            q0 = max(q1, 1.e-8)
            theta1 = t1 * prslki
            rho = prsl1 / (rd * t1 * (1.0 + rvrdm1 * q0))
            qs1 = fpvs_fn(c1xpvs, c2xpvs, tbpvs, t1)
            qs1 = max(eps * qs1 / (prsl1 * epsm1 * qs1), 1.e-8)
            q0 = min(qs1, q0)

            # noah: prepare variables to run noah lsm
            # configuration information
            couple = 1
            ice = 0

            # forcing data
            prcp = rhoh2o * tprcp / delt
            dqsdt2 = qs1 * a23m4 / (t1 - a4)**2

            # canopy characteristics
            ptu = 0.0

            # history variables
            cmc = canopy * 0.001
            snowh = snwdph * 0.001
            sneqv = weasd * 0.001

            if((sneqv != 0.) & (snowh == 0.)):
                snowh = 10.0 * sneqv

            chx = ch * wind
            cmx = cm * wind
            chh = chx * rho
            cmm = cmx

            z0 = zorl / 100.

            tg3, cmc, tsurf, stc0, stc1, stc2, stc3, smc0, smc1, smc2, smc3, \
            slc0, slc1, slc2, slc3, sneqv, chx, cmx, z0, sigmaf, snowh, \
            evap, hflx, evcw, evbs, trans, sbsno, ep, gflux, \
                flx1, flx2, flx3, runoff1, runoff2, snowc, \
                soilm, smcwlt, smcref, smcmax = sflx_fn(  # inputs
                    couple, ice, srflag, delt, zf, zsoil_noah0, zsoil_noah1, zsoil_noah2, zsoil_noah3,
                    dswsfc, snet, dlwflx, sfcemis, prsl1, t1,
                    wind, prcp, q0, qs1, dqsdt2, theta1, ivegsrc,
                    vegtype, soiltyp, slopetyp, shdmin, sfalb, snoalb,
                    bexppert, xlaipert, lheatstrg,
                    tg3, cmc, tsurf, stc0, stc1, stc2, stc3, smc0, smc1, smc2, smc3,
                    slc0, slc1, slc2, slc3, sneqv, chx, cmx, z0, sigmaf, snowh
                )

            # output
            stm = soilm * 1000.0
            snohf = flx1 + flx2 +flx3
            wet1 = smc0 / smcmax
            runoff = runoff1 * 1000.0
            drain = runoff2 * 1000.0
            canopy = cmc * 1000.0
            snwdph = snowh * 1000.0
            weasd = sneqv * 1000.0
            sncovr1 = snowc
            zorl = z0 * 100.

            # compute qsurf
            rch = rho * cp * ch * wind
            qsurf = q1 * evap / (elocp * rch)
            tem = 1.0 / rho
            hflx = hflx * tem * cpinv
            evap = evap * tem * hvapi

        # restore land-related prognostic fields for guess run
        if land & flag_guess:
            weasd = weasd_old
            snwdph = snwdph_old
            tskin = tskin_old
            canopy = canopy_old
            tprcp = tprcp_old
            srflag = srflag_old
            smc0 = smc_old0
            smc1 = smc_old1
            smc2 = smc_old2
            smc3 = smc_old3
            stc0 = stc_old0
            stc1 = stc_old1
            stc2 = stc_old2
            stc3 = stc_old3
            slc0 = slc_old0
            slc1 = slc_old1
            slc2 = slc_old2
            slc3 = slc_old3
        elif land:
            tskin = tsurf


@gtscript.function
def sflx_fn(
    couple, ice, ffrozp, dt, zlvl, sldpth0, sldpth1, sldpth2, sldpth3,
    swdn, swnet, lwdn, sfcems, sfcprs, sfctmp,
    sfcspd, prcp, q2, q2sat, dqsdt2, th2, ivegsrc,
    vegtyp, soiltyp, slopetyp, shdmin, alb, snoalb,
    bexpp, xlaip, lheatstrg,
    tbot, cmc, t1, stc0, stc1, stc2, stc3, smc0, smc1, smc2, smc3, 
    sh2o0, sh2o1, sh2o2, sh2o3, sneqv, ch, cm, z0,
    shdfac, snowh
):
    return tbot, cmc, t1, stc0, stc1, stc2, stc3, smc0, smc1, smc2, smc3, \
    sh2o0, sh2o1, sh2o2, sh2o3, sneqv, ch, cm, z0,\
    shdfac, snowh

# TODO: how to handle access of tbpvs
@gtscript.function
def fpvs_fn(c1xpvs, c2xpvs, tbpvs, t):
    nxpvs = 7501.
    xj = np.minimum(np.maximum(c1xpvs+c2xpvs*t, 1.), nxpvs)
    jx = np.minimum(xj, nxpvs - 1.).astype(int)
    fpvs = tbpvs[jx-1]+(xj-jx)*(tbpvs[jx]-tbpvs[jx-1])

    return fpvs
