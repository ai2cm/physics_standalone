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
    # --- ... subprograms called: redprm, snow_new, csnow, snfrac, alcalc, tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.

    # initialization
    runoff1 = 0.
    runoff2 = 0.
    runoff3 = 0.
    snomlt = 0.

    pc = 0.

    shdfac0 = shdfac

    
    if ivegsrc == 2 and vegtyp == 12:
        # not called
        ice = -1
        shdfac = 0.0

    if ivegsrc == 1 and vegtyp == 14:
        ice = -1
        shdfac = 0.0

    # calculate depth (negative) below ground
    zsoil0 = - sldpth0
    zsoil1 = zsoil0 - sldpth1
    zsoil2 = zsoil1 - sldpth2
    zsoil3 = zsoil2 - sldpth3

    cfactr, cmcmax, rsmin, rsmax, topt, refkdt, kdt,\
        sbeta, shdfac, rgl, hs, zbot, frzx, psisat, slope,\
        snup, salp, bexp, dksat, dwsat, smcmax, smcwlt,\
        smcref, smcdry, f1, quartz, fxexp, rtdis, nroot, \
        czil, xlai, csoil = redprm_fn(vegtyp, soiltyp, slopetyp, sldpth0, zsoil0, shdfac)

    if ivegsrc == 1 and vegtyp == 12:
        rsmin = 400.0 * (1 - shdfac0) + 40.0 * shdfac0
        shdfac = shdfac0
        smcmax = 0.45 * (1 - shdfac0) + smcmax * shdfac0
        smcref = 0.42 * (1 - shdfac0) + smcref * shdfac0
        smcwlt = 0.40 * (1 - shdfac0) + smcwlt * shdfac0
        smcdry = 0.40 * (1 - shdfac0) + smcdry * shdfac0

    if bexpp < 0.:
        # not called
        print("ERROR: case not implemented")
    else:
        bexp = bexp * min(1. + bexpp, 2.)

    xlai = xlai * (1.+xlaip)
    xlai = max(xlai, .75)

    # over sea-ice or glacial-ice, if s.w.e. (sneqv) below threshold
    # lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then
    # set at lower bound and store the source increment in subsurface
    # runoff/baseflow (runoff2).
    if ice == 1:
        print("ERROR: case not implemented")

    elif (ice == -1) and (sneqv < 0.10):
        # TODO: check if it is called
        sneqv = 0.10
        snowh = 1.00

    # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
    # as a flag for non-soil medium
    if ice != 0:
        smc0 = 1.
        smc1 = 1.
        smc2 = 1.
        smc3 = 1.
        sh2o0 = 1.
        sh2o1 = 1.
        sh2o2 = 1.
        sh2o3 = 1.

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
        sncond = csnow(sndens)

    # determine if it's precipitating and what kind of precip it is.
    # if it's prcping and the air temp is colder than 0 c, it's snowing!
    # if it's prcping and the air temp is warmer than 0 c, but the grnd
    # temp is colder than 0 c, freezing rain is presumed to be falling.
    snowng = (prcp > 0.) & (ffrozp > 0.)
    frzgra = (prcp > 0.) & (ffrozp <= 0.) & (t1 <= tfreez)

    # if either prcp flag is set, determine new snowfall (converting
    # prcp rate from kg m-2 s-1 to a liquid equiv snow depth in meters)
    # and add it to the existing snowpack.

    # snowfall
    if snowng:
        sn_new = ffrozp * prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcp1 = (1.-ffrozp) * prcp

    # freezing rain
    if frzgra:
        sn_new = prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcp1 = 0.0

    if snowng or frzgra:

        # update snow density based on new snowfall, using old and new
        # snow.  update snow thermal conductivity
        snowh, sndens = snow_new(sfctmp, sn_new, snowh, sndens)
        sncond = csnow(sndens)

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
            albedo = alb

        else:
            # determine snow fraction cover.
            # determine surface albedo modification due to snowdepth state.
            sncovr = snfrac(sneqv, snup, salp)
            albedo = alcalc(alb, snoalb, sncovr)

    # thermal conductivity for sea-ice case, glacial-ice case
    if ice != 0:
        df1 = 2.2

    else:
        # calculate the subsurface heat flux, which first requires calculation
        # of the thermal diffusivity.
        df1 = tdfcnd(smc[0], quartz, smcmax, sh2o[0])
        if ivegsrc == 1 and vegtyp == 12:
            df1 = 3.24 * (1. - shdfac) + shdfac * df1 * exp(sbeta*shdfac)
        else:
            df1 = df1 * exp(sbeta * shdfac)

    dsoil = -0.5 * zsoil0

    if sneqv == 0.0:
        ssoil = df1 * (t1 - stc0) / dsoil
    else:
        dtot = snowh + dsoil
        frcsno = snowh / dtot
        frcsoi = dsoil / dtot

        # arithmetic mean (parallel flow)
        df1a = frcsno*sncond + frcsoi*df1

        # geometric mean (intermediate between harmonic and arithmetic mean)
        df1 = df1a*sncovr + df1 * (1.0 - sncovr)

        # calculate subsurface heat flux
        ssoil = df1 * (t1 - stc0) / dtot

    # determine surface roughness over snowpack using snow condition
    # from the previous timestep.
    if sncovr > 0.:
        z0 = snowz0(sncovr, z0)

    # calc virtual temps and virtual potential temps needed by
    # subroutines sfcdif and penman.
    t2v = sfctmp * (1.0 + 0.61 * q2)

    # next call routine sfcdif to calculate the sfc exchange coef (ch)
    # for heat and moisture.
    if couple == 0:  # uncoupled mode
        # not called
        print("ERROR: case not implemented")

    else:  # coupled mode

        # surface exchange coefficients computed externally and passed in,
        # hence subroutine sfcdif not called.
        fdown = swnet + lwdn

    # enhance cp as a function of z0 to mimic heat storage
    cpx = cp
    cpx1 = cp1
    cpfac = 1.0
    if lheatstrg and ((ivegsrc == 1 and vegtyp != 13) or ivegsrc == 2):
        # xx1 = (z0 - z0min) / (z0max - z0min)
        # xx2 = 1.0 + min(max(xx1, 0.0), 1.0)
        # cpx = cp * xx2
        # cpx1 = cp1 * xx2
        # cpfac = cp / cpx

        # not called
        print("ERROR: case not implemented")

    # call penman subroutine to calculate potential evaporation (etp),
    # and other partial products and sums save in common/rite for later
    # calculations.
    t24, etp, rch, epsca, rr, flx2 = penman(sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
                                            cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra, ffrozp)

    # call canres to calculate the canopy resistance and convert it
    # into pc if nonzero greenness fraction

    # TODO: check what happens to rc when shdfac <= 0.
    rc = 0.
    rcs = 0.
    rct = 0.
    rcq = 0.
    rcsoil = 0.

    if shdfac > 0.:

        # frozen ground extension: total soil water "smc" was replaced
        # by unfrozen soil water "sh2o" in call to canres below
        rc, pc, rcs, rct, rcq, rcsoil = canres(nsoil, nroot, swdn, ch, q2, q2sat, dqsdt2, sfctmp,
                                               cpx1, sfcprs, sfcems, sh2o, smcwlt, smcref, zsoil, rsmin,
                                               rsmax, topt, rgl, hs, xlai)

    # now decide major pathway branch to take depending on whether
    # snowpack exists or not:
    esnow = 0.

    if sneqv == 0.:
        cmc, t1, stc, sh2o, tbot, \
            eta, smc, ssoil, runoff1, runoff2, runoff3, edir, \
            ec, et, ett, beta, drip, dew, flx1, flx3 = nopac(nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref,
                                                             smcdry, cmcmax, dt, shdfac, sbeta, sfctmp, sfcems,
                                                             t24, th2, fdown, epsca, bexp, pc, rch, rr, cfactr,
                                                             slope, kdt, frzx, psisat, zsoil, dksat, dwsat,
                                                             zbot, ice, rtdis, quartz, fxexp, csoil, ivegsrc, vegtyp,
                                                             cmc, t1, stc, sh2o, tbot, smc)

    else:
        prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh, sh2o, tbot, \
            smc, ssoil, runoff1, runoff2, runoff3, edir, ec, et, \
            ett, snomlt, drip, dew, flx1, flx3, esnow = snopac(nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref, smcdry,
                                                               cmcmax, dt, df1, sfcems, sfctmp, t24, th2, fdown, epsca,
                                                               bexp, pc, rch, rr, cfactr, slope, kdt, frzx, psisat,
                                                               zsoil, dwsat, dksat, zbot, shdfac, ice, rtdis, quartz,
                                                               fxexp, csoil, flx2, snowng, ffrozp, ivegsrc, vegtyp,
                                                               prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh,
                                                               sh2o, tbot, smc)

    # prepare sensible heat (h) for return to parent model
    sheat = -(ch*cp1*sfcprs) / (rd1*t2v) * (th2 - t1)

    # convert units and/or sign of total evap (eta), potential evap (etp),
    # subsurface heat flux (s), and runoffs for what parent model expects
    # convert eta from kg m-2 s-1 to w m-2
    edir = edir * lsubc
    ec = ec * lsubc
    et = et * lsubc

    ett = ett * lsubc
    esnow = esnow * lsubs
    etp = etp * ((1.0 - sncovr)*lsubc + sncovr*lsubs)

    if etp > 0.:
        eta = edir + ec + ett + esnow
    else:
        eta = etp

    beta = eta / etp

    # convert the sign of soil heat flux so that:
    # ssoil>0: warm the surface  (night time)
    # ssoil<0: cool the surface  (day time)
    ssoil = -1.0 * ssoil

    if ice == 0:
        # for the case of land (but not glacial-ice):
        # convert runoff3 (internal layer runoff from supersat) from m
        # to m s-1 and add to subsurface runoff/baseflow (runoff2).
        # runoff2 is already a rate at this point.
        runoff3 = runoff3 / dt
        runoff2 = runoff2 + runoff3

    else:
        # for the case of sea-ice (ice=1) or glacial-ice (ice=-1), add any
        # snowmelt directly to surface runoff (runoff1) since there is no
        # soil medium, and thus no call to subroutine smflx (for soil
        # moisture tendency).
        runoff1 = snomlt / dt

    # total column soil moisture in meters (soilm) and root-zone
    # soil moisture availability (fraction) relative to porosity/saturation
    zsoil_dif = np.append(-zsoil[0], zsoil[:nsoil-1]-zsoil[1:])
    soilm = np.sum(smc*zsoil_dif)
    soilwm = (smcmax-smcwlt) * np.sum(zsoil_dif[:nroot])
    soilww = np.sum((smc[:nroot] - smcwlt) * zsoil_dif[:nroot])
    soilw = soilww / soilwm

    return tbot, cmc, t1, stc0, stc1, stc2, stc3, smc0, smc1, smc2, smc3, \
        sh2o0, sh2o1, sh2o2, sh2o3, sneqv, ch, cm, z0, shdfac, snowh, \
        nroot, albedo, eta, sheat, ec, \
        edir, et, ett, esnow, drip, dew, beta, etp, ssoil, \
        flx1, flx2, flx3, runoff1, runoff2, runoff3, \
        snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq, \
        rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax


# TODO: how to handle access of tbpvs
@gtscript.function
def fpvs_fn(c1xpvs, c2xpvs, tbpvs, t):
    nxpvs = 7501.
    xj = min(max(c1xpvs+c2xpvs*t, 1.), nxpvs)
    jx = min(xj, nxpvs - 1)
    fpvs = tbpvs[jx-1]+(xj-jx)*(tbpvs[jx]-tbpvs[jx-1])

    return fpvs


# *************************************
# 1st level subprograms
# *************************************


def alcalc(
    alb, snoalb, sncovr
):
    # --- ... subprograms called: none

    # calculates albedo using snow effect
    # snoalb: max albedo over deep snow
    albedo = alb + sncovr*(snoalb - alb)
    if (albedo > snoalb):
        albedo = snoalb

    return albedo


def canres(
    nsoil, nroot, swdn, ch, q2, q2sat, dqsdt2, sfctmp,
    cpx1, sfcprs, sfcems, sh2o, smcwlt, smcref, zsoil, rsmin,
    rsmax, topt, rgl, hs, xlai
):
    # --- ... subprograms called: none

    # calculates canopy resistance

    # contribution due to incoming solar radiation
    ff = 0.55 * 2.0 * swdn / (rgl*xlai)
    rcs = (ff + rsmin/rsmax) / (1.0 + ff)
    rcs = max(rcs, 0.0001)

    # contribution due to air temperature at first model level above ground
    rct = 1.0 - 0.0016 * (topt - sfctmp)**2.0
    rct = max(rct, 0.0001)

    # contribution due to vapor pressure deficit at first model level.
    rcq = 1.0 / (1.0 + hs*(q2sat-q2))
    rcq = max(rcq, 0.01)

    # contribution due to soil moisture availability.
    gx = np.maximum(0.0, np.minimum(
        1.0, (sh2o[:nroot] - smcwlt) / (smcref - smcwlt)))

    # use soil depth as weighting factor
    # TODO: check if only until nroot which is 3
    part = np.empty(nroot)
    part[0] = (zsoil[0]/zsoil[nroot-1]) * gx[0]
    part[1:] = ((zsoil[1:nroot] - zsoil[:nroot-1])/zsoil[nroot-1]) * gx[1:]

    rcsoil = max(np.sum(part), 0.0001)

    # determine canopy resistance due to all factors
    rc = rsmin / (xlai*rcs*rct*rcq*rcsoil)
    rr = (4.0*sfcems*sigma1*rd1/cpx1) * (sfctmp**4.0)/(sfcprs*ch) + 1.0
    delta = (lsubc/cpx1) * dqsdt2

    pc = (rr + delta) / (rr*(1.0 + rc*ch) + delta)

    return rc, pc, rcs, rct, rcq, rcsoil


def csnow(sndens):
    # --- ... subprograms called: none

    unit = 0.11631

    c = 0.328 * 10**(2.25*sndens)
    sncond = unit * c

    return sncond


def nopac(
    # inputs
    nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref,
    smcdry, cmcmax, dt, shdfac, sbeta, sfctmp, sfcems,
    t24, th2, fdown, epsca, bexp, pc, rch, rr, cfactr,
    slope, kdt, frzx, psisat, zsoil, dksat, dwsat,
    zbot, ice, rtdis, quartz, fxexp, csoil, ivegsrc, vegtyp,
    # in/outs
    cmc, t1, stc, sh2o, tbot, smc
):
    # --- ... subprograms called: evapo, smflx, tdfcnd, shflx

    # convert etp from kg m-2 s-1 to ms-1 and initialize dew.
    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    dew = 0.0
    edir = 0.0
    edir1 = 0.0
    ec = 0.0
    ec1 = 0.0

    et = init_array(nsoil, "zero")
    et1 = init_array(nsoil, "zero")

    ett = 0.
    ett1 = 0.

    if etp > 0.:
        eta1, edir1, ec1, et1, ett1 = evapo(nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil,
                                            sh2o, smcmax, smcwlt, smcref, smcdry, pc,
                                            shdfac, cfactr, rtdis, fxexp)

        cmc, sh2o, smc, runoff1, runoff2, runoff3, drip = smflx(nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1,
                                                                zsoil, slope, frzx, bexp, dksat, dwsat, shdfac,
                                                                edir1, ec1, et1, cmc, sh2o, smc)

    else:
        # if etp < 0, assume dew forms
        eta1 = 0.0
        dew = -etp1
        prcp1 += dew

        cmc, sh2o, smc, runoff1, runoff2, runoff3, drip = smflx(nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1,
                                                                zsoil, slope, frzx, bexp, dksat, dwsat, shdfac,
                                                                edir1, ec1, et1, cmc, sh2o, smc)

    # convert modeled evapotranspiration fm  m s-1  to  kg m-2 s-1
    eta = eta1 * 1000.0
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et = et1 * 1000.0
    ett = ett1 * 1000.0

    # based on etp and e values, determine beta
    if etp < 0.0:
        beta = 1.0
    elif etp == 0.0:
        beta = 0.0
    else:
        beta = eta / etp

    # get soil thermal diffuxivity/conductivity for top soil lyr, calc.
    df1 = tdfcnd(smc[0], quartz, smcmax, sh2o[0])

    if (ivegsrc == 1) and (vegtyp == 12):
        df1 = 3.24*(1.-shdfac) + shdfac*df1*np.exp(sbeta*shdfac)
    else:
        df1 *= np.exp(sbeta*shdfac)

    # compute intermediate terms passed to routine hrt
    yynum = fdown - sfcems*sigma1*t24
    yy = sfctmp + (yynum/rch + th2 - sfctmp - beta*epsca)/rr
    zz1 = df1/(-0.5*zsoil[0]*rch*rr) + 1.0

    ssoil, stc, t1, tbot, sh2o = shflx(nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot,
                                       psisat, bexp, df1, ice, quartz, csoil, ivegsrc, vegtyp, shdfac,
                                       stc, t1, tbot, sh2o)

    flx1 = 0.0
    flx3 = 0.0

    return cmc, t1, stc, sh2o, tbot, eta, smc, ssoil, runoff1, runoff2, runoff3, edir, \
        ec, et, ett, beta, drip, dew, flx1, flx3


def penman(sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
           cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra, ffrozp):
    # --- ... subprograms called: none

    flx2 = 0.

    # prepare partial quantities for penman equation.
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

@gtscript.function
def redprm_fn(vegtyp, soiltyp, slopetyp, sldpth, zsoil, shdfac):
    # --- ... subprograms called: none

    # if soiltyp > defined_soil:
    #     print("warning: too many soil types, soiltyp =",
    #           soiltyp, "defined_soil = ", defined_soil)

    # if vegtyp > defined_veg:
    #     print("warning: too many veg types")

    # if slopetyp > defined_slope:
    #     print("warning: too many slope types")

    # set-up soil parameters
    bexp = bb[soiltyp]
    dksat = satdk[soiltyp]
    dwsat = satdw[soiltyp]
    f1 = f11[soiltyp]
    kdt = refkdt * dksat / refdk

    psisat = satpsi[soiltyp]
    quartz = qtz[soiltyp]
    smcdry = drysmc[soiltyp]
    smcmax = maxsmc[soiltyp]
    smcref = refsmc[soiltyp]
    smcwlt = wltsmc[soiltyp]

    frzfact = (smcmax / smcref) * (0.412 / 0.468)

    # to adjust frzk parameter to actual soil type: frzk * frzfact
    frzx = frzk * frzfact

    # set-up vegetation parameters
    nroot = nroot_data[vegtyp]
    snup = snupx[vegtyp]
    rsmin = rsmtbl[vegtyp]
    rgl = rgltbl[vegtyp]
    hs = hstbl[vegtyp]
    xlai = lai_data[vegtyp]

    if vegtyp + 1 == bare:
        shdfac = 0.0

    # if nroot > nsoil:
    #     print("warning: too many root layers")

    # calculate root distribution.  present version assumes uniform
    # distribution based on soil layer depths.
    # TODO: check if rtdis size of nsoil is needed or rather just
    rtdis = init_array(nsoil, "zero")
    rtdis[:nroot] = - sldpth[:nroot] / zsoil[nroot-1]

    # set-up slope parameter
    slope = slope_data[slopetyp]

    return cfactr, cmcmax, rsmin, rsmax, topt, refkdt, kdt, \
        sbeta, shdfac, rgl, hs, zbot, frzx, psisat, slope, \
        snup, salp, bexp, dksat, dwsat, smcmax, smcwlt, \
        smcref, smcdry, f1, quartz, fxexp, rtdis, nroot, \
        czil, xlai, csoil \



def sfcdif(
    # inputs
    zlvl, z0, t1v, th2v, sfcspd, czil,
    # in/outs
    cm, ch
):
    # --- ... subprograms called: none

    # sfcdif

    # itrmx  = 5
    # wwst   = 1.2
    # wwst2  = wwst*wwst
    # vkrm   = 0.40
    # excm   = 0.001
    # beta   = 1.0/270.0
    # btg    = beta*gs1
    # elfc   = vkrm*btg
    # wold   = 0.15
    # wnew   = 1.0-wold
    # pihf   = 3.14159265/2.0

    # epsu2  = 1.e-4
    # epsust = 0.07
    # ztmin  = -5.0
    # ztmax  = 1.0
    # hpbl   = 1000.0
    # sqvisc = 258.2

    # ric    = 0.183
    # rric   = 1.0/ric
    # fhneu  = 0.8
    # rfc    = 0.191
    # rfac   = ric/(fhneu*rfc*rfc)

    # # calculates surface layer exchange coefficients

    # # 1. lech's surface functions
    # def pslmu(zz): return -0.96 * np.log(1.0-4.5*zz)
    # def pslms(zz): return zz*rric - 2.076*(1.0 - 1.0/(zz + 1.0))
    # def pslhu(zz): return -0.96 * np.log(1.0-4.5*zz)
    # def pslhs(zz): return zz*rfac - 2.076*(1.0 - 1.0/(zz + 1.0))

    # # 2. paulson's surface functions
    # def pspmu(xx): return -2.0 * np.log((xx + 1.0)*0.5) - \
    #     np.log((xx*xx + 1.0)*0.5) + 2.0*np.atan(xx) - pihf

    # def pspms(yy): return 5.0 * yy
    # def psphu(xx): return -2.0 * np.log((xx*xx + 1.0)*0.5)
    # def psphs(yy): return 5.0 * yy

    # ilech = 0

    # zilfc = -czil * vkrm * sqvisc

    # zu = z0

    # rdz = 1.0 / zlvl
    # cxch = excm * rdz
    # dthv = th2v - t1v
    # du2 = max(sfcspd*sfcspd, epsu2)

    # # beljars correction of ustar
    # btgh = btg * hpbl

    # # if statements to avoid tangent linear problems near zero
    # if btgh*ch*dthv != 0.0:
    #     wstar2 = wwst2 * abs(btgh*ch*dthv)**(2.0/3.0)
    # else:
    #     wstar2 = 0.0

    # ustar = max(np.sqrt(cm*np.sqrt(du2+wstar2)), epsust)

    # # zilitinkevitch approach for zt
    # zt = np.exp(zilfc*np. sqrt(ustar*z0)) * z0

    # zslu = zlvl + zu
    # zslt = zlvl + zt

    # rlogu = np.log( zslu/zu )
    # rlogt = np.log( zslt/zt )

    # rlmo = elfc*ch*dthv / ustar**3

    # for i in range(itrmx):
    #     # 1. monin-obukkhov length-scale
    #     zetalt = max( zslt*rlmo, ztmin )
    #     rlmo   = zetalt / zslt
    #     zetalu = zslu * rlmo
    #     zetau  = zu * rlmo
    #     zetat  = zt * rlmo
    # TODO: remove
    pass


def snfrac(sneqv, snup, salp):
    # --- ... subprograms called: none

    # determine snow fraction cover.
    if sneqv < snup:
        rsnow = sneqv / snup
        sncovr = 1.0 - (np.exp(-salp*rsnow) - rsnow*np.exp(-salp))
    else:
        sncovr = 1.0

    return sncovr


def snopac(
    # inputs
    nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref, smcdry,
    cmcmax, dt, df1, sfcems, sfctmp, t24, th2, fdown, epsca,
    bexp, pc, rch, rr, cfactr, slope, kdt, frzx, psisat,
    zsoil, dwsat, dksat, zbot, shdfac, ice, rtdis, quartz,
    fxexp, csoil, flx2, snowng, ffrozp, ivegsrc, vegtyp,
    # in/outs
    prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh,
    sh2o, tbot, smc
):
    # --- ... subprograms called: evapo, smflx, shflx, snowpack

    # calculates soil moisture and heat flux values and
    # update soil moisture content and soil heat content values for the
    # case when a snow pack is present.

    snoexp = 2.0
    esdmin = 1.e-6

    prcp1 = prcp1 * 0.001
    edir = 0.0
    edir1 = 0.0

    ec = 0.0
    ec1 = 0.0

    runoff1 = 0.0
    runoff2 = 0.0
    runoff3 = 0.0

    drip = 0.0

    et = np.zeros(nsoil)
    et1 = np.zeros(nsoil)

    ett = 0.0
    ett1 = 0.0
    etns = 0.0
    etns1 = 0.0
    esnow = 0.0
    esnow1 = 0.0
    esnow2 = 0.0

    dew = 0.0
    etp1 = etp * 0.001

    if etp < 0.0:
        # dewfall (=frostfall in this case).
        dew = -etp1
        esnow2 = etp1 * dt
        etanrg = etp * ((1.0-sncovr)*lsubc + sncovr*lsubs)

    else:
        # upward moisture flux
        if ice != 0:
            # for sea-ice and glacial-ice case
            esnow = etp
            esnow1 = esnow * 0.001
            esnow2 = esnow1 * dt
            etanrg = esnow * lsubs

        else:
            # for non-glacial land case
            if sncovr < 1.0:
                etns1, edir1, ec1, et1, ett1 = evapo(nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil,
                                                     sh2o, smcmax, smcwlt, smcref, smcdry, pc,
                                                     shdfac, cfactr, rtdis, fxexp)

                edir1 *= 1.0 - sncovr
                ec1 *= 1.0 - sncovr
                et1 *= 1.0 - sncovr
                ett1 *= 1.0 - sncovr
                etns1 *= 1.0 - sncovr

                edir = edir1 * 1000.0
                ec = ec1 * 1000.0
                et = et1 * 1000.0
                ett = ett1 * 1000.0
                etns = etns1 * 1000.0

            esnow = etp * sncovr
            esnow1 = esnow * 0.001
            esnow2 = esnow1 * dt
            etanrg = esnow*lsubs + etns*lsubc

    # if precip is falling, calculate heat flux from snow sfc to newly accumulating precip
    flx1 = 0.0
    if snowng:
        # fractional snowfall/rainfall
        flx1 = (cpice * ffrozp + cph2o1*(1.-ffrozp)) * prcp * (t1 - sfctmp)

    elif prcp > 0.0:
        flx1 = cph2o1 * prcp * (t1 - sfctmp)

    # calculate an 'effective snow-grnd sfc temp' based on heat fluxes between
    # the snow pack and the soil and on net radiation.
    dsoil = -0.5 * zsoil[0]
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((fdown - flx1 - flx2 - sfcems*sigma1*t24) /
            rch + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom

    if t12 <= tfreez:    # no snow melt will occur.

        # set the skin temp to this effective temp
        t1 = t12
        # update soil heat flux
        ssoil = df1 * (t1 - stc[0]) / dtot
        # update depth of snowpack
        sneqv = max(0.0, sneqv-esnow2)
        flx3 = 0.0
        ex = 0.0
        snomlt = 0.0

    else:    # snow melt will occur.
        t1 = tfreez * max(0.01, sncovr**snoexp) + t12 * \
            (1.0 - max(0.01, sncovr**snoexp))
        ssoil = df1 * (t1 - stc[0]) / dtot

        if sneqv - esnow2 <= esdmin:
            # snowpack has sublimated away, set depth to zero.
            sneqv = 0.0
            ex = 0.0
            snomlt = 0.0
            flx3 = 0.0

        else:
            # potential evap (sublimation) less than depth of snowpack
            sneqv -= esnow2
            seh = rch * (t1 - th2)

            t14 = t1 * t1
            t14 = t14 * t14

            flx3 = fdown - flx1 - flx2 - sfcems*sigma1*t14 - ssoil - seh - etanrg
            if flx3 <= 0.0:
                flx3 = 0.0

            ex = flx3 * 0.001 / lsubf

            # snowmelt reduction
            snomlt = ex * dt

            if sneqv - snomlt >= esdmin:
                # retain snowpack
                sneqv -= snomlt
            else:
                # snowmelt exceeds snow depth
                ex = sneqv / dt
                flx3 = ex * 1000.0 * lsubf
                snomlt = sneqv
                sneqv = 0.0

        if ice == 0:
            prcp1 += ex

    if ice == 0:
        # smflx returns updated soil moisture values for non-glacial land.
        cmc, sh2o, smc, runoff1, runoff2, runoff3, drip = smflx(nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1,
                                                                zsoil, slope, frzx, bexp, dksat, dwsat, shdfac,
                                                                edir1, ec1, et1, cmc, sh2o, smc)

    zz1 = 1.0
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    t11 = t1

    # shflx will calc/update the soil temps.
    ssoil1, stc, t11, tbot, sh2o = shflx(nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot,
                                         psisat, bexp, df1, ice, quartz, csoil, ivegsrc, vegtyp,
                                         shdfac, stc, t11, tbot, sh2o)

    # snow depth and density adjustment based on snow compaction.
    if ice == 0:
        if sneqv > 0.0:
            snowh, sndens = snowpack(sneqv, dt, t1, yy, snowh, sndens)

        else:
            sneqv = 0.0
            snowh = 0.0
            sndens = 0.0
            sncovr = 0.0

    elif ice == 1:
        if sneqv >= 0.01:
            snowh, sndens = snowpack(sneqv, dt, t1, yy, snowh, sndens)
        else:
            sneqv = 0.01
            snowh = 0.05
            sncovr = 1.0
    else:
        if sneqv >= 0.10:
            snowh, sndens = snowpack(sneqv, dt, t1, yy, snowh, sndens)
        else:
            sneqv = 0.10
            snowh = 0.50
            sncovr = 1.0
        

    return prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh, sh2o, tbot, \
        smc, ssoil, runoff1, runoff2, runoff3, edir, ec, et, ett, \
        snomlt, drip, dew, flx1, flx3, esnow


def snow_new(
    # inputs
    sfctmp, sn_new,
    # in/outs
    snowh, sndens
):
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


def snowz0(
    # inputs
    sncovr,
    # in/outs
    z0
):
    # --- ... subprograms called: none

    # total roughness length over snow
    z0 = (1.0 - sncovr) * z0 + sncovr * z0
    # TODO: that is totally unnecessary right? z0 = z0

    return z0


def tdfcnd(smc, qz, smcmax, sh2o):
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

    if sh2o.size > 1:
        ake = np.empty(len(sh2o))
        ake[sh2o+0.0005 < smc] = satratio[sh2o+0.0005 < smc]   # frozen
        ake[(sh2o+0.0005 >= smc) & (satratio > 0.1)] = np.log10(satratio[(sh2o +
                                                                          0.0005 >= smc) & (satratio > 0.1)]) + 1.0   # kersten number
        ake[(sh2o+0.0005 >= smc) & (satratio <= 0.1)] = 0.0
    else:
        if sh2o+0.0005 < smc:    # frozen
            ake = satratio
        elif satratio > 0.1:
            # kersten number
            ake = np.log10(satratio) + 1.0
        else:
            ake = 0.0

    # thermal conductivity
    df = ake * (thksat - thkdry) + thkdry

    return df


# *************************************
# 2nd level subprograms
# *************************************


def evapo(
    # inputs
    nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil,
    sh2o, smcmax, smcwlt, smcref, smcdry, pc,
    shdfac, cfactr, rtdis, fxexp
):
    # --- ... subprograms called: devap, transp

    ec1 = 0.0
    ett1 = 0.0
    edir1 = 0.0
    et1 = np.zeros(nsoil)

    if etp1 > 0.0:
        # retrieve direct evaporation from soil surface.
        if shdfac < 1.0:
            edir1 = devap(etp1, sh2o[0], shdfac, smcmax, smcdry, fxexp)

        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.:
            et1 = transp(nsoil, nroot, etp1, sh2o, smcwlt, smcref,
                         cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis)
            ett1 = np.sum(et1)

            # calculate canopy evaporation.
            if cmc > 0.0:
                ec1 = shdfac * ((cmc/cmcmax)**cfactr) * etp1
            else:
                ec1 = 0.0

            # ec should be limited by the total amount of available water on the canopy
            cmc2ms = cmc / dt
            ec1 = min(cmc2ms, ec1)

    eta1 = edir1 + ett1 + ec1

    return eta1, edir1, ec1, et1, ett1


def shflx(
    # inputs
    nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot,
    psisat, bexp, df1, ice, quartz, csoil, ivegsrc, vegtyp, shdfac,
    # in/outs
    stc, t1, tbot, sh2o
):
    # --- ... subprograms called: hstep, hrtice, hrt

    # updates the temperature state of the soil column

    ctfil1 = 0.5
    ctfil2 = 1.0 - ctfil1

    oldt1 = t1
    stsoil = stc

    if ice != 0:  # sea-ice or glacial ice case
        tbot, rhsts, ai, bi, ci = hrtice(
            nsoil, stc, zsoil, yy, zz1, df1, ice, tbot)

        stcf = hstep(nsoil, stc, dt, rhsts, ai, bi, ci)
    else:
        sh2o, rhsts, ai, bi, ci = hrt(nsoil, stc, smc, smcmax, zsoil, yy, zz1, tbot,
                                      zbot, psisat, dt, bexp, df1, quartz, csoil, ivegsrc, vegtyp,
                                      shdfac, sh2o)

        stcf = hstep(nsoil, stc, dt, rhsts, ai, bi, ci)

    stc = stcf

    # update the grnd (skin) temperature in the no snowpack case
    t1 = (yy + (zz1 - 1.0)*stc[0]) / zz1
    t1 = ctfil1*t1 + ctfil2*oldt1
    stc = ctfil1*stc + ctfil2*stsoil

    # calculate surface soil heat flux
    ssoil = df1*(stc[0] - t1) / (0.5*zsoil[0])

    return ssoil, stc, t1, tbot, sh2o


def smflx(
    # inputs
    nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1,
    zsoil, slope, frzx, bexp, dksat, dwsat, shdfac,
    edir1, ec1, et1,
    # in/outs
    cmc, sh2o, smc
):
    # --- ... subprograms called: srt, sstep

    dummy = 0.

    # compute the right hand side of the canopy eqn term
    rhsct = shdfac*prcp1 - ec1

    drip = 0.
    trhsct = dt * rhsct
    excess = cmc + trhsct

    if excess > cmcmax:
        drip = excess - cmcmax

    # pcpdrp is the combined prcp1 and drip (from cmc) that goes into the soil
    pcpdrp = (1.0 - shdfac)*prcp1 + drip/dt

    # store ice content at each soil layer before calling srt and sstep
    # TODO: smc is not declared how to use it here?
    #sice = smc - sh2o
    sice = smc-sh2o  # only a suggestion

    if (pcpdrp*dt) > (0.001*1000.0*(-zsoil[0])*smcmax):
        #     rhstt, runoff1, runoff2, ai, bi, ci = srt(nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat,
        #                                               dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)

        #     sh2ofg, runoff3, smc = sstep(nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice,
        #                                  dummy, rhstt, ai, bi, ci)

        #     sh2oa = (sh2o + sh2ofg) * 0.5

        #     rhstt, runoff1, runoff2, ai, bi, ci = srt(nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat,
        #                                               dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)

        #     sh2ofg, runoff3, smc = sstep(nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice,
        #                                  dummy, rhstt, ai, bi, ci)

        # not called
        print("Error: case not implemented")

    else:
        rhstt, runoff1, runoff2, ai, bi, ci = srt(nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                                                  dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)

        sh2o, runoff3, smc, cmc, rhstt, ai, bi, ci = sstep(nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice,
                                                           cmc, rhstt, ai, bi, ci)

    return cmc, sh2o, smc, runoff1, runoff2, runoff3, drip


def snowpack(
    # inputs
    esd, dtsec, tsnow, tsoil,
    # in/outs
    snowh, sndens
):
    # --- ... subprograms called: none

    # calculates compaction of snowpack under conditions of
    # increasing snow density.

    c1 = 0.01
    c2 = 21.0

    # conversion into simulation units
    snowhc = snowh * 100.0
    esdc = esd * 100.0
    dthr = dtsec / 3600.0
    tsnowc = tsnow - tfreez
    tsoilc = tsoil - tfreez

    # calculating of average temperature of snow pack
    tavgc = 0.5 * (tsnowc + tsoilc)

    # calculating of snow depth and density as a result of compaction
    if esdc > 1.e-2:
        esdcx = esdc
    else:
        esdcx = 1.e-2

    bfac = dthr*c1 * np.exp(0.08*tavgc - c2*sndens)

    # number of terms of polynomial expansion and its accuracy is governed by iteration limit "ipol".
    ipol = 4
    pexp = 0.0

    for j in range(ipol, 0, -1):
        pexp = (1.0 + pexp)*bfac*esdcx/(j+1)
    pexp += 1.

    dsx = sndens * pexp
    # set upper/lower limit on snow density
    dsx = max(min(dsx, 0.40), 0.05)
    sndens = dsx

    # update of snow depth and density depending on liquid water during snowmelt.
    if tsnowc >= 0.0:
        dw = 0.13 * dthr / 24.0
        sndens = min(sndens*(1.0 - dw) + dw, 0.40)

    # calculate snow depth (cm) from snow water equivalent and snow density.
    snowhc = esdc / sndens
    snowh = snowhc * 0.01

    return snowh, sndens


# *************************************
# 3rd level subprograms
# *************************************


def devap(etp1, smc, shdfac, smcmax, smcdry, fxexp):
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


def frh2o(tkelv, smc, sh2o, smcmax, bexp, psis, i):
    # --- ... subprograms called: none

    # calculates amount of supercooled liquid soil water
    # content if temperature is below 273.15k

    # constant parameters
    ck = 8.0
    blim = 5.5
    error = 0.005

    # limits on parameter b
    bx = bexp
    if bexp > blim:
        bx = blim

    nlog = 0
    liqwat = np.empty(smc.size)

    # if temperature not significantly below freezing (t0), sh2o = smc
    j = i & (tkelv <= (tfreez - 1.e-3))
    swl = np.zeros(smc.size)

    # initial guess for swl (frozen content)
    swl[j] = smc[j] - sh2o[j]
    swl[j] = np.maximum(np.minimum(swl[j], smc[j]-0.02), 0.0)

    while nlog < 10 and j.any():
        nlog += 1

        df = np.log((psis*gs2/lsubf) * ((1.0 + ck*swl[j])**2.0) * (smcmax /
                                                                   (smc[j] - swl[j]))**bx) - np.log(-(tkelv[j] - tfreez) / tkelv[j])

        denom = 2.0*ck/(1.0 + ck*swl[j]) + bx/(smc[j] - swl[j])
        swlk = swl[j] - df / denom

        # bounds useful for mathematical solution.
        swlk = np.maximum(np.minimum(swlk, smc[j]-0.02), 0.0)

        # mathematical solution bounds applied.
        dswl = np.abs(swlk - swl[j])
        swl[j] = swlk

        j[j] = j[j] & (dswl > error)

    liqwat = smc - swl

    return liqwat


def hrt(
    # inputs
    nsoil, stc, smc, smcmax, zsoil, yy, zz1, tbot,
    zbot, psisat, dt, bexp, df1, quartz, csoil, ivegsrc, vegtyp,
    shdfac,
    # in/outs
    sh2o
):

    # --- ... subprograms called: tbnd, snksrc, tmpavg
    # calculates the right hand side of the time tendency term
    # of the soil thermal diffusion equation.

    rhsts = np.empty(nsoil)
    ai = np.empty(nsold)
    bi = np.empty(nsold)
    ci = np.empty(nsold)

    csoil_loc = csoil

    if ivegsrc == 1 and vegtyp == 12:
        csoil_loc = 3.0e6*(1.-shdfac)+csoil*shdfac

    #  initialize logical for soil layer temperature averaging.
    # TODO: can itavg be removed?
    itavg = True

    # calc the heat capacity of the top soil layer
    hcpct = sh2o*cph2o2 + (1.0 - smcmax)*csoil_loc + \
        (smcmax - smc)*cp2 + (smc - sh2o)*cpice1

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5*zsoil[1])
    ai[0] = 0.0
    ci[0] = (df1*ddz) / (zsoil[0]*hcpct[0])
    bi[0] = -ci[0] + df1 / (0.5*zsoil[0]*zsoil[0]*hcpct[0]*zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc[0] - stc[1]) / (-0.5*zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (0.5*zsoil[0]*zz1)
    rhsts[0] = (df1*dtsdz - ssoil) / (zsoil[0]*hcpct[0])

    # capture the vertical difference of the heat flux at top and
    # bottom of first soil layer
    qtot = ssoil - df1 * dtsdz

    # calculate frozen water content in 1st soil layer.
    sice = (smc - sh2o > 0.)

    # calculate thermal diffusivity for each layer
    df1n = tdfcnd(smc[1:], quartz, smcmax, sh2o[1:])

    if ivegsrc == 1 and vegtyp == 12:
        df1n = 3.24*(1.-shdfac) + shdfac*df1n

    df1k = np.append(df1, df1n[:-1])

    # calc the vertical soil temp gradient thru each layer
    denom = 0.5 * (zsoil[:-2] - zsoil[2:])
    dtsdz2 = (stc[1:-1] - stc[2:]) / denom
    dtsdz2 = np.append(dtsdz2, (stc[-1] - tbot) /
                       (0.5 * (zsoil[-2] + zsoil[-1]) - zbot))

    # calc the matrix coef, ci, after calc'ng its partial product
    ddz2 = 2.0 / (zsoil[:-2] - zsoil[2:])
    ddz = np.append(ddz, ddz2)
    ci[1:-1] = - df1n[:-1]*ddz2 / ((zsoil[:-2] - zsoil[1:-1]) * hcpct[1:-1])
    ci[-1] = 0.

    if itavg:
        # calculate temp at bottom of each layer
        tsurf = (yy + (zz1-1)*stc[0]) / zz1
        tbk = tbnd(stc, np.append(stc[1:], tbot), zsoil, zbot, nsoil)

    # calculate rhsts
    denom = (zsoil[1:] - zsoil[:-1])*hcpct[1:]
    dtsdz = np.append(dtsdz, dtsdz2[:-1])
    rhsts[1:] = (df1n*dtsdz2 - df1k*dtsdz)/denom

    qtot = np.append(qtot, -1. * denom * rhsts[1:])

    i = sice | (np.append(tsurf, tbk[:-1]) < tfreez) | (
        stc < tfreez) | (tbk < tfreez)
    if itavg:
        tavg = tmpavg(np.append(tsurf, tbk[:-1]), stc, tbk, zsoil, nsoil, i)
    else:
        tavg = np.zeros(nsoil)
        tavg[i] = stc[i]

    tsnsr = np.empty(nsoil)
    tsnsr, sh2o = snksrc(nsoil, i, tavg, smc, smcmax, psisat, bexp, dt,
                         qtot, zsoil, shdfac, sh2o)
    if i[0]:
        rhsts[0] -= tsnsr[0] / (zsoil[0] * hcpct[0])

    i[0] = False
    rhsts[i] -= tsnsr[i] / denom[i[1:]]

    # calc matrix coefs, ai, and bi for this layer.
    ai[1:] = - df1 * ddz / ((zsoil[:-1] - zsoil[1:]) * hcpct[1:])
    bi[1:] = -(ai[1:] + ci[1:])

    return sh2o, rhsts, ai, bi, ci


def hrtice(
    # inputs
    nsoil, stc, zsoil, yy, zz1, df1, ice,
    # in/outs
    tbot
):
    # --- ... subprograms called: none

    # calculates the right hand side of the time tendency
    # term of the soil thermal diffusion equation for sea-ice or glacial-ice

    # set a nominal universal value of specific heat capacity
    if ice == 1:        # sea-ice
        hcpct = 1.72396e+6
        tbot = 271.16
    else:               # glacial-ice
        hcpct = 1.89000e+6

    # set ice pack depth
    if ice == 1:
        zbot = zsoil[-1]
    else:
        zbot = -25.0

    ai = np.zeros(nsoil)
    bi = np.zeros(nsoil)
    ci = np.zeros(nsoil)
    rhsts = np.zeros(nsoil)

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5*zsoil[1])
    ai[0] = 0.0
    ci[0] = (df1*ddz) / (zsoil[0]*hcpct)
    bi[0] = -ci[0] + df1 / (0.5*zsoil[0]*zsoil[0]*hcpct*zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc[0] - stc[1]) / (-0.5*zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (0.5*zsoil[0]*zz1)
    rhsts[0] = (df1*dtsdz - ssoil) / (zsoil[0]*hcpct)

    # the remaining soil layers, repeating the above process
    denom = 0.5 * (zsoil[:-2] - zsoil[2:])
    dtsdz2 = (zsoil[1:-1] - zsoil[2:]) / denom
    ddz2 = 2.0 / (zsoil[:-2] - zsoil[2:])
    ci[1:-1] = -df1*ddz2 / ((zsoil[:-2] - zsoil[1:-1])*hcpct)

    # lowest layer
    dtsdz2 = np.append(dtsdz2, stc[-1] - tbot) / \
        (0.5 * (zsoil[-2]-zsoil[-1]) - zbot)
    ci[-1] = 0.0

    ddz = np.append(ddz, ddz2)
    dtsdz = np.append(dtsdz, dtsdz2[:-1])

    # calc rhsts for this layer after calc'ng a partial product.
    denom = (zsoil[1:] - zsoil[:-1]) * hcpct
    rhsts[1:] = (df1*dtsdz2 - df1*dtsdz) / denom

    # calc matrix coefs
    ai[1:] = - df1*ddz / ((zsoil[:-1] - zsoil[1:]) * hcpct)
    bi[1:] = -(ai[1:] + ci[1:])

    return tbot, rhsts, ai, bi, ci


def hstep(
    # inputs
    nsoil, stcin, dt,
    # in/outs
    rhsts, ai, bi, ci
):
    # --- ... subprograms called: rosr12
    # calculates/updates the soil temperature field.

    # create finite difference values for use in rosr12 routine
    rhsts *= dt
    ai *= dt
    bi = 1. + bi*dt
    ci *= dt

    # solve tri-diagonal matrix-equation
    ci, rhsts = rosr12(nsoil, ai, bi, rhsts, ci)

    # calc/update the soil temps using matrix solution
    stcout = stcin + ci

    return stcout


def rosr12(nsoil, a, b, d, c):
    # --- ... subprograms called: none
    # inverts (solve) the tri-diagonal matrix problem

    p = np.empty(nsoil)
    delta = np.empty(nsoil)

    # initialize eqn coef c for the lowest soil layer
    c[nsoil-1] = 0.

    # solve the coefs for the 1st soil layer
    p[0] = -c[0]/b[0]
    delta[0] = d[0]/b[0]

    for k in range(1, nsoil):
        p[k] = - c[k] / (b[k] + a[k] * p[k-1])
        delta[k] = (d[k] - a[k]*delta[k-1])/(b[k] + a[k]*p[k-1])

    # set p to delta for lowest soil layer
    p[-1] = delta[-1]

    # adjust p for soil layers 2 thru nsoil
    for k in range(nsoil-2, -1, -1):
        p[k] = p[k]*p[k+1] + delta[k]

    return p, delta


def snksrc(nsoil, i, tavg, smc, smcmax, psisat, bexp, dt,
           qtot, zsoil, shdfac, sh2o):

    # --- ... subprograms called: frh2o
    # calculates sink/source term of the termal diffusion equation.

    dz = np.append(-zsoil[0], zsoil[:-1] - zsoil[1:])

    # compute potential or 'equilibrium' unfrozen supercooled free water
    free = frh2o(tavg, smc, sh2o, smcmax, bexp, psisat, i)

    # estimate the new amount of liquid water
    dh2o = 1.0000e3
    xh2o = sh2o + qtot*dt / (dh2o*lsubf*dz)

    # reduce extent of freezing
    j1 = i & (xh2o < sh2o) & (sh2o < free)
    j2 = i & (xh2o < free) & (free <= sh2o)
    xh2o[j1] = sh2o[j1]
    xh2o[j2] = free[j2]

    # then reduce extent of thaw
    j1 = i & (free < sh2o) & (sh2o < xh2o)
    j2 = i & (sh2o <= free) & (free < xh2o)
    xh2o[j1] = sh2o[j1]
    xh2o[j2] = free[j2]

    xh2o = np.maximum(np.minimum(xh2o, smc), 0.0)

    # calculate phase-change heat source/sink term
    tsrc = np.empty(nsoil)
    tsrc[i] = -dh2o * lsubf * dz[i] * (xh2o[i] - sh2o[i]) / dt
    sh2o[i] = xh2o[i]

    return tsrc, sh2o


def srt(
    # inputs
    nsoil, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice
):
    # --- ... subprograms called: wdfcnd

    # determine rainfall infiltration rate and runoff
    cvfrz = 3
    iohinf = 1
    rhstt = np.empty(nsoil)
    ai = np.empty(nsold)
    bi = np.empty(nsold)
    ci = np.empty(nsold)

    sicemax = max(np.max(sice), 0.)

    pddum = pcpdrp
    runoff1 = 0.0

    if pcpdrp != 0:
        # frozen ground version
        dt1 = dt/86400.
        smcav = smcmax - smcwlt
        dmax = -zsoil[0] * smcav
        dmax = np.append(dmax, (zsoil[:-1]-zsoil[1:]) * smcav)
        dmax *= 1.0 - (sh2oa + sice - smcwlt)/smcav

        dice = -zsoil[0] * sice[0] + np.sum((zsoil[:-1]-zsoil[1:]) * sice[1:])
        dd = np.sum(dmax)

        val = 1.0 - np.exp(-kdt*dt1)
        ddt = dd * val

        px = pcpdrp * dt

        if px < 0.0:
            px = 0.0

        infmax = (px*(ddt/(px+ddt)))/dt

        # reduction of infiltration based on frozen ground parameters
        fcr = 1.

        if dice > 1.e-2:
            acrt = cvfrz * frzx / dice
            ialp1 = cvfrz - 1

            j = np.arange(ialp1) + 1
            j_factorial = np.array([np.math.factorial(i) for i in j])
            sum = 1. + np.sum(np.power(acrt, cvfrz - j) /
                         (np.math.factorial(ialp1)/j_factorial))

            fcr = 1. - np.exp(-acrt) * sum

        infmax *= fcr

        # correction of infiltration limitation
        mxsmc = sh2oa[0]

        wdf, wcnd = wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat, sicemax)

        infmax = max(infmax, wcnd)
        infmax = min(infmax, px)

        if pcpdrp > infmax:
            runoff1 = pcpdrp - infmax
            pddum = infmax

    mxsmc = sh2oa
    wdf, wcnd = wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat, sicemax)

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-.5*zsoil[1])
    ai[0] = 0.0
    bi[0] = wdf[0] * ddz / (-zsoil[0])
    ci[0] = -bi[0]

    # calc rhstt for the top layer
    dsmdz = (sh2o[0] - sh2o[1]) / (-.5*zsoil[1])
    rhstt[0] = (wdf[0]*dsmdz + wcnd[0] - pddum + edir + et[0]) / zsoil[0]
    sstt = wdf[0] * dsmdz + wcnd[0] + edir + et[0]

    # the remaining soil layers, repeating the above process
    denom2 = zsoil[:-1] - zsoil[1:]
    denom = (zsoil[:-2] - zsoil[2:])
    dsmdz2 = (sh2o[1:-1] - sh2o[2:]) / (denom * 0.5)
    # calc the matrix coef, ci, after calc'ng its partial product
    ddz2 = 2.0 / denom
    ci[1:-1] = -wdf[1:-1]*ddz2 / denom2[:-1]

    # lowest layer
    dsmdz2 = np.append(dsmdz2, 0.)
    ci[-1] = 0.0

    # slope
    slopx = np.empty(nsoil-1)
    slopx[:-1] = 1.
    slopx[-1] = slope

    # calc rhstt for this layer after calc'ng its numerator
    ddz = np.append(ddz, ddz2)
    dsmdz = np.append(dsmdz, dsmdz2[:-1])

    numer = wdf[1:]*dsmdz2 + slopx*wcnd[1:] - \
        wdf[:-1]*dsmdz - wcnd[:-1] + et[1:]
    rhstt[1:] = -numer / denom2

    # calc matrix coefs
    ai[1:] = - wdf[:-1]*ddz / denom2
    bi[1:] = -(ai[1:] + ci[1:])

    runoff2 = slopx[-1] * wcnd[-1]

    return rhstt, runoff1, runoff2, ai, bi, ci


def sstep(
    # inputs
    nsoil, sh2oin, rhsct, dt, smcmax, cmcmax, zsoil, sice,
    # in/outs
    cmc, rhstt, ai, bi, ci
):
    # --- ... subprograms called: rosr12
    # calculates/updates soil moisture content values and
    # canopy moisture content values.

    sh2oout = np.empty(nsoil)
    smc = np.empty(nsoil)

    # create 'amount' values of variables to be input to the
    # tri-diagonal matrix routine.
    rhstt *= dt
    ai *= dt
    bi = 1. + bi*dt
    ci *= dt

    # solve the tri-diagonal matrix
    ci, rhstt = rosr12(nsoil, ai, bi, rhstt, ci)

    # sum the previous smc value and the matrix solution
    ddz = np.append(-zsoil[0], zsoil[:-1] - zsoil[1:])
    wplus = 0.

    for k in range(nsoil):
        sh2oout[k] = sh2oin[k] + ci[k] + wplus/ddz[k]
        stot = sh2oout[k] + sice[k]

        if stot > smcmax: 
            wplus = (stot-smcmax)*ddz[k]
        else:
            wplus = 0.

        smc[k] = max(min(stot, smcmax), 0.02)
        sh2oout[k] = max(smc[k]-sice[k], 0.0)

    runoff3 = wplus

    # update canopy water content/interception
    cmc += dt * rhsct
    if cmc < 1.e-20:
        cmc = 0.0
    cmc = min(cmc, cmcmax)

    return sh2oout, runoff3, smc, cmc, rhstt, ai, bi, ci


def tbnd(tu, tb, zsoil, zbot, nsoil):
    # --- ... subprograms called: none
    # calculates temperature on the boundary of the
    # layer by interpolation of the middle layer temperature

    # use surface temperature on the top of the first layer
    zup = np.append(0.0, zsoil[:-1])

    # use depth of the constant bottom temperature when interpolate
    # temperature into the last layer boundary
    zb = np.append(zsoil[1:], 2.0*zbot - zsoil[-1])

    # linear interpolation between the average layer temperatures
    tbk = tu + (tb-tu)*(zup-zsoil)/(zup-zb)

    return tbk


def tmpavg(
    # inputs
    tup, tm, tdn, zsoil, nsoil, i
):
    # --- ... subprograms called: none
 
    tavg = np.zeros(nsoil)

    dz = np.append(-zsoil[0], zsoil[:-1] - zsoil[1:])

    dzh = dz * 0.5

    j = i & (tup < tfreez) & (tm < tfreez) & (tdn < tfreez)
    tavg[j] = (tup[j] + 2.0*tm[j] + tdn[j]) / 4.0

    j = i & (tup < tfreez) & (tm < tfreez) & (tdn >= tfreez)
    x0 = (tfreez - tm[j]) * dzh[j] / (tdn[j] - tm[j])
    tavg[j] = 0.5*(tup[j]*dzh[j] + tm[j]*(dzh[j] + x0) +
                   tfreez*(2.*dzh[j]-x0)) / dz[j]

    j = i & (tup < tfreez) & (tm >= tfreez) & (tdn < tfreez)
    xup = (tfreez - tup[j]) * dzh[j] / (tm[j] - tup[j])
    xdn = dzh[j] - (tfreez - tm[j]) * dzh[j] / (tdn[j] - tm[j])
    tavg[j] = 0.5*(tup[j]*xup + tfreez*(2.*dz[j]-xup-xdn)+tdn[j]*xdn) / dz[j]

    j = i &(tup < tfreez) & (tm >= tfreez) & (tdn >= tfreez)
    xup = (tfreez - tup[j]) * dzh[j] / (tm[j] - tup[j])
    tavg[j] = 0.5*(tup[j]*xup + tfreez*(2.*dz[j]-xup)) / dz[j]

    j = i & (tup >= tfreez) & (tm < tfreez) & (tdn < tfreez)
    xup = dzh[j] - (tfreez - tup[j]) * dzh[j] / (tm[j] - tup[j])
    tavg[j] = 0.5*(tfreez * (dz[j] - xup) + tm[j] *
                (dzh[j] + xup) + tdn[j] * dzh[j]) / dz[j]

    j = i & (tup >= tfreez) & (tm < tfreez) & (tdn >= tfreez)
    xup = dzh[j] - (tfreez-tup[j]) * dzh[j] / (tm[j]-tup[j])
    xdn = (tfreez-tm[j]) * dzh[j] / (tdn[j]-tm[j])
    tavg[j] = 0.5 * (tfreez*(2. * dz[j] - xup - xdn) +
                  tm[j] * (xup + xdn)) / dz[j]

    j = i & (tup >= tfreez) & (tm >= tfreez) & (tdn < tfreez)
    xdn = dzh[j] - (tfreez-tm[j]) * dzh[j] / (tdn[j]-tm[j])
    tavg[j] = (tfreez * (dz[j] - xdn) + 0.5*(tfreez + tdn[j]) * xdn) / dz[j]

    return tavg


def transp(
    # inputs
    nsoil, nroot, etp1, smc, smcwlt, smcref,
    cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis
):
    # --- ... subprograms called: none

    # calculates transpiration for the veg class

    # initialize plant transp to zero for all soil layers.
    et1 = np.zeros(nsoil)

    if cmc != 0.0:
        etp1a = shdfac * pc * etp1 * (1.0 - (cmc / cmcmax) ** cfactr)
    else:
        etp1a = shdfac * pc * etp1

    gx = np.maximum(np.minimum(
        (smc[:nroot] - smcwlt) / (smcref - smcwlt), 1.0), 0.0)
    sgx = np.sum(gx) / nroot

    rtx = rtdis[:nroot] + gx - sgx
    gx *= np.maximum(rtx, 0.0)
    denom = sum(gx)

    if denom <= 0.0:
        denom = 1.0

    et1[:gx.size] = etp1a * gx / denom

    return et1


def wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    # --- ... subprograms called: none

    # calc the ratio of the actual to the max psbl soil h2o content of each layer
    factr1 = min(1.0, max(0.0, 0.2/smcmax))
    factr2 = np.minimum(1.0, np.maximum(0.0, smc/smcmax))

    # prep an expntl coef and calc the soil water diffusivity
    expon = bexp + 2.0
    wdf = dwsat * factr2 ** expon

    # frozen soil hydraulic diffusivity.
    if sicemax > 0.:
        vkwgt = 1.0 / (1.0 + (500.0*sicemax)**3.0)
        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon

    # reset the expntl coef and calc the hydraulic conductivity
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * factr2 ** expon

    return wdf, wcnd


# *************************************
# helper functions
# *************************************


def init_array(shape, mode):
    arr = np.empty(shape)
    if mode == "none":
        pass
    if mode == "zero":
        arr[:] = 0.
    elif mode == "nan":
        arr[:] = np.nan
    return arr
