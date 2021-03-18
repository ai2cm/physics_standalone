#!/usr/bin/env python3

import numpy as np

OUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy",
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx",
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf",
            "smcwlt2", "smcref2", "wet1"]


def run(in_dict, in_dict2):
    """run function"""

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = in_dict[key].copy()
        del in_dict[key]

    sfc_drv(**in_dict, **in_dict2, **out_dict)

    return out_dict


def sfc_drv(
    # inputs
    im, km, ps, t1, q1, soiltyp, vegtype, sigmaf,
    sfcemis, dlwflx, dswsfc, snet, delt, tg3, cm, ch,
    prsl1, prslki, zf, land, wind, slopetyp,
    shdmin, shdmax, snoalb, sfalb, flag_iter, flag_guess,
    lheatstrg, isot, ivegsrc,
    bexppert, xlaipert, vegfpert, pertvegf,
    # Inputs to probe for port
    zsoil_noah_ref,
    # in/outs
    weasd, snwdph, tskin, tprcp, srflag, smc, stc, slc,
    canopy, trans, tsurf, zorl,
    # outputs
    sncovr1, qsurf, gflux, drain, evap, hflx, ep, runoff,
    cmm, chh, evbs, evcw, sbsno, snowc, stm, snohf,
    smcwlt2, smcref2, wet1
):
    # --- ... subprograms called: ppfbet, sflx

    # more constant definitions
    cp = 1.0046e+3
    hvap = 2.5e+6
    grav = 9.80665
    rd = 2.8705e+2
    eps = rd/4.6150e+2
    epsm1 = rd/4.6150e+2 - 1.
    rvrdm1 = 4.6150e+2/rd - 1.

    # set constant parameters
    cpinv = 1./cp
    hvapi = 1./hvap
    elocp = hvap/cp
    rhoh2o = 1000.
    a2 = 17.2693882
    a3 = 273.16
    a4 = 35.86
    a23m4 = a2*(a3-a4)
    zsoil_noah = np.array([-0.1, -0.4, -1.0, -2.0])

    # serialbox test
    serialbox_test(zsoil_noah_ref, zsoil_noah, "zsoil_noah")

    # initialize local arrays
    mode = "nan"
    rch = init_array([im], mode)
    rho = init_array([im], mode)
    q0 = init_array([im], mode)
    qs1 = init_array([im], mode)
    theta1 = init_array([im], mode)
    weasd_old = init_array([im], mode)
    snwdph_old = init_array([im], mode)
    tprcp_old = init_array([im], mode)
    srflag_old = init_array([im], mode)
    tskin_old = init_array([im], mode)
    canopy_old = init_array([im], mode)
    et = init_array([km], mode)
    sldpth = init_array([km], mode)
    stsoil = init_array([km], mode)
    smsoil = init_array([km], mode)
    slsoil = init_array([km], mode)
    zsoil = init_array([im, km], mode)
    smc_old = init_array([im, km], mode)
    stc_old = init_array([im, km], mode)
    slc_old = init_array([im, km], mode)

    # save land-related prognostic fields for guess run
    i = land & flag_guess  # TODO: check if boolean
    weasd_old[i] = weasd[i]
    snwdph_old[i] = snwdph[i]
    tskin_old[i] = tskin[i]
    canopy_old[i] = canopy[i]
    tprcp_old[i] = tprcp[i]
    srflag_old[i] = srflag[i]
    smc_old[i, :] = smc[i, :]
    stc_old[i, :] = stc[i, :]
    slc_old[i, :] = slc[i, :]

    # initialization block
    i = flag_iter & land
    ep[i] = 0.
    evap[i] = 0.
    hflx[i] = 0.
    gflux[i] = 0.
    drain[i] = 0.
    canopy[i] = np.maximum(canopy[i], 0.)

    evbs[i] = 0.
    evcw[i] = 0.
    trans[i] = 0.
    sbsno[i] = 0.
    snowc[i] = 0.
    snohf[i] = 0.

    # initialize variables
    # q1=specific humidity at level 1 (kg/kg)
    q0[i] = np.maximum(q1[i], 1.e-8)
    # adiabatic temp at level 1 (k)
    theta1[i] = t1[i] * prslki[i]
    rho[i] = prsl1[i] / (rd*t1[i]*(1.0+rvrdm1*q0[i]))
    qs1[i] = fpvs(t1[i])
    qs1[i] = np.maximum(eps*qs1[i] / (prsl1[i]+epsm1*qs1[i]), 1.e-8)
    q0[i] = np.minimum(qs1[i], q0[i])

    zsoil[i, :] = zsoil_noah[:]

    # noah: prepare variables to run noah lsm
    for i in range(0, im):
        if not(flag_iter[i] & land[i]):
            continue

    # 1. configuration information
        couple = 1
        ffrozp = srflag[i]
        ice = 0
        zlvl = zf[i]
        nsoil = km
        sldpth[0] = - zsoil[i, 0]
        sldpth[1:] = zsoil[i, :km-1] - zsoil[i, 1:]

    # 2. forcing data
        lwdn = dlwflx[i]
        swdn = dswsfc[i]
        solnet = snet[i]
        sfcems = sfcemis[i]

        sfcprs = prsl1[i]
        prcp = rhoh2o * tprcp[i] / delt
        sfctmp = t1[i]
        th2 = theta1[i]
        q2 = q0[i]

    # 3. other forcing data
        sfcspd = wind[i]
        q2sat = qs1[i]
        dqsdt2 = q2sat * a23m4/(sfctmp-a4)**2

    # 4. canopy/soil characteristics
        vtype = vegtype[i]
        stype = soiltyp[i]
        slope = slopetyp[i]
        shdfac = sigmaf[i]

        vegfp = vegfpert[i]
        if(pertvegf[0] > 0.):
            # this condition is never true
            # if it was true ppfbet would be called
            # TODO: include assert
            print("ERROR: case not implemented", pertvegf[0])

        shdmin1d = shdmin[i]
        shdmax1d = shdmax[i]
        snoalb1d = snoalb[i]

        ptu = 0.0
        alb = sfalb[i]
        tbot = tg3[i]

    # 5.history (state) variables
        cmc = canopy[i] * 0.001           # convert from mm to m
        tsea = tsurf[i]                   # clu_q2m_iter

        stsoil = stc[i, :]
        smsoil = smc[i, :]
        slsoil = slc[i, :]

        snowh = snwdph[i] * 0.001         # convert from mm to m
        sneqv = weasd[i] * 0.001         # convert from mm to m
        if ((sneqv != 0.) & (snowh == 0.)):
            # not called
            # TODO: remove?
            snowh = 10.0 * sneqv

        chx = ch[i] * wind[i]              # compute conductance
        cmx = cm[i] * wind[i]
        chh[i] = chx * rho[i]
        cmm[i] = cmx

        z0 = zorl[i]/100.       # outside sflx, roughness uses cm as unit
        bexpp = bexppert[i]
        xlaip = xlaipert[i]

        # call noah lsm
        # TODO: shdfac, snowh are marked as output variables but are declared before. In sflx shdfac value is used.
        nroot, albedo, eta, sheat, ec, \
            edir, et, ett, esnow, drip, dew, beta, etp, ssoil, \
            flx1, flx2, flx3, runoff1, runoff2, runoff3, \
            snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq, \
            rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax = sflx(  # inputs
                nsoil, couple, ice, ffrozp, delt, zlvl, sldpth,
                swdn, solnet, lwdn, sfcems, sfcprs, sfctmp,
                sfcspd, prcp, q2, q2sat, dqsdt2, th2, ivegsrc,
                vtype, soiltyp, slopetyp, shdmin, alb, snoalb,
                bexpp, xlaip, lheatstrg,
                # in/outs
                tbot, cmc, tsea, stsoil, smsoil, slsoil, sneqv, chx, cmx, z0,
                # outputs
                shdfac, snowh  # TODO: are they not in/outs?
            )

    # 6. output
        # TODO: merge return from sflx driectly
        evap[i] = eta
        hflx[i] = sheat
        gflux[i] = ssoil

        evbs[i] = edir
        evcw[i] = ec
        trans[i] = ett
        sbsno[i] = esnow
        snowc[i] = sncovr
        stm[i] = soilm * 1000.0  # unit conversion (from m to kg m-2)
        snohf[i] = flx1 + flx2 + flx3

        smcwlt2[i] = smcwlt
        smcref2[i] = smcref

        ep[i] = etp
        tsurf[i] = tsea

        stc[i, :] = stsoil
        smc[i, :] = smsoil
        slc[i, :] = slsoil
        wet1[i] = smsoil[0] / smcmax

        # unit conversion (from m s-1 to mm s-1 and kg m-2 s-1)
        runoff[i] = runoff1 * 1000.0
        drain[i] = runoff2 * 1000.0

        # unit conversion (from m to mm)
        canopy[i] = cmc * 1000.0
        snwdph[i] = snowh * 1000.0
        weasd[i] = sneqv * 1000.0
        sncovr1[i] = sncovr

        # outside sflx, roughness uses cm as unit (update after snow's effect)
        zorl[i] = z0*100.

    # compute qsurf
    i = flag_iter & land
    rch[i] = rho[i] * cp * ch[i] * wind[i]
    qsurf[i] = q1[i] + evap[i] / (elocp*rch[i])
    tem = 1.0 / rho[i]
    hflx[i] = hflx[i] * tem * cpinv
    evap[i] = evap[i] * tem * hvapi

    # restore land-related prognostic fields for guess run
    i = land & flag_guess
    weasd[i] = weasd_old[i]
    snwdph[i] = snwdph_old[i]
    tskin[i] = tskin_old[i]
    canopy[i] = canopy_old[i]
    tprcp[i] = tprcp_old[i]
    srflag[i] = srflag_old[i]
    smc[i, :] = smc_old[i, :]
    stc[i, :] = stc_old[i, :]
    slc[i, :] = slc_old[i, :]
    i = land & np.logical_not(flag_guess)
    tskin[i] = tsurf[i]


def ppfbet(pr, p, q, iflag, x):
    # is not called
    pass


def sflx(
    # inputs
    nsoil, couple, icein, ffrozp, dt, zlvl, sldpth,
    swdn, swnet, lwdn, sfcems, sfcprs, sfctmp,
    sfcspd, prcp, q2, q2sat, dqsdt2, th2, ivegsrc,
    vegtyp, soiltyp, slopetyp, shdmin, alb, snoalb,
    bexpp, xlaip, lheatstrg,
    # in/outs
    tbot, cmc, t1, stc, smc, sh2o, sneqv, ch, cm, z0,
    # outputs
    shdfac, snowh
):
    # --- ... subprograms called: redprm, snow_new, csnow, snfrac, alcalc, tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.

    nsold = 2
    gs1 = 9.8
    gs2 = 9.81
    trfreez = 2.7315e+2
    lsubc = 2.501e+6
    lsubf = 3.335e5
    lsubs = 2.83e+6
    elcp = 2.4888e+3
    rd1 = 287.04
    cp = 1.0046e+3
    cp1 = 1004.5
    cp2 = 1004.0
    cph2o1 = 4.218e+3
    cpice = 2.1060e+3
    cpice1 = 2.106e6
    sigma1 = 5.67e-8

    nroot = 0
    albedo = 0
    eta = 0
    sheat = 0
    ec = 0
     
    edir = 0
    et = 0
    ett = 0
    esnow = 0
    drip = 0
    dew = 0
    beta = 0
    etp = 0
    ssoil = 0
    
    flx1 = 0
    flx2 = 0
    flx3 = 0
    runoff1 = 0
    runoff2 = 0
    runoff3 = 0
    
    snomlt = 0
    sncovr = 0
    rc = 0
    pc = 0
    rsmin = 0
    xlai = 0
    rcs = 0
    rct = 0
    rcq = 0
    
    rcsoil = 0
    soilw = 0
    soilm = 0
    smcwlt = 0
    smcdry = 0
    smcref = 0
    smcmax = 10
        

    # TODO
    return nroot, albedo, eta, sheat, ec, \
        edir, et, ett, esnow, drip, dew, beta, etp, ssoil, \
        flx1, flx2, flx3, runoff1, runoff2, runoff3, \
        snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq, \
        rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax


# *************************************
# 1st level subprograms
# *************************************


def alcalc(
    # inputs
    alb, snoalb, shdfac, shdmin, sncovr, tsnow,
    # outputs
    albedo
):
    # --- ... subprograms called: none
    # TODO
    pass


def canres(
    # inputs
    nsoil, nroot, swdn, ch, q2, q2sat, dqsdt2, sfctmp,
    cpx1, sfcprs, sfcems, sh2o, smcwlt, smcref, zsoil, rsmin,
    rsmax, topt, rgl, hs, xlai,
    # outputs
    rc, pc, rcs, rct, rcq, rcsoil
):
    # --- ... subprograms called: none
    # TODO
    pass


def csnow(
    # inputs
    sndens,
    # outputs
    sncond
):
    # --- ... subprograms called: none
    # TODO
    pass


def nopac(
    # inputs
    nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref,
    smcdry, cmcmax, dt, shdfac, sbeta, sfctmp, sfcems,
    t24, th2, fdown, epsca, bexp, pc, rch, rr, cfactr,
    slope, kdt, frzx, psisat, zsoil, dksat, dwsat,
    zbot, ice, rtdis, quartz, fxexp, csoil,
    # in/outs
    cmc, t1, stc, sh2o, tbot,
    # outputs
    eta, smc, ssoil, runoff1, runoff2, runoff3, edir,
    ec, et, ett, beta, drip, dew, flx1, flx3
):
    # --- ... subprograms called: evapo, smflx, tdfcnd, shflx
    # TODO
    pass


def penman(
    # inputs
    sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
    cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra,
    # outputs
    t24, etp, rch, epsca, rr, flx2
):
    # --- ... subprograms called: none
    # TODO
    pass


def redprm(
    # inputs
    nsoil, vegtyp, soiltyp, slopetyp, sldpth, zsoil,
    # outputs
    cfactr, cmcmax, rsmin, rsmax, topt, refkdt, kdt,
    sbeta, shdfac, rgl, hs, zbot, frzx, psisat, slope,
    snup, salp, bexp, dksat, dwsat, smcmax, smcwlt,
    smcref, smcdry, f1, quartz, fxexp, rtdis, nroot,
    z0, czil, xlai, csoil
):
    # --- ... subprograms called: none
    # TODO
    pass


def sfcdif(
    # inputs
    zlvl, z0, t1v, th2v, sfcspd, czil,
    # in/outs
    cm, ch
):
    # --- ... subprograms called: none
    # TODO
    pass


def snfrac(
    # inputs
    sneqv, snup, salp, snowh,
    # outputs
    sncovr
):
    # --- ... subprograms called: none
    # TODO
    pass


def snopac(
    # inputs
    nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref, smcdry,
    cmcmax, dt, df1, sfcems, sfctmp, t24, th2, fdown, epsca,
    bexp, pc, rch, rr, cfactr, slope, kdt, frzx, psisat,
    zsoil, dwsat, dksat, zbot, shdfac, ice, rtdis, quartz,
    fxexp, csoil, flx2, snowng,
    # in/outs
    prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh,
    sh2o, tbot, beta,
    # outputs
    smc, ssoil, runoff1, runoff2, runoff3, edir, ec, et,
    ett, snomlt, drip, dew, flx1, flx3, esnow
):
    # --- ... subprograms called: evapo, smflx, shflx, snowpack
    # TODO
    pass


def snow_new(
    # inputs
    sfctmp, sn_new,
    # in/outs
    snowh, sndens
):
    # --- ... subprograms called: none
    # TODO
    pass


def snowz0(
    # inputs
    sncovr,
    # in/outs
    z0
):
    # --- ... subprograms called: none
    # TODO
    pass


def tdfcnd(
    # inputs
    smc, qz, smcmax, sh2o,
    # outputs
    df
):
    # --- ... subprograms called: none
    # TODO
    pass


# *************************************
# 2nd level subprograms
# *************************************


def evapo(
    # inputs
    nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil,
    sh2o, smcmax, smcwlt, smcref, smcdry, pc,
    shdfac, cfactr, rtdis, fxexp,
    # outputs
    eta1, edir1, ec1, et1, ett1
):
    # --- ... subprograms called: devap, transp
    # TODO
    pass


def shflx(
    # inputs
    nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot,
    psisat, bexp, df1, ice, quartz, csoil, vegtyp,
    # in/outs
    stc, t1, tbot, sh2o,
    # outputs
    ssoil
):
    # --- ... subprograms called: hstep, hrtice, hrt
    # TODO
    pass


def smflx(
    # inputs
    nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1,
    zsoil, slope, frzx, bexp, dksat, dwsat, shdfac,
    edir1, ec1, et1,
    # in/outs
    cmc, sh2o,
    # outputs
    smc, runoff1, runoff2, runoff3, drip
):
    # --- ... subprograms called: srt, sstep
    # TODO
    pass


def snowpack(
    # inputs
    esd, dtsec, tsnow, tsoil,
    # in/outs
    snowh, sndens
):
    # --- ... subprograms called: none
    # TODO
    pass


# *************************************
# 3rd level subprograms
# *************************************


def devap(
    # inputs
    etp1, smc, shdfac, smcmax, smcdry, fxexp,
    # outputs
    edir1
):
    # --- ... subprograms called: none
    # TODO
    pass


def frh2o(
    # inputs
    tkelv, smc, sh2o, smcmax, bexp, psis,
    # outputs
    liqwat
):
    # --- ... subprograms called: none
    # TODO
    pass


def hrt(
    # inputs
    nsoil, stc, smc, smcmax, zsoil, yy, zz1, tbot,
    zbot, psisat, dt, bexp, df1, quartz, csoil, vegtyp,
    shdfac,
    # in/outs
    sh2o,
    # outputs
    rhsts, ai, bi, ci
):
    # --- ... subprograms called: tbnd, snksrc, tmpavg
    # TODO
    pass


def hrtice(
    # inputs
    nsoil, stc, zsoil, yy, zz1, df1, ice,
    # in/outs
    tbot,
    # outputs
    rhsts, ai, bi, ci
):
    # --- ... subprograms called: none
    # TODO
    pass


def hstep(
    # inputs
    nsoil, stcin, dt,
    # in/outs
    rhsts, ai, bi, ci,
    # outputs
    stcout
):
    # --- ... subprograms called: rosr12
    # TODO
    pass


def rosr12(
    # inputs
    nsoil, a, b, d,
    # in/outs
    c,
    # outputs
    p, delta
):
    # --- ... subprograms called: none
    # TODO
    pass


def snksrc(
    # inputs
    nsoil, k, tavg, smc, smcmax, psisat, bexp, dt,
    # in/outs
    sh2o,
    # outputs
    tsrc
):
    # --- ... subprograms called: frh2o
    # TODO
    pass


def srt(
    # inputs
    nsoil, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice,
    # outputs
    rhstt, runoff1, runoff2, ai, bi, ci
):
    # --- ... subprograms called: wdfcnd
    # TODO
    pass


def sstep(
    # inputs
    nsoil, sh2oin, rhsct, dt, smcmax, cmcmax, zsoil, sice,
    # in/outs
    cmc, rhstt, ai, bi, ci,
    # outputs
    sh2oout, runoff3, smc
):
    # --- ... subprograms called: rosr12
    # TODO
    pass


def tbnd(
    # inputs
    tu, tb, zsoil, zbot, k, nsoil,
    # outputs
    tbnd1
):
    # --- ... subprograms called: none
    # TODO
    pass


def tmpavg(
    # inputs
    tup, tm, tdn, zsoil, nsoil, k,
    # outputs
    tavg
):
    # --- ... subprograms called: none
    # TODO
    pass


def transp(
    # inputs
    nsoil, nroot, etp1, smc, smcwlt, smcref,
    cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis,
    # outputs
    et1
):
    # --- ... subprograms called: none
    # TODO
    pass


def wdfcnd(
    # inputs
    smc, smcmax, bexp, dksat, dwsat, sicemax,
    # outputs
    wdf, wcnd
):
    # --- ... subprograms called: none
    # TODO
    pass


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


# TODO: check iff correct, this is copied from seaice
# TODO - this should be moved into a shared physics functions module
def fpvs(t):
    """Compute saturation vapor pressure
       t: Temperature [K]
    fpvs: Vapor pressure [Pa]
    """

    # constants
    # TODO - this should be moved into a shared physics constants module
    con_psat = 6.1078e+2
    con_ttp = 2.7316e+2
    con_cvap = 1.8460e+3
    con_cliq = 4.1855e+3
    con_hvap = 2.5000e+6
    con_rv = 4.6150e+2
    con_csol = 2.1060e+3
    con_hfus = 3.3358e+5

    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)

    convert_to_scalar = False
    if np.isscalar(t):
        t = np.array(t)
        convert_to_scalar = True

    fpvs = np.empty_like(t)
    tr = con_ttp / t

    ind1 = t >= tliq
    fpvs[ind1] = con_psat * (tr[ind1]**xponal) * np.exp(xponbl*(1. - tr[ind1]))

    ind2 = t < tice
    fpvs[ind2] = con_psat * (tr[ind2]**xponai) * np.exp(xponbi*(1. - tr[ind2]))

    ind3 = ~np.logical_or(ind1, ind2)
    w = (t[ind3] - tice) / (tliq - tice)
    pvl = con_psat * (tr[ind3]**xponal) * np.exp(xponbl*(1. - tr[ind3]))
    pvi = con_psat * (tr[ind3]**xponai) * np.exp(xponbi*(1. - tr[ind3]))
    fpvs[ind3] = w * pvl + (1. - w) * pvi

    if convert_to_scalar:
        fpvs = fpvs.item()

    return fpvs


def serialbox_test(fortran_sol, py_sol, name):
    if(sum(fortran_sol - py_sol) == 0):
        print(name, "IS CORRECT")
    else:
        print(name, "IS FALSE!!!")
