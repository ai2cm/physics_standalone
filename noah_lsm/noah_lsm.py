#!/usr/bin/env python3

import numpy as np

OUT_VARS = ["weasd", "snwdph", "tskin", "tprcp", "srflag", "smc", "stc", "slc", "canopy", \
            "trans", "tsurf", "zorl", "sncovr1", "qsurf", "gflux", "drain", "evap", "hflx", \
            "ep", "runoff", "cmm", "chh", "evbs", "evcw", "sbsno", "snowc", "stm", "snohf", \
            "smcwlt2", "smcref2", "wet1"]

def run(in_dict):
    """run function"""

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = in_dict[key].copy()
        del in_dict[key]

    sfc_drv(**in_dict, **out_dict)

    return out_dict


def sfc_drv(
    # inputs
    im, km, ps, t1, q1, soiltyp, vegtype, sigmaf,
    sfcemis, dlwflx, dswsfc, snet, delt, tg3, cm, ch,
    prsl1, prslki, zf, land, wind, slopetyp,
    shdmin, shdmax, snoalb, sfalb, flag_iter, flag_guess,
    lheatstrg, isot, ivegsrc,
    bexppert, xlaipert, vegfpert,pertvegf,
    # in/outs
    weasd, snwdph, tskin, tprcp, srflag, smc, stc, slc,
    canopy, trans, tsurf, zorl,
    # outputs
    sncovr1, qsurf, gflux, drain, evap, hflx, ep, runoff,
    cmm, chh, evbs, evcw, sbsno, snowc, stm, snohf,
    smcwlt2, smcref2, wet1
):  
    # --- ... subprograms called: ppfbet, sflx

    #TODO

    # more constant definitions

    # set constant parameters
    
    # initialize arrays

    # save land-related prognostic fields for guess run
    
    # initialization block
    
    # initialize variables

    # 1. configuration information

    # 2. forcing data

    # 3. other forcing data

    # 4. canopy/soil characteristics

    # 5.history (state) variables

    # call noah lsm
    # sflx( # inputs
    #       nsoil, couple, icein, ffrozp, dt, zlvl, sldpth,
    #       swdn, swnet, lwdn, sfcems, sfcprs, sfctmp,
    #       sfcspd, prcp, q2, q2sat, dqsdt2, th2, ivegsrc,
    #       vegtyp, soiltyp, slopetyp,
    #       # in/outs
    #       tbot, cmc, tsea, stsoil, smsoil, slsoil, sneqv, chx, cmx, z0,
    #       #outputs
    #       nroot, shdfac, snowh, albedo, eta, sheat, ec,
            # edir, et, ett, esnow, drip, dew, beta, etp, ssoil,
            # flx1, flx2, flx3, runoff1, runoff2, runoff3,
            # snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq,
            # rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax)

    # 6. output

    # compute qsurf

    # restore land-related prognostic fields for guess run
   
    pass


def ppfbet(pr,p,q,iflag,x):
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
    tbot, cmc, t1, stc, smc, sh2o, sneqv, ch, cm,z0,
    # outputs
    nroot, shdfac, snowh, albedo, eta, sheat, ec,
    edir, et, ett, esnow, drip, dew, beta, etp, ssoil,
    flx1, flx2, flx3, runoff1, runoff2, runoff3,
    snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq,
    rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax
):
    # --- ... subprograms called: redprm, snow_new, csnow, snfrac, alcalc, tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.
    #TODO
    pass


#*************************************
# 1st level subprograms
#*************************************


def alcalc(
    # inputs
    alb, snoalb, shdfac, shdmin, sncovr, tsnow,
    # outputs
    albedo
):  
    # --- ... subprograms called: none
    #TODO
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
    #TODO
    pass 


def csnow(
    # inputs
    sndens,
    # outputs
    sncond
):
    # --- ... subprograms called: none
    #TODO
    pass 


def nopac(
    # inputs
    nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref,
    smcdry, cmcmax, dt, shdfac, sbeta, sfctmp, sfcems,
    t24, th2, fdown, epsca, bexp, pc, rch, rr, cfactr,
    slope, kdt, frzx, psisat, zsoil, dksat, dwsat,
    zbot, ice, rtdis, quartz, fxexp, csoil,
    #in/outs
    cmc, t1, stc, sh2o, tbot,
    # outputs
    eta, smc, ssoil, runoff1, runoff2, runoff3, edir,
    ec, et, ett, beta, drip, dew, flx1, flx3
):
    # --- ... subprograms called: evapo, smflx, tdfcnd, shflx
    #TODO
    pass 


def penman(
    # inputs
    sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,
    cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra,
    # outputs
    t24, etp, rch, epsca, rr, flx2
):
    # --- ... subprograms called: none
    #TODO
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
    #TODO
    pass 


def sfcdif(
    # inputs
    zlvl, z0, t1v, th2v, sfcspd, czil,
    # in/outs
    cm, ch
):
    # --- ... subprograms called: none
    #TODO
    pass 


def snfrac(
    # inputs
    sneqv, snup, salp, snowh,
    # outputs
    sncovr
):
    # --- ... subprograms called: none
    #TODO
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
    #TODO
    pass 


def snow_new(
    # inputs
    sfctmp, sn_new,
    # in/outs
    snowh, sndens
):
    # --- ... subprograms called: none
    #TODO
    pass 


def snowz0(
    # inputs
    sncovr,
    # in/outs
    z0
):
    # --- ... subprograms called: none
    #TODO
    pass 


def tdfcnd(
    # inputs
    smc, qz, smcmax, sh2o,
    # outputs
    df
):
    # --- ... subprograms called: none
    #TODO
    pass 


#*************************************
# 2nd level subprograms
#*************************************


def evapo(
    # inputs
    nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil,
    sh2o, smcmax, smcwlt, smcref, smcdry, pc,
    shdfac, cfactr, rtdis, fxexp,
    # outputs
    eta1, edir1, ec1, et1, ett1
):
    # --- ... subprograms called: devap, transp
    #TODO
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
    #TODO
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
    #TODO
    pass 


def snowpack(
    # inputs
    esd, dtsec, tsnow, tsoil,
    # in/outs
    snowh, sndens
):
    # --- ... subprograms called: none
    #TODO
    pass 


#*************************************
# 3rd level subprograms
#*************************************


def devap(
    # inputs
    etp1, smc, shdfac, smcmax, smcdry, fxexp,
    # outputs
    edir1
):
    # --- ... subprograms called: none
    #TODO
    pass 


def frh2o(
    # inputs
    tkelv, smc, sh2o, smcmax, bexp, psis,
    # outputs
    liqwat
):
    # --- ... subprograms called: none
    #TODO
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
    #TODO
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
    #TODO
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
    #TODO
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
    #TODO
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
    #TODO
    pass 


def srt(
    # inputs
    nsoil, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice,
    # outputs
    rhstt, runoff1, runoff2, ai, bi, ci
):
    # --- ... subprograms called: wdfcnd
    #TODO
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
    #TODO
    pass 


def tbnd(
    # inputs
    tu, tb, zsoil, zbot, k, nsoil,
    # outputs
    tbnd1
):
    # --- ... subprograms called: none
    #TODO
    pass 


def tmpavg(
    # inputs
    tup, tm, tdn, zsoil, nsoil, k,
    # outputs
    tavg
):
    # --- ... subprograms called: none
    #TODO
    pass 


def transp(
    # inputs
    nsoil, nroot, etp1, smc, smcwlt, smcref,
    cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis,
    # outputs
    et1
):
    # --- ... subprograms called: none
    #TODO
    pass 


def wdfcnd(
    # inputs
    smc, smcmax, bexp, dksat, dwsat, sicemax,
    # outputs
    wdf, wcnd
):
    # --- ... subprograms called: none
    #TODO
    pass 