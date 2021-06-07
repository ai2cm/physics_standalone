import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *


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

