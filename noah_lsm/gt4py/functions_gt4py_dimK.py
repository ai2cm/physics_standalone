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
def penman_fn(sfctmp, sfcprs, sfcemis, ch, t2v, th2, prcp, fdown,
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

    fnet = fdown - sfcemis*sigma1*t24 - ssoil

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
        dd += (zsoil[0, 0, -1] - zsoil[0, 0, 0]) * (smcav - (sh2o + sice - smcwlt))
        dice += (zsoil[0, 0, -1] - zsoil[0, 0, 0]) * sice

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
    ddz = 1.0 / (-.5*zsoil[0, 0, 1])
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
    denom2 = zsoil[0, 0, -1] - zsoil[0, 0, 0]
    denom = zsoil[0, 0, -1] - zsoil[0, 0, +1]
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
    denom2 = zsoil[0, 0, -1] - zsoil[0, 0, 0]
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
def rosr12_second_lowerboundary_fn(p, delta):
    p = delta
    return p


@gtscript.function
def rosr12_second_fn(p, delta):
    p = p[0, 0, 0] * p[0, 0, +1] + delta
    return p


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


@gtscript.function
def hrtice_upperboundary_fn(stc, zsoil, yy, zz1, df1, ice):
    # calculates the right hand side of the time tendency
    # term of the soil thermal diffusion equation for sea-ice or glacial-ice

    # set a nominal universal value of specific heat capacity
    if ice == 1:        # sea-ice
        hcpct = 1.72396e+6
    else:               # glacial-ice
        hcpct = 1.89000e+6

    # 1. Layer
    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5*zsoil[0, 0,+1])
    ai = 0.0
    ci = (df1*ddz) / (zsoil*hcpct)
    bi = -ci + df1 / (0.5*zsoil*zsoil*hcpct*zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc[0,0,0] - stc[0,0,+1]) / (-0.5*zsoil[0, 0,+1])
    ssoil = df1 * (stc - yy) / (0.5*zsoil*zz1)
    rhsts = (df1*dtsdz - ssoil) / (zsoil*hcpct)

    return rhsts, ai, bi, ci, dtsdz, ddz, hcpct


@gtscript.function
def hrtice_fn(stc, zsoil, df1, hcpct, dtsdz, ddz):
    # calculates the right hand side of the time tendency
    # term of the soil thermal diffusion equation for sea-ice or glacial-ice

    denom = 0.5 * (zsoil[0, 0,-1] - zsoil[0, 0,+1])
    dtsdz = (stc[0,0,0] - stc[0,0,1])/denom
    ddz = 2.0 / (zsoil[0, 0,-1] - zsoil[0, 0,+1])
    ci = - df1 * ddz / ((zsoil[0, 0,-1] - zsoil[0, 0,0])*hcpct)

    denom = ((zsoil[0, 0,0] - zsoil[0, 0,-1])*hcpct)
    rhsts = (df1*dtsdz - df1*dtsdz[0,0,-1]) / denom

    ai = - df1 * ddz[0,0,-1] / ((zsoil[0, 0,-1] - zsoil[0, 0,0])*hcpct)
    bi = -(ai + ci)

    return rhsts, ai, bi, ci, dtsdz, ddz


@gtscript.function
def hrtice_lowerboundary_fn(stc, zsoil, df1, hcpct, dtsdz, ddz, ice, tbot):
    # calculates the right hand side of the time tendency
    # term of the soil thermal diffusion equation for sea-ice or glacial-ice

    # set ice pack depth
    if ice == 1:
        zbot = zsoil
        tbot = 271.16
    else:
        zbot = -25.0

    dtsdz = (stc[0,0,0] - tbot)/(0.5 * (zsoil[0, 0,-1] - zsoil[0, 0,0]) - zbot)
    ci = 0.0

    denom = (zsoil[0, 0,0] - zsoil[0, 0,-1])*hcpct
    rhsts = (df1*dtsdz - df1*dtsdz[0,0,-1]) / denom

    ai = - df1 * ddz[0,0,-1] / ((zsoil[0, 0,-1] - zsoil[0, 0,0])*hcpct)
    bi = -(ai + ci)

    return rhsts, ai, bi, ci, tbot


@gtscript.function
def tmpavg_fn(tup, tm, tdn, dz):

    dzh = dz * 0.5

    if tup < tfreez:
        if tm < tfreez:
            if tdn < tfreez:
                tavg = (tup + 2.0*tm + tdn) / 4.0
            else:
                x0 = (tfreez - tm) * dzh / (tdn - tm)
                tavg = 0.5*(tup*dzh + tm*(dzh+x0) +
                            tfreez*(2.*dzh-x0)) / dz
        else:
            if tdn < tfreez:
                xup = (tfreez-tup) * dzh / (tm-tup)
                xdn = dzh - (tfreez-tm) * dzh / (tdn-tm)
                tavg = 0.5*(tup*xup + tfreez *
                            (2.*dz-xup-xdn)+tdn*xdn) / dz
            else:
                xup = (tfreez-tup) * dzh / (tm-tup)
                tavg = 0.5*(tup*xup + tfreez*(2.*dz-xup)) / dz
    else:
        if tm < tfreez:
            if tdn < tfreez:
                xup = dzh - (tfreez-tup) * dzh / (tm-tup)
                tavg = 0.5*(tfreez*(dz-xup) + tm *
                            (dzh+xup)+tdn*dzh) / dz
            else:
                xup = dzh - (tfreez-tup) * dzh / (tm-tup)
                xdn = (tfreez-tm) * dzh / (tdn-tm)
                tavg = 0.5 * (tfreez*(2.*dz-xup-xdn) +
                              tm*(xup+xdn)) / dz
        else:
            if tdn < tfreez:
                xdn = dzh - (tfreez-tm) * dzh / (tdn-tm)
                tavg = (tfreez*(dz-xdn) + 0.5*(tfreez+tdn)*xdn) / dz
            else:
                tavg = (tup + 2.0*tm + tdn) / 4.0
    return tavg


@gtscript.function
def frh2o_loop_fn(psisat, ck, swl, smcmax, smc, bx, tavg, error):
    df = log((psisat*gs2/lsubf) * ((1.0 + ck*swl)**2.0) *
             (smcmax / (smc - swl))**bx) - log(-(tavg - tfreez) / tavg)

    denom = 2.0*ck/(1.0 + ck*swl) + bx/(smc - swl)
    swlk = swl - df / denom

    # bounds useful for mathematical solution.
    swlk = max(min(swlk, smc-0.02), 0.0)

    # mathematical solution bounds applied.
    dswl = abs(swlk - swl)
    swl = swlk

    if dswl <= error:
        kcount = False

    free = smc - swl

    return kcount, free, swl


@gtscript.function
def frh2o_fn(psis, bexp, tavg, smc, sh2o, smcmax):
    ### ************ frh2o *********** ###
    # constant parameters
    ck = 8.0
    blim = 5.5
    error = 0.005
    bx = min(bexp, blim)

    kcount = True

    if tavg <= (tfreez - 1.e-3):
        swl = smc - sh2o
        swl = max(min(swl, smc-0.02), 0.0)

        kcount, free, swl = frh2o_loop_fn(
            psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error)

    else:
        free = smc

    return free


@gtscript.function
def snksrc_fn(free, psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dt, dz):

    # estimate the new amount of liquid water
    dh2o = 1.0000e3
    xh2o = sh2o + qtot*dt / (dh2o*lsubf*dz)

    if xh2o < sh2o and xh2o < free:
        if free > sh2o:
            xh2o = sh2o
        else:
            xh2o = free
    if xh2o > sh2o and xh2o > free:
        if free < sh2o:
            xh2o = sh2o
        else:
            xh2o = free

    xh2o = max(min(xh2o, smc), 0.0)
    tsnsr = -dh2o * lsubf * dz * (xh2o - sh2o) / dt
    sh2o = xh2o

    return tsnsr, sh2o



@gtscript.function
def hrt_upperboundary_fn(stc, smc, smcmax, zsoil, yy, zz1, psisat, dt, bexp, df1, 
                         csoil, ivegsrc, vegtype, shdfac, sh2o):
 
    csoil_loc = csoil

    if ivegsrc == 1 and vegtype == 12:
        csoil_loc = 3.0e6*(1.-shdfac)+csoil*shdfac

    # calc the heat capacity of the top soil layer
    hcpct = sh2o*cph2o2 + (1.0 - smcmax)*csoil_loc + \
        (smcmax - smc)*cp2 + (smc - sh2o)*cpice1

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5*zsoil[0, 0,+1])
    ai = 0.0
    ci = (df1*ddz) / (zsoil[0, 0,0]*hcpct)
    bi = -ci + df1 / (0.5*zsoil*zsoil*hcpct*zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc[0,0,0] - stc[0,0,+1]) / (-0.5*zsoil[0, 0,+1])
    ssoil = df1 * (stc[0,0,0] - yy) / (0.5*zsoil[0, 0,0]*zz1)
    rhsts = (df1*dtsdz - ssoil) / (zsoil[0, 0,0]*hcpct)

    # capture the vertical difference of the heat flux at top and
    # bottom of first soil layer
    qtot = ssoil - df1 * dtsdz

    tsurf = (yy + (zz1-1)*stc[0,0,0]) / zz1

    # linear interpolation between the average layer temperatures
    tbk = stc[0,0,0] + (stc[0,0,+1]-stc[0,0,0])*zsoil[0, 0,0]/zsoil[0, 0,+1]
    # calculate frozen water content in 1st soil layer.
    sice = smc[0,0,0] - sh2o[0,0,0]

    df1k = df1

    if sice > 0 or tsurf < tfreez or stc[0,0,0] < tfreez or tbk < tfreez:
        ### ************ tmpavg *********** ###
        dz = -zsoil[0, 0,0]
        tavg = tmpavg_fn(tsurf, stc[0,0,0], tbk, dz)
        ### ************ snksrc *********** ###
        free = frh2o_fn(psisat, bexp, tavg, smc, sh2o, smcmax)
        tsnsr, sh2o = snksrc_fn(
            free, psisat, bexp, tavg, smc[0,0,0], sh2o[0,0,0], smcmax, qtot, dt, dz)
        ### ************ END snksrc *********** ###

        rhsts -= tsnsr / (zsoil[0, 0,0] * hcpct)

    return sh2o, rhsts, ai, bi, ci, free, csoil_loc, tbk, df1k, dtsdz, ddz


@gtscript.function
def hrt_fn(stc, smc, smcmax, zsoil, psisat, dt, bexp, df1, quartz,
            tbk, df1k, dtsdz, ddz, ivegsrc, vegtype, shdfac, sh2o, free, csoil_loc):

    hcpct = sh2o*cph2o2 + (1.0 - smcmax)*csoil_loc + \
        (smcmax - smc)*cp2 + (smc - sh2o)*cpice1

    # calculate thermal diffusivity for each layer
    df1k = tdfcnd_fn(smc, quartz, smcmax, sh2o)

    if ivegsrc == 1 and vegtype == 12:
        df1k = 3.24*(1.-shdfac) + shdfac*df1k

    tbk = stc[0,0,+1] + (stc[0,0,+1]-stc[0,0,0])*(zsoil[0, 0,-1] - zsoil[0, 0,0])/(zsoil[0, 0,-1] - zsoil[0, 0,+1])
    # calc the vertical soil temp gradient thru each layer
    denom = 0.5 * (zsoil[0, 0,-1] - zsoil[0, 0,+1])
    dtsdz = (stc[0,0,0] - stc[0,0,+1]) / denom
    ddz = 2.0 / (zsoil[0, 0,-1] - zsoil[0, 0,+1])

    ci = - df1k*ddz / ((zsoil[0, 0,-1] - zsoil[0, 0,0]) * hcpct)

    # calculate rhsts
    denom = (zsoil[0, 0,0] - zsoil[0, 0,-1])*hcpct
    rhsts = (df1k*dtsdz[0,0,0] - df1k[0,0,-1]*dtsdz[0,0,-1])/denom

    qtot = -1. * denom * rhsts
    sice = smc - sh2o

    if sice > 0 or tbk[0,0,-1] < tfreez or stc < tfreez or tbk[0,0,0] < tfreez:
        ### ************ tmpavg *********** ###
        dz = zsoil[0, 0,-1] - zsoil[0, 0,0]
        tavg = tmpavg_fn(tbk[0,0,-1], stc, tbk[0,0,0], dz)
        ### ************ snksrc *********** ###
        tsnsr, sh2o = snksrc_fn(
            free, psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dt, dz)
        ### ************ END snksrc *********** ###
        rhsts -= tsnsr / denom

    # calc matrix coefs, ai, and bi for this layer.
    ai = - df1 * ddz[0,0,-1] / ((zsoil[0, 0,-1] - zsoil[0, 0,0]) * hcpct)
    bi = - (ai + ci)

    return sh2o, rhsts, ai, bi, ci, tbk, df1k, dtsdz, ddz


@gtscript.function
def hrt_lowerboundary_fn(stc, smc, smcmax, zsoil, psisat, dt, bexp, df1, quartz,
            tbk, df1k, dtsdz, ddz, ivegsrc, vegtype, shdfac, sh2o, free, csoil_loc, tbot, zbot):

    hcpct = sh2o*cph2o2 + (1.0 - smcmax)*csoil_loc + \
        (smcmax - smc)*cp2 + (smc - sh2o)*cpice1

    # calculate thermal diffusivity for each layer
    df1k = tdfcnd_fn(smc, quartz, smcmax, sh2o)

    if ivegsrc == 1 and vegtype == 12:
        df1k = 3.24*(1.-shdfac) + shdfac*df1k

    tbk = stc[0,0,0] + (tbot-stc[0,0,0])*(zsoil[0, 0,-1] - zsoil[0, 0,0])/(zsoil[0, 0,-1] + zsoil[0, 0,0] - 2. * zbot)
    # calc the vertical soil temp gradient thru each layer
    denom = 0.5 * (zsoil[0, 0,-1] + zsoil[0, 0,0]) - zbot
    dtsdz = (stc[0,0,0] - tbot) / denom

    ci = 0.0

    # calculate rhsts
    denom = (zsoil[0, 0,0] - zsoil[0, 0,-1])*hcpct
    rhsts = (df1k*dtsdz[0,0,0] - df1k[0,0,-1]*dtsdz[0,0,-1])/denom

    qtot = -1. * denom * rhsts
    sice = smc - sh2o

    if sice > 0 or tbk[0,0,-1] < tfreez or stc < tfreez or tbk[0,0,0] < tfreez:
        ### ************ tmpavg *********** ###
        dz = zsoil[0, 0,-1] - zsoil[0, 0,0]
        tavg = tmpavg_fn(tbk[0,0,-1], stc, tbk[0,0,0], dz)
        ### ************ snksrc *********** ###
        tsnsr, sh2o = snksrc_fn(
            free, psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dt, dz)
        ### ************ END snksrc *********** ###
        rhsts -= tsnsr / denom

    # calc matrix coefs, ai, and bi for this layer.
    ai = - df1 * ddz[0,0,-1] / ((zsoil[0, 0,-1] - zsoil[0, 0,0]) * hcpct)
    bi = - (ai + ci)

    return sh2o, rhsts, ai, bi, ci


@gtscript.function
def shflx_first_upperboundary_fn(smc, smcmax, dt, yy, zz1,
    zsoil, psisat, bexp, df1, ice, csoil, ivegsrc, vegtype, shdfac, stc, sh2o):

    stsoil = stc

    if ice != 0:  # sea-ice or glacial ice case
        rhsts, ai, bi, ci, dtsdz, ddz, hcpct = hrtice_upperboundary_fn(stc, zsoil, yy, zz1, df1, ice)

    else:
        sh2o, rhsts, ai, bi, ci, free, csoil_loc, tbk, df1k, dtsdz, ddz = hrt_upperboundary_fn(stc, smc, smcmax, zsoil, yy, zz1, psisat, dt, bexp, df1, 
                         csoil, ivegsrc, vegtype, shdfac, sh2o)

    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhsts *= dt

    p, delta = rosr12_first_upperboundary_fn(ai, bi, ci, rhsts)

    return stsoil, dtsdz, ddz, hcpct, sh2o, free, csoil_loc, p, delta, tbk, df1k, dtsdz, ddz


@gtscript.function
def shflx_first_fn(smc, smcmax, dt, zsoil, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, sh2o,
                    hcpct, dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta):

    stsoil = stc

    if ice != 0:  # sea-ice or glacial ice case
        rhsts, ai, bi, ci, dtsdz, ddz = hrtice_fn(stc, zsoil, df1, hcpct, dtsdz, ddz)

    else:
        sh2o, rhsts, ai, bi, ci, tbk, df1k, dtsdz, ddz = hrt_fn(stc, smc, smcmax, zsoil, psisat, dt, bexp, df1, quartz,
            tbk, df1k, dtsdz, ddz, ivegsrc, vegtype, shdfac, sh2o, free, csoil_loc)
    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhsts *= dt

    p, delta = rosr12_first_fn(ai, bi, ci, rhsts, p, delta)

    return stsoil, dtsdz, ddz, hcpct, sh2o, free, csoil_loc, p, delta, tbk, df1k, dtsdz, ddz   

 
@gtscript.function
def shflx_first_lowerboundary_fn(smc, smcmax, dt, zsoil, zbot, tbot, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, sh2o,
                    hcpct, dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta):

    stsoil = stc

    if ice != 0:  # sea-ice or glacial ice case
        rhsts, ai, bi, ci, tbot = hrtice_lowerboundary_fn(stc, zsoil, df1, hcpct, dtsdz, ddz, ice, tbot)

    else:
        sh2o, rhsts, ai, bi, ci = hrt_lowerboundary_fn(stc, smc, smcmax, zsoil, psisat, dt, bexp, df1, quartz,
            tbk, df1k, dtsdz, ddz, ivegsrc, vegtype, shdfac, sh2o, free, csoil_loc, tbot, zbot)
    ai *= dt
    bi = dt*bi + 1.
    ci *= dt
    rhsts *= dt

    p, delta = rosr12_first_fn(ai, bi, ci, rhsts, p, delta)

    return stsoil, sh2o, p, delta, tbot



@gtscript.function
def shflx_second_lowerboundary_fn(p, delta, stc, stsoil):
    ci = rosr12_second_lowerboundary_fn(p, delta)
    stc += ci

    ctfil1 = 0.5
    ctfil2 = 1.0 - ctfil1

    stc = ctfil1*stc + ctfil2*stsoil

    return stc


@gtscript.function
def shflx_second_fn(p, delta, stc, stsoil):
    ci = rosr12_second_fn(p, delta)
    stc += ci

    ctfil1 = 0.5
    ctfil2 = 1.0 - ctfil1

    stc = ctfil1*stc + ctfil2*stsoil

    return stc

@gtscript.function
def shflx_second_upperboundary_fn(p, delta, stc, stsoil, t1, yy, zz1, df1, zsoil):
    ci = rosr12_second_fn(p, delta)
    stc += ci

    ctfil1 = 0.5
    ctfil2 = 1.0 - ctfil1

    oldt1 = t1
    t1 = (yy + (zz1 - 1.0)*stc) / zz1
    t1 = ctfil1*t1 + ctfil2*oldt1

    stc = ctfil1*stc + ctfil2*stsoil
    
    # calculate surface soil heat flux
    ssoil = df1*(stc - t1) / (0.5*zsoil)

    return ssoil, stc, t1


@gtscript.function
def snowpack_fn(esd, dtsec, tsnow, tsoil, snowh, sndens):
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

    bfac = dthr*c1 * exp(0.08*tavgc - c2*sndens)

    # number of terms of polynomial expansion and its accuracy is governed by iteration limit "ipol".
    ipol = 4
    # hardcode loop for ipol = 4
    pexp = 0.0
    pexp = (1.0 + pexp)*bfac*esdcx/5.
    pexp = (1.0 + pexp)*bfac*esdcx/4.
    pexp = (1.0 + pexp)*bfac*esdcx/3.
    pexp = (1.0 + pexp)*bfac*esdcx/2.
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


