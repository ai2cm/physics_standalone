import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *
from functions_gt4py_dimK import *

DT_F2 = gtscript.Field[np.float64]
DT_I2 = gtscript.Field[np.int32]
DT_F = gtscript.Field[gtscript.IJ, np.float64]
DT_I = gtscript.Field[gtscript.IJ, np.int32]
BACKEND = "gtx86"


@gtscript.stencil(backend=BACKEND)
def prepare_sflx(
    zsoil: DT_F2, km: int,
    # output 1D
    zsoil_root: DT_F, q0: DT_F, cmc: DT_F, th2: DT_F, rho: DT_F, qs1: DT_F, ice: DT_I,
    prcp: DT_F, dqsdt2: DT_F, snowh: DT_F, sneqv: DT_F, chx: DT_F, cmx: DT_F, z0: DT_F,
    shdfac: DT_F, kdt: DT_F, frzx: DT_F, sndens: DT_F, snowng: DT_I, prcp1: DT_F,
    sncovr: DT_F, df1: DT_F, ssoil: DT_F, t2v: DT_F, fdown: DT_F, cpx1: DT_F,
    weasd_old: DT_F, snwdph_old: DT_F, tskin_old: DT_F, tprcp_old: DT_F, srflag_old: DT_F,
    canopy_old: DT_F, etp: DT_F, t24: DT_F, rch: DT_F, epsca: DT_F, rr: DT_F, flx2: DT_F, tsea: DT_F,
    # output 2D
    sldpth: DT_F2, rtdis: DT_F2, smc_old: DT_F2, stc_old: DT_F2, slc_old: DT_F2,
    # input output
    cmm: DT_F, chh: DT_F, tsurf: DT_F, t1: DT_F, q1: DT_F, vegtype: DT_I, sigmaf: DT_F,
    sfcemis: DT_F, dlwflx: DT_F, snet: DT_F, cm: DT_F, ch: DT_F,
    prsl1: DT_F, prslki: DT_F, land: DT_I, wind: DT_F, snoalb: DT_F, sfalb: DT_F, flag_iter: DT_I, flag_guess: DT_I,
    bexppert: DT_F, xlaipert: DT_F, fpvs: DT_F,
    weasd: DT_F, snwdph: DT_F, tskin: DT_F, tprcp: DT_F, srflag: DT_F, smc: DT_F2, stc: DT_F2, slc: DT_F2,
    canopy: DT_F, zorl: DT_F, delt: float, ivegsrc: int, bexp: DT_F, dksat: DT_F, rsmin: DT_F, quartz: DT_F, smcdry: DT_F,
    smcmax: DT_F, smcref: DT_F, smcwlt: DT_F, nroot: DT_I, snup: DT_F, xlai: DT_F
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
                sldpth = zsoil[0, 0, -1] - zsoil[0, 0, 0]
                # root distribution
                if nroot <= count:
                    rtdis = 0.0
                    zsoil_root = zsoil[0, 0, -1]
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
                sldpth = zsoil[0, 0, -1] - zsoil[0, 0, 0]
                count = count[0, 0, +1] - 1
                # root distribution
                if nroot <= count:
                    rtdis = 0.0
                    zsoil_root = zsoil[0, 0, -1]
                else:
                    rtdis = - sldpth / zsoil_root

                # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
                # as a flag for non-soil medium
                sh2o = slc
                if ice != 0:
                    smc = 1.
                    sh2o = 1.

        with interval(0, 1):
            ### ------- PREPARATION ------- ###
            # save land-related prognostic fields for guess run
            if land and flag_guess:
                smc_old = smc
                stc_old = stc
                slc_old = slc
            ### ------- END PREPARATION ------- ###

            
            if flag_iter and land:
                # thickness of first soil layer
                sldpth = - zsoil[0, 0, 0]
                # root distribution
                count = count[0, 0, +1] - 1
                if nroot <= count:
                    rtdis = 0.0
                else:
                    rtdis = - sldpth / zsoil_root

                ffrozp = srflag
                dt = delt
                sfctmp = t1
                tsea = tsurf
                q2 = q0
                sfcemis = sfcemis
                sfcprs = prsl1
                th2 = t1 * prslki
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
                frzgra = (prcp > 0.) and (ffrozp <= 0.) and (tsea <= tfreez)

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
                    ssoil = df1 * (tsea - stc) / dsoil
                else:
                    dtot = snowh + dsoil
                    frcsno = snowh / dtot
                    frcsoi = dsoil / dtot

                    # arithmetic mean (parallel flow)
                    df1a = frcsno*sncond + frcsoi*df1

                    # geometric mean (intermediate between harmonic and arithmetic mean)
                    df1 = df1a*sncovr + df1 * (1.0-sncovr)

                    # calculate subsurface heat flux
                    ssoil = df1 * (tsea - stc) / dtot

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

                t24, etp, rch, epsca, rr, flx2 = penman_fn(sfctmp, sfcprs, sfcemis, chx, t2v, th2, prcp, fdown,
                                                           cpx, cpfac, ssoil, q2, q2sat, dqsdt2, snowng, frzgra, ffrozp)


@gtscript.stencil(backend=BACKEND)
def canres(dswsfc: DT_F, nroot: DT_I, count: DT_I, chx: DT_F, q2: DT_F, q2sat: DT_F, dqsdt2: DT_F, sfctmp: DT_F,
           cpx1: DT_F, sfcprs: DT_F, sfcemis: DT_F, sh2o: DT_F2, smcwlt: DT_F, smcref: DT_F, zsoil: DT_F2, rsmin: DT_F,
           rgl: DT_F, hs: DT_F, xlai: DT_F, flag_iter: DT_I, land: DT_I, zsoil_root: DT_F, shdfac: DT_F,
           rc: DT_F, pc: DT_F, rcs: DT_F, rct: DT_F, rcq: DT_F, rcsoil: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:

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
                    else: 
                        gx = 0.0
                    sum = zsoil[0, 0, 0] * gx / zsoil_root

        with interval(1, -1):
            if flag_iter and land:
                if shdfac > 0.0:
                    count += 1

                    if nroot > count:
                        gx = max(0.0, min(1.0, (sh2o-smcwlt)/(smcref - smcwlt)))
                    else: 
                        gx = 0.0

                    sum = sum[0, 0, -1] + \
                        (zsoil[0, 0, 0] - zsoil[0, 0, -1]) * gx / zsoil_root

        with interval(-1, None):
            if flag_iter and land:
                if shdfac > 0.0:
                    count += 1

                    if nroot > count:
                        gx = max(0.0, min(1.0, (sh2o-smcwlt)/(smcref - smcwlt)))
                    else: 
                        gx = 0.0

                    sum = sum[0, 0, -1] + \
                        (zsoil[0, 0, 0] - zsoil[0, 0, -1]) * gx / zsoil_root

                    rcsoil = max(sum, 0.0001)

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
                        (sfctmp**4.0)/(sfcprs*chx) + 1.0
                    delta = (lsubc/cpx1) * dqsdt2

                    pc = (rr + delta) / (rr*(1.0 + rc*chx) + delta)


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_first(etp: DT_F, nroot: DT_I,  smcmax: DT_F, smcwlt: DT_F, smcref: DT_F,
                      smcdry: DT_F, dt: float, shdfac: DT_F,
                      cmc: DT_F, sh2o: DT_F2,
                      flag_iter: DT_I, land: DT_I, sneqv: DT_F, count: DT_I,
                      # output
                      sgx: DT_F, gx: DT_F2, edir1: DT_F, ec1: DT_F, edir: DT_F, ec: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                count = 0

                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    edir1 = 0.0
                    ec1 = 0.0
                    if etp > 0.0:
                        edir1, ec1, sgx, gx = evapo_boundarylayer_fn(nroot, cmc, cmcmax, etp1, dt, sh2o, smcmax, smcwlt,
                                                                     smcref, smcdry, shdfac, cfactr, fxexp, count, sgx)
                    edir = edir1 * 1000.0
                    ec = ec1 * 1000.0
        with interval(1, None):
            if flag_iter and land:
                count += 1
                if sneqv == 0.0:
                    if etp > 0.0:
                        sgx, gx = evapo_first_fn(
                            nroot, etp1, sh2o, smcwlt, smcref, shdfac, count, sgx)


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_second(shdfac: DT_F, etp: DT_F, flag_iter: DT_I, land: DT_I,
                       sneqv: DT_F, count: DT_I, rtdis: DT_F2, sgx: DT_F,
                       # output
                       denom: DT_F, gx: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)

        with interval(1, -1):
            count += 1
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)
        
        with interval(-1, None):
            count += 1
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)

                        if denom <= 0.0:
                            denom = 1.0


@gtscript.stencil(backend=BACKEND)
def nopac_evapo_third(shdfac: DT_F, etp: DT_F, flag_iter: DT_I, land: DT_I,
                      sneqv: DT_F, count: DT_I, gx: DT_F2, cmc: DT_F, pc: DT_F,
                      denom: DT_F, edir1: DT_F, ec1: DT_F,
                      # output
                      et1: DT_F2, ett1: DT_F, eta1: DT_F, et: DT_F2, ett: DT_F, eta: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                et = et1 * 1000.0
        with interval(1, -1):
            count += 1
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                et = et1 * 1000.0

        with interval(-1, None):
            count += 1
            if flag_iter and land:
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)

                        eta1 = edir1 + ett1 + ec1
                    et = et1 * 1000.0
                    eta = eta1 * 1000.0
                    ett = ett1 * 1000.0


@gtscript.stencil(backend=BACKEND)
def nopac_smflx_first(smcmax: DT_F, dt: float, smcwlt: DT_F, prcp: DT_F, prcp1: DT_F, zsoil: DT_F2, shdfac: DT_F, ec1: DT_F,
                      cmc: DT_F, sh2o: DT_F2, smc: DT_F2, flag_iter: DT_I, land: DT_I, etp: DT_F, sneqv: DT_F,
                      # output
                      sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, rhsct: DT_F, drip: DT_F, sice: DT_F2, dew: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    prcp1 = prcp * 0.001
                    if etp <= 0.0:
                        etp1 = etp * 0.001
                        dew = -etp1
                        prcp1 += dew

                    rhsct, drip, sice, sicemax, dd, dice, pcpdrp, sice = smflx_first_upperboundary_fn(
                        dt, smcmax, smcwlt, cmcmax, prcp1, zsoil, ec1, cmc,
                        sh2o, smc, shdfac, sicemax, dd, dice, pcpdrp, rhsct)

        with interval(1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    sicemax, dd, dice, sice = smflx_first_fn(smcmax, smcwlt, zsoil, sh2o, smc, pcpdrp, sicemax,
                                                             dd, dice)


@gtscript.stencil(backend=BACKEND)
def nopac_smflx_second(smcmax: DT_F, etp: DT_F, dt: float, bexp: DT_F, kdt: DT_F, frzx: DT_F, dksat: DT_F, dwsat: DT_F, slope: DT_F,
                       zsoil: DT_F2, sh2o: DT_F2, flag_iter: DT_I, land: DT_I, sneqv: DT_F, sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, edir1: DT_F, et1: DT_F2,
                       ai: DT_F2, bi: DT_F2,
                       # outpus
                       rhstt: DT_F2, ci: DT_F2, runoff1: DT_F, runoff2: DT_F, dsmdz: DT_F2, ddz: DT_F2, wdf: DT_F2, wcnd: DT_F2, p: DT_F2, delta: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    rhstt, ci, runoff1, dsmdz, ddz, wdf, wcnd, p, delta, ai, bi = smflx_second_upperboundary_fn(edir1, et1, sh2o, pcpdrp, zsoil, dwsat,
                                                                                                        dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice, ai, bi,
                                                                                                        rhstt, runoff1, ci, dsmdz, ddz, wdf, wcnd, p, delta)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv == 0.0:
                    rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta, ai, bi = smflx_second_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, ai, bi,
                                                                                    rhstt, ci, dsmdz, dsmdz[0,0,-1], ddz, ddz[0, 0, -1], wdf, wdf[0,0,-1], wcnd, wcnd[0,0,-1], p, delta, p[0,0,-1], delta[0,0,-1])

        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    rhstt, ci, runoff2, p, delta, ai, bi, dsmdz, ddz, wdf, wcnd = smflx_second_lowerboundary_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, slope, ai, bi,
                                                                                    rhstt, ci, runoff2, dsmdz, dsmdz[0,0,-1], ddz, ddz[0, 0, -1], wdf, wdf[0,0,-1], wcnd, wcnd[0,0,-1], p, delta, p[0,0,-1], delta[0,0,-1])


@gtscript.stencil(backend=BACKEND)
def rosr12_second(delta: DT_F2, p: DT_F2, flag_iter: DT_I, land: DT_I, ci: DT_F2):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD):
        with interval(-1, None):
            if flag_iter and land:
                ci = delta

        with interval(0, -1):
            if flag_iter and land:
                ci = p[0, 0, 0] * ci[0, 0, +1] + delta


@gtscript.stencil(backend=BACKEND)
def nopac_smflx_third(dt: float, smcmax: DT_F, flag_iter: DT_I, land: DT_I, sice: DT_F2, zsoil: DT_F2, rhsct: DT_F, ci: DT_F2,
                      sneqv: DT_F,
                      # outpus
                      sh2o: DT_F2, smc: DT_F2, cmc: DT_F, runoff3: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:            
                    wplus, smc, sh2o = sstep_upperboundary_fn(
                        sh2o, smc, smcmax, sice, ci, zsoil)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv == 0.0:
                    wplus, smc, sh2o = sstep_fn(
                        sh2o, smc, smcmax, sice, ci, zsoil, wplus, wplus[0,0,-1])

        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    wplus, smc, sh2o = sstep_fn(
                        sh2o, smc, smcmax, sice, ci, zsoil, wplus, wplus[0,0,-1])
                    runoff3 = wplus

                    # update canopy water content/interception
                    cmc += dt * rhsct
                    if cmc < 1.e-20:
                        cmc = 0.0
                    cmc = min(cmc, cmcmax)


@gtscript.stencil(backend=BACKEND)
def nopac_shflx_first(land: DT_I, flag_iter: DT_I, sneqv: DT_F, etp: DT_F, eta: DT_F, smc: DT_F2, quartz: DT_F, smcmax: DT_F, ivegsrc: int,
                      vegtype: DT_I, shdfac: DT_F, fdown: DT_F, sfcemis: DT_F, t24: DT_F, sfctmp: DT_F, rch: DT_F, th2: DT_F, 
                      epsca: DT_F, rr: DT_F, zsoil: DT_F2, dt: float, psisat: DT_F, bexp: DT_F, ice: DT_I, stc: DT_F2, 
                      hcpct:DT_F2, tbk: DT_F2, df1k: DT_F2, dtsdz: DT_F2, ddz: DT_F2, csoil_loc: DT_F,
                      # output
                      sh2o: DT_F2, stsoil: DT_F2, rhsts: DT_F2, ai: DT_F2, bi: DT_F2, ci: DT_F2, p: DT_F2, delta: DT_F2, tbot: DT_F, yy: DT_F, zz1: DT_F, df1: DT_F, beta: DT_F):
    from __gtscript__ import FORWARD, computation, interval
    # from __externals__ import tdfcnd_fn, shflx_first_upperboundary_fn, shflx_first_lowerboundary_fn

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    # based on etp and e values, determine beta
                    if etp < 0.0:
                        beta = 1.0
                    elif etp == 0.0:
                        beta = 0.0
                    else:
                        beta = eta / etp

                    # get soil thermal diffuxivity/conductivity for top soil lyr, calc.
                    df1 = tdfcnd_fn(smc, quartz, smcmax, sh2o)

                    if (ivegsrc == 1) and (vegtype == 12):
                        df1 = 3.24*(1.-shdfac) + shdfac*df1*exp(sbeta*shdfac)
                    else:
                        df1 *= exp(sbeta*shdfac)

                    # compute intermediate terms passed to routine hrt
                    yynum = fdown - sfcemis*sigma1*t24
                    yy = sfctmp + (yynum/rch + th2 - sfctmp - beta*epsca)/rr
                    zz1 = df1/(-0.5*zsoil*rch*rr) + 1.0

                    stsoil, rhsts, ai, bi, ci, dtsdz, ddz, hcpct, sh2o, free, csoil_loc, p, delta, tbk, df1k, dtsdz, ddz = shflx_first_upperboundary_fn(smc, smcmax, dt, yy, zz1,
                                                                                                                                     zsoil, psisat, bexp, df1, hcpct, tbk, df1k, dtsdz, ddz,
                                                                                                                            ice, csoil, ivegsrc, vegtype, shdfac, stc, sh2o, p, delta, rhsts, ai, bi, ci)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv == 0.0:

                    stsoil, rhsts, ai, bi, ci, dtsdz, ddz, hcpct, sh2o, free, p, delta, tbk, df1k, dtsdz, ddz = shflx_first_fn(smc, smcmax, dt, zsoil, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, stc[0,0,+1], sh2o,
                    hcpct, tbk[0,0,-1], df1k[0,0,-1], dtsdz[0,0,-1], ddz[0,0,-1], p[0,0,-1], delta[0,0,-1], dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta, rhsts, ai, bi, ci)
        
        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:

                    stsoil, rhsts, ai, bi, ci, sh2o, p, delta, tbot = shflx_first_lowerboundary_fn(smc, smcmax, dt, zsoil, zbot, tbot, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, sh2o,
                    hcpct, tbk[0,0,-1], df1k[0,0,-1], dtsdz[0,0,-1], ddz[0,0,-1], p[0,0,-1], delta[0,0,-1], dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta, rhsts, ai, bi, ci)
            

@gtscript.stencil(backend=BACKEND)
def nopac_shflx_second(land: DT_I, flag_iter: DT_I, sneqv: DT_F, zsoil: DT_F2, stsoil: DT_F2, p: DT_F2, delta: DT_F2, 
                        yy: DT_F, zz1: DT_F, df1: DT_F,
                        # output
                        ssoil: DT_F, tsea: DT_F, stc: DT_F2):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD):
        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    stc, p = shflx_second_lowerboundary_fn(p, delta, stc, stsoil)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv == 0.0:
                    stc, p = shflx_second_fn(p, p[0,0,+1], delta, stc, stsoil)

        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
                    ssoil, stc, tsea = shflx_second_upperboundary_fn(p, p[0,0,+1], delta, stc, stsoil, tsea, yy, zz1, df1, zsoil)


@gtscript.stencil(backend=BACKEND)
def snopac_evapo_first(ice: DT_I, sncovr: DT_F, etp: DT_F, nroot: DT_I, smcmax: DT_F, smcwlt: DT_F, smcref: DT_F,
                      smcdry: DT_F, dt: float, shdfac: DT_F,cmc: DT_F, sh2o: DT_F2,
                      flag_iter: DT_I, land: DT_I, sneqv: DT_F, count: DT_I,
                      # output
                      sgx: DT_F, gx: DT_F2, edir1: DT_F, ec1: DT_F, edir: DT_F, ec: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                count = 0
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    edir1 = 0.0
                    ec1 = 0.0

                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        edir1, ec1, sgx, gx = evapo_boundarylayer_fn(nroot, cmc, cmcmax, etp1, dt, sh2o, smcmax, smcwlt,
                                                                     smcref, smcdry, shdfac, cfactr, fxexp, count, sgx)


        with interval(1, None):
            if flag_iter and land:
                count += 1
                if sneqv != 0.0:
                    etp1 = etp * 0.001

                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        sgx, gx = evapo_first_fn(
                            nroot, etp1, sh2o, smcwlt, smcref, shdfac, count, sgx)


@gtscript.stencil(backend=BACKEND)
def snopac_evapo_second(ice: DT_I, sncovr: DT_F, shdfac: DT_F, etp: DT_F, flag_iter: DT_I, land: DT_I,
                       sneqv: DT_F, count: DT_I, rtdis: DT_F2, sgx: DT_F,
                       # output
                       denom: DT_F, gx: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0
            if flag_iter and land:
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)

        with interval(1, -1):
            count += 1
            if flag_iter and land:
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)
        
        with interval(-1, None):
            count += 1
            if flag_iter and land:
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        denom, gx = evapo_second_fn(
                            etp1, shdfac, rtdis, gx, sgx, denom)

                        if denom <= 0.0:
                            denom = 1.0


@gtscript.stencil(backend=BACKEND)
def snopac_evapo_third(ice: DT_I, sncovr: DT_F, edir: DT_F, ec: DT_F, shdfac: DT_F, etp: DT_F, flag_iter: DT_I, land: DT_I,
                      sneqv: DT_F, count: DT_I, gx: DT_F2, cmc: DT_F, pc: DT_F,
                      denom: DT_F, edir1: DT_F, ec1: DT_F,
                      # output
                      et1: DT_F2, ett1: DT_F, eta1: DT_F, et: DT_F2, ett: DT_F, eta: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0
            if flag_iter and land:
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                        et1 *= 1.0 - sncovr
                    et = et1 * 1000.0
                count += 1

        with interval(1, -1):
            if flag_iter and land:
                count += 1
                if sneqv != 0.0:
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                        et1 *= 1.0 - sncovr
                        
                    et = et1 * 1000.0

        with interval(-1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    count += 1
                    etp1 = etp * 0.001
                    if etp >= 0.0 and ice == 0 and sncovr < 1.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)

                        eta1 = edir1 + ett1 + ec1
                            
                        eta1 *= 1.0 - sncovr
                        ett1 *= 1.0 - sncovr
                        et1 *= 1.0 - sncovr
                        edir1 *= 1.0 - sncovr
                        ec1 *= 1.0 - sncovr

                    et = et1 * 1000.0
                    eta = eta1 * 1000.0
                    ett = ett1 * 1000.0
                    edir = edir1 * 1000.0
                    ec = ec1 * 1000.0


@gtscript.stencil(backend=BACKEND)
def snopac_smflx_first(smcmax: DT_F, dt: float, smcwlt: DT_F, prcp: DT_F, prcp1: DT_F, zsoil: DT_F2, shdfac: DT_F, ec1: DT_F,
                      cmc: DT_F, sh2o: DT_F2, smc: DT_F2, flag_iter: DT_I, land: DT_I, etp: DT_F,
                      sncovr: DT_F, ice: DT_I, snowng: DT_I, ffrozp: DT_F, sfctmp: DT_F, eta: DT_F, snowh: DT_F,
                      df1: DT_F, rr: DT_F, rch: DT_F, fdown: DT_F, flx2: DT_F, sfcemis: DT_F, t24: DT_F, th2: DT_F, stc: DT_F2,
                      # output
                      sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, rhsct: DT_F, drip: DT_F, sice: DT_F2, dew: DT_F,
                      tsea: DT_F, sneqv: DT_F, sneqv_new: DT_F, flx1: DT_F, flx3: DT_F, esnow: DT_F, ssoil: DT_F, snomlt: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                
                if sneqv != 0.0:
                    sneqv_new = sneqv
                    snoexp = 2.0
                    esdmin = 1.e-6
                    prcp1 = prcp1 * 0.001
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
                            esnow = etp * sncovr
                            esnow1 = esnow * 0.001
                            esnow2 = esnow1 * dt
                            etanrg = esnow*lsubs + eta*lsubc


                    # if precip is falling, calculate heat flux from snow sfc to newly accumulating precip
                    flx1 = 0.0
                    if snowng:
                        # fractional snowfall/rainfall
                        flx1 = (cpice * ffrozp + cph2o1*(1.-ffrozp)) * prcp * (tsea - sfctmp)

                    elif prcp > 0.0:
                        flx1 = cph2o1 * prcp * (tsea - sfctmp)

                    # calculate an 'effective snow-grnd sfc temp' based on heat fluxes between
                    # the snow pack and the soil and on net radiation.
                    dsoil = -0.5 * zsoil
                    dtot = snowh + dsoil
                    denom = 1.0 + df1 / (dtot * rr * rch)
                    t12a = ((fdown - flx1 - flx2 - sfcemis*sigma1*t24) /
                            rch + th2 - sfctmp - etanrg / rch) / rr
                    t12b = df1 * stc / (dtot * rr * rch)
                    t12 = (sfctmp + t12a + t12b) / denom

                    if t12 <= tfreez:    # no snow melt will occur.

                        # set the skin temp to this effective temp
                        tsea = t12
                        # update soil heat flux
                        ssoil = df1 * (tsea - stc) / dtot
                        # update depth of snowpack
                        sneqv_new = max(0.0, sneqv_new-esnow2)
                        flx3 = 0.0
                        ex = 0.0
                        snomlt = 0.0

                    else:    # snow melt will occur.
                        tsea = tfreez * max(0.01, sncovr**snoexp) + t12 * \
                            (1.0 - max(0.01, sncovr**snoexp))
                        ssoil = df1 * (tsea - stc) / dtot

                        if sneqv_new - esnow2 <= esdmin:
                            # snowpack has sublimated away, set depth to zero.
                            sneqv_new = 0.0
                            ex = 0.0
                            snomlt = 0.0
                            flx3 = 0.0

                        else:
                            # potential evap (sublimation) less than depth of snowpack
                            sneqv_new -= esnow2
                            seh = rch * (tsea - th2)

                            t14 = tsea * tsea
                            t14 = t14 * t14

                            flx3 = fdown - flx1 - flx2 - sfcemis*sigma1*t14 - ssoil - seh - etanrg
                            if flx3 <= 0.0:
                                flx3 = 0.0

                            ex = flx3 * 0.001 / lsubf

                            # snowmelt reduction
                            snomlt = ex * dt

                            if sneqv_new - snomlt >= esdmin:
                                # retain snowpack
                                sneqv_new -= snomlt
                            else:
                                # snowmelt exceeds snow depth
                                ex = sneqv_new / dt
                                flx3 = ex * 1000.0 * lsubf
                                snomlt = sneqv_new
                                sneqv_new = 0.0

                        if ice == 0:
                            prcp1 += ex

                    if ice == 0:
                        rhsct, drip, sice, sicemax, dd, dice, pcpdrp, sice = smflx_first_upperboundary_fn(
                            dt, smcmax, smcwlt, cmcmax, prcp1, zsoil, ec1, cmc,
                            sh2o, smc, shdfac, sicemax, dd, dice, pcpdrp, rhsct)

        with interval(1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0:
                        sicemax, dd, dice, sice = smflx_first_fn(smcmax, smcwlt, zsoil, sh2o, smc, pcpdrp, sicemax,
                                                                dd, dice)


@gtscript.stencil(backend=BACKEND)
def snopac_smflx_second(ice: DT_I, smcmax: DT_F, etp: DT_F, dt: float, bexp: DT_F, kdt: DT_F, frzx: DT_F, dksat: DT_F, dwsat: DT_F, slope: DT_F,
                       zsoil: DT_F2, sh2o: DT_F2, flag_iter: DT_I, land: DT_I, sneqv: DT_F, sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, edir1: DT_F, et1: DT_F2,
                       ai: DT_F2, bi: DT_F2,
                       # outpus
                       rhstt: DT_F2, ci: DT_F2, runoff1: DT_F, runoff2: DT_F, dsmdz: DT_F2, ddz: DT_F2, wdf: DT_F2, wcnd: DT_F2, p: DT_F2, delta: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:

                        rhstt, ci, runoff1, dsmdz, ddz, wdf, wcnd, p, delta, ai, bi = smflx_second_upperboundary_fn(edir1, et1, sh2o, pcpdrp, zsoil, dwsat,
                                                                                                            dksat, smcmax, bexp, dt, kdt, frzx, sicemax, dd, dice, ai, bi,
                                                                                                            rhstt, runoff1, ci, dsmdz, ddz, wdf, wcnd, p, delta)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:
                        rhstt, ci, dsmdz, ddz, wdf, wcnd, p, delta, ai, bi = smflx_second_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, ai, bi,
                                                                                     rhstt, ci, dsmdz, dsmdz[0,0,-1], ddz, ddz[0, 0, -1], wdf, wdf[0,0,-1], wcnd, wcnd[0,0,-1], p, delta, p[0,0,-1], delta[0,0,-1])

        with interval(-1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:
                        rhstt, ci, runoff2, p, delta, ai, bi, dsmdz, ddz, wdf, wcnd = smflx_second_lowerboundary_fn(et1, sh2o, zsoil, dwsat, dksat, smcmax, bexp, dt, sicemax, slope, ai, bi,
                                                                                     rhstt, ci, runoff2, dsmdz, dsmdz[0,0,-1], ddz, ddz[0, 0, -1], wdf, wdf[0,0,-1], wcnd, wcnd[0,0,-1], p, delta,  p[0,0,-1], delta[0,0,-1])


@gtscript.stencil(backend=BACKEND)
def snopac_smflx_third(ice: DT_I,  dt: float, smcmax: DT_F, flag_iter: DT_I, land: DT_I, sice: DT_F2,
                      zsoil: DT_F2, rhsct: DT_F, ci: DT_F2, sneqv: DT_F,
                      # outpus
                      sh2o: DT_F2, smc: DT_F2, cmc: DT_F, runoff3: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:
                        wplus, smc, sh2o = sstep_upperboundary_fn(
                            sh2o, smc, smcmax, sice, ci, zsoil)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:
                        wplus, smc, sh2o = sstep_fn(
                            sh2o, smc, smcmax, sice, ci, zsoil, wplus, wplus[0,0,-1])

        with interval(-1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    if ice == 0.0:
                        wplus, smc, sh2o = sstep_fn(
                            sh2o, smc, smcmax, sice, ci, zsoil, wplus, wplus[0,0,-1])

                        runoff3 = wplus

                        # update canopy water content/interception
                        cmc += dt * rhsct
                        if cmc < 1.e-20:
                            cmc = 0.0
                        cmc = min(cmc, cmcmax)



@gtscript.stencil(backend=BACKEND)
def snopac_shflx_first(ssoil: DT_F, land: DT_I, flag_iter: DT_I, sneqv: DT_F, etp: DT_F, eta: DT_F, smc: DT_F2, quartz: DT_F, smcmax: DT_F, ivegsrc: int,
                      vegtype: DT_I, shdfac: DT_F, fdown: DT_F, sfcemis: DT_F, t24: DT_F, sfctmp: DT_F, rch: DT_F, th2: DT_F,
                      epsca: DT_F, rr: DT_F, zsoil: DT_F2, dt: float, psisat: DT_F, bexp: DT_F, ice: DT_I, stc: DT_F2, hcpct:DT_F2, tbk: DT_F2, df1k: DT_F2, dtsdz: DT_F2, ddz: DT_F2, csoil_loc: DT_F,
                      # output
                      sh2o: DT_F2, stsoil: DT_F2, rhsts: DT_F2, ai: DT_F2, bi: DT_F2, ci: DT_F2, p: DT_F2, delta: DT_F2, tbot: DT_F, yy: DT_F, zz1: DT_F, df1: DT_F, beta: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv != 0.0:

                    zz1 = 1.0
                    yy = stc - 0.5 * ssoil * zsoil * zz1 / df1

                    stsoil, rhsts, ai, bi, ci, dtsdz, ddz, hcpct, sh2o, free, csoil_loc, p, delta, tbk, df1k, dtsdz, ddz = shflx_first_upperboundary_fn(smc, smcmax, dt, yy, zz1,
                                                                                                                                     zsoil, psisat, bexp, df1, hcpct, tbk, df1k, dtsdz, ddz, ice, csoil, ivegsrc, vegtype, shdfac, stc, sh2o, p, delta, rhsts, ai, bi, ci,)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv != 0.0:
                    stsoil, rhsts, ai, bi, ci, dtsdz, ddz, hcpct, sh2o, free, p, delta, tbk, df1k, dtsdz, ddz = shflx_first_fn(smc, smcmax, dt, zsoil, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, stc[0,0,+1], sh2o, 
                    hcpct, tbk[0,0,-1], df1k[0,0,-1], dtsdz[0,0,-1], ddz[0,0,-1], p[0,0,-1], delta[0,0,-1], dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta, rhsts, ai, bi, ci,)

        with interval(-1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    stsoil, rhsts, ai, bi, ci, sh2o, p, delta, tbot = shflx_first_lowerboundary_fn(smc, smcmax, dt, zsoil, zbot, tbot, psisat, bexp, df1, ice, quartz, ivegsrc, vegtype, shdfac, stc, sh2o, 
                    hcpct, tbk[0,0,-1], df1k[0,0,-1], dtsdz[0,0,-1], ddz[0,0,-1], p[0,0,-1], delta[0,0,-1], dtsdz, ddz, tbk, df1k, free, csoil_loc, p, delta, rhsts, ai, bi, ci,)


@gtscript.stencil(backend=BACKEND)
def snopac_shflx_second(ice: DT_I, sneqv_new: DT_F, sncovr: DT_F, dt: float, snowh: DT_F, sndens: DT_F, land: DT_I, flag_iter: DT_I,
                        sneqv: DT_F, zsoil: DT_F2, stsoil: DT_F2, p: DT_F2, delta: DT_F2,
                        yy: DT_F, zz1: DT_F, df1: DT_F,
                        # output
                        ssoil: DT_F, tsea: DT_F, stc: DT_F2):
    from __gtscript__ import BACKWARD, computation, interval

    with computation(BACKWARD):
        with interval(-1, None):
            if flag_iter and land:
                if sneqv != 0.0:
                    stc, p = shflx_second_lowerboundary_fn(p, delta, stc, stsoil)

        with interval(1, -1):
            if flag_iter and land:
                if sneqv != 0.0:
                    stc, p = shflx_second_fn(p, p[0,0,+1], delta, stc, stsoil)

        with interval(0, 1):
            if flag_iter and land:
                if sneqv != 0.0:
                    ssoil1, stc, t11 = shflx_second_upperboundary_fn(p, p[0,0,+1], delta, stc, stsoil, tsea, yy, zz1, df1, zsoil)

                    # snow depth and density adjustment based on snow compaction.
                    if ice == 0:
                        if sneqv_new > 0.0:
                            snowh, sndens = snowpack_fn(
                                sneqv_new, dt, tsea, yy, snowh, sndens)

                        else:
                            sneqv_new = 0.0
                            snowh = 0.0
                            sndens = 0.0
                            sncovr = 0.0

                    elif ice == 1:
                        if sneqv_new >= 0.01:
                            snowh, sndens = snowpack_fn(
                                sneqv_new, dt, tsea, yy, snowh, sndens)
                        else:
                            sneqv_new = 0.01
                            snowh = 0.05
                            sncovr = 1.0
                    else:
                        if sneqv_new >= 0.10:
                            snowh, sndens = snowpack_fn(
                                sneqv_new, dt, tsea, yy, snowh, sndens)
                        else:
                            sneqv_new = 0.10
                            snowh = 0.50
                            sncovr = 1.0
                    sneqv = sneqv_new


@gtscript.stencil(backend=BACKEND)
def cleanup_sflx(
    land: DT_I, flag_iter: DT_I, eta: DT_F, ssoil: DT_F, edir: DT_F, ec: DT_F, ett: DT_F, esnow: DT_F, sncovr: DT_F, soilm: DT_F,
    flx1: DT_F, flx2: DT_F, flx3: DT_F, etp: DT_F, runoff1: DT_F, runoff2: DT_F,
    cmc: DT_F, snowh: DT_F, sneqv: DT_F, z0: DT_F, rho: DT_F, chx: DT_F, wind: DT_F, q1: DT_F, flag_guess: DT_I,
    weasd_old: DT_F, snwdph_old: DT_F, tskin_old: DT_F, canopy_old: DT_F, tprcp_old: DT_F, srflag_old: DT_F,
    smc_old: DT_F2, slc_old: DT_F2, stc_old: DT_F2, th2: DT_F, tsea: DT_F, et: DT_F2, ice: DT_I, runoff3: DT_F, dt: float,
    sfcprs: DT_F, t2v: DT_F, snomlt: DT_F, zsoil: DT_F2, smcmax: DT_F, smcwlt: DT_F, smcref: DT_F, ch: DT_F,
    # output
    weasd: DT_F, snwdph: DT_F, tskin: DT_F, tprcp: DT_F, srflag: DT_F, smc: DT_F2,
    stc: DT_F2, slc: DT_F2, canopy: DT_F, trans: DT_F, tsurf: DT_F, zorl: DT_F,
    sncovr1: DT_F, qsurf: DT_F, gflux: DT_F, drain: DT_F, evap: DT_F, hflx: DT_F, ep: DT_F, runoff: DT_F,
    cmm: DT_F, chh: DT_F, evbs: DT_F, evcw: DT_F, sbsno: DT_F, snowc: DT_F, stm: DT_F, snohf: DT_F,
    smcwlt2: DT_F, smcref2: DT_F, wet1: DT_F
    ):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0,1):
            if flag_iter and land:
                # prepare sensible heat (h) for return to parent model
                sheat = -(chx*cp1*sfcprs) / (rd1*t2v) * (th2 - tsea)

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

                soilm = - smc * zsoil
                soilww = - (smc - smcwlt) * zsoil

                # output
                tsurf = tsea
                evap = eta
                hflx = sheat
                gflux = ssoil

                evbs = edir
                evcw = ec
                trans = ett
                sbsno = esnow
                snowc = sncovr
                snohf = flx1 + flx2 + flx3

                smcwlt2 = smcwlt
                smcref2 = smcref

                ep = etp
                wet1 = smc / smcmax

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

        with interval(1, -1):
            if flag_iter and land:
                et = et * lsubc

                soilm += smc * (zsoil[0, 0,-1] - zsoil[0, 0,0])
                soilww += (smc - smcwlt) * (zsoil[0, 0,-1] - zsoil[0, 0,0])

                # restore land-related prognostic fields for guess run
            if land and flag_guess:
                smc = smc_old
                stc = stc_old
                slc = slc_old

        with interval(-1, None):
            if flag_iter and land:
                et = et * lsubc

                soilm += smc * (zsoil[0, 0,-1] - zsoil[0, 0,0])
                soilww += (smc - smcwlt) * (zsoil[0, 0,-1] - zsoil[0, 0,0])
                soilwm = - (smcmax-smcwlt) * zsoil
                soilw = soilww / soilwm
                stm = soilm * 1000.0


                # restore land-related prognostic fields for guess run
            if land and flag_guess:
                smc = smc_old
                stc = stc_old
                slc = slc_old


@gtscript.stencil(backend=BACKEND)
def testing_canopy(flag_iter: DT_I, land: DT_I, cmc: DT_F, canopy: DT_F, et: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0,1):
            if flag_iter and land:
                canopy = cmc * 1000.0

        with interval(1,None):
            if flag_iter and land:
                et = et * lsubc
