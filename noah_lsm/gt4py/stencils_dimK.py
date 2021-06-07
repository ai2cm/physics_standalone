import gt4py as gt
import numpy as np
from gt4py import gtscript
from physcons import *
from functions_gt4py_dimK import *

DT_F2 = gtscript.Field[np.float64]
DT_I2 = gtscript.Field[np.int32]
DT_F = gtscript.Field[gtscript.IJ, np.float64]
DT_I = gtscript.Field[gtscript.IJ, np.int32]
DT_FK = gtscript.Field[gtscript.IK, np.float64]
BACKEND = "numpy"

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
                      sgx: DT_F, gx: DT_F2, edir1: DT_F, ec1: DT_F, edir: DT_F, ec: DT_F):
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
                    edir = edir1
                    ec = ec1
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
                      et1: DT_F2, ett1: DT_F, eta1: DT_F, et: DT_F2, ett: DT_F, eta: DT_F):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            count = 0
        with interval(...):
            if flag_iter and land:
                shdfac = sigmaf
                if sneqv == 0.0:
                    etp1 = etp * 0.001
                    if etp > 0.0:
                        et1, ett1 = evapo_third_fn(
                            etp1, shdfac, cmc, cmcmax, pc, cfactr, gx, denom, ett1)
                et = et1 * 1000.0
                count += 1

        with interval(-1, None):
            if flag_iter and land:
                if sneqv == 0.0:
                    if etp > 0.0:
                        eta1 = edir1 + ett1 + ec1
                    eta = eta1 * 1000.0
                    ett = ett1 * 1000.0



@gtscript.stencil(backend=BACKEND)
def nopac_smflx_first(dt: float, smcmax: DT_F, smcwlt: DT_F, prcp: DT_F, prcp1: DT_F, zsoil: DT_FK, shdfac: DT_F, ec1: DT_F,
                      cmc: DT_F, sh2o: DT_F2, smc: DT_F2, flag_iter: DT_I, land: DT_I, etp: DT_F, sneqv: DT_F,
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
def nopac_smflx_second(etp: DT_F, smcmax: DT_F, dt: float, bexp: DT_F, kdt: DT_F, frzx: DT_F, dksat: DT_F, dwsat: DT_F, slope: DT_F,
                       zsoil: DT_FK, sh2o: DT_F2, flag_iter: DT_I, land: DT_I, sneqv: DT_F, sicemax: DT_F, dd: DT_F, dice: DT_F, pcpdrp: DT_F, edir1: DT_F, et1: DT_F2,
                       # outpus
                       rhstt: DT_F2, ci: DT_F2, runoff1: DT_F, runoff2: DT_F, dsmdz: DT_F2, ddz: DT_F2, wdf: DT_F2, wcnd: DT_F2, p: DT_F2, delta: DT_F2):
    from __gtscript__ import FORWARD, computation, interval

    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if sneqv == 0.0:
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

