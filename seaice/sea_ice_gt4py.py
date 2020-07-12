#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import gt4py as gt
from gt4py import gtscript

# BACKEND = "gtcuda"
# BACKEND = "gtx86"
BACKEND = "numpy"

DT_F = gtscript.Field[np.float64]
DT_I = gtscript.Field[np.int32]

SCALAR_VARS = ["delt", "cimin"]

IN_VARS = ["ps", "t1", "q1", "sfcemis", "dlwflx", "sfcnsw", "sfcdsw", "srflag",
    "cm", "ch", "prsl1", "prslki", "islimsk", "wind", "flag_iter"]

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc0", "stc1", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

# mathematical constants
EULER = 2.7182818284590452353602874713527

# physical constants
HVAP = 2.5e6
RV = 4.615e2
PSAT = 6.1078e2
TTP = 2.7316e2
CVAP = 1.846e3
CLIQ = 4.1855e3
CSOL = 2.106e3
HFUS = 3.3358e5

# fvps constants
FPVS_XPONAL = -(CVAP - CLIQ) / RV
FPVS_XPONBL = -(CVAP - CLIQ) / RV + HVAP / (RV * TTP)
FPVS_XPONAI = -(CVAP - CSOL) / RV
FPVS_XPONBI = -(CVAP - CSOL) / RV + (HVAP + HFUS) / (RV * TTP)

# other constants
INIT_VALUE = 0.  # TODO - this should be float("NaN")


def numpy_to_gt4py_storage(arr, name, backend="numpy"):
    data = np.reshape(arr, (arr.shape[0], 1, 1))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    return gt.storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def gt4py_to_numpy_storage(arr, name):
    data = arr.view(np.ndarray)
    return np.reshape(data, (data.shape[0]))


def run(in_dict, backend=BACKEND):
    """run function"""

    # special handling of stc
    stc = in_dict.pop("stc")
    in_dict["stc0"] = stc[:, 0]
    in_dict["stc1"] = stc[:, 1]

    # setup storages
    scalar_dict = {k: in_dict[k] for k in SCALAR_VARS}
    out_dict = {k: numpy_to_gt4py_storage(in_dict[k].copy(), k, backend=backend) for k in OUT_VARS}
    in_dict = {k: numpy_to_gt4py_storage(in_dict[k], k) for k in IN_VARS}

    # compile stencil
    sfc_sice = gtscript.stencil(definition=sfc_sice_defs, backend=backend, externals={})

    # call sea-ice parametrization
    sfc_sice(**scalar_dict, **in_dict, **out_dict)

    # convert back to numpy for validation
    out_dict = {k: gt4py_to_numpy_storage(out_dict[k], k) for k in OUT_VARS}

    # special handling of stc
    stc[:, 0] = out_dict.pop("stc0")[:]
    stc[:, 1] = out_dict.pop("stc1")[:]
    out_dict["stc"] = stc

    return out_dict


@gtscript.function
def exp_fn(a):
    return EULER ** a


@gtscript.function
def fpvs_fn(t):
    """Compute saturation vapor pressure
       t: Temperature 
    fpvs: Vapor pressure [Pa]
    """

    tr = TTP / t

    # over liquid
    pvl = PSAT * (tr ** FPVS_XPONAL) * exp_fn(FPVS_XPONBL * (1.0 - tr))

    # over ice
    pvi = PSAT * (tr ** FPVS_XPONAI) * exp_fn(FPVS_XPONBI * (1.0 - tr))

    # determine regime weight
    w = (t - TTP + 20.) / 20.

    fpvs = INIT_VALUE
    if w >= 1.0:
        fpvs = pvl
    elif w < 0.0:
        fpvs = pvi
    else:
        fpvs = w * pvl + (1.0 - w) * pvi

    return fpvs


@gtscript.function
def ice3lay(fice,flag,hfi,hfd, sneti, focn, delt, snowd, hice, \
            stc0, stc1, tice, snof, snowmt, gflux):
    """TODO - write a nice docstring here"""

    # TODO - move constants outside and capitalize (see fpvs)
    # constant parameters
    ds   = 330.     # snow (ov sea ice) density (kg/m^3)
    dw   = 1000.    # fresh water density  (kg/m^3)
    dsdw = ds / dw
    dwds = dw / ds
    ks   = 0.31     # conductivity of snow   (w/mk)
    i0   = 0.3      # ice surface penetrating solar fraction
    ki   = 2.03     # conductivity of ice  (w/mk)
    di   = 917.     # density of ice   (kg/m^3)
    didw = di / dw
    dsdi = ds / di
    ci   = 2054.    # heat capacity of fresh ice (j/kg/k)
    li   = 3.34e5   # latent heat of fusion (j/kg-ice)
    si   = 1.       # salinity of sea ice
    mu   = 0.054    # relates freezing temp to salinity
    tfi  = -mu * si # sea ice freezing temp = -mu*salinity
    tfw  = -1.8     # tfw - seawater freezing temp (c)
    tfi0 = tfi - 0.0001
    dici = di * ci
    dili = di * li
    dsli = ds * li
    ki4  = ki * 4.
    t0c    = 2.7315e+2
    
    hdi = 0.
    ip  = 0.
    tsf = 0.
    ai  = 0.
    bi  = 0.
    k12 = 0.
    k32 = 0.
    wrk = 0.
    wrk1 = 0.
    a10 = 0.
    b10 = 0.
    a1  = 0.
    b1  = 0.
    c1  = 0.
    tmelt = 0.
    bmelt = 0.
    h1 = 0.
    h2 = 0.
    dh = 0.
    f1 = 0.

    dt2  = 2. * delt
    dt4  = 4. * delt
    dt6  = 6. * delt
    dt2i = 1. / dt2

    if flag:
        snowd = snowd  * dwds
        hdi = (dsdw * snowd + didw * hice)
    
        if hice < hdi:
            snowd = snowd + hice - hdi
            hice  = hice + (hdi - hice) * dsdi
    
        snof = snof * dwds
        tice = tice - t0c
        # min(stc0-tc0, tfi0)
        # TODO - replace "min(x,y)" with "x if x<y else y"
        stc0 = stc0 - t0c if stc0 - t0c < tfi0 else tfi0
        # min(stc1-tc0, tfi0)
        stc1 = (stc1 - t0c >= tfi0)*tfi0 + (stc1 - t0c < tfi0)*(stc1 - t0c)     # degc
    
        ip = i0 * sneti # ip +v here (in winton ip=-i0*sneti)
    
        if snowd > 0.:
            tsf = 0.
            ip  = 0.
        else:
            tsf = tfi
            ip  = i0 * sneti  # ip +v here (in winton ip=-i0*sneti)
    
        # min(tice, tsf)
        tice = (tice >= tsf)*tsf + (tice < tsf)*tice
    
        # compute ice temperature
    
        bi = hfd
        ai = hfi - sneti + ip - tice * bi # +v sol input here
        k12 = ki4 * ks / (ks * hice + ki4 * snowd)
        k32 = (ki + ki) / hice
    
        wrk = 1. / (dt6 * k32 + dici * hice)
        a10 = dici * hice * dt2i + \
            k32 * (dt4 * k32 + dici * hice) * wrk
        b10 = -di * hice * (ci * stc0 + li * tfi / \
                stc0) * dt2i - ip - k32 * \
                (dt4 * k32 * tfw + dici * hice * stc1) * wrk
    
        wrk1 = k12 / (k12 + bi)
        a1 = a10 + bi * wrk1
        b1 = b10 + ai * wrk1
        c1   = dili * tfi * dt2i * hice
    
        stc0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
        tice = (k12 * stc0 - ai) / (k12 + bi)
    
        if tice > tsf:
            a1 = a10 + k12
            b1 = b10 - k12 * tsf
            stc0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
            tice = tsf
            tmelt = (k12 * (stc0 - tsf) - (ai + bi * tsf)) * delt
        else:
            tmelt = 0.
            snowd = snowd + snof * delt
    
        stc1 = (dt2 * k32 * (stc0 + tfw + tfw) + \
            dici * hice * stc1) * wrk
        bmelt = (focn + ki4 * (stc1 - tfw) / hice) * delt
    
    #  --- ...  resize the ice ...
    
        h1 = 0.5 * hice
        h2 = 0.5 * hice
    
    #  --- ...  top ...
        if tmelt <= snowd * dsli:
            snowmt = tmelt / dsli
            snowd = snowd - snowmt
        else:
            snowmt = snowd
            h1 = h1 - (tmelt - snowd * dsli) / \
                    (di * (ci - li / stc0) * (tfi - stc0))
            snowd = 0.
    
    #  --- ...  and bottom
    
        if bmelt < 0.:
            dh = -bmelt / (dili + dici * (tfi - tfw))
            stc1 = (h2 * stc1 + dh * tfw) / (h2 + dh)
            h2 = h2 + dh
        else:
            h2 = h2 - bmelt / (dili + dici * (tfi - stc1))
    
    #  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow
    
        hice = h1 + h2
    
        # begin if_hice_block
        if hice > 0.:
            if h1 > 0.5 * hice:
                f1 = 1. - 2. * h2 / hice
                stc1 = f1 * (stc0 + li*tfi/ \
                        (ci*stc0)) + (1. - f1)*stc1
        
                if stc1 > tfi:
                    hice = hice - h2 * ci*(stc1 - tfi)/(li*delt)
                    stc1 = tfi
        
            else:
                f1 = 2*h1/hice
                stc0 = f1*(stc0 + li*tfi/ \
                        (ci*stc0)) + (1. - f1)*stc1
                stc0= (stc0 - (stc0 * stc0 - 4.0*tfi*li/ci)**0.5) * 0.5
    
            k12 = ki4*ks / (ks*hice + ki4*snowd)
            gflux = k12*(stc0 - tice)
        
        else:
            snowd = snowd + (h1*(ci*(stc0 - tfi)\
                    - li*(1. - tfi/stc0)) + h2*(ci*\
                    (stc1 - tfi) - li)) / li
            # max(0,m snowd*dsdi)
            hice = (0. >= snowd*dsdi)*0. + (0. < snowd*dsdi)*snowd*dsdi
            snowd = 0.
            stc0 = tfw
            stc1 = tfw
            gflux = 0.
    
        # end if_hice_block
        gflux = fice * gflux
        snowmt = snowmt * dsdw
        snowd = snowd * dsdw
        tice = tice + t0c
        stc0 = stc0 + t0c
        stc1 = stc1 + t0c

        return snowd, hice, stc0, stc1, tice, snof, snowmt, gflux


def sfc_sice_defs(
    ps: DT_F,
    t1: DT_F,
    q1: DT_F,
    sfcemis: DT_F,
    dlwflx: DT_F,
    sfcnsw: DT_F,
    sfcdsw: DT_F,
    srflag: DT_F,
    cm: DT_F,
    ch: DT_F,
    prsl1: DT_F,
    prslki: DT_F,
    islimsk: DT_I,
    wind: DT_F,
    flag_iter: DT_I,
    hice: DT_F,
    fice: DT_F,
    tice: DT_F,
    weasd: DT_F,
    tskin: DT_F,
    tprcp: DT_F,
    stc0: DT_F,
    stc1: DT_F,
    ep: DT_F,
    snwdph: DT_F,
    qsurf: DT_F,
    cmm: DT_F,
    chh: DT_F,
    evap: DT_F,
    hflx: DT_F,
    gflux: DT_F,
    snowmt: DT_F,
    *,
    delt: float,
    cimin: float
):
    """TODO: write docstring for this function!"""

    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):

        # TODO - move constants outside and capitalize (see fpvs)
        # constants definition
        cp     = 1.0046e+3
        hvap   = 2.5e+6
        sbc    = 5.6704e-8
        tgice  = 2.712e+2
        rv     = 4.615e+2
        rd     = 2.8705e+2
        eps    = rd / rv
        epsm1  = rd / rv - 1.
        rvrdm1 = rv / rd - 1.
        t0c    = 2.7315e+2

        # constant parameterts
        cpinv  = 1. / cp
        hvapi  = 1. / hvap
        elocp  = hvap / cp
        himax  = 8.          # maximum ice thickness allowed
        himin  = 0.1         # minimum ice thickness required
        hsmax  = 2.          # maximum snow depth allowed
        timin  = 173.        # minimum temperature allowed for snow/ice
        albfw  = 0.06        # albedo for lead
        dsi    = 1. / 0.33

        # arrays
        # TODO - only initialize the arrays which are defined in an if-statement
        q0      = 0.
        theta1  = 0.
        rho     = 0.
        qs1     = 0.
        ffw     = 0.
        snowd   = 0.
        rch     = 0.
        evapi   = 0.
        evapw   = 0.
        sneti   = 0.
        snetw   = 0.
        t12     = 0.
        t14     = 0.
        hfi     = 0.
        hfd     = 0.
        focn    = 0.
        snof    = 0.
        hflxi   = 0.
        hflxw   = 0.
        tem     = 0.
    
    #  --- ...  set flag for sea-ice
    
        # TODO - gt4py supports only the "and" statement and not the "&"
        flag = (islimsk == 2) and flag_iter

        if flag_iter and (islimsk < 2):
            hice = 0.
            fice = 0.

        qs1 = fpvs_fn(t1)
        if flag:
            if srflag > 0.:
                ep = ep * (1. - srflag)
                weasd = weasd + 1.e3 * tprcp * srflag
                tprcp = tprcp * (1. - srflag)

    #  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
    #           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
    #           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
    #           is sat. hum. at surface
    #           convert slrad to the civilized unit from langley minute-1 k-4

    #         dlwflx has been given a negative sign for downward longwave
    #         sfcnsw is the net shortwave flux (direction: dn-up)
            # TODO - max(q1, 1.0e-8)
            q0     = (q1 >= 1.0e-8)*q1 + (q1 < 1.0e-8)*1.0e-8
            theta1 = t1 * prslki
            rho    = prsl1 / (rd * t1 * (1. + rvrdm1 * q0))

            # TODO - max(eps * qs1 / (prsl1 + epsm1 * qs1), 1.e-8)
            qs1 = (eps * qs1 / (prsl1 + epsm1 * qs1) >= 1.e-8)*(eps * qs1 / (prsl1 + epsm1 * qs1)) \
                  + (eps * qs1 / (prsl1 + epsm1 * qs1) < 1.e-8)*1.e-8
            # q0     = min(qs1, q0)
            q0 = (qs1 < q0)*qs1 + (qs1 >= q0)*q0

            if fice < cimin:
                # print("warning: ice fraction is low:", fice)
                #fice = cimin
                tice = tgice
                tskin= tgice
                # print('fix ice fraction: reset it to:', fice)

            fice = fice*(fice > cimin) + cimin*(fice <= cimin)

        qssi = fpvs_fn(tice)
        qssw = fpvs_fn(tgice)
        if flag:
            ffw  = 1.0 - fice
            qssi = eps * qssi / (ps + epsm1 * qssi)
            qssw = eps * qssw / (ps + epsm1 * qssw)

    #  --- ...  snow depth in water equivalent is converted from mm to m unit

            snowd = weasd * 0.001

    #  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
    #           soil is allowed to interact with the atmosphere.
    #           we should eventually move to a linear combination of soil and
    #           snow under the condition of patchy snow.

    #  --- ...  rcp = rho cp ch v

            cmm = cm * wind
            chh = rho * ch * wind
            rch = chh * cp

    #  --- ...  sensible and latent heat flux over open water & sea ice

            evapi = elocp * rch * (qssi - q0)
            evapw = elocp * rch * (qssw - q0)

            snetw = sfcdsw * (1. - albfw)
            # snetw = np.minimum(3. * sfcnsw / (1. + 2. * ffw), snetw)
            snetw = (3. * sfcnsw / (1. + 2. * ffw) < snetw) * (3. * sfcnsw / (1. + 2. * ffw)) \
                    + (3. * sfcnsw / (1. + 2. * ffw) >= snetw)*snetw
            sneti = (sfcnsw - ffw * snetw) / fice

            t12 = tice * tice
            t14 = t12 * t12

        #  --- ...  hfi = net non-solar and upir heat flux @ ice surface

            hfi = -dlwflx + sfcemis * sbc * t14 + evapi + \
                    rch * (tice - theta1)
            hfd = 4. * sfcemis * sbc * tice * t12 + \
                    (1. + elocp * eps * hvap * qs1 / (rd * t12)) * rch

            t12 = tgice * tgice
            t14 = t12 * t12

        #  --- ...  hfw = net heat flux @ water surface (within ice)

            focn = 2.   # heat flux from ocean - should be from ocn model
            snof = 0.   # snowfall rate - snow accumulates in gbphys

            # hice = np.maximum(np.minimum(hice, himax), himin)
            hice = (((hice < himax)*hice + (hice >= himax)*himax) >= himin)*((hice < himax)*hice + (hice > himax)*himax) + \
                   (((hice < himax)*hice + (hice >= himax)*himax) < himin)*himin
            # snowd = np.minimum(snowd, hsmax)
            snowd = (snowd >= hsmax)*hsmax + (snowd < hsmax)*snowd

            if snowd > 2. * hice:
                # print('warning: too much snow :', snowd[i])
                snowd = hice + hice
                # print('fix: decrease snow depth to:', snowd[i])

        # run the 3-layer ice model
        snowd, hice, stc0, stc1, tice, snof, snowmt, gflux = ice3lay(
                fice, flag, hfi, hfd, sneti, focn, delt,
                snowd, hice, stc0, stc1, tice, snof, snowmt, gflux)

        if flag:
            if tice < timin:
                # print('warning: snow/ice temperature is too low:', tice, ' i=', i)
                tice = timin
                # print('fix snow/ice temperature: reset it to:', timin)

            if stc0 < timin:
                # print('warning: layer 1 ice temp is too low:', stsice[i, 0], ' i=', i)
                stc0 = timin
                # print('fix layer 1 ice temp: reset it to:', timin)

            if stc1 < timin:
                # print('warning: layer 2 ice temp is too low:', stsice[i, 1], 'i=', i)
                stc1 = timin
                # print('fix layer 2 ice temp: reset it to:', timin)

            tskin = tice * fice + tgice * ffw
            stc0 = (stc0 >= t0c)*t0c + (stc0 < t0c)*stc0
            stc1 = (stc1 >= t0c)*t0c + (stc1 < t0c)*stc1

        #  --- ...  calculate sensible heat flux (& evap over sea ice)

            hflxi = rch * (tice - theta1)
            hflxw = rch * (tgice - theta1)
            hflx = fice * hflxi + ffw * hflxw
            evap = fice * evapi + ffw * evapw

        #  --- ...  the rest of the output

            qsurf = q1 + evap / (elocp * rch)

        #  --- ...  convert snow depth back to mm of water equivalent

            weasd = snowd * 1000.
            snwdph = weasd * dsi             # snow depth in mm

            hflx = hflx / rho * cpinv
            evap = evap / rho * hvapi

