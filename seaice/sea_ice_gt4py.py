#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import gt4py as gt
from gt4py import gtscript

# backend = "gtcuda"
backend = "gtx86"
dtype = np.float64
dtype_int = np.int32
dtype_bool = np.bool

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]
ONED_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):
    """run function"""

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = in_dict[key].copy()

    shape = (in_dict['im'], 1, 1)

    flag_iter = gt.storage.from_array(np.reshape(in_dict['flag_iter'], shape).astype(dtype_int), backend, (0, 0, 0))
    islimsk = gt.storage.from_array(np.reshape(in_dict['islimsk'], shape), backend, (0, 0, 0))
    hice = gt.storage.from_array(np.reshape(in_dict['hice'], shape), backend, (0, 0, 0))
    fice = gt.storage.from_array(np.reshape(in_dict['fice'], shape), backend, (0, 0, 0))
    weasd = gt.storage.from_array(np.reshape(in_dict['weasd'], shape), backend, (0, 0, 0))
    tprcp = gt.storage.from_array(np.reshape(in_dict['tprcp'], shape), backend, (0, 0, 0))
    srflag = gt.storage.from_array(np.reshape(in_dict['srflag'], shape), backend, (0, 0, 0))
    ep = gt.storage.from_array(np.reshape(in_dict['ep'], shape), backend, (0, 0, 0))
    stc0 = gt.storage.from_array(np.reshape(in_dict['stc'][:,0], shape), backend, (0, 0, 0))
    stc1 = gt.storage.from_array(np.reshape(in_dict['stc'][:,1], shape), backend, (0, 0, 0))
    q1 = gt.storage.from_array(np.reshape(in_dict['q1'], shape), backend, (0, 0, 0))
    t1 = gt.storage.from_array(np.reshape(in_dict['t1'], shape), backend, (0, 0, 0))
    prslki = gt.storage.from_array(np.reshape(in_dict['prslki'], shape), backend, (0, 0, 0))
    prsl1 = gt.storage.from_array(np.reshape(in_dict['prsl1'], shape), backend, (0, 0, 0))
    ps = gt.storage.from_array(np.reshape(in_dict['ps'], shape), backend, (0, 0, 0))
    tskin = gt.storage.from_array(np.reshape(in_dict['tskin'], shape), backend, (0, 0, 0))
    tice = gt.storage.from_array(np.reshape(in_dict['tice'], shape), backend, (0, 0, 0))
    cmm = gt.storage.from_array(np.reshape(in_dict['cmm'], shape), backend, (0, 0, 0))
    chh = gt.storage.from_array(np.reshape(in_dict['chh'], shape), backend, (0, 0, 0))
    cm = gt.storage.from_array(np.reshape(in_dict['cm'], shape), backend, (0, 0, 0))
    ch = gt.storage.from_array(np.reshape(in_dict['ch'], shape), backend, (0, 0, 0))
    wind = gt.storage.from_array(np.reshape(in_dict['wind'], shape), backend, (0, 0, 0))
    sfcdsw = gt.storage.from_array(np.reshape(in_dict['sfcdsw'], shape), backend, (0, 0, 0))
    sfcnsw = gt.storage.from_array(np.reshape(in_dict['sfcnsw'], shape), backend, (0, 0, 0))
    sfcemis = gt.storage.from_array(np.reshape(in_dict['sfcemis'], shape), backend, (0, 0, 0))
    dlwflx = gt.storage.from_array(np.reshape(in_dict['dlwflx'], shape), backend, (0, 0, 0))
    evap = gt.storage.from_array(np.reshape(in_dict['evap'], shape), backend, (0, 0, 0))
    snwdph = gt.storage.from_array(np.reshape(in_dict['snwdph'], shape), backend, (0, 0, 0))
    hflx = gt.storage.from_array(np.reshape(in_dict['hflx'], shape), backend, (0, 0, 0))
    qsurf = gt.storage.from_array(np.reshape(in_dict['qsurf'], shape), backend, (0, 0, 0))
    gflux = gt.storage.from_array(np.reshape(in_dict['gflux'], shape), backend, (0, 0, 0))
    snowmt = gt.storage.from_array(np.reshape(in_dict['snowmt'], shape), backend, (0, 0, 0))

    sfc_sice(
        srflag=srflag,
        islimsk=islimsk,
        flag_iter=flag_iter,
        hice=hice,
        fice=fice,
        weasd=weasd,
        tprcp=tprcp,
        ep=ep,
        stc0=stc0,
        stc1=stc1,
        q1=q1,
        t1=t1,
        prslki=prslki,
        prsl1=prsl1,
        ps=ps,
        tskin=tskin,
        tice=tice,
        cmm=cmm,
        chh=chh,
        cm=cm,
        ch=ch,
        wind=wind,
        sfcdsw=sfcdsw,
        sfcnsw=sfcnsw,
        sfcemis=sfcemis,
        dlwflx=dlwflx,
        evap=evap,
        hflx=hflx,
        snwdph=snwdph,
        qsurf=qsurf,
        gflux=gflux,
        snowmt=snowmt,
        im=in_dict['im'],
        cimin=in_dict['cimin'],
        delt=in_dict['delt'],
        lprnt=in_dict['lprnt'],
        ipr=in_dict['ipr'],
        origin=(0, 0, 0), domain=shape)

    for key in ONED_VARS:
        out_dict[key] = np.reshape(locals()[key], shape[0])
    out_dict['stc'][:,0] = np.reshape(stc0, shape[0])
    out_dict['stc'][:,1] = np.reshape(stc1, shape[0])

    return out_dict


# TODO - this should be moved into a shared physics functions module
@gtscript.function
def fpvs(t):
# TODO: improve doc string
    """Compute saturation vapor pressure
       t: Temperature [K]
    fpvs: Vapor pressure [Pa]
    """

    fpvs = 0.
    w = 0.
    pvl = 0.
    pvi = 0.

    # TODO: how does the exp function word?!
    e = 2.718281828459045

    # constants
    # TODO - this should be moved into a shared physics constants module
    con_psat = 6.1078e+2
    con_ttp  = 2.7316e+2
    con_cvap = 1.8460e+3
    con_cliq = 4.1855e+3
    con_hvap = 2.5000e+6
    con_rv   = 4.6150e+2
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

    # convert_to_scalar = False
    # if np.isscalar(t):
    #     t = np.array(t)
    #     convert_to_scalar = True

    # fpvs = np.empty_like(t)
    tr = con_ttp / t

    if (t >= tliq):
        fpvs = con_psat * (tr**xponal) * e**(xponbl*(1. - tr))
    elif (t < tice):
        fpvs = con_psat * (tr**xponai) * e**(xponbi*(1. - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = con_psat * (tr**xponal) * e**(xponbl*(1. - tr))
        pvi = con_psat * (tr**xponai) * e**(xponbi*(1. - tr))
        fpvs = w * pvl + (1. - w) * pvi

    # if convert_to_scalar:
    #     fpvs = fpvs.item()

    return fpvs


# TODO: can ice3lay be defined as a stencil?!
# @gtscript.stencil(backend=backend, verbose=True) 
@gtscript.function
def ice3lay(
    im,kmi,fice,flag,hfi,hfd, sneti, focn, delt, lprnt, ipr, \
    # in/outputs
    snowd, hice, stsice0, stsice1, tice, snof, snowmt, gflux):
    # flag: gtscript.Field[dtype_int],
    # snowd: gtscript.Field[dtype],
    # hice: gtscript.Field[dtype],
    # *,
    # delt: float,
    # ):
    """function ice3lay"""

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

    # TODO: move variable definition to separate file 
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
    
        if (hice < hdi):
            snowd = snowd + hice - hdi
            hice  = hice + (hdi - hice) * dsdi
    
        snof = snof * dwds
        tice = tice - t0c
        stsice0 = (stsice0 - t0c > tfi0)*tfi0 + (stsice0 - t0c < tfi0)*(stsice0 - t0c)     # degc
        stsice1 = (stsice1 - t0c > tfi0)*tfi0 + (stsice1 - t0c < tfi0)*(stsice1 - t0c)     # degc
    
        ip = i0 * sneti # ip +v here (in winton ip=-i0*sneti)
    
        if (snowd > 0.):
            tsf = 0.
            ip  = 0.
        else:
            tsf = tfi
            ip  = i0 * sneti  # ip +v here (in winton ip=-i0*sneti)
    
        tice = (tice > tsf)*tsf + (tice < tsf)*tice
    
        # compute ice temperature
    
        bi = hfd
        ai = hfi - sneti + ip - tice * bi # +v sol input here
        k12 = ki4 * ks / (ks * hice + ki4 * snowd)
        k32 = (ki + ki) / hice
    
        wrk = 1. / (dt6 * k32 + dici * hice)
        a10 = dici * hice * dt2i + \
            k32 * (dt4 * k32 + dici * hice) * wrk
        b10 = -di * hice * (ci * stsice0 + li * tfi / \
                stsice0) * dt2i - ip - k32 * \
                (dt4 * k32 * tfw + dici * hice * stsice1) * wrk
    
        wrk1 = k12 / (k12 + bi)
        a1 = a10 + bi * wrk1
        b1 = b10 + ai * wrk1
        c1   = dili * tfi * dt2i * hice
    
        stsice0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
        tice = (k12 * stsice0 - ai) / (k12 + bi)
    
        if (tice > tsf):
            a1 = a10 + k12
            b1 = b10 - k12 * tsf
            stsice0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
            tice = tsf
            tmelt = (k12 * (stsice0 - tsf) - (ai + bi * tsf)) * delt
        else:
            tmelt = 0.
            snowd = snowd + snof * delt
    
        stsice1 = (dt2 * k32 * (stsice0 + tfw + tfw) + \
            dici * hice * stsice1) * wrk
        bmelt = (focn + ki4 * (stsice1 - tfw) / hice) * delt
    
    #  --- ...  resize the ice ...
    
        h1 = 0.5 * hice
        h2 = 0.5 * hice
    
    #  --- ...  top ...
        if (tmelt <= snowd * dsli):
            snowmt = tmelt / dsli
            snowd = snowd - snowmt
        else:
            snowmt = snowd
            h1 = h1 - (tmelt - snowd * dsli) / \
                    (di * (ci - li / stsice0) * (tfi - stsice0))
            snowd = 0.
    
    #  --- ...  and bottom
    
        if (bmelt < 0.):
            dh = -bmelt / (dili + dici * (tfi - tfw))
            stsice1 = (h2 * stsice1 + dh * tfw) / (h2 + dh)
            h2 = h2 + dh
        else:
            h2 = h2 - bmelt / (dili + dici * (tfi - stsice1))
    
    #  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow
    
        hice = h1 + h2
    
        # begin if_hice_block
        if (hice > 0.):
            if (h1 > 0.5*hice):
                f1 = 1. - 2*h2/hice
                stsice1 = f1 * (stsice0 + li*tfi/ \
                        (ci*stsice0)) + (1. - f1)*stsice1
        
                if (stsice1 > tfi):
                    hice = hice - h2 * ci*(stsice1 - tfi)/(li*delt)
                    stsice1 = tfi
        
            else:
                f1 = 2*h1/hice
                stsice0 = f1*(stsice0 + li*tfi/ \
                        (ci*stsice0)) + (1. - f1)*stsice1
                stsice0= (stsice0 - (stsice0 * stsice0 - 4.0*tfi*li/ci)**0.5) * 0.5
    
            k12 = ki4*ks / (ks*hice + ki4*snowd)
            gflux = k12*(stsice0 - tice)
        
        else:
            snowd = snowd + (h1*(ci*(stsice0 - tfi)\
                    - li*(1. - tfi/stsice0)) + h2*(ci*\
                    (stsice1 - tfi) - li)) / li
            hice = (0. > snowd*dsdi)*0. + (0. < snowd*dsdi)*snowd*dsdi
            snowd = 0.
            stsice0 = tfw
            stsice1 = tfw
            gflux = 0.
    
        # end if_hice_block
        gflux = fice * gflux
        snowmt = snowmt * dsdw
        snowd = snowd * dsdw
        tice = tice     + t0c
        stsice0 = stsice0 + t0c
        stsice1 = stsice1 + t0c

        return snowd, hice, stsice0, stsice1, tice, snof, snowmt, gflux


@gtscript.stencil(backend=backend, verbose=True, externals={"fpvs":fpvs, "ice3lay":ice3lay})
def sfc_sice(
    ps: gtscript.Field[dtype], 
    t1: gtscript.Field[dtype],
    q1: gtscript.Field[dtype],
    sfcemis: gtscript.Field[dtype],
    dlwflx: gtscript.Field[dtype],
    sfcnsw: gtscript.Field[dtype],
    sfcdsw: gtscript.Field[dtype],
    srflag: gtscript.Field[dtype],
    cm: gtscript.Field[dtype],
    ch: gtscript.Field[dtype],
    prsl1: gtscript.Field[dtype],
    prslki: gtscript.Field[dtype],
    islimsk: gtscript.Field[dtype_int],
    wind: gtscript.Field[dtype],
    flag_iter: gtscript.Field[dtype_int],
    hice: gtscript.Field[dtype],
    fice: gtscript.Field[dtype],
    tice: gtscript.Field[dtype],
    weasd: gtscript.Field[dtype],
    tskin: gtscript.Field[dtype],
    tprcp: gtscript.Field[dtype],
    stc0: gtscript.Field[dtype],
    stc1: gtscript.Field[dtype],
    ep: gtscript.Field[dtype],
    snwdph: gtscript.Field[dtype],
    qsurf: gtscript.Field[dtype],
    cmm: gtscript.Field[dtype],
    chh: gtscript.Field[dtype],
    evap: gtscript.Field[dtype],
    hflx: gtscript.Field[dtype],
    gflux: gtscript.Field[dtype],
    snowmt: gtscript.Field[dtype],
    *,
    im: int,
    # km: int,
    delt: float,
    lprnt: int,
    ipr: int,
    cimin: float,
):
    """TODO: write docstring for this function!"""
#     from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):
        # constant definition
        # TODO - this should be moved into a shared physics constants / physics functions module
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
        kmi    = 2
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
#         mode = "nan"
# TODO: initialization of arrays necessary?
        q0      = 0.
        stsice0 = 0.
        stsice1 = 0.
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
    
        flag = (islimsk == 2)*flag_iter

        # TODO: is '*' a good replace for logical 'and'? Or how does the 'and' work in gt4py?
        if flag_iter*(islimsk < 2):
            hice = 0.
            fice = 0.

        if flag:
            if (srflag > 0.):
                ep = ep * (1. - srflag)
                weasd = weasd + 1.e3 * tprcp * srflag
                tprcp = tprcp * (1. - srflag)

    #  --- ...  update sea ice temperature
            stsice0 = stc0[0,0,0]
            stsice1 = stc1[0,0,0]

    #  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
    #           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
    #           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
    #           is sat. hum. at surface
    #           convert slrad to the civilized unit from langley minute-1 k-4

    #         dlwflx has been given a negative sign for downward longwave
    #         sfcnsw is the net shortwave flux (direction: dn-up)

            q0     = (q1 > 1.0e-8)*q1 + (q1 < 1.0e-8)*1.0e-8  # max(q1, 1.0e-8)
            theta1 = t1 * prslki
            rho    = prsl1 / (rd * t1 * (1. + rvrdm1 * q0))
            # TODO: function call fpvs does not work!

        qs1 = fpvs(t=t1)
        if flag:
            # qs1    = max(eps * qs1 / (prsl1 + epsm1 * qs1), 1.e-8)
            qs1 = (eps * qs1 / (prsl1 + epsm1 * qs1) > 1.e-8)*(eps * qs1 / (prsl1 + epsm1 * qs1)) \
                  + (eps * qs1 / (prsl1 + epsm1 * qs1) < 1.e-8)*1.e-8
            # q0     = min(qs1, q0)
            q0 = (qs1 < q0)*qs1 + (qs1 > q0)*q0

            if fice < cimin:
        #         # TODO: print statement possible?
        #         # print("warning: ice fraction is low:", fice[i])
        #         fice = cimin
                tice = tgice
                tskin= tgice
                # print('fix ice fraction: reset it to:', fice[i])

            fice = fice*(fice > cimin) + cimin*(fice < cimin)

            ffw  = 1.0 - fice

        qssi = fpvs(tice)
        if flag:
            qssi = eps * qssi / (ps + epsm1 * qssi)
        qssw = fpvs(tgice)
        if flag:
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
                    + (3. * sfcnsw / (1. + 2. * ffw) > snetw)*snetw
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
            # TODO: check all min/max for <= > or < >=
            hice = (((hice < himax)*hice + (hice > himax)*himax) > himin)*((hice < himax)*hice + (hice > himax)*himax) + \
                   (((hice < himax)*hice + (hice > himax)*himax) < himin)*himin
            # snowd = np.minimum(snowd, hsmax)
            snowd = (snowd >= hsmax)*hsmax + (snowd < hsmax)*snowd

            if (snowd > 2. * hice):
                # TODO: print statement?
                # print('warning: too much snow :', snowd[i])
                snowd = hice + hice
                # print('fix: decrease snow depth to:', snowd[i])

            # run the 3-layer ice model
        snowd, hice, stsice0, stsice1, tice, snof, snowmt, gflux = ice3lay(
                im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr,
                snowd, hice, stsice0, stsice1, tice, snof, snowmt, gflux)

        # ice3lay(
        #     flag=flag,
        #     snowd=snowd, 
        #     # hice, 
        #     # delt=delt,
        #     origin=(0, 0, 0), domain=(im, 1, 1))

        if flag:
            if (tice < timin):
                # TODO: print statement?
                # print('warning: snow/ice temperature is too low:', tice, ' i=', i)
                tice = timin
                # print('fix snow/ice temperature: reset it to:', timin)

            if (stsice0 < timin):
                # TODO: print statement?
                # print('warning: layer 1 ice temp is too low:', stsice[i, 0], ' i=', i)
                stsice0 = timin
                # print('fix layer 1 ice temp: reset it to:', timin)

            if (stsice1 < timin):
                # TODO: print statement?
                # print('warning: layer 2 ice temp is too low:', stsice[i, 1], 'i=', i)
                stsice1 = timin
                # print('fix layer 2 ice temp: reset it to:', timin)

            tskin = tice * fice + tgice * ffw

            # stc[i, 0:kmi] = np.minimum(stsice[i, 0:kmi], t0c)
            stc0 = (stsice0 >= t0c)*t0c + (stsice0 < t0c)*stsice0
            stc1 = (stsice1 >= t0c)*t0c + (stsice1 < t0c)*stsice1

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

            tem = 1. / rho
            hflx = hflx * tem * cpinv
            evap = evap * tem * hvapi



def init_array(shape, mode):
    arr = np.empty(shape)
    if mode == "none":
        pass
    if mode == "zero":
        arr[:] = 0.
    elif mode == "nan":
        arr[:] = np.nan
    return arr
