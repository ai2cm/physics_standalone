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
#     sfc_sice(
#         in_dict['im'], in_dict['km'], in_dict['ps'],
#         in_dict['t1'], in_dict['q1'], in_dict['delt'],
#         in_dict['sfcemis'], in_dict['dlwflx'], in_dict['sfcnsw'],
#         in_dict['sfcdsw'], in_dict['srflag'], in_dict['cm'],
#         in_dict['ch'], in_dict['prsl1'], in_dict['prslki'],
#         in_dict['islimsk'], in_dict['wind'], in_dict['flag_iter'],
#         in_dict['lprnt'], in_dict['ipr'], in_dict['cimin'],
#         out_dict['hice'], out_dict['fice'], out_dict['tice'],
#         out_dict['weasd'], out_dict['tskin'], out_dict['tprcp'],
#         out_dict['stc'], out_dict['ep'], out_dict['snwdph'],
#         out_dict['qsurf'], out_dict['cmm'], out_dict['chh'],
#         out_dict['evap'], out_dict['hflx'], out_dict['gflux'],
#         out_dict['snowmt'],
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

    # for key in OUT_VARS:
    #     out_dict[key] = locasl()[key]
    out_dict['hice'] = np.reshape(hice, shape[0])

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
    
    # vecotorize constants
    # mode = "nan"
    # ip = init_array([im], mode)
    # tsf = init_array([im], mode)
    # ai = init_array([im], mode)
    # k12 = init_array([im], mode)
    # k32 = init_array([im], mode)
    # a1 = init_array([im], mode)
    # b1 = init_array([im], mode)
    # c1 = init_array([im], mode)
    # tmelt = init_array([im], mode)
    # h1 = init_array([im], mode)
    # h2 = init_array([im], mode)
    # bmelt = init_array([im], mode)
    # dh = init_array([im], mode)
    # f1 = init_array([im], mode)
    # wrk = init_array([im], mode)
    # wrk1 = init_array([im], mode)
    # bi = init_array([im], mode)
    # a10 = init_array([im], mode)
    # b10 = init_array([im], mode)
    # hdi = 0.

    # dt2  = 2. * delt
    # dt4  = 4. * delt
    # dt6  = 6. * delt
    # dt2i = 1. / dt2

    if flag:
        snowd = snowd  * dwds
#         #     hdi = (dsdw * snowd + didw * hice)
#     
#         i = flag & (hice < hdi)
#         snowd[i] = snowd[i] + hice[i] - hdi[i]
#         hice[i]  = hice[i] + (hdi[i] - hice[i]) * dsdi
#     
#         i = flag
#         snof[i] = snof[i] * dwds
#         tice[i] = tice[i] - t0c
#         stsice[i, 0] = np.minimum(stsice[i, 0] - t0c, tfi0)     # degc
#         stsice[i, 1] = np.minimum(stsice[i, 1] - t0c, tfi0)     # degc
#     
#         ip[i] = i0 * sneti[i] # ip +v here (in winton ip=-i0*sneti)
#     
#         i = flag & (snowd > 0.)
#         tsf[i] = 0.
#         ip[i]  = 0.
#     
#         i = flag & ~i
#         tsf[i] = tfi
#         ip[i] = i0 * sneti[i]  # ip +v here (in winton ip=-i0*sneti)
#     
#         i = flag
#         tice[i] = np.minimum(tice[i], tsf[i])
#     
#         # compute ice temperature
#     
#         bi[i] = hfd[i]
#         ai[i] = hfi[i] - sneti[i] + ip[i] - tice[i] * bi[i] # +v sol input here
#         k12[i] = ki4 * ks / (ks * hice[i] + ki4 * snowd[i])
#         k32[i] = (ki + ki) / hice[i]
#     
#         wrk[i] = 1. / (dt6 * k32[i] + dici * hice[i])
#         a10[i] = dici * hice[i] * dt2i + \
#             k32[i] * (dt4 * k32[i] + dici * hice[i]) * wrk[i]
#         b10[i] = -di * hice[i] * (ci * stsice[i, 0] + li * tfi / \
#                 stsice[i, 0]) * dt2i - ip[i] - k32[i] * \
#                 (dt4 * k32[i] * tfw + dici * hice[i] * stsice[i,1]) * wrk[i]
#     
#         wrk1[i] = k12[i] / (k12[i] + bi[i])
#         a1[i] = a10[i] + bi[i] * wrk1[i]
#         b1[i] = b10[i] + ai[i] * wrk1[i]
#         c1[i]   = dili * tfi * dt2i * hice[i]
#     
#         stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
#             (a1[i] + a1[i])
#         tice[i] = (k12[i] * stsice[i, 0] - ai[i]) / (k12[i] + bi[i])
#     
#         i = flag & (tice > tsf)
#         a1[i] = a10[i] + k12[i]
#         b1[i] = b10[i] - k12[i] * tsf[i]
#         stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
#             (a1[i] + a1[i])
#         tice[i] = tsf[i]
#         tmelt[i] = (k12[i] * (stsice[i, 0] - tsf[i]) - (ai[i] + bi[i] * tsf[i])) * delt
#     
#         i = flag & ~i
#         tmelt[i] = 0.
#         snowd[i] = snowd[i] + snof[i] * delt
#     
#         i = flag
#         stsice[i, 1] = (dt2 * k32[i] * (stsice[i, 0] + tfw + tfw) + \
#             dici * hice[i] * stsice[i, 1]) * wrk[i]
#         bmelt[i] = (focn[i] + ki4 * (stsice[i, 1] - tfw) / hice[i]) * delt
#     
#     #  --- ...  resize the ice ...
#     
#         h1[i] = 0.5 * hice[i]
#         h2[i] = 0.5 * hice[i]
#     
#     #  --- ...  top ...
#         i = flag & (tmelt <= snowd * dsli)
#         snowmt[i] = tmelt[i] / dsli
#         snowd[i] = snowd[i] - snowmt[i]
#     
#         i = flag & ~i
#         snowmt[i] = snowd[i]
#         h1[i] = h1[i] - (tmelt[i] - snowd[i] * dsli) / \
#                 (di * (ci - li / stsice[i, 0]) * (tfi - stsice[i, 0]))
#         snowd[i] = 0.
#     
#     #  --- ...  and bottom
#     
#         i = flag & (bmelt < 0.)
#         dh[i] = -bmelt[i] / (dili + dici * (tfi - tfw))
#         stsice[i, 1] = (h2[i] * stsice[i, 1] + dh[i] * tfw) / (h2[i] + dh[i])
#         h2[i] = h2[i] + dh[i]
#     
#         i = flag & ~i
#         h2[i] = h2[i] - bmelt[i] / (dili + dici * (tfi - stsice[i, 1]))
#     
#     #  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow
#     
#         i = flag
#         hice[i] = h1[i] + h2[i]
#     
#         # begin if_hice_block
#         i = flag & (hice > 0.)
#         # begin if_h1_block
#         j = i & (h1 > 0.5*hice)
#         f1[j] = 1. - 2*h2[j]/hice[j]
#         stsice[j, 1] = f1[j] * (stsice[j, 0] + li*tfi/ \
#                 (ci*stsice[j,0])) + (1. - f1[j])*stsice[j,1]
#     
#         # begin if_stsice_block
#         k = j & (stsice[:,1] > tfi)
#         hice[k] = hice[k] - h2[k]* ci*(stsice[k, 1] - tfi)/(li*delt)
#         stsice[k, 1] = tfi
#         # end if_stsice_block
#     
#         # else if_h1_block
#         j = flag & ~j
#         f1[j] = 2*h1[j]/hice[j]
#         stsice[j, 0] = f1[j]*(stsice[j,0] + li*tfi/ \
#                 (ci*stsice[j,0])) + (1. - f1[j])*stsice[j,1]
#         stsice[j,0]= (stsice[j,0] - np.sqrt(stsice[j,0]\
#                 *stsice[j,0] - 4.0*tfi*li/ci)) * 0.5
#         # end if_h1_block
#     
#         k12[i] = ki4*ks / (ks*hice[i] + ki4*snowd[i])
#         gflux[i] = k12[i]*(stsice[i,0] - tice[i])
#         
#         # else if_hice_block
#         i = flag & ~i
#         snowd[i] = snowd[i] + (h1[i]*(ci*(stsice[i, 0] - tfi)\
#                 - li*(1. - tfi/stsice[i, 0])) + h2[i]*(ci*\
#                 (stsice[i, 1] - tfi) - li)) / li
#         hice[i] = np.maximum(0., snowd[i]*dsdi)
#         snowd[i] = 0.
#         stsice[i, 0] = tfw
#         stsice[i, 1] = tfw
#         gflux[i] = 0.
#     
#         # end if_hice_block
#         i = flag
#         gflux[i] = fice[i] * gflux[i]
#         snowmt[i] = snowmt[i] * dsdw
#         snowd[i] = snowd[i] * dsdw
#         tice[i] = tice[i]     + t0c
#         stsice[i,0] = stsice[i,0] + t0c
#         stsice[i,1] = stsice[i,1] + t0c

        return snowd

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
        snowd = ice3lay(im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr,
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
