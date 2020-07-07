#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):
    """run function"""

    ser = in_dict['serializer']
    sp = in_dict['savepoint']

    # TODO - implement sea-ice model
    hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, snowmt, \
    gflux, cmm, chh, evap, hflx = sfc_sice(
        in_dict['im'], in_dict['km'], in_dict['ps'],
        in_dict['t1'], in_dict['q1'], in_dict['delt'],
        in_dict['sfcemis'], in_dict['dlwflx'], in_dict['sfcnsw'],
        in_dict['sfcdsw'], in_dict['srflag'], in_dict['cm'],
        in_dict['ch'], in_dict['prsl1'], in_dict['prslki'],
        in_dict['islimsk'], in_dict['wind'], in_dict['flag_iter'],
        in_dict['lprnt'], in_dict['ipr'], in_dict['cimin'],
        in_dict['hice'], in_dict['fice'], in_dict['tice'],
        in_dict['weasd'], in_dict['tskin'], in_dict['tprcp'],
        in_dict['stc'], in_dict['ep'], in_dict['snwdph'], in_dict['qsurf'], in_dict['cmm'],
        in_dict['chh'], in_dict['evap'], in_dict['hflx'], 
        ser, sp)

    #d = dict([(k, locals()[k]) for k in ('hice', 'fice', 'tice', 'weasd', 'tskin', 'tprcp',
    #                                 'stc', 'ep', 'snwdph', 'qsurf', 'snowmt', 'gflux',
    #                                 'cmm', 'chh', 'evap', 'hflx')])
    d = {}
    for i in ('hice', 'fice', 'tice', 'weasd', 'stc', 'ep', 'snwdph', 'qsurf', 'snowmt',
            'gflux', 'cmm', 'chh', 'evap', 'hflx'):
        d[i] = locals()[i]
    #print(d)
    in_dict.update(d)
    return {key: in_dict.get(key, None) for key in OUT_VARS}

def init_array(shape, mode):
    arr = np.empty(shape)
    if mode == "zero":
        arr[:] = 0.
    if mode == "nan":
        arr[:] = np.nan
    return arr

def sfc_sice(im, km, ps, t1, q1, delt, sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,
             cm, ch, prsl1, prslki, islimsk, wind, flag_iter, lprnt, ipr, cimin,
             hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, cmm, chh, 
             evap, hflx,
             ser, sp):
    """run function"""

    # constant definition
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
    mode = "nan"
    ffw    = init_array([im], mode)
    evapi  = init_array([im], mode)
    evapw  = init_array([im], mode)
    sneti  = init_array([im], mode)
    snetw  = init_array([im], mode)
    hfd    = init_array([im], mode)
    hfi    = init_array([im], mode)
    focn   = init_array([im], mode)
    snof   = init_array([im], mode)
    rch    = init_array([im], mode)
    rho    = init_array([im], mode)
    snowd  = init_array([im], mode)
    theta1 = init_array([im], mode)
    flag   = init_array([im], mode)
    q0     = init_array([im], mode)
    qs1    = init_array([im], mode)
    stsice = init_array([im, kmi], mode)
    snowmt = init_array([im], "zero")
    gflux  = init_array([im], "zero")

#  --- ...  set flag for sea-ice

    flag = (islimsk == 2) & flag_iter
    i = flag_iter & (islimsk < 2)
    hice[i] = 0.
    fice[i] = 0.

    i = flag & (srflag > 0.)
    ep[i] = ep[i] * (1. - srflag[i])
    weasd[i] = weasd[i] + 1.e3 * tprcp[i] * srflag[i]
    tprcp[i] = tprcp[i] * (1. - srflag[i])

#  --- ...  update sea ice temperature
    i = flag
    stsice[i, :] = stc[i, 0:kmi]

#  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
#           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
#           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
#           is sat. hum. at surface
#           convert slrad to the civilized unit from langley minute-1 k-4

#         dlwflx has been given a negative sign for downward longwave
#         sfcnsw is the net shortwave flux (direction: dn-up)
    q0[i]     = np.maximum(q1[i], 1.0e-8)
    theta1[i] = t1[i] * prslki[i]
    rho[i]    = prsl1[i] / (rd * t1[i] * (1. + rvrdm1 * q0[i]))
    qs1[i]    = fpvs(t1[i])
    qs1[i]    = np.maximum(eps * qs1[i] / (prsl1[i] + epsm1 * qs1[i]), 1.e-8)
    q0[i]     = np.minimum(qs1[i], q0[i])

    i = flag & (fice < cimin)
    if any(i):
        print("warning: ice fraction is low:", fice[i])
        fice[i] = cimin
        tice[i] = tgice
        tskin[i]= tgice
        print('fix ice fraction: reset it to:', fice[i])

    i = flag
    ffw[i]    = 1.0 - fice[i]

    qssi = fpvs(tice[i])
    qssi = eps * qssi / (ps[i] + epsm1 * qssi)
    qssw = fpvs(tgice)
    qssw = eps * qssw / (ps[i] + epsm1 * qssw)

#  --- ...  snow depth in water equivalent is converted from mm to m unit

    snowd[i] = weasd[i] * 0.001

#  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
#           soil is allowed to interact with the atmosphere.
#           we should eventually move to a linear combination of soil and
#           snow under the condition of patchy snow.

#  --- ...  rcp = rho cp ch v

    cmm[i] = cm[i] * wind[i]
    chh[i] = rho[i] * ch[i] * wind[i]
    rch[i] = chh[i] * cp

#  --- ...  sensible and latent heat flux over open water & sea ice

    evapi[i] = elocp * rch[i] * (qssi - q0[i])
    evapw[i] = elocp * rch[i] * (qssw - q0[i])

    snetw[i] = sfcdsw[i] * (1. - albfw)
    snetw[i] = np.minimum(3. * sfcnsw[i] / (1. + 2. * ffw[i]), snetw[i])
    sneti[i] = (sfcnsw[i] - ffw[i] * snetw[i]) / fice[i]

    t12 = tice[i] * tice[i]
    t14 = t12 * t12

#  --- ...  hfi = net non-solar and upir heat flux @ ice surface

    hfi[i] = -dlwflx[i] + sfcemis[i] * sbc * t14 + evapi[i] + \
            rch[i] * (tice[i] - theta1[i])
    hfd[i] = 4. * sfcemis[i] * sbc * tice[i] * t12 + \
            (1. + elocp * eps * hvap * qs1[i] / (rd * t12)) * rch[i]

    t12 = tgice * tgice
    t14 = t12 * t12

#  --- ...  hfw = net heat flux @ water surface (within ice)

    focn[i] = 2.   # heat flux from ocean - should be from ocn model
    snof[i] = 0.    # snowfall rate - snow accumulates in gbphys

    hice[i] = np.maximum(np.minimum(hice[i], himax), himin)
    snowd[i] = np.minimum(snowd[i], hsmax)

# TODO: write more efficiently, save mask as new variable?
    i = flag & (snowd > 2. * hice)
    if any(i):
        print('warning: too much snow :', snowd[i])
        snowd[i] = hice[i] + hice[i]
        print('fix: decrease snow depth to:', snowd[i])

    fice_ref = ser.read("fice", sp)
    assert np.allclose(fice, fice_ref, equal_nan=True)
    hfi_ref = ser.read("hfi", sp)
    assert np.allclose(hfi, hfi_ref, equal_nan=True)
    hfd_ref = ser.read("hfd", sp)
    assert np.allclose(hfd, hfd_ref, equal_nan=True)
    sneti_ref = ser.read("sneti", sp)
    assert np.allclose(sneti, sneti_ref, equal_nan=True)
    focn_ref = ser.read("focn", sp)
    assert np.allclose(focn, focn_ref, equal_nan=True)
    delt_ref = ser.read("delt", sp)
    assert np.allclose(delt, delt_ref, equal_nan=True)
    lprnt_ref = ser.read("lprnt", sp)
    assert np.allclose(lprnt, lprnt_ref, equal_nan=True)
    ipr_ref = ser.read("ipr", sp)
    assert np.allclose(ipr, ipr_ref, equal_nan=True)
    snowd_ref = ser.read("snowd", sp)
    assert np.allclose(snowd, snowd_ref, equal_nan=True)
    hice_ref = ser.read("hice", sp)
    assert np.allclose(hice, hice_ref, equal_nan=True)
    stsice_ref = ser.read("stsice", sp)
    assert np.allclose(stsice, stsice_ref, equal_nan=True)
    tice_ref = ser.read("tice", sp)
    assert np.allclose(tice, tice_ref, equal_nan=True)
    snof_ref = ser.read("snof", sp)
    assert np.allclose(snof, snof_ref, equal_nan=True)
    snowmt_ref = ser.read("snowmt", sp)
    assert np.allclose(snowmt, snowmt_ref, equal_nan=True)
    gflux_ref = ser.read("gflux", sp)
    assert np.allclose(gflux, gflux_ref, equal_nan=True)

# call function ice3lay
    snowd, hice, stsice, tice, snof, snowmt, gflux = \
        ice3lay(im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr, 
                snowd, hice, stsice, tice, snof, snowmt, gflux, ser, sp)

    # ser_snowd = ser.read("snowd", sp)
    # assert np.allclose(ser_snowd, snowd, equal_nan=True)
    # ser_hice = ser.read("hice", sp)
    # assert np.allclose(ser_hice, hice, equal_nan=True)
    # ser_snof = ser.read("snof", sp)
    # assert np.allclose(ser_snof, snof, equal_nan=True)

    if any(tice[flag] < timin):
        # TODO: print indices (i) of temp-warnings
        print('warning: snow/ice temperature is too low:',
              tice[flag][tice[flag] < timin]) # , ' i=', i)
        tice[flag] = timin
        print('fix snow/ice temperature: reset it to:', tice[flag])

    if any(stsice[flag,0] < timin):
        print('warning: layer 1 ice temp is too low:',stsice[flag,0])
# TODO: print indices (i) of temp-warning             ' i=',i
        stsice[flag,0] = timin
        print('fix layer 1 ice temp: reset it to:', stsice[flag,0])

    if any(stsice[flag,1] < timin):
        print('warning: layer 2 ice temp is too low:',stsice[flag,1])
        stsice[flag,1] = timin
        print('fix layer 2 ice temp: reset it to:',stsice[flag,1])

    tskin[flag] = tice[flag]*fice[flag] + tgice*ffw[flag]

    # TODO: dimension mismatch
    stc[flag,0:kmi] = np.minimum(stsice[flag,0:kmi], t0c)

#  --- ...  calculate sensible heat flux (& evap over sea ice)

    hflxi      = rch[flag] * (tice[flag] - theta1[flag])
    hflxw      = rch[flag] * (tgice      - theta1[flag])
    hflx[flag] = fice[flag]* hflxi       + ffw[flag]*hflxw
    evap[flag] = fice[flag]* evapi[flag] + ffw[flag]*evapw[flag]

#  --- ...  the rest of the output

    qsurf[flag] = q1[flag] + evap[flag] / (elocp*rch[flag])

#  --- ...  convert snow depth back to mm of water equivalent

    weasd[flag]  = snowd[flag] * 1000.0
    snwdph[flag] = weasd[flag] * dsi             # snow depth in mm

    tem     = 1.0 / rho[flag]
    hflx[flag] = hflx[flag] * tem * cpinv
    evap[flag] = evap[flag] * tem * hvapi

    return hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, \
           snowmt, gflux, cmm, chh, evap, hflx


def ice3lay(im,kmi,fice,flag,hfi,hfd, sneti, focn, delt, lprnt, ipr, \
            snowd, hice, stsice, tice, snof, snowmt, gflux,
            ser, sp):
    """function ice3lay"""
    # constant parameters
    ds   = 330.0    # snow (ov sea ice) density (kg/m^3)
    dw   = 1000.0   # fresh water density  (kg/m^3)
    dsdw = ds / dw
    dwds = dw / ds
    ks   = 0.31     # conductivity of snow   (w/mk)
    i0   = 0.3      # ice surface penetrating solar fraction
    ki   = 2.03     # conductivity of ice  (w/mk)
    di   = 917.0    # density of ice   (kg/m^3)
    didw = di / dw
    dsdi = ds / di
    ci   = 2054.0   # heat capacity of fresh ice (j/kg/k)
    li   = 3.34e5   # latent heat of fusion (j/kg-ice)
    si   = 1.0      # salinity of sea ice
    mu   = 0.054    # relates freezing temp to salinity
    tfi  = -mu * si # sea ice freezing temp = -mu*salinity
    tfw  = -1.8     # tfw - seawater freezing temp (c)
    tfi0 = tfi - 0.0001
    dici = di * ci
    dili = di * li
    dsli = ds * li
    ki4  = ki * 4.0
    # TODO: move variable definition to separate file 
    t0c    = 2.7315e+2
    
    # vecotorize constants
    mode = "nan"
    ip = init_array([im], mode)
    tsf = init_array([im], mode)
    ai = init_array([im], mode)
    k12 = init_array([im], mode)
    k32 = init_array([im], mode)
    a1 = init_array([im], mode)
    b1 = init_array([im], mode)
    c1 = init_array([im], mode)
    tmelt = init_array([im], mode)
    h1 = init_array([im], mode)
    h2 = init_array([im], mode)
    bmelt = init_array([im], mode)
    dh = init_array([im], mode)
    f1 = init_array([im], mode)
    hdi = init_array([im], mode)
    wrk = init_array([im], mode)
    wrk1 = init_array([im], mode)
    bi = init_array([im], mode)
    a10 = init_array([im], mode)
    b10 = init_array([im], mode)

    dt2  = 2. * delt
    dt4  = 4. * delt
    dt6  = 6. * delt
    dt2i = 1. / dt2

    i = flag
    snowd[i] = snowd[i]  * dwds
    hdi[i] = (dsdw * snowd[i] + didw * hice[i])

    i = flag & (hice < hdi)
    snowd[i] = snowd[i] + hice[i] - hdi[i]
    hice[i]  = hice[i] + (hdi[i] - hice[i]) * dsdi

    i = flag
    snof[i] = snof[i] * dwds
    tice[i] = tice[i] - t0c
    stsice[i, 0] = np.minimum(stsice[i, 0] - t0c, tfi0)     # degc
    stsice[i, 1] = np.minimum(stsice[i, 1] - t0c, tfi0)     # degc

    ip[i] = i0 * sneti[i] # ip +v here (in winton ip=-i0*sneti)

    i = flag & (snowd > 0.)
    tsf[i] = 0.
    ip[i]  = 0.

    i = flag & ~i
    tsf[i] = tfi
    ip[i] = i0 * sneti[i]  # ip +v here (in winton ip=-i0*sneti)

    i = flag
    tice[i] = np.minimum(tice[i], tsf[i])

    # compute ice temperature

    bi[i] = hfd[i]
    ai[i] = hfi[i] - sneti[i] + ip[i] - tice[i] * bi[i] # +v sol input here
    k12[i] = ki4 * ks / (ks * hice[i] + ki4 * snowd[i])
    k32[i] = (ki + ki) / hice[i]

    wrk[i] = 1. / (dt6 * k32[i] + dici * hice[i])
    a10[i] = dici * hice[i] * dt2i + \
        k32[i] * (dt4 * k32[i] + dici * hice[i]) * wrk[i]
    b10[i] = -di * hice[i] * (ci * stsice[i, 0] + li * tfi / \
            stsice[i, 0]) * dt2i - ip[i] - k32[i] * \
            (dt4 * k32[i] * tfw + dici * hice[i] * stsice[i,1]) * wrk[i]

    wrk1[i] = k12[i] / (k12[i] + bi[i])
    a1[i] = a10[i] + bi[i] * wrk1[i]
    b1[i] = b10[i] + ai[i] * wrk1[i]
    c1[i]   = dili * tfi * dt2i * hice[i]

    stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
        (a1[i] + a1[i])
    tice[i] = (k12[i] * stsice[i, 0] - ai[i]) / (k12[i] + bi[i])

    i = flag & (tice > tsf)
    a1[i] = a10[i] + k12[i]
    b1[i] = b10[i] - k12[i] * tsf[i]
    stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
        (a1[i] + a1[i])
    tice[i] = tsf[i]
    tmelt[i] = (k12[i] * (stsice[i, 0] - tsf[i]) - (ai[i] + bi[i] * tsf[i])) * delt

    i = flag & ~i
    tmelt[i] = 0.
    snowd[i] = snowd[i] + snof[i] * delt

    i = flag
    stsice[i, 1] = (dt2 * k32[i] * (stsice[i, 0] + tfw + tfw) + \
        dici * hice[i] * stsice[i, 1]) * wrk[i]
    bmelt[i] = (focn[i] + ki4 * (stsice[i, 1] - tfw) / hice[i]) * delt

#  --- ...  resize the ice ...

    h1[i] = 0.5 * hice[i]
    h2[i] = 0.5 * hice[i]

#  --- ...  top ...
    i = flag & (tmelt <= snowd * dsli)
    snowmt[i] = tmelt[i] / dsli
    snowd[i] = snowd[i] - snowmt[i]

    i = flag & ~i
    snowmt[i] = snowd[i]
    h1[i] = h1[i] - (tmelt[i] - snowd[i] * dsli) / \
            (di * (ci - li / stsice[i, 0]) * (tfi - stsice[i, 0]))
    snowd[i] = 0.

#  --- ...  and bottom

    i = flag & (bmelt < 0.)
    dh[i] = -bmelt[i] / (dili + dici * (tfi - tfw))
    stsice[i, 1] = (h2[i] * stsice[i, 1] + dh[i] * tfw) / (h2[i] + dh[i])
    h2[i] = h2[i] + dh[i]

    i = flag & ~i
    h2[i] = h2[i] - bmelt[i] / (dili + dici * (tfi - stsice[i, 1]))

    ser_hice = ser.read("hice2", sp)
    assert np.allclose(ser_hice, hice, equal_nan=True)
    ser_snowmt = ser.read("snowmt2", sp)
    assert np.allclose(ser_snowmt, snowmt, equal_nan=True)
    ser_gflux = ser.read("gflux2", sp)
    assert np.allclose(ser_gflux, gflux, equal_nan=True)
    ser_k32 = ser.read("k32", sp)
    assert np.allclose(ser_k32, k32, equal_nan=True)
    ser_a1 = ser.read("a1", sp)
    assert np.allclose(ser_a1, a1, equal_nan=True)
    ser_ai = ser.read("ai", sp)
    assert np.allclose(ser_ai, ai, equal_nan=True)
    ser_bi = ser.read("bi", sp)
    assert np.allclose(ser_bi, bi, equal_nan=True)
    ser_c1 = ser.read("c1", sp)
    assert np.allclose(ser_c1, c1, equal_nan=True)
    ser_ip = ser.read("ip", sp)
    assert np.allclose(ser_ip, ip, equal_nan=True)
    ser_k12 = ser.read("k12", sp)
    assert np.allclose(ser_k12, k12, equal_nan=True)
    ser_tsf = ser.read("tsf", sp)
    assert np.allclose(ser_tsf, tsf, equal_nan=True)
    ser_tice = ser.read("tice2", sp)
    assert np.allclose(ser_tice, tice, equal_nan=True)
    ser_b1 = ser.read("b1", sp)
    assert np.allclose(ser_b1, b1, equal_nan=True)
    ser_stsice = ser.read("stsice2", sp)
    assert np.allclose(ser_stsice, stsice, equal_nan=True)

#  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

    hice[flag] = h1[flag] + h2[flag]


    # begin if_hice_block
    # begin if_h1_block

    f1[flag & (hice > 0.) & (h1 > 0.5*hice)] = 1.-\
            (h2[flag][hice[flag]>0.][h1[flag]>0.5*hice[flag]]+\
            h2[flag][hice[flag]>0.][h1[flag]>0.5*hice[flag]])\
            / hice[flag][hice[flag]>0.][h1[flag]>0.5*hice[flag]]
    stsice[flag & (hice > 0.) & (h1 > 0.5*hice), 1] = \
            f1[flag][hice[flag]>0.][h1[flag]>0.5*hice[flag]] * \
            (stsice[flag,0][hice[flag]>0.][h1[flag]>0.5*hice[flag]]\
            + li*tfi/ \
            (ci* \
            stsice[flag,0][hice[flag]>0.][h1[flag]>0.5*hice[flag]]))\
            + (1. - \
            f1[flag][hice[flag]>0.][h1[flag]>0.5*hice[flag]]) * \
            stsice[flag,1][hice[flag]>0.][h1[flag]>0.5*hice[flag]]

    # begin if_stsice_block

    hice[flag & (hice > 0.) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)] \
            = hice[flag & (hice > 0.) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)] - \
            h2[flag & (hice > 0.) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)]* \
            ci*(stsice[flag & (hice > 0.) & (h1 > 0.5*hice) & \
            (stsice[:,1]> tfi), 1] - tfi)/(li*delt)

    stsice[flag & (hice > 0.) & (h1 > 0.5*hice) & (stsice[:,1] > tfi), 1] = tfi

    # end if_stsice_block

    # else if_h1_block

    # hice[flag] > 0.
    # h1[flag] <= 0.5*hice[flag]

    f1[flag & (hice > 0.) & (h1 <= 0.5*hice)] = \
            (h1[flag][hice[flag]>0.][h1[flag][hice[flag]>0.]<=0.5*hice[flag][hice[flag]>0.]]+\
            h1[flag][hice[flag]>0.][h1[flag][hice[flag]>0.]<=0.5*hice[flag][hice[flag]>0.]]) / \
            hice[flag][hice[flag]>0.][h1[flag][hice[flag]>0.]<=0.5*hice[flag][hice[flag]>0.]]

    stsice[flag & (hice > 0.) & (h1 <= 0.5*hice), 0] = \
            f1[flag][hice[flag]>0.][h1[flag]<=0.5*hice[flag]]*(\
            stsice[flag,0][hice[flag]>0.][h1[flag]<=0.5*hice[flag]]\
            + li*tfi/(ci*stsice[flag,0][hice[flag]>0.]\
            [h1[flag]<=0.5*hice[flag]]))+(1. - \
            f1[flag][hice[flag]>0.][h1[flag]<=0.5*hice[flag]])\
            *stsice[flag,1][hice[flag]>0.][h1[flag]<=0.5*hice[flag]]

    stsice[flag & (hice > 0.),0][h1[flag]<=0.5*hice[flag]]= (\
            stsice[flag,0][hice[flag]>0.][h1[flag]<=0.5*hice[flag]]\
            - np.sqrt(stsice[flag,0][hice[flag]>0.]\
            [h1[flag]<=0.5*hice[flag]]*stsice[flag,0]\
            [hice[flag]>0.][h1[flag]<=0.5*hice[flag]] \
            - 4.0*tfi*li/ci)) * 0.5

    # end if_h1_block

    k12[flag & (hice > 0.)] = ki4*ks / (ks* \
            hice[flag][hice[flag]>0.] + ki4* \
            snowd[flag & (hice > 0.)])

    gflux[flag & (hice > 0.)] = k12[flag & (hice > 0.)] * \
            (stsice[flag & (hice > 0.),0] -\
            tice[flag & (hice > 0.)])

    # else if_hice_block

    snowd[flag & (hice <= 0.)] = snowd[flag & (hice <=0.)] + \
            (h1[flag & (hice <= 0.)]*(ci*\
            (stsice[flag & (hice <= 0.), 0] - tfi)- li*(1. - tfi/ \
            stsice[flag & (hice <= 0.), 0])) +\
            h2[flag & (hice <= 0.)]*(ci*\
            (stsice[flag & (hice <= 0.), 1] - tfi) - li)) / li

    hice[flag & (hice <= 0.)] = np.maximum(0., \
            snowd[flag & (hice <= 0.)]*dsdi)

    snowd[flag & (hice <= 0.)] = 0.

    stsice[flag & (hice <= 0.), 0] = tfw
    stsice[flag & (hice <= 0.), 1] = tfw

    gflux[flag & (hice <= 0.)] = 0.

    # end if_hice_block

    gflux[flag] = fice[flag] * gflux[flag]
    snowmt[flag] = snowmt[flag] * dsdw
    snowd[flag] = snowd[flag] * dsdw
    tice[flag] = tice[flag]     + t0c
    stsice[flag,0] = stsice[flag,0] + t0c
    stsice[flag,1] = stsice[flag,1] + t0c

    # end if_flag_block


    return snowd, hice, stsice, tice, snof, snowmt, gflux



# TODO - this hsould be moved into a shared physics functions module
def fpvs(t):
    """Compute saturation vapor pressure
       t: Temperature [K]
    fpvs: Vapor pressure [Pa]
    """

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
