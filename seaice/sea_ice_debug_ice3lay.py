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

def sfc_sice(im, km, ps, t1, q1, delt, sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,
             cm, ch, prsl1, prslki, islimsk, wind, flag_iter, lprnt, ipr, cimin,
             hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, cmm, chh, 
             evap, hflx,
             ser, sp):
    """run function"""

    # constant definition
    # TODO - this should be moved into a shared physics constants / physics functions module
    cp     = 1.0046e+3
    hvap   = 2.5000e+6
    sbc    = 5.6704e-8
    tgice  = 2.7120e+2
    rv     = 4.6150e+2
    rd     = 2.8705e+2
    eps    = rd/rv
    epsm1  = rd/rv - 1.
    rvrdm1 = rv/rd - 1.
    t0c    = 2.7315e+2

    # constant parameterts
    kmi    = 2
    zero   = 0.
    one    = 1.
    cpinv  = one/cp
    hvapi  = one/hvap
    elocp  = hvap/cp
    himax  = 8.          # maximum ice thickness allowed
    himin  = 0.          # minimum ice thickness required
    hsmax  = 2.          # maximum snow depth allowed
    timin  = 173.        # minimum temperature allowed for snow/ice
    albfw  = 0.          # albedo for lead
    dsi    = one/0.33

    # arrays
    ffw    = np.zeros(im)
    evapi  = np.zeros(im)
    evapw  = np.zeros(im)
    sneti  = np.zeros(im)
    snetw  = np.zeros(im)
    hfd    = np.zeros(im)
    hfi    = np.zeros(im)
    focn   = np.zeros(im)
    snof   = np.zeros(im)
    rch    = np.zeros(im)
    rho    = np.zeros(im)
    snowd  = np.zeros(im)
    theta1 = np.zeros(im)
    flag   = np.zeros(im)
    q0     = np.zeros(im)
    qs1     = np.zeros(im)
    stsice = np.zeros([im[0], kmi])

#  --- ...  set flag for sea-ice

    flag = (islimsk == 2) & flag_iter
    hice[flag_iter & (islimsk < 2)] = zero
    fice[flag_iter & (islimsk < 2)] = zero

    # TODO: if flag.any():
# TODO: save mask "flag & srflag > zero" as logical array?
    ep[flag & ((srflag > zero))]    = ep[flag & ((srflag > zero))]* \
            (one-srflag[flag & ((srflag > zero))])
    weasd[flag & (srflag > zero)] = weasd[flag & (srflag > zero)] + \
            1.e3*tprcp[flag & (srflag > zero)]*srflag[flag & (srflag > zero)]
    tprcp[flag & (srflag > zero)] = tprcp[flag & (srflag > zero)]* \
            (one-srflag[flag & (srflag > zero)])
#  --- ...  update sea ice temperature
    #TODO: shape error! 
    stsice[flag, 0:kmi] = stc[flag, 0:kmi]
    
#  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
#           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
#           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
#           is sat. hum. at surface
#           convert slrad to the civilized unit from langley minute-1 k-4

#         dlwflx has been given a negative sign for downward longwave
#         sfcnsw is the net shortwave flux (direction: dn-up)
    q0[flag]     = np.maximum(q1[flag], 1.0e-8)
    theta1[flag] = t1[flag] * prslki[flag]
    rho[flag]    = prsl1[flag] / (rd*t1[flag]*(one+rvrdm1*q0[flag]))
    qs1[flag] = fpvs(t1[flag])
    qs1[flag] = np.maximum(eps*qs1[flag] / (prsl1[flag] + epsm1*qs1[flag]), 1.e-8)
    q0[flag]  = np.minimum(qs1[flag], q0[flag])

    if any(fice[flag] < cimin):
        print("warning: ice fraction is low:", fice[flag][fice[flag] < cimin])
        fice[flag][fice[flag] < cimin] = cimin
        tice[flag][fice[flag] < cimin] = tgice
        tskin[flag][fice[flag] < cimin]= tgice
        print('fix ice fraction: reset it to:', fice[flag][fice[flag] < cimin])

    ffw[flag]    = 1.0 - fice[flag]

    qssi = fpvs(tice[flag])
    qssi = eps*qssi / (ps[flag] + epsm1*qssi)
    qssw = fpvs(tgice)
    qssw = eps*qssw / (ps[flag] + epsm1*qssw)

#  --- ...  snow depth in water equivalent is converted from mm to m unit

    snowd[flag] = weasd[flag] * 0.001

#  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
#           soil is allowed to interact with the atmosphere.
#           we should eventually move to a linear combination of soil and
#           snow under the condition of patchy snow.

#  --- ...  rcp = rho cp ch v

    cmm[flag] = cm[flag]  * wind[flag]
    chh[flag] = rho[flag] * ch[flag] * wind[flag]
    rch[flag] = chh[flag] * cp

#  --- ...  sensible and latent heat flux over open water & sea ice

    evapi[flag] = elocp * rch[flag] * (qssi - q0[flag])
    evapw[flag] = elocp * rch[flag] * (qssw - q0[flag])

    snetw[flag] = sfcdsw[flag] * (one - albfw)
    snetw[flag] = np.minimum(3.*sfcnsw[flag]/(one+2.*ffw[flag]), snetw[flag])
    sneti[flag] = (sfcnsw[flag] - ffw[flag]*snetw[flag]) / fice[flag]

    t12 = tice[flag] * tice[flag]
    t14 = t12 * t12

#  --- ...  hfi = net non-solar and upir heat flux @ ice surface

    hfi[flag] = -dlwflx[flag] + sfcemis[flag]*sbc*t14 + evapi[flag] + \
            rch[flag]*(tice[flag] - theta1[flag])
    hfd[flag] = 4.*sfcemis[flag]*sbc*tice[flag]*t12 + \
            (one + elocp*eps*hvap*qs1[flag]/(rd*t12)) * rch[flag]


    t12 = tgice * tgice
    t14 = t12 * t12

#  --- ...  hfw = net heat flux @ water surface (within ice)

    focn[flag] = 2.   # heat flux from ocean - should be from ocn model
    snof[flag] = zero    # snowfall rate - snow accumulates in gbphys

    hice[flag] = np.maximum( np.minimum( hice[flag], himax ), himin )
    snowd[flag] = np.minimum( snowd[flag], hsmax )

# TODO: write more efficiently, save mask as new variable?
    if any(snowd[flag] > (2.*hice[flag])):
        print('warning: too much snow :',
              snowd[flag][snowd[flag] > (2.*hice[flag])])
        snowd[flag][snowd[flag] > (2.*hice[flag])] = \
                hice[flag][snowd[flag] > (2.*hice[flag])] + \
                hice[flag][snowd[flag] > (2.*hice[flag])]
        print('fix: decrease snow depth to:',
              snowd[flag][snowd[flag] > (2.*hice[flag])])

    hfi = ser.read("hfi", sp)
    hfd = ser.read("hfd", sp)
    sneti = ser.read("sneti", sp)
    focn = ser.read("focn", sp)
    delt = ser.read("delt", sp)
    lprnt = ser.read("lprnt", sp)
    ipr = ser.read("ipr", sp)
    snowd = ser.read("snowd", sp)
    hice = ser.read("hice", sp)
    stsice = ser.read("stsice", sp)
    tice = ser.read("tice", sp)
    snof = ser.read("snof", sp)
    snowmt = ser.read("snowmt", sp)
    hice = ser.read("hice", sp)
    gflux = ser.read("gflux", sp)

# call function ice3lay
    snowd, hice, stsice, tice, snof, snowmt, gflux = \
            ice3lay(im, kmi, fice, flag, hfi, hfd,
                    sneti, focn, delt, lprnt, ipr,
		    snowd, hice, stsice, tice, snof,
                    snowmt, gflux,
                    ser, sp)

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
    dw   =1000.0   # fresh water density  (kg/m^3)
    dsdw = ds/dw
    dwds = dw/ds
    ks   = 0.31    # conductivity of snow   (w/mk)
    i0   = 0.3      # ice surface penetrating solar fraction
    ki   = 2.03    # conductivity of ice  (w/mk)
    di   = 917.0   # density of ice   (kg/m^3)
    didw = di/dw
    dsdi = ds/di
    ci   = 2054.0  # heat capacity of fresh ice (j/kg/k)
    li   = 3.34e5     # latent heat of fusion (j/kg-ice)
    si   = 1.0      # salinity of sea ice
    mu   = 0.054   # relates freezing temp to salinity
    tfi  = -mu*si     # sea ice freezing temp = -mu*salinity
    tfw  = -1.8     # tfw - seawater freezing temp (c)
    tfi0 = tfi-0.0001
    dici = di*ci
    dili = di*li
    dsli = ds*li
    ki4  = ki*4.0
    zero = 0.0
    one  = 1.0
    # TODO: move variable definition to separate file 
    t0c    = 2.7315e+2
    
    # vecotorize constants
    ip = np.zeros(im)
    ip[:] = np.nan
    tsf = np.zeros(im)
    tsf[:] = np.nan
    ai = np.zeros(im)
    ai[:] = np.nan
    k12 = np.zeros(im)
    k12[:] = np.nan
    k32 = np.zeros(im)
    k32[:] = np.nan
    a1 = np.zeros(im)
    a1[:] = np.nan
    b1 = np.zeros(im)
    b1[:] = np.nan
    c1 = np.zeros(im)
    c1[:] = np.nan
    tmelt = np.zeros(im)
    tmelt[:] = np.nan
    h1 = np.zeros(im)
    h1[:] = np.nan
    h2 = np.zeros(im)
    h2[:] = np.nan
    bmelt = np.zeros(im)
    bmelt[:] = np.nan
    dh = np.zeros(im)
    dh[:] = np.nan
    f1 = np.zeros(im)
    f1[:] = np.nan
    hdi = np.zeros(im)
    hdi[:] = np.nan
    wrk = np.zeros(im)
    wrk[:] = np.nan
    wrk1 = np.zeros(im)
    wrk1[:] = np.nan
    bi = np.zeros(im)
    bi[:] = np.nan
    a10 = np.zeros(im)
    a10[:] = np.nan
    b10 = np.zeros(im)
    b10[:] = np.nan

    dt2  = 2. * delt
    dt4  = 4. * delt
    dt6  = 6. * delt
    dt2i = one / dt2

    snowd[flag] = snowd[flag]  * dwds
    hdi[flag] = (dsdw*snowd[flag] + didw * hice[flag])
    
    snowd[flag & (hice < hdi)] = snowd[flag & (hice < hdi)] + \
        hice[flag & (hice < hdi)] - hdi[flag & (hice < hdi)]
    hice[flag & (hice < hdi)]  = hice[flag & (hice < hdi)] + \
            (hdi[flag & (hice < hdi)] - hice[flag & (hice < hdi)]) * dsdi   
    
    snof[flag] = snof[flag] * dwds
    tice[flag] = tice[flag] - t0c
    stsice[flag,0] = np.minimum(stsice[flag,0] - t0c, tfi0)     # degc
    stsice[flag,1] = np.minimum(stsice[flag,1] - t0c, tfi0)     # degc
    
    ip[flag] = i0 * sneti[flag] # ip +v here (in winton ip=-i0*sneti)

    tsf[flag & (snowd > zero)] = zero
    ip[flag & (snowd > zero)]  = zero

    tsf[flag & (snowd <= zero)] = tfi
    ip[flag & (snowd <= zero)] = i0 * sneti[flag & (snowd <= zero)]  # ip +v here (in winton ip=-i0*sneti)

    tice[flag] = np.minimum(tice[flag], tsf[flag])    
    
    # compute ice temperature
    bi[flag] = hfd[flag]
    ai[flag] = hfi[flag] - sneti[flag] + ip[flag] - tice[flag]*hfd[flag]
  # +v sol input here
    k12[flag] = ki4*ks / (ks*hice[flag] + ki4*snowd[flag])
    k32[flag] = (ki+ki) / hice[flag]

    wrk[flag] = one / (dt6*k32[flag] + dici*hice[flag])
    a10[flag] = dici*hice[flag]*dt2i + k32[flag]*(dt4*k32[flag] + dici*hice[flag])*wrk[flag]

    b10[flag] = -di*hice[flag] * (ci*stsice[flag,0] + li*tfi/ \
            stsice[flag,0]) * dt2i - ip[flag] - k32[flag]*\
            (dt4*k32[flag]*tfw + dici*hice[flag]*stsice[flag,1]) \
            * wrk[flag]
  
    wrk1[flag] = k12[flag] / (k12[flag] + bi[flag])  
    
    a1[flag] = a10[flag] + bi[flag] * wrk1[flag]
    b1[flag] = b10[flag] + ai[flag] * wrk1[flag]
    c1[flag]   = dili * tfi * dt2i * hice[flag]

    stsice[flag,0] = -(np.sqrt(b1[flag]*b1[flag] - 4.0*a1[flag] * \
            c1[flag]) + b1[flag])/(a1[flag]+a1[flag])
    tice[flag] = (k12[flag]*stsice[flag,0] - ai[flag]) / (k12[flag] + \
            hfd[flag])

    a1[flag & (tice > tsf)] = a10[flag & (tice > tsf)] + k12[flag & (tice > tsf)]
    b1[flag & (tice > tsf)] = b10[flag & (tice > tsf)] - k12[flag & (tice > tsf)]*tsf[flag & (tice > tsf)]
    stsice[flag & (tice > tsf), 0] = \
            -(np.sqrt(b1[flag & (tice > tsf)] * \
            b1[flag & (tice > tsf)] -\
            4.0*a1[flag & (tice > tsf)] *\
            c1[flag & (tice > tsf)]) + \
            b1[flag & (tice > tsf)])/ \
            (a1[flag & (tice > tsf)] + \
            a1[flag & (tice > tsf)])
    tice[flag & (tice > tsf)] = tsf[flag & (tice > tsf)]
    tmelt[flag & (tice > tsf)] = \
            (k12[flag & (tice > tsf)] * \
            (stsice[flag & (tice > tsf), 0] - \
            tsf[flag & (tice > tsf)]) - \
            (ai[flag & (tice > tsf)] + \
            hfd[flag & (tice > tsf)] *\
            tsf[flag & (tice > tsf)])) * \
            delt
              
    tmelt[flag & (tice <= tsf)] = zero
    snowd[flag & (tice <= tsf)] = \
            snowd[flag & (tice <= tsf)] + \
            snof[flag & (tice <= tsf)] * \
            delt


    stsice[flag,1] = (dt2*k32[flag]*(stsice[flag,0] + tfw + tfw) \
            +  dici*hice[flag]*stsice[flag,1]) * one / \
            (dt6*k32[flag] + dici*hice[flag])


    bmelt[flag] = (focn[flag] + \
            ki4*(stsice[flag,1] - tfw)/hice[flag]) * delt

#  --- ...  resize the ice ...

    h1[flag] = 0.5 * hice[flag]
    h2[flag] = 0.5 * hice[flag]


#  --- ...  top ...
                      
    snowmt[flag & (tmelt <= snowd*dsli)] = \
            tmelt[flag][tmelt[flag]<=snowd[flag]*dsli]  / dsli
    snowd[flag & (tmelt <= snowd*dsli)] = \
            snowd[flag][tmelt[flag]<=snowd[flag]*dsli] -\
            snowmt[flag][tmelt[flag]<=snowd[flag]*dsli]
          
    snowmt[flag & (tmelt > snowd*dsli)] = \
             snowd[flag][tmelt[flag]>snowd[flag]*dsli]
    h1[flag & (tmelt > snowd*dsli)] = \
            h1[flag][tmelt[flag]>snowd[flag]*dsli] - \
            (tmelt[flag][tmelt[flag]>snowd[flag]*dsli] - \
            snowd[flag][tmelt[flag]>snowd[flag]*dsli]*dsli) / \
            (di * (ci - li/ \
            stsice[flag,0][tmelt[flag]>snowd[flag]*dsli]) *\
            (tfi - stsice[flag,0][tmelt[flag]>snowd[flag]*dsli]))
    snowd[flag & (tmelt > snowd*dsli)] = zero
        

#  --- ...  and bottom


    dh[flag & (bmelt < zero)] = -bmelt[flag][bmelt[flag] < zero] \
            / (dili + dici*(tfi - tfw))
    stsice[flag & (bmelt < zero), 1]=(h2[flag][bmelt[flag] < zero]\
            *stsice[flag,1][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero]*tfw) / \
            (h2[flag][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero])
    h2[flag & (bmelt < zero)] = h2[flag][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero]
    
    h2[flag & (bmelt <= zero)] = h2[flag][bmelt[flag] <= zero] - \
            bmelt[flag][bmelt[flag] <= zero] / \
            (dili + dici*(tfi - stsice[flag,1][bmelt[flag] <= zero]))

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
    # print('fortran\n', ser_b1[~np.isclose(b1,ser_b1, equal_nan=True)], '\n python \n', b1[~np.isclose(b1,ser_b1, equal_nan=True)])
    assert np.allclose(ser_b1, b1, equal_nan=True)
    ser_stsice = ser.read("stsice2", sp)
    assert np.allclose(ser_stsice, stsice, equal_nan=True)

          

#  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

    hice[flag] = h1[flag] + h2[flag]

          
    # begin if_hice_block
    # begin if_h1_block

    f1[flag & (hice > zero) & (h1 > 0.5*hice)] = one-\
            (h2[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]+\
            h2[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]])\
            / hice[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]
    stsice[flag & (hice > zero) & (h1 > 0.5*hice), 1] = \
            f1[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]] * \
            (stsice[flag,0][hice[flag]>zero][h1[flag]>0.5*hice[flag]]\
            + li*tfi/ \
            (ci* \
            stsice[flag,0][hice[flag]>zero][h1[flag]>0.5*hice[flag]]))\
            + (one - \
            f1[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]) * \
            stsice[flag,1][hice[flag]>zero][h1[flag]>0.5*hice[flag]]

    # begin if_stsice_block

    hice[flag & (hice > zero) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)] \
            = hice[flag & (hice > zero) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)] - \
            h2[flag & (hice > zero) & (h1 > 0.5*hice) & (stsice[:,1] > tfi)]* \
            ci*(stsice[flag & (hice > zero) & (h1 > 0.5*hice) & \
            (stsice[:,1]> tfi), 1] - tfi)/(li*delt)

    stsice[flag & (hice > zero) & (h1 > 0.5*hice) & (stsice[:,1] > tfi), 1] = tfi
              
    # end if_stsice_block

    # else if_h1_block
    
    # hice[flag] > zero
    # h1[flag] <= 0.5*hice[flag]

    f1[flag & (hice > zero) & (h1 <= 0.5*hice)] = \
            (h1[flag][hice[flag]>zero][h1[flag][hice[flag]>zero]<=0.5*hice[flag][hice[flag]>zero]]+\
            h1[flag][hice[flag]>zero][h1[flag][hice[flag]>zero]<=0.5*hice[flag][hice[flag]>zero]]) / \
            hice[flag][hice[flag]>zero][h1[flag][hice[flag]>zero]<=0.5*hice[flag][hice[flag]>zero]]

    stsice[flag & (hice > zero) & (h1 <= 0.5*hice), 0] = \
            f1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]*(\
            stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]\
            + li*tfi/(ci*stsice[flag,0][hice[flag]>zero]\
            [h1[flag]<=0.5*hice[flag]]))+(one - \
            f1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]])\
            *stsice[flag,1][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]

    stsice[flag & (hice > zero),0][h1[flag]<=0.5*hice[flag]]= (\
            stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]\
            - np.sqrt(stsice[flag,0][hice[flag]>zero]\
            [h1[flag]<=0.5*hice[flag]]*stsice[flag,0]\
            [hice[flag]>zero][h1[flag]<=0.5*hice[flag]] \
            - 4.0*tfi*li/ci)) * 0.5

    # end if_h1_block

    k12[flag & (hice > zero)] = ki4*ks / (ks* \
            hice[flag][hice[flag]>zero] + ki4* \
            snowd[flag & (hice > zero)])

    gflux[flag & (hice > zero)] = k12[flag & (hice > zero)] * \
            (stsice[flag & (hice > zero),0] -\
            tice[flag & (hice > zero)])
    
    # else if_hice_block

    snowd[flag & (hice <= zero)] = snowd[flag & (hice <=zero)] + \
            (h1[flag & (hice <= zero)]*(ci*\
            (stsice[flag & (hice <= zero), 0] - tfi)- li*(one - tfi/ \
            stsice[flag & (hice <= zero), 0])) +\
            h2[flag & (hice <= zero)]*(ci*\
            (stsice[flag & (hice <= zero), 1] - tfi) - li)) / li

    hice[flag & (hice <= zero)] = np.maximum(zero, \
            snowd[flag & (hice <= zero)]*dsdi)

    snowd[flag & (hice <= zero)] = zero

    stsice[flag & (hice <= zero), 0] = tfw
    stsice[flag & (hice <= zero), 1] = tfw

    gflux[flag & (hice <= zero)] = zero
    
    # end if_hice_block

    gflux[flag] = fice[flag] * gflux[flag]
    snowmt[flag] = snowmt[flag] * dsdw
    snowd[flag] = snowd[flag] * dsdw
    tice[flag] = tice[flag]     + t0c
    stsice[flag,0] = stsice[flag,0] + t0c
    stsice[flag,1] = stsice[flag,1] + t0c
    
    # end if_flag_block


    return snowd, hice, stsice, tice, snof, snowmt, gflux



def fpvs(t):
    """Compute saturation vapor pressure over liquid
    t: temperature in Kelvin
    """

    # constant definition
    # TODO - this should be moved into a shared physics constants / physics functions module
    psat     = 6.1078e+2
    rv       = 4.6150e+2
    ttp      = 2.7316e+2
    cvap     = 1.8460e+3
    csol     = 2.1060e+3
    hvap     = 2.5000e+6
    hfus     = 3.3358e+5
    dldt     = cvap-csol
    heat     = hvap+hfus
    xpona    = -dldt/rv
    xponb    = -dldt/rv+heat/(rv*ttp)
    tr       = ttp/t

    return psat*(tr**xpona)*np.exp(xponb*(1.-tr))
