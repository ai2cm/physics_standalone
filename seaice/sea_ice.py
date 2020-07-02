#!/usr/bin/env python3

import numpy as np

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):
        
    # TODO - implement sea-ice model
    hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx = sfc_sice(in_dict['im'], in_dict['km'], in_dict['ps'], in_dict['t1'], in_dict['q1'], in_dict['delt'], in_dict['sfcemis'], in_dict['dlwflx'], in_dict['sfcnsw'], in_dict['sfcdsw'], in_dict['srflag'], in_dict['cm'], in_dict['ch'], in_dict['prsl1'], in_dict['prslki'], in_dict['islimks'], in_dict['wind'], in_dict['flat_iter'], in_dict['lprnt'], in_dict['ipr'], in_dict['cimin'], in_dict['hice'], in_dict['fice'], in_dict['tice'], in_dict['weasd'], in_dict['tskin'], in_dict['tprcp'], in_dict['stc'], in_dict['ep'])
    
    d = dict(((k, eval(k)) for k in (hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx)))
    in_dict.update(d)
    return {key: in_dict.get(key, None) for key in OUT_VARS}

def sfc_sice(im, km, ps, t1, q1, delt, sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,
        cm, ch, prsl1, prslki, islimks, wind, flat_iter, lprnt, ipr, cimin,
        hice, fice, tice, weasd, tskin, tprcp, stc, ep):

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
    ffw = np.zeros(im)
    evapi = np.zeros(im)
    evapw = np.zeros(im)
    sneti = np.zeros(im)
    snetw = np.zeros(im)
    hfd = np.zeros(im)
    hfi = np.zeros(im)
    focn = np.zeros(im)
    snof = np.zeros(im)
    rch = np.zeros(im)
    rho = np.zeros(im)
    snowd = np.zeros(im)
    theta1 = np.zeros(im)
    flag = np.zeros(im)    

    stsice = np.zeros(im, kmi)

    return hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx
#  --- ...  set flag for sea-ice

    flag = (islimsk == 2) and flag_iter
    hice[flag_iter and (islimsk < 2)] = zero
    fice[flag_iter and (islimsk < 2)] = zero

# TODO: save mask "flag and srflag > zero" as logical array?
    ep[flag and srflag > zero]    = ep[flag and srflag > zero]*(one-srflag[flag and srflag > zero])
    weasd[flag and srflag > zero] = weasd[flag and srflag > zero] + 1.e3*tprcp[flag and srflag > zero]*srflag[flag and srflag > zero]
    tprcp[flag and srflag > zero] = tprcp[flag and srflag > zero]*(one-srflag[flag and srflag > zero])
#  --- ...  update sea ice temperature

    stsice[flag, :] = stci[flag, :]

#  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
#           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
#           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
#           is sat. hum. at surface
#           convert slrad to the civilized unit from langley minute-1 k-4

#         dlwflx has been given a negative sign for downward longwave
#         sfcnsw is the net shortwave flux (direction: dn-up)

    q0[flag]  = np.max(q1[flag], 1.0e-8)
    theta1[flag] = t1[flag] * prslki[flag]
    rho[flag]    = prsl1[flag] / (rd*t1[flag]*(one+rvrdm1*q0))
    qs1       = fpvs(t1[flag])
    qs1       = np.max(eps*qs1 / (prsl1[flag] + epsm1*qs1), 1.e-8)
    q0        = min(qs1, q0)
  
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

    evapi[flag] = elocp * rch[flag] * (qssi - q0)
    evapw[flag] = elocp * rch[flag] * (qssw - q0)

    snetw[flag] = sfcdsw[flag] * (one - albfw)
    snetw[flag] = np.min(3.*sfcnsw[flag]/(one+2.*ffw[flag]), snetw[flag])
    sneti[flag] = (sfcnsw[flag] - ffw[flag]*snetw[flag]) / fice[flag]

    t12 = tice[flag] * tice[flag]
    t14 = t12 * t12

#  --- ...  hfi = net non-solar and upir heat flux @ ice surface

    hfi[flag] = -dlwflx[flag] + sfcemis[flag]*sbc*t14 + evapi[flag] + rch[flag]*(tice[flag] - theta1[flag])
    hfd[flag] = 4.*sfcemis[flag]*sbc*tice[flag]*t12 + (one + elocp*eps*hvap*qs1/(rd*t12)) * rch[flag]


    t12 = tgice * tgice
    t14 = t12 * t12

#  --- ...  hfw = net heat flux @ water surface (within ice)

    focn[flag] = 2.   # heat flux from ocean - should be from ocn model
    snof[flag] = zero    # snowfall rate - snow accumulates in gbphys
  
    hice[flag] = np.max( np.min( hice[flag], himax ), himin )
    snowd[flag] = np.min( snowd[flag], hsmax )

# TODO: write more efficiently, save mask as new variable?
    if any(snowd[flag] > (2.*hice[flag])):
        print('warning: too much snow :', snowd[flag][snowd[flag] > (2.*hice[flag])])
        snowd[flag][snowd[flag] > (2.*hice[flag])] = hice[flag][snowd[flag] > (2.*hice[flag])] + hice[flag][snowd[flag] > (2.*hice[flag])]
        print('fix: decrease snow depth to:',snowd[flag][snowd[flag] > (2.*hice[flag])])

# call function ice3lay
    snowd, hice, stsice, tice, snof, snowmt, gflux = ice3lay(                                                      
           im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr)

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

    stc[flag,:] = np.min(stsice[flag,k], t0c)

#  --- ...  calculate sensible heat flux (& evap over sea ice)

    hflxi      = rch[flag] * (tice[flag] - theta1[flag])
    hflxw      = rch[flag] * (tgice      - theta1[flag])
    hflx[flag] = fice[flag]* hflxi       + ffw[flag]*hflxw
    evap[flag] = fice[flag]* evapi[flag] + ffw[flag]*evapw[flag]
#  
#  --- ...  the rest of the output

    qsurf[flag] = q1[flag] + evap[flag] / (elocp*rch[flag])

#  --- ...  convert snow depth back to mm of water equivalent

    weasd[flag]  = snowd[flag] * 1000.0
    snwdph[flag] = weasd[flag] * dsi             # snow depth in mm

    tem     = 1.0 / rho[flag]
    hflx[flag] = hflx[flag] * tem * cpinv
    evap[flag] = evap[flag] * tem * hvapi

    return hice, fice, tice, weasd, tskin, tprcp, ep, snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx


def fpvs(t):
    # Compute saturation vapor pressure over liquid
    # t: temperature in Kelvin

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
