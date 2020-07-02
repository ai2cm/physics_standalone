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

def sfc_sice(im, km, ps, t1, q1, delt, sfcemis, dlwflx, sfcnsw, sfcdsw, srflag, cm, ch, prsl1, prslki, islimks, wind, flat_iter, lprnt, ipr, cimin, hice, fice, tice, weasd, tskin, tprcp, stc, ep):

    # constant parameterts
    kmi = 2
    zero = 0.
    one = 1.
    cpinv = one/cp
    hvapi = one/hvap
    elocp = hvap/cp
    himax = 8.          # maximum ice thickness allowed
    himin = 0.          # minimum ice thickness required
    hsmax = 2.          # maximum snow depth allowed
    timin = 173.        # minimum temperature allowed for snow/ice
    albfw = 0.          # albedo for lead
    dsi   = one/0.33
    
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


