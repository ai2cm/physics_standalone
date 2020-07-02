#!/usr/bin/env python3

import numpy as np

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):
        
    # TODO - implement sea-ice model
    #for keys, values in in_dict.items():
    #print(keys)
    #print(values)
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

return hice, fice, tice, weasd, tskin, tprcp, ep, snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx


