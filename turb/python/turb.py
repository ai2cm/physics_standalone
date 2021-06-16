#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import math
from mfscu import mfscu, assert_test
from mfpblt import mfpblt
from routines import gpvs, fpvs

OUT_VARS = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]

#def run(in_dict, compare_dict):
def run(in_dict):
    """run function"""

    compare_dict = []

    dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl = satmedmfvdif(in_dict['im'], in_dict['ix'], in_dict['km'], 
                 in_dict['ntrac'], in_dict['ntcw'], in_dict['ntiw'], in_dict['ntke'], 
                 in_dict['dv'], in_dict['du'], in_dict['tdt'], in_dict['rtg'], 
                 in_dict['u1'], in_dict['v1'], in_dict['t1'], in_dict['q1'], 
                 in_dict['swh'], in_dict['hlw'], in_dict['xmu'], in_dict['garea'],
                 in_dict['psk'], in_dict['rbsoil'], in_dict['zorl'],
                 in_dict['u10m'], in_dict['v10m'], in_dict['fm'], in_dict['fh'],
                 in_dict['tsea'], in_dict['heat'], in_dict['evap'], in_dict['stress'],
                 in_dict['spd1'], in_dict['kpbl'],
                 in_dict['prsi'], in_dict['del'], in_dict['prsl'], in_dict['prslk'], in_dict['phii'],
                 in_dict['phil'], in_dict['delt'],
                 in_dict['dspheat'], in_dict['dusfc'], in_dict['dvsfc'], in_dict['dtsfc'], 
                 in_dict['dqsfc'], in_dict['hpbl'],
                 in_dict['kinver'], in_dict['xkzm_m'], in_dict['xkzm_h'], in_dict['xkzm_s'],
                 compare_dict)

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = np.zeros(1, dtype=np.float64)

    out_dict["dv"] = dv
    out_dict["du"] = du
    out_dict["tdt"] = tdt
    out_dict["rtg"] = rtg
    out_dict["kpbl"] = kpbl
    out_dict["dusfc"] = dusfc
    out_dict["dvsfc"] = dvsfc
    out_dict["dtsfc"] = dtsfc
    out_dict["dqsfc"] = dqsfc
    out_dict["hpbl"] = hpbl
    
    return out_dict


def satmedmfvdif(im, ix, km, ntrac, ntcw, ntiw, ntke,
                 dv, du, tdt, rtg, u1, v1, t1, q1, swh, hlw, xmu, garea,
                 psk, rbsoil, zorl, u10m, v10m, fm, fh, tsea, heat, evap, stress, spd1, kpbl,
                 prsi, del_, prsl, prslk, phii, phil, delt,
                 dspheat, dusfc, dvsfc, dtsfc, dqsfc, hpbl,
                 kinver, xkzm_m, xkzm_h, xkzm_s,
                 compare_dict):

    # Physics Constants used from physcon module

    grav  = 9.80665e0
    rd    = 2.87050e2
    cp    = 1.00460e3
    rv    = 4.61500e2
    hvap  = 2.50000e6
    hfus  = 3.33580e5
    fv    = rv/rd - 1.0
    eps   = rd/rv
    epsm1 = rd/rv - 1.0

    gravi  = 1.0/grav
    g      = grav
    gocp   = g/cp
    cont   = cp/g
    conq   = hvap/g
    conw   = 1.0/g
    elocp  = hvap/cp
    el2orc = hvap*hvap/(rv*cp)
    wfac   = 7.0
    cfac   = 4.5
    gamcrt = 3.0
    gamcrq = 0.0
    sfcfrac = 0.1
    vk     = 0.4
    rimin  = -100.0
    rbcr   = 0.25
    zolcru = -0.02
    tdzmin = 1.0e-3
    rlmn   = 30.0
    rlmx   = 500.0
    elmx   = 500.0
    prmin  = 0.25
    prmax  = 4.0
    prtke  = 1.0
    prscu  = 0.67
    f0     = 1.0e-4
    crbmin = 0.15
    crbmax = 0.35
    tkmin  = 1.0e-9
    dspfac = 0.5
    dspmax = 10.0
    qmin   = 1.0e-8
    qlmin  = 1.0e-12
    zfmin  = 1.0e-8
    aphi5  = 5.0
    aphi16 = 16.0
    elmfac = 1.0
    elefac = 1.0
    cql    = 100.0
    dw2min = 1.0e-4
    dkmax  = 1000.0
    xkgdx  = 25000.0
    qlcr   = 3.5e-5
    zstblmax = 2500.0
    xkzinv = 0.15
    h1     = 0.33333333
    ck0    = 0.4
    ck1    = 0.15
    ch0    = 0.4
    ch1    = 0.15
    ce0    = 0.4
    rchck  = 1.5
    cdtn   = 25.0

    # Local arrays
    lcld  = np.zeros(shape=im, dtype=np.int64)
    kcld  = np.zeros(shape=im, dtype=np.int64)
    krad  = np.zeros(shape=im, dtype=np.int64)
    mrad  = np.zeros(shape=im, dtype=np.int64)
    kx1   = np.zeros(shape=im, dtype=np.int64)
    kpblx = np.zeros(shape=im, dtype=np.int64)

    tke  = np.zeros(shape=(im,km))
    tkeh = np.zeros(shape=(im,km-1))

    theta  = np.zeros(shape=(im,km))
    thvx   = np.zeros(shape=(im,km))
    thlvx  = np.zeros(shape=(im,km))
    qlx    = np.zeros(shape=(im,km))
    thetae = np.zeros(shape=(im,km))
    thlx   = np.zeros(shape=(im,km))
    slx    = np.zeros(shape=(im,km))
    svx    = np.zeros(shape=(im,km))
    qtx    = np.zeros(shape=(im,km))
    tvx    = np.zeros(shape=(im,km))
    pix    = np.zeros(shape=(im,km))

    radx = np.zeros(shape=(im,km-1))
    dku  = np.zeros(shape=(im,km-1))
    dkt  = np.zeros(shape=(im,km-1))
    dkq  = np.zeros(shape=(im,km-1))
    cku  = np.zeros(shape=(im,km-1))
    ckt  = np.zeros(shape=(im,km-1))

    plyr = np.zeros(shape=(im,km))
    rhly = np.zeros(shape=(im,km))
    cfly = np.zeros(shape=(im,km))
    qstl = np.zeros(shape=(im,km))

    dtdz1   = np.zeros(shape=im)
    gdx     = np.zeros(shape=im)
    phih    = np.zeros(shape=im)
    phim    = np.zeros(shape=im)
    prn     = np.zeros(shape=(im,km-1))
    rbdn    = np.zeros(shape=im)
    rbup    = np.zeros(shape=im)
    thermal = np.zeros(shape=im)
    ustart  = np.zeros(shape=im)
    ustar   = np.zeros(shape=im)
    wstar   = np.zeros(shape=im)
    hpblx   = np.zeros(shape=im)
    ust3    = np.zeros(shape=im)
    wst3    = np.zeros(shape=im)
    z0      = np.zeros(shape=im)
    crb     = np.zeros(shape=im)
    hgamt   = np.zeros(shape=im)
    hgamq   = np.zeros(shape=im)
    wscale  = np.zeros(shape=im)
    vpert   = np.zeros(shape=im)
    zol     = np.zeros(shape=im)
    sflux   = np.zeros(shape=im)
    radj    = np.zeros(shape=im)
    tx1     = np.zeros(shape=im)
    tx2     = np.zeros(shape=im)

    radmin = np.zeros(shape=im)

    zi      = np.zeros(shape=(im,km+1))
    zl      = np.zeros(shape=(im,km))
    zm      = np.zeros(shape=(im,km))
    xkzo    = np.zeros(shape=(im,km-1))
    xkzmo   = np.zeros(shape=(im,km-1))
    xkzm_hx = np.zeros(shape=im)
    xkzm_mx = np.zeros(shape=im)
    rdzt    = np.zeros(shape=(im,km-1))
    al      = np.zeros(shape=(im,km-1))
    ad      = np.zeros(shape=(im,km))
    au      = np.zeros(shape=(im,km-1))
    f1      = np.zeros(shape=(im,km))
    f2      = np.zeros(shape=(im,km*(ntrac-1)))

    elm    = np.zeros(shape=(im,km))
    ele    = np.zeros(shape=(im,km))
    rle    = np.zeros(shape=(im,km-1))
    ckz    = np.zeros(shape=(im,km))
    chz    = np.zeros(shape=(im,km))
    diss   = np.zeros(shape=(im,km-1))
    prod   = np.zeros(shape=(im,km-1))
    bf     = np.zeros(shape=(im,km-1))
    shr2   = np.zeros(shape=(im,km-1))
    xlamue = np.zeros(shape=(im,km-1))
    xlamde = np.zeros(shape=(im,km-1))
    gotvx  = np.zeros(shape=(im,km))
    rlam   = np.zeros(shape=(im,km-1))

    tcko = np.zeros(shape=(im,km))
    qcko = np.zeros(shape=(im,km,ntrac))
    ucko = np.zeros(shape=(im,km))
    vcko = np.zeros(shape=(im,km))
    buou = np.zeros(shape=(im,km))
    xmf  = np.zeros(shape=(im,km))

    tcdo = np.zeros(shape=(im,km))
    qcdo = np.zeros(shape=(im,km,ntrac))
    ucdo = np.zeros(shape=(im,km))
    vcdo = np.zeros(shape=(im,km))
    buod = np.zeros(shape=(im,km))
    xmfd = np.zeros(shape=(im,km))

    pblflg = np.zeros(shape=im, dtype=np.bool)
    sfcflg = np.zeros(shape=im, dtype=np.bool)
    flg    = np.zeros(shape=im, dtype=np.bool)
    scuflg = np.zeros(shape=im, dtype=np.bool)
    pcnvflg = np.zeros(shape=im, dtype=np.bool)

    dt2 = delt
    rdt = 1.0/ dt2

    # Comment from Fortran : The code is written assuming ntke=ntrac
    #                        If ntrac > ntke, the code needs to be modified

    c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx = gpvs()

    ntrac1 = ntrac - 1
    km1 = km - 1
    kmpbl = int(km / 2)
    kmscu = int(km / 2)

    zi[:,:-1] = phii[:,:-1] * gravi
    zl[:,:] = phil[:,:] * gravi
    # These variables are zeroed in the Fortran code but already initialized to zero
    # in the Python code : xmf, xmfd, buou, buod
    ckz[:,:] = ck1
    chz[:,:] = ch1

    zi[:,km] = phii[:,km] * gravi
    zm[:,:]    = zi[:,1:]
    gdx[:] = np.sqrt(garea[:])

    tke[:,:] = np.maximum(q1[:,:,ntke-1],tkmin)

    tkeh[:,:] = 0.5 * (tke[:,:-1] + tke[:,1:])

    rdzt[:,:] = 1.0 / (zl[:,1:] - zl[:,:-1])
    prn[:,:]  = 1.0

    kx1[:] = 1
    tx1[:] = 1.0 / prsi[:,0]
    tx2[:] = tx1[:]

    tem = 1.0 / (xkgdx-5.0)
    tem1 = (xkzm_h - 0.01) * tem
    tem2 = (xkzm_m - 0.01) * tem
    ptem = gdx[:] - 5.0

    if_else_array = gdx[:] >= xkgdx

    xkzm_hx[:] = (if_else_array * xkzm_h) + ~if_else_array * (0.01 + tem1 * ptem)
    xkzm_mx[:] = (if_else_array * xkzm_m) + ~if_else_array * (0.01 + tem2 * ptem)

    # xkzo and xkzmo are already set to 0.0 in their respective initialization

    for k in range(km1):
        for i in range(im):
            if k < kinver[i]:
                ptem = prsi[i,k+1] * tx1[i]
                tem1 = 1.0 - ptem
                tem1 = tem1*tem1 * 10.0
                xkzo[i,k] = xkzm_hx[i] * min(1.0,math.exp(-tem1))

                if(ptem >= xkzm_s):
                    xkzmo[i,k] = xkzm_mx[i]
                    kx1[i] = k+1
                else:
                    if k == kx1[i] and k > 1:
                        tx2[i] = 1.0 / prsi[i,k]
                    tem1 = 1.0 - prsi[i,k+1] * tx2[i]
                    tem1 = tem1 * tem1 * 5.0
                    xkzmo[i,k] = xkzm_mx[i] * min(1.0,math.exp(-tem1))
    
    z0[:] = 0.01 * zorl[:]
    dusfc[:] = 0.0
    dvsfc[:] = 0.0
    dtsfc[:] = 0.0
    dqsfc[:] = 0.0
    kpbl[:] = 1
    hpbl[:] = 0.0
    kpblx[:] = 1 - 1 # May need to think whether the -1 is necessary
    hpblx[:] = 0.0
    pblflg[:] = True
    sfcflg[:] = True
    sfcflg[:] = sfcflg[:] * ~(rbsoil[:] > 0.0)
    pcnvflg[:] = False
    scuflg[:] = True
    radmin[:] = 0.0
    mrad[:] = km1
    krad[:] = 1 - 1   # May need to think whether the -1 is necessary
    lcld[:] = km1 - 1 # May need to think whether the -1 is necessary
    kcld[:] = km1 - 1 # May need to think whether the -1 is necessary

    # for k in range(km):
    #     for i in range(im):
    #         pix[i,k] = psk[i] / prslk[i,k]
    #         theta[i,k] = t1[i,k] * pix[i,k]
    #         if ntiw > 0:
    #             tem = max(q1[i,k,ntcw-1], qlmin)
    #             tem1 = max(q1[i,k,ntiw-1], qlmin)
    #             qlx[i,k] = tem+tem1
    #             ptem = hvap*tem + (hvap+hfus)*tem1
    #             slx[i,k] = cp * t1[i,k] + phil[i,k] - ptem
    #         else:
    #             qlx[i,k] = max(q1[i,k,ntcw-1], qlmin)
    #             slx[i,k] = cp * t1[i,k] + phil[i,k] - hvap*qlx[i,k]
    #         tem2 = 1.0 + fv * max(q1[i,k,0],qmin) - qlx[i,k]
    #         thvx[i,k] = theta[i,k] * tem2
    #         tvx[i,k] = t1[i,k] * tem2
    #         qtx[i,k] = max(q1[i,k,0],qmin)+qlx[i,k]
    #         thlx[i,k] = theta[i,k] - pix[i,k]*elocp*qlx[i,k]
    #         thlvx[i,k] = thlx[i,k] * (1.0 + fv * qtx[i,k])
    #         svx[i,k] = cp * tvx[i,k]
    #         ptem1 = elocp * pix[i,k] * max(q1[i,k,0],qmin)
    #         thetae[i,k] = theta[i,k] + ptem1
    #         gotvx[i,k] = g / tvx[i,k]
    
    for k in range(km):
        pix[:,k] = psk[:] / prslk[:,k]
        theta[:,k] = t1[:,k] * pix[:,k]
        if ntiw > 0:
            tem = np.maximum(q1[:,k,ntcw-1], qlmin)
            tem1 = np.maximum(q1[:,k,ntiw-1], qlmin)
            qlx[:,k] = tem+tem1
            ptem = hvap*tem + (hvap+hfus)*tem1
            slx[:,k] = cp * t1[:,k] + phil[:,k] - ptem
        else:
            qlx[:,k] = np.maximum(q1[:,k,ntcw-1], qlmin)
            slx[:,k] = cp * t1[:,k] + phil[:,k] - hvap*qlx[:,k]
        tem2 = 1.0 + fv * np.maximum(q1[:,k,0],qmin) - qlx[:,k]
        thvx[:,k] = theta[:,k] * tem2
        tvx[:,k] = t1[:,k] * tem2
        qtx[:,k] = np.maximum(q1[:,k,0],qmin)+qlx[:,k]
        thlx[:,k] = theta[:,k] - pix[:,k]*elocp*qlx[:,k]
        thlvx[:,k] = thlx[:,k] * (1.0 + fv * qtx[:,k])
        svx[:,k] = cp * tvx[:,k]
        ptem1 = elocp * pix[:,k] * np.maximum(q1[:,k,0],qmin)
        thetae[:,k] = theta[:,k] + ptem1
        gotvx[:,k] = g / tvx[:,k]  

    tem1 = (tvx[:,1:] - tvx[:,:-1]) * rdzt[:,:]
    tem1 = tem1 > 1e-5
    xkzo[:,:] = xkzo[:,:]*~tem1 + tem1*np.minimum(xkzo,xkzinv)
    xkzmo[:,:] = xkzmo[:,:]*~tem1 + tem1*np.minimum(xkzmo,xkzinv)

    # np.testing.assert_array_equal(xkzo,compare_dict['xkzo'])
    # np.testing.assert_array_equal(xkzmo,compare_dict['xkzmo'])
    # np.testing.assert_array_equal(slx,compare_dict['slx'])
    # np.testing.assert_array_equal(thvx,compare_dict['thvx'])
    # np.testing.assert_array_equal(thlvx,compare_dict['thlvx'])
    # np.testing.assert_array_equal(svx,compare_dict['svx'])
    # np.testing.assert_array_equal(thetae,compare_dict['thetae'])
    # np.testing.assert_array_equal(gotvx,compare_dict['gotvx'])

    plyr[:,:] = 0.01 * prsl[:,:]
    for k in range(km):
        for i in range(im):
            es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, t1[i,k])
            qs = max(qmin, eps * es / (plyr[i,k] + epsm1*es))
            rhly[i,k] = max(0.0, min(1.0, max(qmin, q1[i,k,0])/qs))
            qstl[i,k] = qs

    # np.testing.assert_array_equal(plyr,compare_dict['plyr'])
    # np.testing.assert_array_equal(rhly,compare_dict['rhly'])
    # np.testing.assert_array_equal(qstl,compare_dict['qstl'])

    clwt = 1.0e-6 * (plyr * 0.001)
    filter0 = qlx > clwt

    onemrh = np.maximum(1e-10, 1.0-rhly)
    tem1 = np.minimum(np.maximum((onemrh*qstl)**0.49, 0.0001),1.0)
    tem1 = cql / tem1
    value0 = np.maximum(np.minimum(tem1*qlx, 50.0),0.0)
    tem2 = np.sqrt(np.sqrt(rhly))
    cfly[:,:] = np.minimum(np.maximum(tem2*(1.0-np.exp(-value0)),0.0),1.0)
    cfly = cfly * filter0

    #np.testing.assert_array_equal(cfly,compare_dict['cfly'])

    tem = 0.5 * (svx[:,:-1] + svx[:,1:])
    tem1 = 0.5 * (t1[:,:-1] + t1[:,1:])
    tem2 = 0.5 * (qstl[:,:-1] + qstl[:,1:])
    cfh = np.minimum(cfly[:,1:], 0.5*(cfly[:,:-1] + cfly[:,1:]))
    alp = g / tem
    gamma = el2orc * tem2 / (tem1**2.0)
    epsi = tem1 / elocp
    beta = (1.0 + gamma*epsi*(1.0+fv)) / (1.0 + gamma)
    chx = cfh * alp * beta + (1.0 - cfh) * alp
    cqx = cfh * alp * hvap * (beta - epsi)
    cqx = cqx + (1.0 - cfh) * fv * g
    ptem1 = (slx[:,1:] - slx[:,:-1]) * rdzt
    ptem2 = (qtx[:,1:] - qtx[:,:-1]) * rdzt
    bf[:,:] = chx * ptem1 + cqx * ptem2

    # np.testing.assert_array_equal(svx,compare_dict['svx'])
    # np.testing.assert_array_equal(t1,compare_dict['t1'])
    # np.testing.assert_array_equal(qstl,compare_dict['qstl'])
    # np.testing.assert_array_equal(cfly,compare_dict['cfly'])
    # np.testing.assert_array_equal(slx,compare_dict['slx'])
    # np.testing.assert_array_equal(qtx,compare_dict['qtx'])
    # np.testing.assert_array_equal(bf,compare_dict['bf'])

    tem = zi[:,1:-1] - zi[:,:-2]
    radx[:,:] = tem *(np.transpose(np.multiply(np.transpose(swh[:,:-1]),xmu)) + hlw[:,:-1])

    #np.testing.assert_array_equal(radx,compare_dict['radx'])

    sflux[:] = heat + evap*fv*theta[:,0]

    pblflg[:] = np.logical_and(~np.logical_or(~sfcflg, sflux <=0), pblflg)

    # np.testing.assert_array_equal(pblflg,compare_dict['pblflg'])
    # np.testing.assert_array_equal(sflux,compare_dict['sflux'])

    thermal[:] = pblflg*thlvx[:,0] + ~pblflg*(tsea * (1.0 + fv*np.maximum(q1[:,0,0],qmin)))

    # np.testing.assert_array_equal(thermal,compare_dict['thermal'])
    tem1 = 1.0e-7*(np.maximum(np.sqrt(u10m**2 + v10m**2),1.0) / (f0 * z0))
    crb[:] = 0.16 * (tem1 ** (-0.18))
    crb[:] = pblflg*rbcr + ~pblflg*(np.maximum(np.minimum(crb,crbmax),crbmin))

    # np.testing.assert_array_equal(crb,compare_dict['crb'])

    dtdz1[:] = dt2/(zi[:,1] - zi[:,0])
    ustar[:] = np.sqrt(stress)

    shr2[:,:] = np.maximum((u1[:,:-1]-u1[:,1:])**2.0 + (v1[:,:-1] -v1[:,1:])**2.0,dw2min) * rdzt * rdzt

    # np.testing.assert_array_equal(dtdz1,compare_dict['dtdz1'])
    # np.testing.assert_array_equal(ustar,compare_dict['ustar'])
    # np.testing.assert_array_equal(shr2,compare_dict['shr2'])

    flg[:] = False
    rbup[:] = rbsoil

    for k in range(kmpbl):
        # for i in range(im):
        #     if ~flg[i]:
        #         rbdn[i] = rbup[i]
        #         spdk2 = max((u1[i,k]**2 + v1[i,k]**2), 1.0)
        #         rbup[i] = (thlvx[i,k] - thermal[i]) * (g*zl[i,k]/thlvx[i,0])/spdk2
        #         kpblx[i] = k
        #         flg[i] = rbup[i] > crb[i]
        rbdn[:] = rbup[:]* ~flg + rbdn[:] * flg
        spdk2 = np.maximum(u1[:,k]**2 + v1[:,k]**2,1.0)
        rbup[:] = ~flg* ((thlvx[:,k] - thermal[:]) * (g*zl[:,k]/thlvx[:,0])/spdk2) + flg*rbup
        kpblx[:] = ~flg*k + flg*kpblx
        flg[:] = ~flg*(rbup > crb) + flg

    #np.testing.assert_array_equal(rbdn,compare_dict['rbdn'])
    # np.testing.assert_allclose(rbdn,compare_dict['rbdn'],rtol=1e-15, atol=0)
    # #np.testing.assert_array_equal(rbup,compare_dict['rbup'])
    # np.testing.assert_allclose(rbup,compare_dict['rbup'],rtol=1e-15, atol=0)
    # np.testing.assert_array_equal(flg,compare_dict['flg'])
    # np.testing.assert_array_equal(kpblx,compare_dict['kpblx']-1)

    # Note : kpblx and kpbl are adjusted based off of Python indices, not Fortran indices

    for i in range(im):
        if kpblx[i] > 0:
            k = kpblx[i]
            if rbdn[i] >= crb[i]:
                rbint = 0.0
            elif rbup[i] <= crb[i]:
                rbint = 1.0
            else:
                rbint = (crb[i]-rbdn[i])/(rbup[i]-rbdn[i])
            hpblx[i] = zl[i,k-1] + rbint*(zl[i,k]-zl[i,k-1])
            if hpblx[i] < zi[i,kpblx[i]]:
                kpblx[i] = kpblx[i] - 1
        else:
            hpblx[i] = zl[i,0]
            kpblx[i] = 0
        hpbl[i] = hpblx[i]
        kpbl[i] = kpblx[i]
        if kpbl[i] <= 0:
            pblflg[i] = False

    # np.testing.assert_array_equal(hpblx,compare_dict['hpblx'])
    # np.testing.assert_array_equal(kpblx,compare_dict['kpblx']-1)
    # np.testing.assert_array_equal(hpbl,compare_dict['hpbl'])
    # np.testing.assert_array_equal(kpbl,compare_dict['kpbl']-1)
    # np.testing.assert_array_equal(pblflg,compare_dict['pblflg'])

    zol[:] = np.maximum(rbsoil*fm*fm/fh, rimin)
    zol[:] = sfcflg*np.minimum(zol,-zfmin) + ~sfcflg*np.maximum(zol, zfmin)

    zol1 = zol*sfcfrac*hpbl/zl[:,0]

    phih[:] = sfcflg*(np.sqrt(np.complex128(1.0/(1.0-aphi16*zol1))))
    phim[:] = np.sqrt(phih)

    phim[:] = phim[:] + ~sfcflg*(1.0+aphi5*zol1)
    phih[:] = phih[:] + ~sfcflg*phim[:]

    # np.testing.assert_array_equal(zol,compare_dict['zol'])
    # np.testing.assert_array_equal(phih,compare_dict['phih'])
    # np.testing.assert_array_equal(phim,compare_dict['phim'])

    pcnvflg[:] = pblflg*(zol < zolcru)
    #wst3[:] = pblflg*(gotvx[:,0]*sflux*hpbl)
    #wstar[:] = pblflg*(wst3**h1)
    #ust3[:] = pblflg*(ustar**3)
    wscale[:] = pblflg*((ustar**3.0 + wfac*vk*(gotvx[:,0]*sflux*hpbl)*sfcfrac)**h1)
    ptem = pblflg*(ustar/aphi5)
    wscale[:] = pblflg*(np.maximum(wscale,ptem))

    # Get rid of nans in computation
    where_are_nans = np.isnan(wscale)
    wscale[where_are_nans] = 0.0

    # np.testing.assert_array_equal(pcnvflg,compare_dict['pcnvflg'])
    # np.testing.assert_array_equal(wscale,compare_dict['wscale'])

    hgamt[:] = pcnvflg * (heat/wscale)
    hgamq[:] = pcnvflg * (evap/wscale)
    vpert[:] = pcnvflg * (hgamt + hgamq*fv*theta[:,0])
    vpert[:] = pcnvflg * np.maximum(vpert,0.0)

    # Reset the nan values to zero
    where_are_nans = np.isnan(hgamt)
    hgamt[where_are_nans] = 0.0
    where_are_nans = np.isnan(hgamq)
    hgamq[where_are_nans] = 0.0
    where_are_nans = np.isnan(vpert)
    vpert[where_are_nans] = 0.0
    where_are_nans = np.isnan(thermal)

    tem = pcnvflg * np.minimum(cfac*vpert,gamcrt)
    thermal = pcnvflg * (thermal + tem) + ~pcnvflg * thermal

    # np.testing.assert_array_equal(hgamt,compare_dict['hgamt'])
    # np.testing.assert_array_equal(hgamq,compare_dict['hgamq'])
    # np.testing.assert_array_equal(vpert,compare_dict['vpert'])
    # np.testing.assert_array_equal(thermal,compare_dict['thermal'])

    flg[:] = True
    flg[:] = np.logical_and(flg, ~pcnvflg)
    rbup[:] = pcnvflg*rbsoil + ~pcnvflg*rbup

    # np.testing.assert_array_equal(flg,compare_dict['flg'])
    # np.testing.assert_array_equal(rbup,compare_dict['rbup'])

    for k in range(1,kmpbl):
        for i in range(im):
            if ~flg[i]:
                rbdn[i] = rbup[i]
                spdk2 = max((u1[i,k]**2 + v1[i,k]**2),1.0)
                rbup[i] = (thlvx[i,k] - thermal[i]) * (g*zl[i,k]/thlvx[i,0])/spdk2
                kpbl[i] = k
                flg[i] = rbup[i] > crb[i]

    # np.testing.assert_allclose(rbdn,compare_dict["rbdn"], rtol=1e-15, atol=0)
    # np.testing.assert_array_equal(rbup,compare_dict["rbup"])
    # np.testing.assert_array_equal(kpbl,compare_dict["kpbl"]-1)
    # np.testing.assert_array_equal(flg,compare_dict["flg"])

    for i in range(im):
        if pcnvflg[i]:
            k = kpbl[i]
            if rbdn[i] >= crb[i]:
                rbint = 0.0
            elif rbup[i] <= crb[i]:
                rbint = 1.0
            else:
                rbint = (crb[i]-rbdn[i])/(rbup[i]-rbdn[i])
            hpbl[i] = zl[i,k-1] + rbint * (zl[i,k] - zl[i,k-1])
            if hpbl[i] < zi[i,kpbl[i]]:
                kpbl[i] = kpbl[i] - 1
            if kpbl[i] <= 0:
                pcnvflg[i] = False
                pblflg[i] = False

    # np.testing.assert_array_equal(hpbl,compare_dict["hpbl"])
    # np.testing.assert_array_equal(kpbl,compare_dict["kpbl"]-1)
    # np.testing.assert_array_equal(pcnvflg,compare_dict["pcnvflg"])
    # np.testing.assert_array_equal(pblflg,compare_dict["pblflg"])

    flg[:] = scuflg

    for k in range(km1):
        evalu = np.logical_and(flg,zl[:,k]>=zstblmax)
        lcld[:] = evalu * k + ~evalu*lcld
        flg[:] = evalu*False + ~evalu*flg

    # np.testing.assert_array_equal(flg,compare_dict["flg"])
    # np.testing.assert_array_equal(lcld,compare_dict["lcld"]-1)

    flg[:] = scuflg

    for k in range(kmscu-1,-1,-1):
        # for i in range(im):
        #     if flg[i] and (k <= lcld[i]):
        #         if qlx[i,k] >= qlcr:
        #             kcld[i] = k
        #             flg[i] = False
        evalu = np.logical_and(np.logical_and(flg,k <= lcld),qlx[:,k]>=qlcr)
        kcld[:] = evalu*k + ~evalu*kcld
        flg[:] = evalu*False + ~evalu*flg

    # np.testing.assert_array_equal(flg,compare_dict["flg"])
    # np.testing.assert_array_equal(kcld,compare_dict["kcld"]-1)

    evalu = np.logical_and(scuflg,kcld == (km1-1))  # Adjustment made to km1 based on Python indexing
    scuflg[:] = np.logical_and(~evalu,scuflg)

    #np.testing.assert_array_equal(scuflg,compare_dict["scuflg"])

    flg[:] = scuflg

    for k in range(kmscu-1, -1, -1):
        for i in range(im):
            if flg[i] and (k <= kcld[i]):
                if qlx[i,k] >= qlcr:
                    if radx[i,k] < radmin[i]:
                        radmin[i] = radx[i,k]
                        krad[i] = k
                else:
                    flg[i] = False

    # np.testing.assert_array_equal(radmin,compare_dict["radmin"])
    # np.testing.assert_array_equal(krad,compare_dict["krad"]-1)
    # np.testing.assert_array_equal(flg,compare_dict["flg"])


    scuflg[:] = np.logical_and(~np.logical_and(scuflg,krad <= 0),scuflg)
    scuflg[:] = np.logical_and(~np.logical_and(scuflg,radmin >= 0.0), scuflg)

    # np.testing.assert_array_equal(scuflg,compare_dict["scuflg"])

    for k in range(km):
        tcko[:,k] = pcnvflg*t1[:,k] + ~pcnvflg*tcko[:,k]
        ucko[:,k] = pcnvflg*u1[:,k] + ~pcnvflg*ucko[:,k]
        vcko[:,k] = pcnvflg*v1[:,k] + ~pcnvflg*vcko[:,k]

        tcdo[:,k] = scuflg*t1[:,k] + ~scuflg*tcdo[:,k]
        ucdo[:,k] = scuflg*u1[:,k] + ~scuflg*ucdo[:,k]
        vcdo[:,k] = scuflg*v1[:,k] + ~scuflg*vcdo[:,k]

    # np.testing.assert_array_equal(tcko,compare_dict["tcko"])
    # np.testing.assert_array_equal(ucko,compare_dict["ucko"])
    # np.testing.assert_array_equal(vcko,compare_dict["vcko"])
    # np.testing.assert_array_equal(tcdo,compare_dict["tcdo"])
    # np.testing.assert_array_equal(ucdo,compare_dict["ucdo"])
    # np.testing.assert_array_equal(vcdo,compare_dict["vcdo"])

    for kk in range(ntrac1):
        for k in range(km):
            qcko[:, k, kk] = pcnvflg*q1[:,k,kk] + ~pcnvflg*qcko[:,k,kk]
            qcdo[:, k, kk] = scuflg *q1[:,k,kk] + ~scuflg *qcdo[:,k,kk]

    # np.testing.assert_array_equal(qcko,compare_dict["qcko"])
    # np.testing.assert_array_equal(qcdo,compare_dict["qcdo"])

    kpbl, hpbl, buou, xmf, tcko, qcko, \
    ucko, vcko, xlamue = mfpblt(im,ix,km,kmpbl,ntcw,ntrac1,dt2,
                                pcnvflg,zl,zm,q1,t1,u1,v1,plyr,pix,thlx,thvx,
                                gdx,hpbl,kpbl,vpert,buou,xmf,
                                tcko,qcko,ucko,vcko,xlamue,
                                g, gocp, elocp, el2orc,
                                compare_dict)
    
    # np.testing.assert_array_equal(kpbl,compare_dict["kpbl"]-1)
    # np.testing.assert_array_equal(hpbl,compare_dict["hpbl"])
    # np.testing.assert_array_equal(buou,compare_dict["buou"])
    # xmf[np.isnan(xmf)] = 0.0
    # np.testing.assert_array_equal(xmf,compare_dict["xmf"])
    # test = compare_dict["tcko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(tcko,test)
    # np.testing.assert_array_equal(qcko,compare_dict["qcko"])
    # test = compare_dict["ucko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(ucko,test)
    # test = compare_dict["vcko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(vcko,test)
    # test = compare_dict["xlamue"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(xlamue,test)
    
    radj, mrad, buod, xmfd, tcdo, \
    qcdo, ucdo, vcdo, xlamde = mfscu(im,ix,km,kmscu,ntcw,ntrac1,dt2, 
                                     scuflg,zl,zm,q1,t1,u1,v1,plyr,pix,
                                     thlx,thvx,thlvx,gdx,thetae,radj,
                                     krad,mrad,radmin,buod,xmfd,
                                     tcdo,qcdo,ucdo,vcdo,xlamde,
                                     g, gocp, elocp, el2orc,
                                     compare_dict)

    for k in range(kmpbl):
        # for i in range(im):
        #     if(k < kpbl[i]):
        #         tem = phih[i] / phim[i]
        #         ptem = -3.0 * (max(zi[i,k+1]- sfcfrac*hpbl[i],0.0))**2 / hpbl[i]**2
        #         if pcnvflg[i]:
        #             prn[i,k] = 1.0 + (tem-1.0)*math.exp(ptem)
        #         else:
        #             prn[i,k] = tem
                
        #         prn[i,k] = min(prn[i,k],prmax)
        #         prn[i,k] = max(prn[i,k],prmin)

        #         ckz[i,k] = ck1 + (ck0-ck1)*math.exp(ptem)
        #         ckz[i,k] = min(ckz[i,k],ck0)
        #         ckz[i,k] = max(ckz[i,k],ck1)

        #         chz[i,k] = ch1 + (ch0-ch1)*math.exp(ptem)
        #         chz[i,k] = min(chz[i,k],ch0)
        #         chz[i,k] = max(chz[i,k],ch1)
        evalu = k < kpbl
        tem = phih / phim
        ptem = -3.0 * (np.maximum(zi[:,k+1]- sfcfrac*hpbl[:],0.0))**2 / hpbl[:]**2
        prn[:,k] = evalu*(pcnvflg*(1.0 + (tem-1.0) * np.exp(ptem)) + ~pcnvflg*tem) + ~evalu*prn[:,k]
        prn[:,k] = evalu*(np.minimum(prn[:,k],prmax)) + ~evalu*prn[:,k]
        prn[:,k] = evalu*(np.maximum(prn[:,k],prmin)) + ~evalu*prn[:,k]

        ckz[:,k] = evalu*(ck1 + (ck0-ck1)*np.exp(ptem)) + ~evalu*ckz[:,k]
        ckz[:,k] = evalu*(np.minimum(ckz[:,k],ck0)) + ~evalu*ckz[:,k]
        ckz[:,k] = evalu*(np.maximum(ckz[:,k],ck1)) + ~evalu*ckz[:,k]

        chz[:,k] = evalu*(ch1 + (ch0-ch1) * np.exp(ptem)) + ~evalu*chz[:,k]
        chz[:,k] = evalu*(np.minimum(chz[:,k],ch0)) + ~evalu*chz[:,k]
        chz[:,k] = evalu*(np.maximum(chz[:,k],ch1)) + ~evalu*chz[:,k]

    # np.testing.assert_allclose(prn, compare_dict["prn"],rtol=1e-15, atol=0)
    # np.testing.assert_allclose(ckz, compare_dict["ckz"],rtol=1e-15,atol=0)
    # np.testing.assert_allclose(chz, compare_dict["chz"],rtol=1e-15, atol=0)

    for k in range(km1):
        # for i in range(im):
        #     zlup = 0.0
        #     bsum = 0.0
        #     mlenflg = True
        #     for n in range(k,km1):
        #         if mlenflg:
        #             dz = zl[i,n+1] - zl[i,n]
        #             ptem = gotvx[i,n] * (thvx[i,n+1] - thvx[i,k]) * dz
        #             bsum = bsum + ptem
        #             zlup = zlup + dz
        #             if bsum >= tke[i,k]:
        #                 if ptem >= 0.0:
        #                     tem2 = max(ptem,zfmin)
        #                 else:
        #                     tem2 = min(ptem,-zfmin)
        #                 ptem1 = (bsum - tke[i,k]) / tem2
        #                 zlup = zlup - ptem1 * dz
        #                 zlup = max(zlup,0.0)
        #                 mlenflg= False
        #     zldn = 0.0
        #     bsum = 0.0
        #     mlenflg = True
        #     for n in range(k,-1,-1):
        #         if mlenflg:
        #             if n == 0:
        #                 dz = zl[i,0]
        #                 tem1 = tsea[i] * (1.0+fv*max(q1[i,0,0],qmin))
        #             else:
        #                 dz = zl[i,n] - zl[i,n-1]
        #                 tem1 = thvx[i,n-1]
        #             ptem = gotvx[i,n] * (thvx[i,k] - tem1) * dz
        #             bsum = bsum + ptem
        #             zldn = zldn + dz
        #             if bsum >= tke[i,k]:
        #                 if ptem >= 0.0:
        #                     tem2 = max(ptem,zfmin)
        #                 else:
        #                     tem2 = min(ptem,-zfmin)
        #                 ptem1 = (bsum - tke[i,k]) / tem2
        #                 zldn = zldn - ptem1 * dz
        #                 zldn = max(zldn,0.0)
        #                 mlenflg = False

        #     tem = 0.5 * (zi[i,k+1] - zi[i,k])  
        #     tem1 = min(tem,rlmn)

        #     ptem2 = min(zlup,zldn)
        #     rlam[i,k] = elmfac * ptem2
        #     rlam[i,k] = max(rlam[i,k],tem1)
        #     rlam[i,k] = min(rlam[i,k],rlmx)

        #     ptem2 = math.sqrt(zlup*zldn)
        #     ele[i,k] = elefac * ptem2
        #     ele[i,k] = max(ele[i,k],tem1)
        #     ele[i,k] = min(ele[i,k],elmx)

        zlup = np.zeros(im)
        bsum = np.zeros(im)
        mlenflg = np.ones(im,dtype=bool)
        for n in range(k,km1):
            dz = zl[:,n+1] - zl[:,n]
            ptem = gotvx[:,n] * (thvx[:,n+1] - thvx[:,k]) * dz
            bsum[:] = mlenflg*(bsum + ptem) + ~mlenflg*bsum
            zlup[:] = mlenflg*(zlup + dz) + ~mlenflg*zlup

            evalu0 = np.logical_and(mlenflg,(bsum >= tke[:,k]))
            evalu1 = ptem >= 0.0

            tem2 = evalu1*np.maximum(ptem,zfmin) + ~evalu1*np.minimum(ptem,-zfmin)
            ptem1 = (bsum  - tke[:,k]) / tem2
            zlup[:] = evalu0*np.maximum(zlup - ptem1 * dz,0.0) + ~evalu0*zlup

            mlenflg[:] = ~np.logical_or(~mlenflg,evalu0)

        zldn = np.zeros(im)
        bsum[:] = 0.0
        mlenflg[:] = True

        for n in range(k,-1,-1):
            evalu = (n == 0)
            dz = evalu*zl[:,0] + (not evalu)*(zl[:,n] - zl[:,n-1])
            tem1 = evalu*(tsea * (1.0+fv*np.maximum(q1[:,0,0],qmin))) + (not evalu)*thvx[:,n-1]

            ptem = gotvx[:,n] * (thvx[:,k] - tem1) * dz
            bsum[:] = mlenflg*(bsum+ptem) + ~mlenflg*bsum
            zldn[:] = mlenflg*(zldn+dz)   + ~mlenflg*zldn

            evalu0 = np.logical_and(mlenflg,bsum >= tke[:,k])
            evalu1 = (ptem >= 0.0)
            tem2 = evalu1*np.maximum(ptem,zfmin) + ~evalu1*np.minimum(ptem,-zfmin)
            ptem1 = (bsum - tke[:,k]) / tem2
            zldn[:] = evalu0*(np.maximum(zldn-ptem1*dz,0.0)) + ~evalu0*zldn
            mlenflg[:] = ~np.logical_or(~mlenflg,evalu0)

        tem = 0.5 * (zi[:,k+1] - zi[:,k])
        tem1 = np.minimum(tem,rlmn)

        ptem2 = np.minimum(zlup,zldn)
        rlam[:,k] = np.minimum(np.maximum(elmfac*ptem2,tem1),rlmx)

        ptem2 = np.sqrt(zlup*zldn)
        ele[:,k] = np.minimum(np.maximum(elefac*ptem2,tem1),elmx)

    # np.testing.assert_array_equal(rlam,compare_dict["rlam"])
    # assert_test(ele,compare_dict["ele"])

    for k in range(km1):
        # for i in range(im):
        #     tem = vk * zl[i,k]
        #     if zol[i] < 0.0:
        #         ptem = 1.0 - 100.0 * zol[i]
        #         ptem1 = ptem**0.2
        #         zk = tem * ptem1
        #     elif zol[i] >= 1.0:
        #         zk = tem / 3.7
        #     else:
        #         ptem = 1.0 + 2.7 * zol[i]
        #         zk = tem / ptem
        #     elm[i,k] = zk * rlam[i,k] / (rlam[i,k] + zk)

        #     dz = zi[i,k+1] - zi[i,k]
        #     tem = max(gdx[i],dz)
        #     elm[i,k] = min(elm[i,k], tem)
        #     ele[i,k] = min(ele[i,k], tem)

        tem = vk * zl[:,k]
        evalu0 = zol < 0.0
        evalu1 = zol >= 1.0
        evalu2 = np.logical_and(~evalu0,~evalu1)

        evalu0 = evalu0*(tem*(1.0-100.0*zol)**0.2)
        evalu0[np.isnan(evalu0)] = 0.0
        evalu1 = evalu1*(tem/3.7)
        evalu2 = evalu2*(tem/(1.0+2.7*zol))
        zk = evalu0 + evalu1 + evalu2
        elm[:,k] = zk * rlam[:,k] / (rlam[:,k] + zk)
        dz = zi[:,k+1] - zi[:,k]
        tem = np.maximum(gdx,dz)
        elm[:,k] = np.minimum(elm[:,k],tem)
        ele[:,k] = np.minimum(ele[:,k],tem)
    
    # assert_test(elm,compare_dict["elm"])
    # assert_test(ele,compare_dict["ele"])

    elm[:,km-1] = elm[:,km1-1]
    ele[:,km-1] = ele[:,km1-1]

    # assert_test(elm,compare_dict["elm"])
    # assert_test(ele,compare_dict["ele"])  

    for k in range(km1):
        # for i in range(im):
        #     tem = 0.5 * (elm[i,k] + elm[i,k+1])
        #     tem = tem * math.sqrt(tkeh[i,k])
        #     if k < kpbl[i]:
        #         if pblflg[i]:
        #             dku[i,k] = ckz[i,k] * tem
        #             dkt[i,k] = dku[i,k] / prn[i,k]
        #         else:
        #             dkt[i,k] = chz[i,k] * tem
        #             dku[i,k] = dkt[i,k] * prn[i,k]
        #     else:
        #         ri = max(bf[i,k]/shr2[i,k],rimin)
        #         if ri < 0.0:
        #             dku[i,k] = ck1 * tem
        #             dkt[i,k] = rchck * dku[i,k]
        #         else:
        #             dkt[i,k] = ch1 * tem
        #             prnum = 1.0 + 2.1*ri
        #             prnum = min(prnum,prmax)
        #             dku[i,k] = dkt[i,k] * prnum
            
        #     if scuflg[i]:
        #         if (k >= mrad[i]) and (k < krad[i]):
        #             tem1 = ckz[i,k] * tem
        #             ptem1 = tem1 / prscu
        #             dku[i,k] = max(dku[i,k], tem1)
        #             dkt[i,k] = max(dkt[i,k], ptem1)
            
        #     dkq[i,k] = prtke * dkt[i,k]
        #     dkt[i,k] = min(dkt[i,k],dkmax)
        #     dkt[i,k] = max(dkt[i,k],xkzo[i,k])
        #     dkq[i,k] = min(dkq[i,k],dkmax)
        #     dkq[i,k] = max(dkq[i,k],xkzo[i,k])
        #     dku[i,k] = min(dku[i,k],dkmax)
        #     dku[i,k] = max(dku[i,k],xkzmo[i,k])

        tem = 0.5 * (elm[:,k] + elm[:,k+1])
        tem = tem * np.sqrt(tkeh[:,k])

        evalu0 = (k < kpbl)
        dku_term1 = np.logical_and(evalu0, pblflg) * (ckz[:,k] * tem)
        dkt_term1 = dku_term1 / prn[:,k]
        dkt_term2 = np.logical_and(evalu0,~pblflg) * (ckz[:,k] * tem)
        dku_term2 = dkt_term2 * prn[:,k]

        ri = np.maximum(bf[:,k]/shr2[:,k], rimin)

        dku_term3 = np.logical_and(~evalu0, ri < 0.0) * (ck1 * tem)
        dkt_term3 = rchck * dku_term3
        dkt_term4 = np.logical_and(~evalu0, ~(ri < 0.0)) * (ch1 * tem)
        dku_term4 =  dkt_term4 * np.minimum(1.0+2.1*ri,prmax)

        dku[:,k] = dku_term1 + dku_term2 + dku_term3 + dku_term4
        dkt[:,k] = dkt_term1 + dkt_term2 + dkt_term3 + dkt_term4

        evalu0 = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))

        tem1 = ckz[:,k] * tem
        ptem1 = tem1 / prscu
        dku[:,k] = evalu0*(np.maximum(dku[:,k], tem1)) + ~evalu0*dku[:,k]
        dkt[:,k] = evalu0*(np.maximum(dkt[:,k], ptem1)) + ~evalu0*dkt[:,k]

        dkq[:,k] = prtke * dkt[:,k]
        dkt[:,k] = np.minimum(dkt[:,k],dkmax)
        dkt[:,k] = np.maximum(dkt[:,k],xkzo[:,k])
        dkq[:,k] = np.minimum(dkq[:,k],dkmax)
        dkq[:,k] = np.maximum(dkq[:,k],xkzo[:,k])
        dku[:,k] = np.minimum(dku[:,k],dkmax)
        dku[:,k] = np.maximum(dku[:,k],xkzmo[:,k])
    
    # assert_test(dkt,compare_dict["dkt"])
    # assert_test(dkq,compare_dict["dkq"])
    # assert_test(dku,compare_dict["dku"])

    # for i in range(im):
    #     if scuflg[i]:
    #         k = krad[i]
    #         tem = bf[i,k] / gotvx[i,k]
    #         tem1 = max(tem,tdzmin)
    #         ptem = radj[i] / tem1
    #         dkt[i,k] = dkt[i,k] + ptem
    #         dku[i,k] = dku[i,k] + ptem
    #         dkq[i,k] = dkq[i,k] + ptem

    tem = bf[range(im),krad] / gotvx[range(im),krad]
    tem1 = np.maximum(tem,tdzmin)
    ptem = radj / tem1

    dkt[range(im),krad] = scuflg*(dkt[range(im),krad] + ptem) + ~scuflg*dkt[range(im),krad]
    dku[range(im),krad] = scuflg*(dku[range(im),krad] + ptem) + ~scuflg*dku[range(im),krad]
    dkq[range(im),krad] = scuflg*(dkq[range(im),krad] + ptem) + ~scuflg*dkq[range(im),krad]
            

    # assert_test(dkt,compare_dict["dkt"])
    # assert_test(dkq,compare_dict["dkq"])
    # assert_test(dku,compare_dict["dku"])

    # for k in range(km1):
    #     for i in range(im):
    #         if k == 0:
    #             tem = -dkt[i,0] * bf[i,0]
    #             ptem1 = 0.0
                
    #             if scuflg[i] and (mrad[i] == 0):
    #                 ptem2 = xmfd[i,0] * buod[i,0]
    #             else:
    #                 ptem2 = 0.0
                
    #             tem = tem + ptem1 + ptem2
    #             buop = 0.5 * (gotvx[i,0] * sflux[i] + tem)
    #             tem1 = dku[i,0] * shr2[i,0]
    #             tem = (u1[i,1] - u1[i,0]) * rdzt[i,0]
    #             ptem1 = 0.0

    #             if scuflg[i] and (mrad[i] == 0):
    #                 ptem = ucdo[i,0] + ucdo[i,1] - u1[i,0] - u1[i,1]
    #                 ptem = 0.5 * tem * xmfd[i,0] * ptem
    #             else:
    #                 ptem = 0.0
                
    #             ptem1 = ptem1 + ptem

    #             tem = (v1[i,1] - v1[i,0]) * rdzt[i,0]
    #             ptem2 = 0.0

    #             if scuflg[i] and (mrad[i] == 0):
    #                 ptem = vcdo[i,0] + vcdo[i,1] - v1[i,0] - v1 [i,1]
    #                 ptem = 0.5 * tem * xmfd[i,0] * ptem
    #             else:
    #                 ptem = 0.0

    #             ptem2 = ptem2 + ptem

    #             tem2 = stress[i] * ustar[i] * phim[i] / (vk*zl[i,0])
    #             shrp = 0.5 * (tem1 + ptem1 + ptem2 + tem2)
            
    #         else:
    #             tem1 = -dkt[i,k-1] * bf[i,k-1]
    #             tem2 = -dkt[i,k] * bf[i,k]
    #             tem = 0.5 * (tem1 + tem2)

    #             if pcnvflg[i] and (k <= kpbl[i]):
    #                 ptem = 0.5 * (xmf[i,k-1] + xmf[i,k])
    #                 ptem1 = ptem * buou[i,k]
    #             else:
    #                 ptem1 = 0.0

    #             if scuflg[i]:
    #                 if (k >= mrad[i]) and (k < krad[i]):
    #                     ptem0 = 0.5 * (xmfd[i,k-1] + xmfd[i,k])
    #                     ptem2 = ptem0 * buod[i,k]
    #                 else:
    #                     ptem2 = 0.0
    #             else:
    #                 ptem2 = 0.0

    #             buop = tem + ptem1 + ptem2

    #             tem1 = dku[i,k-1] * shr2[i,k-1]
    #             tem2 = dku[i,k] * shr2[i,k]
    #             tem = 0.5 * (tem1 + tem2)
    #             tem1 = (u1[i,k+1] - u1[i,k]) * rdzt[i,k]
    #             tem2 = (u1[i,k] - u1[i,k-1]) * rdzt[i,k-1]

    #             if pcnvflg[i] and (k <= kpbl[i]):
    #                 ptem = xmf[i,k] * tem1 + xmf[i,k-1] * tem2
    #                 ptem1 = 0.5 * ptem * (u1[i,k] - ucko[i,k])
    #             else:
    #                 ptem1 = 0.0

    #             if scuflg[i]:
    #                 if (k >= mrad[i]) and (k < krad[i]):
    #                     ptem0 = xmfd[i,k] * tem1 + xmfd[i,k-1] * tem2
    #                     ptem2 = 0.5 * ptem0 * (ucdo[i,k] - u1[i,k])
    #                 else:
    #                     ptem2 = 0.0
    #             else:
    #                 ptem2 = 0.0
                
    #             shrp = tem + ptem1 + ptem2
    #             tem1 = (v1[i,k+1] - v1[i,k])   * rdzt[i,k]
    #             tem2 = (v1[i,k]   - v1[i,k-1]) * rdzt[i,k-1]

    #             if pcnvflg[i] and (k <= kpbl[i]):
    #                 ptem = xmf[i,k] * tem1 + xmf[i,k-1] * tem2
    #                 ptem1 = 0.5 * ptem * (v1[i,k] - vcko[i,k])
    #             else:
    #                 ptem1 = 0.0

    #             if scuflg[i]:
    #                 if (k >= mrad[i]) and (k < krad[i]):
    #                     ptem0 = xmfd[i,k] * tem1 + xmfd[i,k-1] * tem2
    #                     ptem2 = 0.5 * ptem0 * (vcdo[i,k] - v1[i,k])
    #                 else:
    #                     ptem2 = 0.0
    #             else:
    #                 ptem2 = 0.0

    #             shrp = shrp + ptem1 + ptem2

    #         prod[i,k] = buop + shrp

    xmfd[np.isnan(xmfd)] = 0.0
    xmf[np.isnan(xmf)] = 0.0

    tem = -dkt[:,0] * bf[:,0]
    ptem1 = 0.0

    ptem2 = np.logical_and(scuflg,mrad==0)*(xmfd[:,0] * buod[:,0])
    tem = tem + ptem1 + ptem2

    buop = 0.5 * (gotvx[:,0] * sflux + tem)
    tem1 = dku[:,0] * shr2[:,0]
    tem = (u1[:,1] - u1[:,0]) * rdzt[:,0]
    ptem1 = 0.0

    ptem = np.logical_and(scuflg,mrad==0)*(0.5 * tem * xmfd[:,0] * (ucdo[:,0] + ucdo[:,1] - u1[:,0] - u1[:,1]))
    ptem1 = ptem1 + ptem

    tem = (v1[:,1] - v1[:,0]) * rdzt[:,0]
    ptem2 = 0.0

    ptem = np.logical_and(scuflg,mrad==0)*(0.5 * tem * xmfd[:,0] * (vcdo[:,0] + vcdo[:,1] - v1[:,0] - v1 [:,1]))
    ptem2 = ptem2 + ptem

    tem2 = stress * ustar * phim / (vk*zl[:,0])
    shrp = 0.5 * (tem1 + ptem1 + ptem2 + tem2)

    prod[:,0] = buop + shrp

    for k in range(1,km1):
        tem1 = -dkt[:,k-1] * bf[:,k-1]
        tem2 = -dkt[:,k] * bf[:,k]
        tem = 0.5 * (tem1 + tem2)

        ptem1 = np.logical_and(pcnvflg,k <= kpbl)*((0.5 * (xmf[:,k-1] + xmf[:,k]))* buou[:,k])

        ptem2 = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))*((0.5 * (xmfd[:,k-1] + xmfd[:,k]))* buod[:,k])

        buop = tem + ptem1 + ptem2

        tem1 = dku[:,k-1] * shr2[:,k-1]
        tem2 = dku[:,k] * shr2[:,k]
        tem = 0.5 * (tem1 + tem2)
        tem1 = (u1[:,k+1] - u1[:,k]) * rdzt[:,k]
        tem2 = (u1[:,k] - u1[:,k-1]) * rdzt[:,k-1]

        ptem1 = np.logical_and(pcnvflg,k <= kpbl)*(0.5 * (xmf[:,k] * tem1 + xmf[:,k-1] * tem2) * (u1[:,k] - ucko[:,k]))

        ptem2 = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))*(0.5 * (xmfd[:,k] * tem1 + xmfd[:,k-1] * tem2) * (ucdo[:,k] - u1[:,k]))
        
        shrp = tem + ptem1 + ptem2
        tem1 = (v1[:,k+1] - v1[:,k])   * rdzt[:,k]
        tem2 = (v1[:,k]   - v1[:,k-1]) * rdzt[:,k-1]

        ptem1 = np.logical_and(pcnvflg,k <= kpbl)*(0.5 * (xmf[:,k] * tem1 + xmf[:,k-1] * tem2) * (v1[:,k] - vcko[:,k]))

        ptem2 = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))*(0.5 * (xmfd[:,k] * tem1 + xmfd[:,k-1] * tem2) * (vcdo[:,k] - v1[:,k]))

        shrp = shrp + ptem1 + ptem2

        prod[:,k] = buop + shrp

    # assert_test(prod,compare_dict["prod"])

    rle[:,:] = ce0 / ele[:,:-1]

    # assert_test(rle,compare_dict["rle"])

    kk = max(round(dt2/cdtn),1)
    dtn = dt2 / kk

    for n in range(kk):
        # tem = np.sqrt(tke[:,:km1])
        # diss[:,:] = rle * tke[:,:km1] * tem
        # tem1 = rle * tke[:,:km1] * tem
        # diss[:,:] = np.maximum(np.minimum(diss,tem1),0.0)
        # tke[:,:km1] = np.maximum(tke[:,:km1] + dtn * (prod - diss), tkmin)
        for k in range(km1):
            tem = np.sqrt(tke[:,k])
            diss[:,k] = rle[:,k] * tke[:,k] * tem
            tem1 = prod[:,k] + tke[:,k] / dtn
            diss[:,k] = np.maximum(np.minimum(diss[:,k],tem1),0.0)
            tke[:,k] = np.maximum(tke[:,k] + dtn * (prod[:,k] - diss[:,k]), tkmin)

            # for i in range(im):
            #     tem = math.sqrt(tke[i,k])
            #     diss[i,k] = rle[i,k] * tke[i,k] * tem
            #     tem1 = prod[i,k] + tke[i,k] / dtn
            #     diss[i,k] = max(min(diss[i,k],tem1),0.0)
            #     tke[i,k] = tke[i,k] + dtn * (prod[i,k]-diss[i,k])
            #     tke[i,k] = max(tke[i,k],tkmin)


    # assert_test(diss,compare_dict["diss"])
    # assert_test(tke,compare_dict["tke"])


    for k in range(km):
        qcko[:,k,ntke-1] = pcnvflg*tke[:,k] + ~pcnvflg*qcko[:,k,ntke-1]
        qcdo[:,k,ntke-1] = scuflg*tke[:,k] + ~scuflg*qcdo[:,k,ntke-1]
        # for i in range(im):
        #     if pcnvflg[i]:
        #         qcko[i,k,ntke-1] = tke[i,k]
        #     if scuflg[i]:
        #         qcdo[i,k,ntke-1] = tke[i,k]

    # assert_test(qcko,compare_dict["qcko"])
    # assert_test(qcdo,compare_dict["qcdo"])

    for k in range(1,kmpbl):
        evalu = np.logical_and(pcnvflg,k <= kpbl)
        dz = zl[:,k] - zl[:,k-1]
        tem = 0.5 * xlamue[:,k-1] * dz
        factor = 1.0 + tem
        qcko[:,k,ntke-1] = evalu*(((1.0 - tem) * qcko[:,k-1,ntke-1]+tem*(tke[:,k] + tke[:,k-1]))/factor) + ~evalu*qcko[:,k,ntke-1]
        # for i in range(im):
        #     if pcnvflg[i] and (k <= kpbl[i]):
        #         dz = zl[i,k] - zl[i,k-1]
        #         tem = 0.5 * xlamue[i,k-1] * dz
        #         factor = 1.0 + tem
        #         qcko[i,k,ntke-1] = ((1.0 - tem) * qcko[i,k-1,ntke-1]+tem*(tke[i,k] + tke[i,k-1]))/factor

    # assert_test(qcko,compare_dict["qcko"])

    for k in range(kmscu-1,-1,-1):
        for i in range(im):
            if scuflg[i] and k < krad[i]:
                if k >= mrad[i]:
                    dz = zl[i,k+1] - zl[i,k]
                    tem = 0.5 * xlamde[i,k] * dz
                    factor = 1.0 + tem
                    qcdo[i,k,ntke-1] = ((1.0-tem) * qcdo[i,k+1,ntke-1] + tem*(tke[i,k] + tke[i,k+1]))/factor

    # assert_test(qcdo,compare_dict["qcdo"])

    ad[:,0] = 1.0
    f1[:,0] = tke[:,0]

    # assert_test(ad,compare_dict["ad"])
    # assert_test(f1,compare_dict["f1"])

    for k in range(km1):
        dtodsd  = dt2/del_[:,k]
        dtodsu  = dt2/del_[:,k+1]
        dsig    = prsl[:,k]-prsl[:,k+1]
        rdz     = rdzt[:,k]
        tem1    = dsig * dkq[:,k] * rdz
        dsdz2   = tem1 * rdz
        au[:,k] = -dtodsd*dsdz2
        al[:,k] = -dtodsu*dsdz2
        ad[:,k] = ad[:,k]-au[:,k]
        ad[:,k+1]= 1.-al[:,k]
        tem2    = dsig * rdz

        evalu = np.logical_and(pcnvflg, k < kpbl)
        ptem      = 0.5 * tem2 * xmf[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        tem       = tke[:,k] + tke[:,k+1]
        ptem      = qcko[:,k,ntke-1] + qcko[:,k+1,ntke-1]
        f1[:,k]   = evalu*(f1[:,k]-(ptem-tem)*ptem1) + ~evalu*f1[:,k]
        f1[:,k+1] = evalu*(tke[:,k+1]+(ptem-tem)*ptem2) + ~evalu*tke[:,k+1]

        evalu = np.logical_and(scuflg,np.logical_and(k >= mrad,k < krad))
        ptem      = 0.5 * tem2 * xmfd[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        tem       = tke[:,k] + tke[:,k+1]
        ptem      = qcdo[:,k,ntke-1] + qcdo[:,k+1,ntke-1]
        f1[:,k]   = evalu*(f1[:,k] + (ptem - tem) * ptem1) + ~evalu*f1[:,k]
        f1[:,k+1] = evalu*(f1[:,k+1] - (ptem - tem) * ptem2) + ~evalu*f1[:,k+1]

        # for i in range(im):
        #     dtodsd  = dt2/del_[i,k]
        #     dtodsu  = dt2/del_[i,k+1]
        #     dsig    = prsl[i,k]-prsl[i,k+1]
        #     rdz     = rdzt[i,k]
        #     tem1    = dsig * dkq[i,k] * rdz
        #     dsdz2   = tem1 * rdz
        #     au[i,k] = -dtodsd*dsdz2
        #     al[i,k] = -dtodsu*dsdz2
        #     ad[i,k] = ad[i,k]-au[i,k]
        #     ad[i,k+1]= 1.-al[i,k]
        #     tem2    = dsig * rdz
    
        #     if pcnvflg[i] and k < kpbl[i]:
        #         ptem      = 0.5 * tem2 * xmf[i,k]
        #         ptem1     = dtodsd * ptem
        #         ptem2     = dtodsu * ptem
        #         tem       = tke[i,k] + tke[i,k+1]
        #         ptem      = qcko[i,k,ntke-1] + qcko[i,k+1,ntke-1]
        #         f1[i,k]   = f1[i,k]-(ptem-tem)*ptem1
        #         f1[i,k+1] = tke[i,k+1]+(ptem-tem)*ptem2
        #     else:
        #         f1[i,k+1] = tke[i,k+1]

        #     if scuflg[i]:
        #         if k >= mrad[i] and k < krad[i]:
        #             ptem      = 0.5 * tem2 * xmfd[i,k]
        #             ptem1     = dtodsd * ptem
        #             ptem2     = dtodsu * ptem
        #             tem       = tke[i,k] + tke[i,k+1]
        #             ptem      = qcdo[i,k,ntke-1] + qcdo[i,k+1,ntke-1]
        #             f1[i,k]   = f1[i,k] + (ptem - tem) * ptem1
        #             f1[i,k+1] = f1[i,k+1] - (ptem - tem) * ptem2


    # assert_test(f1,compare_dict["f1"])
    # assert_test(au,compare_dict["au"])
    # assert_test(al,compare_dict["al"])
    # assert_test(ad,compare_dict["ad"])

    au, f1 = tridit(im,km,1,al,ad,au,f1,au,f1, compare_dict)

    # assert_test(f1,compare_dict["f1"])
    # assert_test(au,compare_dict["au"])


    qtend = (f1 - q1[:,:,ntke-1]) * rdt
    rtg[:,:,ntke-1] = rtg[:,:,ntke-1] + qtend

    # assert_test(rtg,compare_dict["rtg"])

    for i in range(im):
        ad[i,0] = 1.0
        f1[i,0] = t1[i,0] + dtdz1[i] * heat[i]
        f2[i,0] = q1[i,0,0] + dtdz1[i] * evap[i]

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for i in range(im):
                f2[i,is_] = q1[i,0,kk]

    # assert_test(f1,compare_dict["f1"])
    # assert_test(f2,compare_dict["f2"])
    # assert_test(ad,compare_dict["ad"])

    for k in range(km1):
        dtodsd  = dt2/del_[:,k]
        dtodsu  = dt2/del_[:,k+1]
        dsig    = prsl[:,k]-prsl[:,k+1]
        rdz     = rdzt[:,k]
        tem1    = dsig * dkt[:,k] * rdz
        dsdzt   = tem1 * gocp
        dsdz2   = tem1 * rdz
        au[:,k] = -dtodsd*dsdz2
        al[:,k] = -dtodsu*dsdz2
        ad[:,k] = ad[:,k]-au[:,k]
        ad[:,k+1]= 1.-al[:,k]
        tem2    = dsig * rdz
        # for i in range(im):
        #     dtodsd  = dt2/del_[i,k]
        #     dtodsu  = dt2/del_[i,k+1]
        #     dsig    = prsl[i,k]-prsl[i,k+1]
        #     rdz     = rdzt[i,k]
        #     tem1    = dsig * dkt[i,k] * rdz
        #     dsdzt   = tem1 * gocp
        #     dsdz2   = tem1 * rdz
        #     au[i,k] = -dtodsd*dsdz2
        #     al[i,k] = -dtodsu*dsdz2
        #     ad[i,k] = ad[i,k]-au[i,k]
        #     ad[i,k+1]= 1.-al[i,k]
        #     tem2    = dsig * rdz

        evalu = np.logical_and(pcnvflg, k<kpbl)
        ptem      = 0.5 * tem2 * xmf[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        tem       = t1[:,k] + t1[:,k+1]
        ptem      = tcko[:,k] + tcko[:,k+1]
        f1[:,k]   = evalu*(f1[:,k]+dtodsd*dsdzt-(ptem-tem)*ptem1) + ~evalu*(f1[:,k]+dtodsd*dsdzt)
        f1[:,k+1] = evalu*(t1[:,k+1]-dtodsu*dsdzt+(ptem-tem)*ptem2) + ~evalu*(t1[:,k+1]-dtodsu*dsdzt)
        tem       = q1[:,k,0] + q1[:,k+1,0]
        ptem      = qcko[:,k,0] + qcko[:,k+1,0]
        f2[:,k]   = evalu*(f2[:,k] - (ptem - tem) * ptem1) + ~evalu*f2[:,k]
        f2[:,k+1] = evalu*(q1[:,k+1,0] + (ptem - tem) * ptem2) + ~evalu*q1[:,k+1,0]

            # if pcnvflg[i] and k < kpbl[i]:
            #     ptem      = 0.5 * tem2 * xmf[i,k]
            #     ptem1     = dtodsd * ptem
            #     ptem2     = dtodsu * ptem
            #     tem       = t1[i,k] + t1[i,k+1]
            #     ptem      = tcko[i,k] + tcko[i,k+1]
            #     f1[i,k]   = f1[i,k]+dtodsd*dsdzt-(ptem-tem)*ptem1
            #     f1[i,k+1] = t1[i,k+1]-dtodsu*dsdzt+(ptem-tem)*ptem2
            #     tem       = q1[i,k,0] + q1[i,k+1,0]
            #     ptem      = qcko[i,k,0] + qcko[i,k+1,0]
            #     f2[i,k]   = f2[i,k] - (ptem - tem) * ptem1
            #     f2[i,k+1] = q1[i,k+1,0] + (ptem - tem) * ptem2
            # else:
            #     f1[i,k]   = f1[i,k]+dtodsd*dsdzt
            #     f1[i,k+1] = t1[i,k+1]-dtodsu*dsdzt
            #     f2[i,k+1] = q1[i,k+1,0]


        evalu = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))
        ptem      = 0.5 * tem2 * xmfd[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        ptem      = tcdo[:,k] + tcdo[:,k+1]
        tem       = t1[:,k] + t1[:,k+1]
        f1[:,k]   = evalu*(f1[:,k] + (ptem - tem) * ptem1) + ~evalu*f1[:,k]
        f1[:,k+1] = evalu*(f1[:,k+1] - (ptem - tem) * ptem2) + ~evalu*f1[:,k+1]
        tem       = q1[:,k,0] + q1[:,k+1,0]
        ptem      = qcdo[:,k,0] + qcdo[:,k+1,0]
        f2[:,k]   = evalu*(f2[:,k] + (ptem - tem) * ptem1) + ~evalu*f2[:,k]
        f2[:,k+1] = evalu*(f2[:,k+1] - (ptem - tem) * ptem2) + ~evalu*f2[:,k+1]
            # if scuflg[i]:
            #     if k >= mrad[i] and k < krad[i]:
            #         ptem      = 0.5 * tem2 * xmfd[i,k]
            #         ptem1     = dtodsd * ptem
            #         ptem2     = dtodsu * ptem
            #         ptem      = tcdo[i,k] + tcdo[i,k+1]
            #         tem       = t1[i,k] + t1[i,k+1]
            #         f1[i,k]   = f1[i,k] + (ptem - tem) * ptem1
            #         f1[i,k+1] = f1[i,k+1] - (ptem - tem) * ptem2
            #         tem       = q1[i,k,0] + q1[i,k+1,0]
            #         ptem      = qcdo[i,k,0] + qcdo[i,k+1,0]
            #         f2[i,k]   = f2[i,k] + (ptem - tem) * ptem1
            #         f2[i,k+1] = f2[i,k+1] - (ptem - tem) * ptem2
    
    # assert_test(au,compare_dict["au"])
    # assert_test(al,compare_dict["al"])
    # assert_test(f1,compare_dict["f1"])
    # assert_test(f2,compare_dict["f2"])

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for k in range(km1):
                evalu = np.logical_and(pcnvflg, k < kpbl)
                dtodsd = dt2/del_[:,k]
                dtodsu = dt2/del_[:,k+1]
                dsig  = prsl[:,k]-prsl[:,k+1]
                tem   = dsig * rdzt[:,k]
                ptem  = 0.5 * tem * xmf[:,k]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1  = qcko[:,k,kk] + qcko[:,k+1,kk]
                tem2  = q1[:,k,kk] + q1[:,k+1,kk]
                f2[:,k+is_] = evalu*(f2[:,k+is_] - (tem1 - tem2) * ptem1) + ~evalu*f2[:,k+is_]
                f2[:,k+1+is_]= evalu*(q1[:,k+1,kk] + (tem1 - tem2) * ptem2) + ~evalu*q1[:,k+1,kk]

                evalu = np.logical_and(scuflg,np.logical_and(k >= mrad, k < krad))
                dtodsd = dt2/del_[:,k]
                dtodsu = dt2/del_[:,k+1]
                dsig  = prsl[:,k]-prsl[:,k+1]
                tem   = dsig * rdzt[:,k]
                ptem  = 0.5 * tem * xmfd[:,k]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1  = qcdo[:,k,kk] + qcdo[:,k+1,kk]
                tem2  = q1[:,k,kk] + q1[:,k+1,kk]
                f2[:,k+is_]  = evalu*(f2[:,k+is_] + (tem1 - tem2) * ptem1) + ~evalu*f2[:,k+is_]
                f2[:,k+1+is_]= evalu*(f2[:,k+1+is_] - (tem1 - tem2) * ptem2) + ~evalu*f2[:,k+1+is_]

    # assert_test(f2,compare_dict["f2"])

    au, f1, f2 = tridin(im,km,ntrac1,al,ad,au,f1,f2,au,f1,f2, compare_dict)

    # assert_test(au,compare_dict["au"])
    # assert_test(f1,compare_dict["f1"])
    # assert_test(f2,compare_dict["f2"])

    for k in range(km):
        ttend = (f1[:,k] - t1[:,k]) * rdt
        qtend = (f2[:,k] - q1[:,k,0]) * rdt
        tdt[:,k] = tdt[:,k] + ttend
        rtg[:,k,0] = rtg[:,k,0] + qtend
        dtsfc[:] = dtsfc + cont * del_[:,k]*ttend
        dqsfc[:] = dqsfc + conq * del_[:,k]*qtend

    # assert_test(tdt,compare_dict["tdt"])
    # assert_test(rtg,compare_dict["rtg"])
    # assert_test(dtsfc,compare_dict["dtsfc"])
    # assert_test(dqsfc,compare_dict["dqsfc"])

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for k in range(km):
                rtg[:,k,kk] = rtg[:,k,kk] + ((f2[:,k+is_] - q1[:,k,kk])*rdt)

    # assert_test(rtg,compare_dict["rtg"])

    if dspheat:
        tdt[:,:km1] = tdt[:,:km1] + dspfac * (diss[:,:km1] / cp)

    # assert_test(diss,compare_dict["diss"])
    # assert_test(tdt,compare_dict["tdt"])

    ad[:,0] = 1.0 + dtdz1 * stress / spd1
    f1[:,0] = u1[:,0]
    f2[:,0] = v1[:,0]

    # assert_test(ad,compare_dict["ad"])
    # assert_test(f1,compare_dict["f1"])
    # assert_test(f2,compare_dict["f2"])

    for k in range(km1):
        dtodsd  = dt2/del_[:,k]
        dtodsu  = dt2/del_[:,k+1]
        dsig    = prsl[:,k]-prsl[:,k+1]
        rdz     = rdzt[:,k]
        tem1    = dsig * dku[:,k] * rdz
        dsdz2   = tem1*rdz
        au[:,k] = -dtodsd*dsdz2
        al[:,k] = -dtodsu*dsdz2
        ad[:,k] = ad[:,k]-au[:,k]
        ad[:,k+1]= 1.-al[:,k]
        tem2    = dsig * rdz

        evalu = np.logical_and(pcnvflg, k < kpbl)
        ptem      = 0.5 * tem2 * xmf[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        tem       = u1[:,k] + u1[:,k+1]
        ptem      = ucko[:,k] + ucko[:,k+1]
        f1[:,k]   = evalu*(f1[:,k] - (ptem - tem) * ptem1) + ~evalu*f1[:,k]
        f1[:,k+1] = evalu*(u1[:,k+1] + (ptem - tem) * ptem2) + ~evalu*u1[:,k+1]
        tem       = v1[:,k] + v1[:,k+1]
        ptem      = vcko[:,k] + vcko[:,k+1]
        f2[:,k]   = evalu*(f2[:,k] - (ptem - tem) * ptem1) + ~evalu*f2[:,k]
        f2[:,k+1] = evalu*(v1[:,k+1] + (ptem - tem) * ptem2) + ~evalu*v1[:,k+1]

        evalu = np.logical_and(scuflg,np.logical_and(k >= mrad,k < krad))
        ptem      = 0.5 * tem2 * xmfd[:,k]
        ptem1     = dtodsd * ptem
        ptem2     = dtodsu * ptem
        tem       = u1[:,k] + u1[:,k+1]
        ptem      = ucdo[:,k] + ucdo[:,k+1]
        f1[:,k]   = evalu*(f1[:,k] + (ptem - tem) *ptem1) + ~evalu*f1[:,k]
        f1[:,k+1] = evalu*(f1[:,k+1] - (ptem - tem) *ptem2) + ~evalu*f1[:,k+1]
        tem       = v1[:,k] + v1[:,k+1]
        ptem      = vcdo[:,k] + vcdo[:,k+1]
        f2[:,k]   = evalu*(f2[:,k] + (ptem - tem) * ptem1) + ~evalu*f2[:,k]
        f2[:,k+1] = evalu*(f2[:,k+1] - (ptem - tem) * ptem2) + ~evalu*f2[:,k+1]

        # for i in range(im):

        #     dtodsd  = dt2/del_[i,k]
        #     dtodsu  = dt2/del_[i,k+1]
        #     dsig    = prsl[i,k]-prsl[i,k+1]
        #     rdz     = rdzt[i,k]
        #     tem1    = dsig * dku[i,k] * rdz
        #     dsdz2   = tem1*rdz
        #     au[i,k] = -dtodsd*dsdz2
        #     al[i,k] = -dtodsu*dsdz2
        #     ad[i,k] = ad[i,k]-au[i,k]
        #     ad[i,k+1]= 1.-al[i,k]
        #     tem2    = dsig * rdz

        #     if pcnvflg[i] and (k < kpbl[i]):
        #         ptem      = 0.5 * tem2 * xmf[i,k]
        #         ptem1     = dtodsd * ptem
        #         ptem2     = dtodsu * ptem
        #         tem       = u1[i,k] + u1[i,k+1]
        #         ptem      = ucko[i,k] + ucko[i,k+1]
        #         f1[i,k]   = f1[i,k] - (ptem - tem) * ptem1
        #         f1[i,k+1] = u1[i,k+1] + (ptem - tem) * ptem2
        #         tem       = v1[i,k] + v1[i,k+1]
        #         ptem      = vcko[i,k] + vcko[i,k+1]
        #         f2[i,k]   = f2[i,k] - (ptem - tem) * ptem1
        #         f2[i,k+1] = v1[i,k+1] + (ptem - tem) * ptem2
        #     else:
        #         f1[i,k+1] = u1[i,k+1]
        #         f2[i,k+1] = v1[i,k+1]

        #     if scuflg[i]:
        #         if (k >= mrad[i]) and (k < krad[i]):
        #             ptem      = 0.5 * tem2 * xmfd[i,k]
        #             ptem1     = dtodsd * ptem
        #             ptem2     = dtodsu * ptem
        #             tem       = u1[i,k] + u1[i,k+1]
        #             ptem      = ucdo[i,k] + ucdo[i,k+1]
        #             f1[i,k]   = f1[i,k] + (ptem - tem) *ptem1
        #             f1[i,k+1] = f1[i,k+1] - (ptem - tem) *ptem2
        #             tem       = v1[i,k] + v1[i,k+1]
        #             ptem      = vcdo[i,k] + vcdo[i,k+1]
        #             f2[i,k]   = f2[i,k] + (ptem - tem) * ptem1
        #             f2[i,k+1] = f2[i,k+1] - (ptem - tem) * ptem2

    # assert_test(f1,compare_dict["f1"])
    # assert_test(au,compare_dict["au"])
    # assert_test(al,compare_dict["al"])
    # assert_test(ad,compare_dict["ad"])
    # assert_test(f2,compare_dict["f2"])


    au, f1, f2 = tridi2(im,km,al,ad,au,f1,f2,au,f1,f2, compare_dict)

    # assert_test(au,compare_dict["au"])
    # assert_test(f1,compare_dict["f1"])
    # assert_test(f2,compare_dict["f2"])   

    for k in range(km):
        utend = (f1[:,k]-u1[:,k])*rdt
        vtend = (f2[:,k]-v1[:,k])*rdt
        du[:,k]  = du[:,k]+utend
        dv[:,k]  = dv[:,k]+vtend
        dusfc[:] = dusfc+conw*del_[:,k]*utend
        dvsfc[:] = dvsfc+conw*del_[:,k]*vtend

    # assert_test(du,compare_dict["du"])
    # assert_test(dv,compare_dict["dv"])
    # assert_test(dusfc,compare_dict["dusfc"])
    # assert_test(dvsfc,compare_dict["dvsfc"])

    hpbl[:] = hpblx
    kpbl[:] = kpblx
    
    # assert_test(du,compare_dict["du"])
    # assert_test(dv,compare_dict["dv"])
    # assert_test(tdt,compare_dict["tdt"])
    # assert_test(rtg,compare_dict["rtg"])
    # assert_test(hpbl,compare_dict["hpbl"])
    # assert_test(kpbl,compare_dict["kpbl"]-1)
    # assert_test(dusfc,compare_dict["dusfc"])
    # assert_test(dvsfc,compare_dict["dvsfc"])
    # assert_test(dqsfc,compare_dict["dqsfc"])
    # assert_test(dtsfc,compare_dict["dtsfc"])

    return dv, du, tdt, rtg, kpbl+1, dusfc, dvsfc, dtsfc, dqsfc, hpbl

def tridit(l, n, nt, cl, cm, cu, rt, au, at, compare_dict):

    fk = np.zeros(l)
    fkk = np.zeros((l,cl.shape[1]))


    # for i in range(l):
    #     fk[i] = 1/cm[i,0]
    #     au[i,0] = fk[i] * cu[i,0]

    fk[:l] = 1/cm[:l,0]
    au[:l,0] = fk[:l] * cu[:l,0]

    # assert_test(au,compare_dict["au0"])

    for k in range(nt):
        is_ = k * n
        at[:l, is_] = fk[:l] * rt[:l,is_]
        # for i in range(l):
        #     at[i,is_] = fk[i] * rt[i,is_]

    # assert_test(at,compare_dict["at0"])

    for k in range(1,n-1):
        fkk[:l,k] = 1 / (cm[:l,k] - cl[:l,k-1] * au[:l,k-1])
        au[:l,k]  = fkk[:l,k] * cu[:l,k]
        # for i in range(l):
        #     fkk[i,k] = 1 / (cm[i,k] - cl[i,k-1] * au[i,k-1])
        #     au[i,k]  = fkk[i,k] * cu[i,k]

    # assert_test(au,compare_dict["au1"])

    for kk in range(nt):
        for k in range(1,n-1):
            at[:l,k+is_] = fkk[:l,k] * (rt[:l,k+is_] - cl[:l,k-1] * at[:l,k+is_-1])
            # for i in range(l):
            #     at[i,k+is_] = fkk[i,k] * (rt[i,k+is_] - cl[i,k-1] * at[i,k+is_-1])

    # assert_test(at,compare_dict["at1"])
    
    # for i in range(l):
    #     fk[i] = 1.0/(cm[i,n-1] - cl[i,n-2] * au[i,n-2])

    fk[:l] = 1.0 / (cm[:l,n-1] - cl[:l,n-2] * au[:l, n-2])
    # assert_test(fk,compare_dict["fk"])

    for k in range(nt):
        is_ = k * n
        at[:l, n+is_-1] = fk[:l] * (rt[:l,n+is_-1] - cl[:l,n-2] * at[:l,n+is_-2])
        # for i in range(l):
        #     at[i,n+is_-1] = fk[i] * (rt[i,n+is_-1] - cl[i,n-2] * at[i,n+is_-2])

    # assert_test(at,compare_dict["at2"])

    for kk in range(nt):
        is_ = kk * n
        for k in range(n-2,-1,-1):
            for i in range(l):
                at[i,k+is_] = at[i, k+is_] - au[i,k] * at[i,k+is_+1]

    # assert_test(at,compare_dict["at3"])

    return au, at

def tridin(l,n,nt,cl,cm,cu,r1,r2,au,a1,a2, compare_dict):

    fk = np.zeros(l)
    fkk = np.zeros((l,n-1))

    # for i in range(l):
    #     fk[i] = 1 / cm[i,0]
    #     au[i,0] = fk[i] * cu[i,0]
    #     a1[i,0] = fk[i] * r1[i,0]

    fk[:] = 1.0 / cm[:,0]
    au[:,0] = fk * cu[:,0]
    a1[:,0] = fk * r1[:,0]
    
    # assert_test(fk,compare_dict["fk"])
    # assert_test(au,compare_dict["au"])
    # assert_test(a1,compare_dict["a1"])

    for k in range(nt):
        is_ = k * n
        a2[:,is_] = fk * r2[:,is_]

    # assert_test(a2,compare_dict["a2"])

    # for k in range(1,n-1):
    #     fkk = (1.0/(cm[:,k] - cl[:,k-1] * au[:,k-1]))
    #     au[:,k] = fkk * cu[:,k]
    #     a1[:,k] = fkk * (r1[:,k] - cl[:,k-1] * a1[:,k-1])

    for k in range(1,n-1):
        fkk[:,k] = (1.0/(cm[:,k] - cl[:,k-1] * au[:,k-1]))
        au[:,k] = fkk[:,k] * cu[:,k]
        a1[:,k] = fkk[:,k] * (r1[:,k] - cl[:,k-1] * a1[:,k-1])

    # assert_test(au,compare_dict["au"])
    # assert_test(a1,compare_dict["a1"])

    for kk in range(nt):
        is_ = kk * n
        for k in range(1,n-1):
            a2[:,k+is_] = fkk[:,k] * (r2[:,k+is_] - cl[:,k-1] * a2[:,k+is_-1])
            # for i in range(l):
            #     a2[i,k+is_] = fkk[i,k] * (r2[i,k+is_] - cl[i,k-1] * a2[i,k+is_-1])

    # assert_test(a2,compare_dict["a2"])

    fk[:] = 1 / (cm[:,n-1] - cl[:,n-2] * au[:,n-2])
    a1[:,n-1] = fk * (r1[:,n-1] - cl[:,n-2] * a1[:,n-2])

    # assert_test(fk,compare_dict["fk"])
    # assert_test(a1,compare_dict["a1"])

    for k in range(nt):
        is_ = k * n
        a2[:,n+is_-1] = fk * (r2[:,n+is_-1] - cl[:,n-2]*a2[:,n+is_-2])

    # assert_test(a2,compare_dict["a2"])

    for k in range(n-2,-1,-1):
        a1[:,k] = a1[:,k] - au[:,k] * a1[:,k+1]

    # assert_test(a1,compare_dict["a1"])

    for kk in range(nt):
        is_ = kk * n
        for k in range(n-2,-1,-1):
            a2[:,k+is_] = a2[:,k+is_] - au[:,k]*a2[:,k+is_+1]

    # assert_test(a2,compare_dict["a2"])

    return au, a1, a2

def tridi2(l,n,cl,cm,cu,r1,r2,au,a1,a2,compare_dict):

    fk = 1/cm[:,0]
    au[:,0] = fk*cu[:,0]
    a1[:,0] = fk*r1[:,0]
    a2[:,0] = fk*r2[:,0]

    # assert_test(au,compare_dict["au"])
    # assert_test(a1,compare_dict["a1"])
    # assert_test(a2[:,0],compare_dict["a2"][:,0])
        
    for k in range(1,n-1):
        fk = 1.0/(cm[:,k] - cl[:,k-1]*au[:,k-1])
        au[:,k] = fk * cu[:,k]
        a1[:,k] = fk * (r1[:,k] - cl[:,k-1] * a1[:,k-1])
        a2[:,k] = fk * (r2[:,k] - cl[:,k-1] * a2[:,k-1])

    # assert_test(au,compare_dict["au"])
    # assert_test(a1,compare_dict["a1"])
    # assert_test(a2[:,1:n-1],compare_dict["a2"][:,1:n-1])

    fk = 1.0/(cm[:,n-1] - cl[:,-1] * au[:,n-2])
    a1[:,n-1] = fk * (r1[:,n-1] - cl[:,-1] * a1[:,n-2])
    a2[:,n-1] = fk * (r2[:,n-1] - cl[:,-1] * a2[:,n-2])

    # assert_test(a1,compare_dict["a1"])
    # assert_test(a2[:,n-1],compare_dict["a2"][:,n-1])

    for k in range(n-2,-1,-1):
        a1[:,k] = a1[:,k] - au[:,k] * a1[:,k+1]
        a2[:,k] = a2[:,k] - au[:,k] * a2[:,k+1]

    # assert_test(a1,compare_dict["a1"])
    # assert_test(a2[:,n-1],compare_dict["a2"][:,n-1])
  
    return au, a1, a2