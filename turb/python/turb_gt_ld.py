#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import math
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage  as gt_storage

from time import perf_counter

# Physics Constants used from physcon module

grav  = 9.80665e0
rd    = 2.87050e2
cp    = 1.00460e3
rv    = 4.61500e2
hvap  = 2.50000e6
hfus  = 3.33580e5
wfac   = 7.0
cfac   = 4.5
gamcrt = 3.0
# gamcrq = 0.0
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
# dspmax = 10.0
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
xmin   = 180.0
xmax   = 330.0

con_ttp  = 2.7316e2
con_cvap = 1.8460e3
con_cliq = 4.1855e3
con_hvap = 2.5000e6
con_rv   = 4.6150e2
con_csol = 2.1060e3
con_hfus = 3.3358e5
con_psat = 6.1078e2

OUT_VARS = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]

backend = "gtx86"
F_TYPE = np.float64
I_TYPE = np.int32
B_TYPE = np.bool

def run(in_dict, compare_dict, region_timings):
    """run function"""

    #compare_dict = []

    dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl = satmedmfvdif_gt(in_dict['im'], in_dict['ix'], in_dict['km'], 
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
                 compare_dict, region_timings)

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

def satmedmfvdif_gt(im, ix, km, ntrac, ntcw, ntiw, ntke,
                 dv, du, tdt, rtg, u1, v1, t1, q1, swh, hlw, xmu, garea,
                 psk, rbsoil, zorl, u10m, v10m, fm, fh, tsea, heat, evap, stress, spd1, kpbl,
                 prsi, del_, prsl, prslk, phii, phil, delt,
                 dspheat, dusfc, dvsfc, dtsfc, dqsfc, hpbl,
                 kinver, xkzm_m, xkzm_h, xkzm_s,
                 compare_dict, region_timings):

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
   
    dt2 = delt
    rdt = 1.0/ dt2
    ntrac1 = ntrac - 1
    km1 = km - 1
    kmpbl = int(km / 2)
    kmscu = int(km / 2)

    # 3D GT storage
    qcko       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1,ntrac), default_origin=(0,0,0))
    qcdo       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1,ntrac), default_origin=(0,0,0))
    f2         = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,1,km*(ntrac-1)), default_origin=(0,0,0))
    pcnvflg_v2 = gt_storage.zeros(backend=backend, dtype=B_TYPE, shape=(im,km+1,ntrac), default_origin=(0,0,0))
    scuflg_v2  = gt_storage.zeros(backend=backend, dtype=B_TYPE, shape=(im,km+1,ntrac), default_origin=(0,0,0))
    q1_gt      = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1,ntrac), default_origin=(0,0,0))

    # 2D Lower Dimensional Storages
    ad      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ad_p1   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    al      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    au      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    bf      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    buod    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    buou    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    cfly    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    chz     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ckt     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    cku     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ckz     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    diss    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    dkq     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    dkt     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    dku     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ele     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    elm     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    f1      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    f1_p1   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    f2_km   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    f2_p1   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    gotvx   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    index_mask = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    pix     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    plyr    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    prn     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    prod    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qcdo_0    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qcdo_ntke = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qcko_0    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qcko_ntke = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qlx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qstl    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    qtx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    radx    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rdzt    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rhly    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rlam    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rle     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    shr2    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    slx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    svx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    tcdo    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    tcko    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    theta   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thetae  = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thlvx   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thlvx_0 = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thlx    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thvx    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    tke     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    tkeh    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    tvx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ucdo    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ucko    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    vcdo    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    vcko    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xkzo    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xkzmo   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xlamde  = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xlamue  = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xmf     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xmfd    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    zi      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    zl      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    zm      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))

    # 1D Arrays represented as 2D storages
    crb     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    dtdz1   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    flg     = gt_storage.zeros(backend=backend,dtype=B_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    gdx     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    hpblx   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    kcld    = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    kpblx   = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    krad    = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    kx1     = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    lcld    = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    mrad    = gt_storage.zeros(backend=backend,dtype=I_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    pblflg  = gt_storage.zeros(backend=backend,dtype=B_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    pcnvflg = gt_storage.zeros(backend=backend,dtype=B_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    phih    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    phim    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    radj    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    radmin  = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rbdn    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    rbup    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    scuflg  = gt_storage.zeros(backend=backend,dtype=B_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    sfcflg  = gt_storage.zeros(backend=backend,dtype=B_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    sflux   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    thermal = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    ustar   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    vpert   = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    wscale  = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xkzm_hx = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    xkzm_mx = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    z0      = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    zl_0    = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    zol     = gt_storage.zeros(backend=backend,dtype=F_TYPE,shape=(im,km+1),default_origin=(0,0),mask=(True,False,True))
    
    # Changing 1D Numpy Arrays into GT4Py 2D Storages
    dqsfc  = numpy_to_gt4py_storage_1D(dqsfc,         backend, km+1)
    dtsfc  = numpy_to_gt4py_storage_1D(dtsfc,         backend, km+1)
    dusfc  = numpy_to_gt4py_storage_1D(dusfc,         backend, km+1)
    dvsfc  = numpy_to_gt4py_storage_1D(dvsfc,         backend, km+1)
    evap   = numpy_to_gt4py_storage_1D(evap,          backend, km+1)
    fh     = numpy_to_gt4py_storage_1D(fh,            backend, km+1)
    fm     = numpy_to_gt4py_storage_1D(fm,            backend, km+1)
    garea  = numpy_to_gt4py_storage_1D(garea,         backend, km+1)
    heat   = numpy_to_gt4py_storage_1D(heat,          backend, km+1)
    hpbl   = numpy_to_gt4py_storage_1D(hpbl,          backend, km+1)
    kinver = numpy_to_gt4py_storage_1D(kinver,        backend, km+1)
    kpbl   = numpy_to_gt4py_storage_1D(kpbl,          backend, km+1)
    psk    = numpy_to_gt4py_storage_1D(psk,           backend, km+1)
    rbsoil = numpy_to_gt4py_storage_1D(rbsoil,        backend, km+1)
    spd1   = numpy_to_gt4py_storage_1D(spd1,          backend, km+1)
    stress = numpy_to_gt4py_storage_1D(stress,        backend, km+1)
    tsea   = numpy_to_gt4py_storage_1D(tsea,          backend, km+1)
    tx1    = numpy_to_gt4py_storage_1D(1.0/prsi[:,0], backend, km+1)
    tx2    = numpy_to_gt4py_storage_1D(1.0/prsi[:,0], backend, km+1)
    u10m   = numpy_to_gt4py_storage_1D(u10m,          backend, km+1)
    v10m   = numpy_to_gt4py_storage_1D(v10m,          backend, km+1)
    xmu    = numpy_to_gt4py_storage_1D(xmu,           backend, km+1)
    zorl   = numpy_to_gt4py_storage_1D(zorl,          backend, km+1)

    # Changing 2D Numpy Arrays into GT4Py 2D Storages
    del_    = numpy_to_gt4py_storage_2D(del_,            backend, km+1)
    du      = numpy_to_gt4py_storage_2D(du,             backend, km+1)
    dv      = numpy_to_gt4py_storage_2D(dv,             backend, km+1)
    hlw     = numpy_to_gt4py_storage_2D(hlw,            backend, km+1)
    phii    = numpy_to_gt4py_storage_2D(phii,           backend, km+1)
    phil    = numpy_to_gt4py_storage_2D(phil,           backend, km+1)
    prsi    = numpy_to_gt4py_storage_2D(prsi,           backend, km+1)
    prsl    = numpy_to_gt4py_storage_2D(prsl,           backend, km+1)
    prslk   = numpy_to_gt4py_storage_2D(prslk,          backend, km+1)
    swh     = numpy_to_gt4py_storage_2D(swh,            backend, km+1)
    q1_0    = numpy_to_gt4py_storage_2D(q1[:,:,0],      backend, km+1)
    q1_ntcw = numpy_to_gt4py_storage_2D(q1[:,:,ntcw-1], backend, km+1)
    q1_ntiw = numpy_to_gt4py_storage_2D(q1[:,:,ntiw-1], backend, km+1)
    q1_ntke = numpy_to_gt4py_storage_2D(q1[:,:,ntke-1], backend, km+1)    
    t1      = numpy_to_gt4py_storage_2D(t1,             backend, km+1)
    u1      = numpy_to_gt4py_storage_2D(u1,             backend, km+1)
    v1      = numpy_to_gt4py_storage_2D(v1,             backend, km+1)
    
    zero_3d_storage = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=(im,km+1,1), dtype=F_TYPE)

    mask_init(index_mask=index_mask, 
              domain=(im,1,km+1))

    ts = perf_counter()

    init(chz=chz,
            ckz=ckz,
            dqsfc=dqsfc, 
            dtsfc=dtsfc,
            dusfc=dusfc, 
            dvsfc=dvsfc, 
            elocp=elocp,
            eps=eps,
            fv=fv,
            g=g,
            garea=garea,
            gdx=gdx,
            gotvx=gotvx,
            gravi=gravi,
            hpbl=hpbl,
            hpblx=hpblx, 
            kpbl=kpbl,
            kcld=kcld, 
            kinver=kinver, 
            km1=km1, 
            kpblx=kpblx, 
            krad=krad,
            kx1=kx1,
            lcld=lcld,
            mask=index_mask,
            mrad=mrad,
            ntiw=ntiw,
            pblflg=pblflg,
            pcnvflg=pcnvflg,
            phii=phii,
            phil=phil,
            pix=pix,
            plyr=plyr,
            prn=prn,
            prsi=prsi,
            prsl=prsl,
            prslk=prslk,
            psk=psk,
            q1_0=q1_0,
            q1_ntcw=q1_ntcw,
            q1_ntiw=q1_ntiw,
            q1_ntke=q1_ntke,
            qlx=qlx,
            qstl=qstl,
            qtx=qtx,
            radmin=radmin,
            rbsoil=rbsoil,
            rdzt=rdzt,
            rhly=rhly,
            scuflg=scuflg,
            sfcflg=sfcflg,
            slx=slx,
            svx=svx,
            t1=t1,
            theta=theta, 
            thetae=thetae,
            thlvx=thlvx,
            thlx=thlx,
            thvx=thvx,
            tke=tke,
            tkeh=tkeh,
            tkmin=tkmin,
            tvx=tvx,
            tx1=tx1,
            tx2=tx2,
            xkzm_h=xkzm_h,
            xkzm_m=xkzm_m, 
            xkzm_s=xkzm_s,            
            xkzm_hx=xkzm_hx,
            xkzm_mx=xkzm_mx,
            xkzmo=xkzmo,
            xkzo=xkzo,
            z0=z0,
            zi=zi,
            zl=zl,
            zm=zm,
            zorl=zorl,
            domain=(im,1,km))
    
    te = perf_counter()

    region_timings[0] += te-ts

    print("Region 1 Time : " + str(te-ts))

    #print("Past init...")
    part2(bf=bf,
          cfly=cfly,
          elocp=elocp,
          el2orc=el2orc,
          fv=fv,
          g=g,
          hlw=hlw,
          plyr=plyr,
          qlx=qlx,
          qstl=qstl,
          qtx=qtx,
          radx=radx,
          rdzt=rdzt,
          rhly=rhly,
          slx=slx,
          svx=svx,
          swh=swh,
          t1=t1,
          xmu=xmu,
          zi=zi,
          domain=(im,1,km+1))

    #print("Past part2...")

    part2a(crb=crb,
           dtdz1=dtdz1,
           evap=evap,
           heat=heat,
           pblflg=pblflg,
           q1=q1_0,
           sfcflg=sfcflg,
           sflux=sflux,
           stress=stress,
           thermal=thermal,
           theta=theta,
           thlvx=thlvx,
           tsea=tsea,
           u10m=u10m,
           ustar=ustar,
           v10m=v10m,
           z0=z0,
           zi=zi,
           fv=fv,
           dt2=dt2,
           domain=(im,1,1))

    #print("Past part2a...")

    part3(rbsoil=rbsoil,
          rbup=rbup,
          rdzt=rdzt,
          shr2=shr2,
          u1=u1,
          v1=v1,
          domain=(im,1,km+1))

    #print("Past part3...")

    part3a(crb=crb,
           flg=flg,
           g=g,
           kpblx=kpblx,
           mask=index_mask,
           rbdn=rbdn,
           rbup=rbup,
           thermal=thermal,
           thlvx=thlvx,
           thlvx_0=thlvx_0,
           u1=u1,
           v1=v1,
           zl=zl,
           domain=(im,1,kmpbl))

    #print("Past part3a...")

    #te = perf_counter()

    #print("Region 1 Time : " + str(te-ts))

    #region_timings[0] += te-ts

    zl_0[:,0] = zl[:,0].reshape((im))

    ts = perf_counter()

    part3a1(crb=crb,
            hpbl=hpbl,
            hpblx=hpblx,
            kpbl=kpbl,
            kpblx=kpblx,
            mask=index_mask,
            pblflg=pblflg,
            rbdn=rbdn,
            rbup=rbup,
            zi=zi,
            zl=zl,
            zl_0=zl_0,
            domain=(im,1,km))

    # for i in range(im):
    #     if kpblx[i,0,0] > 0:
    #         k = kpblx[i,0,0]
    #         if rbdn[i,0,0] >= crb[i,0,0]:
    #             rbint = 0.0
    #         elif rbup[i,0,0] <= crb[i,0,0]:
    #             rbint = 1.0
    #         else:
    #             rbint = (crb[i,0,0]-rbdn[i,0,0])/(rbup[i,0,0]-rbdn[i,0,0])
    #         hpblx[i,0,0] = zl[i,0,k-1] + rbint*(zl[i,0,k]-zl[i,0,k-1])
    #         #if hpblx[i,0,0] < zi[i,0,kpblx[i,0,0]]:
    #         if hpblx[i,0,0] < zi[i,0,k]:
    #             kpblx[i,0,0] = kpblx[i,0,0] - 1
    #     else:
    #         hpblx[i,0,0] = zl[i,0,0]
    #         kpblx[i,0,0] = 0
    #     hpbl[i,0,0] = hpblx[i,0,0]
    #     kpbl[i,0,0] = kpblx[i,0,0]
    #     if kpbl[i,0,0] <= 0:
    #         pblflg[i,0,0] = False

    #print("Past part3a1...")
    part3b(crb=crb,
           evap=evap,
           fh=fh,
           flg=flg,
           fm=fm,
           fv=fv,
           gotvx=gotvx,
           heat=heat,
           hpbl=hpbl,
           hpblx=hpblx,
           kpbl=kpbl,
           kpblx=kpblx,
           phih=phih,
           phim=phim,
           pblflg=pblflg,
           pcnvflg=pcnvflg,
           rbdn=rbdn,
           rbsoil=rbsoil,
           rbup=rbup,
           sfcflg=sfcflg,
           sflux=sflux,
           thermal=thermal,
           theta=theta,
           ustar=ustar,
           vpert=vpert,
           wscale=wscale,
           zl=zl,
           zol=zol,
           domain=(im,1,1))

    #print("Past part3b...")

    part3c(crb=crb,
           flg=flg,
           g=g,
           kpblx=kpbl,
           mask=index_mask,
           rbdn=rbdn,
           rbup=rbup,
           thermal=thermal,
           thlvx=thlvx,
           thlvx_0=thlvx_0,
           u1=u1,
           v1=v1,
           zl=zl,
           domain=(im,1,kmpbl))

    #print("Past part3c...")

    part3c1(crb=crb,
            hpbl=hpbl,
            kpbl=kpbl,
            mask=index_mask,
            pblflg=pblflg,
            pcnvflg=pcnvflg,
            rbdn=rbdn,
            rbup=rbup,
            zi=zi,
            zl=zl,
            domain=(im,1,km))

    # for i in range(im):
    #     if pcnvflg[i,0,0]:
    #         k = kpbl[i,0,0]
    #         if rbdn[i,0,0] >= crb[i,0,0]:
    #             rbint = 0.0
    #         elif rbup[i,0,0] <= crb[i,0,0]:
    #             rbint = 1.0
    #         else:
    #             rbint = (crb[i,0,0]-rbdn[i,0,0])/(rbup[i,0,0]-rbdn[i,0,0])
            
    #         hpbl[i,0,0] = zl[i,0,k-1] + rbint*(zl[i,0,k]-zl[i,0,k-1])
            
    #         #if hpbl[i,0,0] < zi[i,0,kpbl[i,0,0]]:
    #         if hpbl[i,0,0] < zi[i,0,k]:
    #             kpbl[i,0,0] = kpbl[i,0,0] - 1        
            
    #         if kpbl[i,0,0] <= 0:
    #             pblflg[i,0,0] = False
    #             pcnvflg[i,0,0] = False

    #print("Past part3c1...")

    part3d(flg=flg,
           lcld=lcld,
           mask=index_mask,
           scuflg=scuflg,
           zl=zl,
           domain=(im,1,km1))

    #print("Past part3d...")

    part3e(flg=flg,
           kcld=kcld,
           krad=krad,
           lcld=lcld,
           km1=km1,
           mask=index_mask,
           radmin=radmin,
           radx=radx,
           qlx=qlx,
           scuflg=scuflg,
           domain=(im,1,kmscu))

    #print("Past part3e...")

    te = perf_counter()

    region_timings[1] += te-ts

    print("Region 2 Time : ", str(te-ts))

    q1_gt[:,:-1,:] = q1[:,:,:]

    ts = perf_counter()

    part4(pcnvflg=pcnvflg,
          scuflg=scuflg,
          t1=t1,
          tcdo=tcdo,
          tcko=tcko,
          u1=u1,
          ucdo=ucdo,
          ucko=ucko,
          v1=v1,
          vcdo=vcdo,
          vcko=vcko,
          domain=(im,1,km+1))

    te = perf_counter()

    region_timings[2] += te-ts

    print("Region 3 Time : ", str(te-ts))

    #print("Past part4...")

    pcnvflg_v2[:,:,0] = pcnvflg[:,:]
    scuflg_v2[:,:,0]  = scuflg[:,:]


    ts = perf_counter()

    part4a(pcnvflg_v2=pcnvflg_v2,
           q1=q1_gt,
           qcdo=qcdo,
           qcko=qcko,
           scuflg_v2=scuflg_v2,
           domain=(im,km,ntrac1))

    te = perf_counter()

    region_timings[3] += te-ts

    print("Region 4 Time : ", str(te-ts))

    #print("Past part4a...")

    kpbl, hpbl, buou, xmf, tcko, qcko, \
    ucko, vcko, xlamue = mfpblt(im,ix,km,kmpbl,ntcw,ntrac1,dt2,
                                pcnvflg,zl,zm,q1,q1_0,q1_ntcw,t1,u1,v1,plyr,pix,thlx,thvx,
                                gdx,hpbl,kpbl,vpert,buou,xmf,
                                tcko,qcko,ucko,vcko,xlamue,
                                g, gocp, elocp, el2orc, index_mask,
                                compare_dict)

    print("Past mfpblt...")

    radj, mrad, buod, xmfd, tcdo, \
    qcdo, ucdo, vcdo, xlamde = mfscu(im,ix,km,kmscu,ntcw,ntrac1,dt2, 
                                     scuflg,zl,zm,q1,q1_0, q1_ntcw, t1,u1,v1,plyr,pix,
                                     thlx,thvx,thlvx,gdx,thetae,radj,
                                     krad,mrad,radmin,buod,xmfd,
                                     tcdo,qcdo,ucdo,vcdo,xlamde,
                                     g, gocp, elocp, el2orc, index_mask,
                                     compare_dict)

    print("Past mfscu...")

    part5(chz=chz,
          ckz=ckz,
          hpbl=hpbl,
          kpbl=kpbl,
          mask=index_mask,
          pcnvflg=pcnvflg,
          phih=phih,
          phim=phim,
          prn=prn,
          zi=zi,
          domain=(im,1,kmpbl))

    print("Past part5...")

    for k in range(km1):
        for i in range(im):
            zlup = 0.0
            bsum = 0.0
            mlenflg = True
            for n in range(k,km1):
                if mlenflg:
                    dz = zl[i,n+1] - zl[i,n]
                    ptem = gotvx[i,n] * (thvx[i,n+1] - thvx[i,k]) * dz
                    bsum = bsum + ptem
                    zlup = zlup + dz
                    if bsum >= tke[i,k]:
                        if ptem >= 0.0:
                            tem2 = max(ptem,zfmin)
                        else:
                            tem2 = min(ptem,-zfmin)
                        ptem1 = (bsum - tke[i,k]) / tem2
                        zlup = zlup - ptem1 * dz
                        zlup = max(zlup,0.0)
                        mlenflg= False
            zldn = 0.0
            bsum = 0.0
            mlenflg = True
            for n in range(k,-1,-1):
                if mlenflg:
                    if n == 0:
                        dz = zl[i,0]
                        tem1 = tsea[i,0] * (1.0+fv*max(q1[i,0,0],qmin))
                    else:
                        dz = zl[i,n] - zl[i,n-1]
                        tem1 = thvx[i,n-1]
                    ptem = gotvx[i,n] * (thvx[i,k] - tem1) * dz
                    bsum = bsum + ptem
                    zldn = zldn + dz
                    if bsum >= tke[i,k]:
                        if ptem >= 0.0:
                            tem2 = max(ptem,zfmin)
                        else:
                            tem2 = min(ptem,-zfmin)
                        ptem1 = (bsum - tke[i,k]) / tem2
                        zldn = zldn - ptem1 * dz
                        zldn = max(zldn,0.0)
                        mlenflg = False

            tem = 0.5 * (zi[i,k+1] - zi[i,k])  
            tem1 = min(tem,rlmn)

            ptem2 = min(zlup,zldn)
            rlam[i,k] = elmfac * ptem2
            rlam[i,k] = max(rlam[i,k],tem1)
            rlam[i,k] = min(rlam[i,k],rlmx)

            ptem2 = math.sqrt(zlup*zldn)
            ele[i,k] = elefac * ptem2
            ele[i,k] = max(ele[i,k],tem1)
            ele[i,k] = min(ele[i,k],elmx)

    print("Past python stencil...")

    ts = perf_counter()

    part6(bf=bf,
          chz=chz,
          ckz=ckz,
          dku=dku,
          dkt=dkt,
          dkq=dkq,
          ele=ele,
          elm=elm,
          gdx=gdx,
          kpbl=kpbl,
          mask=index_mask,
          mrad=mrad,
          krad=krad,
          pblflg=pblflg,
          prn=prn,
          rlam=rlam,
          scuflg=scuflg,
          shr2=shr2,
          tkeh=tkeh,
          xkzo=xkzo,
          xkzmo=xkzmo,
          zi=zi,
          zl=zl,
          zol=zol,
          domain=(im,1,km))

    #print("Past part6...")

    part6a(bf=bf,
           dkq=dkq,
           dkt=dkt,
           dku=dku,
           gotvx=gotvx,
           krad=krad,
           mask=index_mask,
           radj=radj,
           scuflg=scuflg,
           domain=(im,1,km))

    # for i in range(im):
    #     if scuflg[i,0,0]:
    #         k = krad[i,0,0]
    #         tem = bf[i,0,k] / gotvx[i,0,k]
    #         tem1 = max(tem,tdzmin)
    #         ptem = radj[i,0,0] / tem1
    #         dkt[i,0,k] = dkt[i,0,k] + ptem
    #         dku[i,0,k] = dku[i,0,k] + ptem
    #         dkq[i,0,k] = dkq[i,0,k] + ptem

    #print("Past python stencil...")

    part7(bf=bf,
          buod=buod,
          buou=buou,
          dkt=dkt,
          dku=dku,
          ele=ele,
          gotvx=gotvx,
          krad=krad,
          kpbl=kpbl,
          mask=index_mask,
          mrad=mrad,
          pcnvflg=pcnvflg,
          phim=phim,
          prod=prod,
          rdzt=rdzt,
          rle=rle,
          scuflg=scuflg,
          sflux=sflux,
          shr2=shr2,
          stress=stress,
          u1=u1,
          ucdo=ucdo,
          ucko=ucko,
          ustar=ustar,
          v1=v1,
          vcdo=vcdo,
          vcko=vcko,
          xmf=xmf,
          xmfd=xmfd,
          zl=zl,
          domain=(im,1,km1))

    #print("Past part7...")

    kk = max(round(dt2/cdtn),1)
    dtn = dt2 / kk

    for n in range(kk):
        part8(diss=diss,
              prod=prod,
              rle=rle,
              tke=tke,
              dtn=dtn,
              domain=(im,1,km1))

    te = perf_counter()

    region_timings[4] += te-ts

    print("Region 5 Time : ", str(te-ts))

    #print("Past part8...")

    qcko_ntke[:,:] = qcko[:,:,ntke-1].reshape((im,km+1))
    qcdo_ntke[:,:] = qcdo[:,:,ntke-1].reshape((im,km+1))

    ts = perf_counter()

    part9(pcnvflg=pcnvflg,
          qcdo_ntke=qcdo_ntke,
          qcko_ntke=qcko_ntke,
          scuflg=scuflg,
          tke=tke,
          domain=(im,1,km))

    #print("Past part9...")

    part10(kpbl=kpbl,
           mask=index_mask,
           pcnvflg=pcnvflg,
           qcko_ntke=qcko_ntke,
           tke=tke,
           xlamue=xlamue,
           zl=zl,
           domain=(im,1,kmpbl))

    #print("Past part10...")

    part11(ad=ad,
           f1=f1,
           krad=krad,
           mask=index_mask,
           mrad=mrad,
           qcdo_ntke=qcdo_ntke,
           scuflg=scuflg,
           tke=tke,
           xlamde=xlamde,
           zl=zl,
           domain=(im,1,kmscu))

    #print("Past part11...")

    part12(ad=ad,
           ad_p1=ad_p1,
           al=al,
           au=au,
           del_=del_,
           dkq=dkq,
           dt2=dt2,
           f1=f1,
           f1_p1=f1_p1,
           kpbl=kpbl,
           krad=krad,
           mask=index_mask,
           mrad=mrad,
           pcnvflg=pcnvflg,
           prsl=prsl,
           qcdo_ntke=qcdo_ntke,
           qcko_ntke=qcko_ntke,
           rdzt=rdzt,
           scuflg=scuflg,
           tke=tke,
           xmf=xmf,
           xmfd=xmfd,
           domain=(im,1,km))

    te = perf_counter()

    region_timings[5] += te-ts

    print("Region 6 Time : ", str(te-ts))

    #print("Past part12...")

    au, f1 = tridit(im,km,1,al,ad,au,f1,au,f1, compare_dict)

    print("Past tridit...")

    qtend = (f1[:,:-1] - q1[:,:,ntke-1]) * rdt
    rtg[:,:,ntke-1] = rtg[:,:,ntke-1] + qtend

    for i in range(im):
        ad[i,0] = 1.0
        f1[i,0] = t1[i,0] + dtdz1[i,0] * heat[i,0]
        f2[i,0] = q1[i,0,0] + dtdz1[i,0] * evap[i,0]

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for i in range(im):
                f2[i,0,is_] = q1[i,0,kk]

    f2_km[:,:-1] = f2[:,0,0:km].reshape((im,km))
    qcdo_0[:,:] = qcdo[:,:,0].reshape((im,km+1))
    qcko_0[:,:] = qcko[:,:,0].reshape((im,km+1))

    print("Past python stencil...")

    ts = perf_counter()

    part13(ad=ad,
           ad_p1=ad_p1,
           al=al,
           au=au,
           del_=del_,
           dkt=dkt,
           f1=f1,
           f1_p1=f1_p1,
           f2=f2_km,
           f2_p1=f2_p1,
           kpbl=kpbl,
           krad=krad,
           mask=index_mask,
           mrad=mrad,
           pcnvflg=pcnvflg,
           prsl=prsl,
           q1=q1_0,
           qcdo=qcdo_0,
           qcko=qcko_0,
           rdzt=rdzt,
           scuflg=scuflg,
           tcdo=tcdo,
           tcko=tcko,
           t1=t1,
           xmf=xmf,
           xmfd=xmfd,
           dt2=dt2,
           gocp=gocp,
           domain=(im,1,km))

    te = perf_counter()

    region_timings[6] += te-ts

    print("Region 7 Time : ", str(te-ts))

    #print("Past part13...")

    f2[:,0,0:km] = f2_km[:,0:km]

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for k in range(km1):
                for i in range(im):
                    if pcnvflg[i,0] and k < kpbl[i,0]:
                        dtodsd = dt2/del_[i,k]
                        dtodsu = dt2/del_[i,k+1]
                        dsig  = prsl[i,k]-prsl[i,k+1]
                        tem   = dsig * rdzt[i,k]
                        ptem  = 0.5 * tem * xmf[i,k]
                        ptem1 = dtodsd * ptem
                        ptem2 = dtodsu * ptem
                        tem1  = qcko[i,k,kk] + qcko[i,k+1,kk]
                        tem2  = q1[i,k,kk] + q1[i,k+1,kk]
                        f2[i,0,k+is_] = f2[i,0,k+is_] - (tem1 - tem2) * ptem1
                        f2[i,0,k+1+is_]= q1[i,k+1,kk] + (tem1 - tem2) * ptem2
                    else:
                        f2[i,0,k+1+is_] = q1[i,k+1,kk]

                    if scuflg[i,0] and k >= mrad[i,0] and k < krad[i,0]:
                        dtodsd = dt2/del_[i,k]
                        dtodsu = dt2/del_[i,k+1]
                        dsig  = prsl[i,k]-prsl[i,k+1]
                        tem   = dsig * rdzt[i,k]
                        ptem  = 0.5 * tem * xmfd[i,k]
                        ptem1 = dtodsd * ptem
                        ptem2 = dtodsu * ptem
                        tem1  = qcdo[i,k,kk] + qcdo[i,k+1,kk]
                        tem2  = q1[i,k,kk] + q1[i,k+1,kk]
                        f2[i,0,k+is_]  = f2[i,0,k+is_] + (tem1 - tem2) * ptem1
                        f2[i,0,k+1+is_]= f2[i,0,k+1+is_] - (tem1 - tem2) * ptem2

    print("Past python stencil...")

    au, f1, f2 = tridin(im,km,ntrac1,al,ad,au,f1,f2,au,f1,f2, compare_dict)

    print("Past tridin...")

    for k in range(km):
        ttend = (f1[:,k] - t1[:,k]) * rdt
        qtend = (f2[:,0,k] - q1[:,k,0]) * rdt
        tdt[:,k] = tdt[:,k] + ttend
        rtg[:,k,0] = rtg[:,k,0] + qtend
        dtsfc[:,0] = dtsfc[:,0] + cont * del_[:,k]*ttend
        dqsfc[:,0] = dqsfc[:,0] + conq * del_[:,k]*qtend

    if ntrac1 >= 2:
        for kk in range(1,ntrac1):
            is_ = kk * km
            for k in range(km):
                rtg[:,k,kk] = rtg[:,k,kk] + ((f2[:,0,k+is_] - q1[:,k,kk])*rdt)

    tdt = numpy_to_gt4py_storage_2D(tdt,backend, km+1)
    f2_km[:,:-1] = f2[:,0,0:km].reshape((im,km))

    print("Past python stencil...")

    ts = perf_counter()

    part14(ad=ad,
           ad_p1=ad_p1,
           al=al,
           au=au,
           del_=del_,
           diss=diss,
           dku=dku,
           dtdz1=dtdz1,
           f1=f1,
           f1_p1=f1_p1,
           f2=f2_km,
           f2_p1=f2_p1,
           kpbl=kpbl,
           krad=krad,
           mask=index_mask,
           mrad=mrad,
           pcnvflg=pcnvflg,
           prsl=prsl,
           rdzt=rdzt,
           scuflg=scuflg,
           spd1=spd1,
           stress=stress,
           tdt=tdt,
           u1=u1,
           ucdo=ucdo,
           ucko=ucko,
           v1=v1,
           vcdo=vcdo,
           vcko=vcko,
           xmf=xmf,
           xmfd=xmfd,
           dspheat=dspheat,
           dt2=dt2,
           domain=(im,1,km))

    #print("Past part14...")

    tridi2(im, km, al, ad, au, f1, f2_km, au, f1, f2_km, compare_dict)

    #print("Past tridi2...")

    part15(del_=del_,
           du=du,
           dusfc=dusfc,
           dv=dv,
           dvsfc=dvsfc,
           f1=f1,
           f2=f2_km,
           hpbl=hpbl,
           hpblx=hpblx,
           kpbl=kpbl,
           kpblx=kpblx,
           mask=index_mask,
           u1=u1,
           v1=v1,
           conw=conw,
           rdt=rdt,
           domain=(im,1,km))

    te = perf_counter()

    region_timings[7] += te-ts

    print("Region 8 Time : ", str(te-ts))

    # Increment counter
    region_timings[8] += 1

    dv = storage_to_numpy(dv, (im,km))
    du = storage_to_numpy(du, (im,km))
    tdt = storage_to_numpy(tdt,(im,km))
    kpbl = storage_to_numpy(kpbl,im)
    dusfc = storage_to_numpy(dusfc,im)
    dvsfc = storage_to_numpy(dvsfc,im)
    dtsfc = storage_to_numpy(dtsfc,im)
    dqsfc = storage_to_numpy(dqsfc,im)
    hpbl = storage_to_numpy(hpbl, im)

    kpbl[:] = kpbl + 1

    return dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl

@gtscript.stencil(backend=backend)
def mask_init(index_mask : gtscript.Field[I_TYPE, gtscript.IK]):
    with computation(FORWARD), interval(1,None):
        index_mask = index_mask[0,-1] + 1

def numpy_to_gt4py_storage_2D(arr, backend, k_depth):
    """convert numpy storage to gt4py 2D storage"""
    if arr.dtype == "bool":
        arr = data.astype(np.int32)
    # Enforce that arrays are at least of length k_depth in the "k" direction
    if arr.shape[1] < k_depth:
        Z = np.zeros((arr.shape[0],k_depth-arr.shape[1]))
        arr = np.hstack((arr,Z))
    return gt_storage.from_array(arr, backend=backend, default_origin=((0, 0)), mask=(True,False,True))

def numpy_to_gt4py_storage_1D(arr, backend, k_depth):
    """convert 1d numpy storage to gt4py 2d storage"""
    if arr.dtype == "bool":
        arr = arr.astype(np.int32)
    arr = np.repeat(arr[:,np.newaxis], k_depth, axis=1)
    return gt_storage.from_array(arr, backend=backend, default_origin=(0,0),mask=(True,False,True))

def storage_to_numpy(gt_storage, array_dim):
    if isinstance(array_dim,tuple):
        np_tmp = np.zeros(array_dim)
        np_tmp[:,:] = gt_storage[0:array_dim[0],0:array_dim[1]]
    else:
        np_tmp = np.zeros(array_dim)
        np_tmp[:] = gt_storage[0:array_dim,0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp

def storage_to_numpy_and_assert_equal(gt_storage, numpy_array):
    if numpy_array.ndim == 1:
        temp = gt_storage[0:numpy_array.shape[0],0,0]
        temp2 = np.zeros(temp.shape)
        temp2[:] = temp
        np.testing.assert_allclose(temp2,numpy_array, rtol=1e-14,atol=0)
        #np.testing.assert_array_equal(temp2,numpy_array)
    elif numpy_array.ndim == 2:
        temp = gt_storage.reshape(gt_storage.shape[0],gt_storage.shape[2])
        temp = temp[0:numpy_array.shape[0], 0:numpy_array.shape[1]]
        temp2 = np.zeros(temp.shape)
        temp2[:,:] = temp
        np.testing.assert_allclose(temp2,numpy_array,rtol=1e-14,atol=0)
        #np.testing.assert_array_equal(temp2,numpy_array)

@gtscript.function
def fpvsx(t):
    con_ttp  = 2.7316e2
    con_cvap = 1.8460e3
    con_cliq = 4.1855e3
    con_hvap = 2.5000e6
    con_rv   = 4.6150e2
    con_csol = 2.1060e3
    con_hfus = 3.3358e5
    con_psat = 6.1078e2

    tliq   = con_ttp
    tice   = con_ttp - 20.0
    dldtl  = con_cvap - con_cliq
    heatl  = con_hvap
    xponal = -dldtl/con_rv
    xponbl = -dldtl/con_rv + heatl/(con_rv*con_ttp)
    dldti  = con_cvap - con_csol
    heati  = con_hvap + con_hfus
    xponai = -dldti/con_rv
    xponbi = -dldti/con_rv + heati/(con_rv*con_ttp)

    tr = con_ttp/t
    w = (t-tice)/(tliq-tice)
    pvl=con_psat*(tr**xponal)*exp(xponbl*(1.0-tr))
    pvi=con_psat*(tr**xponai)*exp(xponbi*(1.0-tr))
    tmp1 = tr**xponal
    tmp2 = tr**xponai
    tmp3 = exp(xponbl*(1.0-tr))
    tmp4 = exp(xponbi*(1.0-tr))

    fpvsx = 0.0
    if t > tliq:
        fpvsx = con_psat * tmp1 * tmp3
    elif t < tice:
        fpvsx = con_psat * tmp2 * tmp4
    else:
        fpvsx=w*pvl + (1.0-w) * pvi

    return fpvsx

# @gtscript.function
# def min_fn(a, b):
#     return a if a < b else b

# @gtscript.function
# def max_fn(a,b):
#     return a if a > b else b

@gtscript.function
def fpvs(t):   
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax-xmin)/(nxpvs-1)
    c2xpvs = 1.0/xinc
    c1xpvs= 1.0 - (xmin*c2xpvs)

    tmp = max(c1xpvs+c2xpvs*t,1.0)
    xj = min(tmp,nxpvs)
    jx = min(xj,nxpvs-1.0)
    jx = floor(jx)

    # Convert jx to "x"
    x = xmin + (jx*xinc)
    xm = xmin + ((jx-1)*xinc)

    fpvs = fpvsx(xm) + (xj-jx) *(fpvsx(x) - fpvsx(xm))

    return fpvs

@gtscript.stencil(backend=backend)
def init(chz      : gtscript.Field[F_TYPE, gtscript.IK],
            ckz      : gtscript.Field[F_TYPE, gtscript.IK],
            dqsfc    : gtscript.Field[F_TYPE, gtscript.IK],
            dtsfc    : gtscript.Field[F_TYPE, gtscript.IK],
            dusfc    : gtscript.Field[F_TYPE, gtscript.IK],
            dvsfc    : gtscript.Field[F_TYPE, gtscript.IK],
            garea    : gtscript.Field[F_TYPE, gtscript.IK],
            gdx      : gtscript.Field[F_TYPE, gtscript.IK],
            gotvx    : gtscript.Field[F_TYPE, gtscript.IK],
            hpbl     : gtscript.Field[F_TYPE, gtscript.IK],
            hpblx    : gtscript.Field[F_TYPE, gtscript.IK],
            kinver   : gtscript.Field[I_TYPE, gtscript.IK],
            kcld     : gtscript.Field[I_TYPE, gtscript.IK],
            kpbl     : gtscript.Field[I_TYPE, gtscript.IK],
            kpblx    : gtscript.Field[I_TYPE, gtscript.IK],
            krad     : gtscript.Field[I_TYPE, gtscript.IK],
            kx1      : gtscript.Field[I_TYPE, gtscript.IK],
            lcld     : gtscript.Field[I_TYPE, gtscript.IK],
            mask     : gtscript.Field[I_TYPE, gtscript.IK],
            mrad     : gtscript.Field[I_TYPE, gtscript.IK],
            pblflg   : gtscript.Field[B_TYPE, gtscript.IK],
            pcnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
            phii     : gtscript.Field[F_TYPE, gtscript.IK],
            phil     : gtscript.Field[F_TYPE, gtscript.IK],            
            pix      : gtscript.Field[F_TYPE, gtscript.IK],
            plyr     : gtscript.Field[F_TYPE, gtscript.IK],
            prn      : gtscript.Field[F_TYPE, gtscript.IK],
            prsi     : gtscript.Field[F_TYPE, gtscript.IK],
            prsl     : gtscript.Field[F_TYPE, gtscript.IK],
            prslk    : gtscript.Field[F_TYPE, gtscript.IK],
            psk      : gtscript.Field[F_TYPE, gtscript.IK],
            q1_0     : gtscript.Field[F_TYPE, gtscript.IK],
            q1_ntcw  : gtscript.Field[F_TYPE, gtscript.IK],
            q1_ntiw  : gtscript.Field[F_TYPE, gtscript.IK],
            q1_ntke  : gtscript.Field[F_TYPE, gtscript.IK],
            qlx      : gtscript.Field[F_TYPE, gtscript.IK],
            qstl     : gtscript.Field[F_TYPE, gtscript.IK],
            qtx      : gtscript.Field[F_TYPE, gtscript.IK],
            radmin   : gtscript.Field[F_TYPE, gtscript.IK],
            rbsoil   : gtscript.Field[F_TYPE, gtscript.IK],
            rhly     : gtscript.Field[F_TYPE, gtscript.IK],
            rdzt     : gtscript.Field[F_TYPE, gtscript.IK],
            scuflg   : gtscript.Field[B_TYPE, gtscript.IK],
            sfcflg   : gtscript.Field[B_TYPE, gtscript.IK],
            slx      : gtscript.Field[F_TYPE, gtscript.IK],
            svx      : gtscript.Field[F_TYPE, gtscript.IK],
            t1       : gtscript.Field[F_TYPE, gtscript.IK],
            theta    : gtscript.Field[F_TYPE, gtscript.IK],
            thetae   : gtscript.Field[F_TYPE, gtscript.IK],
            thlvx    : gtscript.Field[F_TYPE, gtscript.IK],
            thlx     : gtscript.Field[F_TYPE, gtscript.IK],
            thvx     : gtscript.Field[F_TYPE, gtscript.IK],
            tke      : gtscript.Field[F_TYPE, gtscript.IK],
            tkeh     : gtscript.Field[F_TYPE, gtscript.IK],
            tvx      : gtscript.Field[F_TYPE, gtscript.IK],
            tx1      : gtscript.Field[F_TYPE, gtscript.IK],
            tx2      : gtscript.Field[F_TYPE, gtscript.IK],
            xkzm_hx  : gtscript.Field[F_TYPE, gtscript.IK],
            xkzm_mx  : gtscript.Field[F_TYPE, gtscript.IK],
            xkzo     : gtscript.Field[F_TYPE, gtscript.IK],
            xkzmo    : gtscript.Field[F_TYPE, gtscript.IK],
            z0       : gtscript.Field[F_TYPE, gtscript.IK],
            zi       : gtscript.Field[F_TYPE, gtscript.IK],
            zorl     : gtscript.Field[F_TYPE, gtscript.IK],
            zl       : gtscript.Field[F_TYPE, gtscript.IK],
            zm       : gtscript.Field[F_TYPE, gtscript.IK],
            *,
            elocp    : float,
            eps      : float,
            fv       : float,
            g        : float,
            gravi    : float,
            km1      : int,
            ntiw     : int,
            tkmin    : float,
            xkzm_h   : float,
            xkzm_m   : float,
            xkzm_s   : float):
    with computation(PARALLEL), interval(...):
        zi = phii[0,0] * gravi
        zl = phil[0,0] * gravi
        tke = max(q1_ntke[0,0], tkmin)
        ckz = ck1
        chz = ch1
        gdx = sqrt(garea[0,0])
        prn = 1.0
        kx1 = 0.0
        zm  = zi[0,1]
        tkeh = 0.5 * (tke[0,0] + tke[0,1])
        rdzt = 1.0 / (zl[0,1] - zl[0,0])

        if gdx[0,0] >= xkgdx:
            xkzm_hx = xkzm_h
            xkzm_mx = xkzm_m
        else:
            xkzm_hx = 0.01 + ((xkzm_h - 0.01) * (1.0 / (xkgdx - 5.0))) * (gdx[0,0] - 5.0)
            xkzm_mx = 0.01 + ((xkzm_m - 0.01) * (1.0 / (xkgdx - 5.0))) * (gdx[0,0] - 5.0)
    
        ptem = prsi[0,1] * tx1[0,0]
        tem1 = 1.0 - ptem
        tem1 = tem1 * tem1 * 10.0
        tem2 = min(1.0, exp(-tem1))

        if mask[0,0] == kx1[0,0] and mask[0,0] > 0:
            tx2 = 1.0 / prsi[0,0]
        tem1 = 1.0 - prsi[0,1] * tx2[0,0]
        tem1 = tem1 * tem1 * 5.0
        tem1 = min(1.0, exp(-tem1))
        if mask[0,0] < kinver[0,0]:
            xkzo = xkzm_hx[0,0] * tem2

            if ptem >= xkzm_s:
                xkzmo = xkzm_mx[0,0]
                kx1 = mask[0,0] + 1
            else:
                xkzmo = xkzm_mx[0,0] * tem1

        z0 = 0.01 * zorl[0,0]
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl  = 1
        hpbl  = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = 1
        sfcflg = 1
        if rbsoil[0,0] > 0.0:
            sfcflg = 0
        pcnvflg = 0
        scuflg  = 1
        radmin  = 0.0
        mrad    = km1
        krad    = 1 - 1
        lcld    = km1 - 1
        kcld    = km1 - 1

        pix   = psk[0,0] / prslk[0,0]
        theta = t1[0,0] * pix[0,0]
        if ntiw > 0:
            tem  = max(q1_ntcw[0,0], qlmin)
            tem1 = max(q1_ntiw[0,0], qlmin)
            qlx = tem + tem1
            slx = cp * t1[0,0] + phil[0,0] - (hvap * tem + (hvap + hfus) * tem1)
        else:
            qlx = max(q1_ntcw[0,0], qlmin)
            slx = cp * t1[0,0] + phil[0,0] - hvap*qlx[0,0]

        tem = 1.0 + fv*max(q1_0[0,0],qmin) - qlx[0,0]
        thvx = theta[0,0] * tem
        tvx  = t1[0,0] * tem
        qtx = max(q1_0[0,0],qmin) + qlx[0,0]
        thlx = theta[0,0] - pix[0,0]*elocp*qlx[0,0]
        thlvx = thlx[0,0] * (1.0 + fv * qtx[0,0])
        svx = cp * tvx[0,0]
        thetae = theta[0,0] + elocp * pix[0,0] * max(q1_0[0,0],qmin)
        gotvx = g / tvx[0,0]

        tem = (tvx[0,1] - tvx[0,0]) * rdzt[0,0]
        if tem > 1.0e-5:
            xkzo = min(xkzo[0,0],xkzinv)
            xkzmo = min(xkzmo[0,0], xkzinv)

        plyr = 0.01 * prsl[0,0]
        es   = 0.01 * fpvs(t1[0,0])
        qs = max(qmin, eps * es / (plyr[0,0] + (eps-1)*es))
        tmp = max(qmin, q1_0[0,0])/qs
        tmp = min(1.0, tmp)
        rhly = max(0.0, tmp)
        qstl = qs

@gtscript.stencil(backend=backend)
def part2(bf   : gtscript.Field[F_TYPE, gtscript.IK],
          cfly : gtscript.Field[F_TYPE, gtscript.IK],
          hlw  : gtscript.Field[F_TYPE, gtscript.IK],
          plyr : gtscript.Field[F_TYPE, gtscript.IK],
          qlx  : gtscript.Field[F_TYPE, gtscript.IK],
          qstl : gtscript.Field[F_TYPE, gtscript.IK],
          qtx  : gtscript.Field[F_TYPE, gtscript.IK],
          radx : gtscript.Field[F_TYPE, gtscript.IK],  
          rdzt : gtscript.Field[F_TYPE, gtscript.IK],
          rhly : gtscript.Field[F_TYPE, gtscript.IK],
          slx  : gtscript.Field[F_TYPE, gtscript.IK],
          svx  : gtscript.Field[F_TYPE, gtscript.IK],
          swh  : gtscript.Field[F_TYPE, gtscript.IK],
          t1   : gtscript.Field[F_TYPE, gtscript.IK],
          xmu  : gtscript.Field[F_TYPE, gtscript.IK],
          zi   : gtscript.Field[F_TYPE, gtscript.IK],
          *,
          elocp  : float,
          el2orc : float,
          fv     : float,
          g      : float):
    
    with computation(PARALLEL):
        with interval(...):
            cfly = 0.0
            clwt = 1.0e-6 * (plyr[0,0]*0.001)
            onemrh = max(1.0e-10, 1.0-rhly[0,0])
            tem1   = max((onemrh*qstl[0,0])**0.49,0.0001)
            tem1   = min(tem1,1.0)
            tem1   = cql / tem1
            val    = min(tem1*qlx[0,0], 50.0)
            val    = max(val,0.0)
            tem2   = sqrt(sqrt(rhly[0,0]))
            tem1   = max(tem2*(1.0-exp(-val)), 0.0)
            tem1   = min(tem1, 1.0)
            if qlx[0,0] > clwt:
                cfly   = tem1
        
        with interval(0,-2):
            tem1 = 0.5 * (t1[0,0]   + t1[0,1])
            cfh  = min(cfly[0,1], 0.5 * (cfly[0,0] + cfly[0,1]))
            alp = g / (0.5 * (svx[0,0]  + svx[0,1]))
            gamma = el2orc * (0.5 * (qstl[0,0] + qstl[0,1])) / (tem1**2)
            epsi = tem1 / elocp
            beta = (1.0 + gamma*epsi * (1.0+fv)) / (1.0 + gamma)
            cqx  = cfh * alp * hvap * (beta - epsi)
            cqx  = cqx + (1.0 - cfh) * fv * g

            bf   = (cfh * alp * beta + (1.0 - cfh) * alp) \
                   * ((slx[0,1] - slx[0,0]) * rdzt[0,0]) \
                   + cqx * ((qtx[0,1] - qtx[0,0]) * rdzt[0,0])

            radx = (zi[0,1] - zi[0,0]) * (swh[0,0] * xmu[0,0]  + hlw[0,0])

@gtscript.stencil(backend=backend)
def part2a(crb    : gtscript.Field[F_TYPE, gtscript.IK],
           dtdz1  : gtscript.Field[F_TYPE, gtscript.IK],
           evap   : gtscript.Field[F_TYPE, gtscript.IK],
           heat   : gtscript.Field[F_TYPE, gtscript.IK],
           pblflg : gtscript.Field[B_TYPE, gtscript.IK],
           q1     : gtscript.Field[F_TYPE, gtscript.IK],
           sfcflg : gtscript.Field[B_TYPE, gtscript.IK],
           sflux  : gtscript.Field[F_TYPE, gtscript.IK],
           stress : gtscript.Field[F_TYPE, gtscript.IK],           
           thermal: gtscript.Field[F_TYPE, gtscript.IK],
           theta  : gtscript.Field[F_TYPE, gtscript.IK],
           thlvx  : gtscript.Field[F_TYPE, gtscript.IK],
           tsea   : gtscript.Field[F_TYPE, gtscript.IK],
           u10m   : gtscript.Field[F_TYPE, gtscript.IK],
           ustar  : gtscript.Field[F_TYPE, gtscript.IK],
           v10m   : gtscript.Field[F_TYPE, gtscript.IK],
           z0     : gtscript.Field[F_TYPE, gtscript.IK],
           zi     : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           dt2    : float,
           fv     : float):
    with computation(PARALLEL), interval(...):
        sflux = heat[0,0] + evap[0,0] * fv * theta[0,0]

        if sfcflg[0,0] == 0 or sflux[0,0] <= 0.0:
            pblflg = 0

        if pblflg[0,0]:
            thermal = thlvx[0,0]
            crb     = rbcr
        else:
            tem1 = sqrt(u10m[0,0]**2 + v10m[0,0]**2)
            tem1 = max(tem1,1.0)
            tem1 = 1e-7 * (tem1 / (f0 * z0[0,0]))
            tem1 = 0.16 * (tem1 ** (-0.18))
            tem1 = min(tem1,crbmax)
            tem1 = max(tem1,crbmin)
            thermal = tsea[0,0] * (1.0 + fv * (max(q1[0,0],qmin)))
            crb = tem1

        dtdz1 = dt2 / (zi[0,1] - zi[0,0])
        ustar = sqrt(stress[0,0])

@gtscript.stencil(backend=backend)
def part3(rbsoil: gtscript.Field[F_TYPE, gtscript.IK],
          rbup  : gtscript.Field[F_TYPE, gtscript.IK],
          rdzt  : gtscript.Field[F_TYPE, gtscript.IK],
          shr2  : gtscript.Field[F_TYPE, gtscript.IK],
          u1    : gtscript.Field[F_TYPE, gtscript.IK],
          v1    : gtscript.Field[F_TYPE, gtscript.IK]):
    
    with computation(PARALLEL), interval(0,-2):
        dw2 = (u1[0,0] - u1[0,1])**2 + (v1[0,0] - v1[0,1])**2
        shr2 = max(dw2, dw2min)*rdzt[0,0]*rdzt[0,0]

    with computation(PARALLEL), interval(...):
        rbup = rbsoil[0,0]

@gtscript.stencil(backend=backend)
def part3a(crb     : gtscript.Field[F_TYPE, gtscript.IK],
           flg     : gtscript.Field[B_TYPE, gtscript.IK],
           kpblx   : gtscript.Field[I_TYPE, gtscript.IK],
           mask    : gtscript.Field[I_TYPE, gtscript.IK],
           rbdn    : gtscript.Field[F_TYPE, gtscript.IK],
           rbup    : gtscript.Field[F_TYPE, gtscript.IK],
           thermal : gtscript.Field[F_TYPE, gtscript.IK],
           thlvx   : gtscript.Field[F_TYPE, gtscript.IK],
           thlvx_0 : gtscript.Field[F_TYPE, gtscript.IK],
           u1      : gtscript.Field[F_TYPE, gtscript.IK],
           v1      : gtscript.Field[F_TYPE, gtscript.IK],
           zl      : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           g       : float):

    with computation(FORWARD):
        with interval(0,1):
            thlvx_0 = thlvx[0,0]

        with interval(1,None):
            thlvx_0 = thlvx_0[0,-1]
            crb = crb[0,-1]
            thermal = thermal[0,-1]

    with computation(FORWARD):
        with interval(0,1):
            if flg[0,0] == 0:
                tem  = u1[0,0]**2 + v1[0,0]**2
                spdk = max(tem,1.0)
                rbdn = rbup[0,0]
                rbup = (thlvx[0,0] - thermal[0,0]) * (g*zl[0,0]/thlvx_0[0,0])/spdk
                kpblx = mask[0,0]
                flg = rbup[0,0] > crb[0,0]

        with interval(1,None):
            if flg[0,-1] == 0:
                tem  = u1[0,0]**2 + v1[0,0]**2
                spdk = max(tem,1.0)
                rbdn = rbup[0,-1]
                rbup = (thlvx[0,0] - thermal[0,0]) * (g*zl[0,0]/thlvx_0[0,0])/spdk
                kpblx = mask[0,0]
                flg = rbup[0,0] > crb[0,0]
            else:
                rbdn = rbdn[0,-1]
                rbup = rbup[0,-1]
                kpblx = kpblx[0,-1]
                flg = flg[0,-1]

    with computation(BACKWARD), interval(0,-1):
        rbdn = rbdn[0,1]
        rbup = rbup[0,1]
        kpblx = kpblx[0,1]
        flg = flg[0,1]

@gtscript.stencil(backend=backend)
def part3a1(crb        : gtscript.Field[F_TYPE, gtscript.IK],
            hpbl       : gtscript.Field[F_TYPE, gtscript.IK],
            hpblx      : gtscript.Field[F_TYPE, gtscript.IK],
            kpbl       : gtscript.Field[I_TYPE, gtscript.IK],
            kpblx      : gtscript.Field[I_TYPE, gtscript.IK],
            mask       : gtscript.Field[I_TYPE, gtscript.IK],
            pblflg     : gtscript.Field[B_TYPE, gtscript.IK],
            rbdn       : gtscript.Field[F_TYPE, gtscript.IK],
            rbup       : gtscript.Field[F_TYPE, gtscript.IK],
            zi         : gtscript.Field[F_TYPE, gtscript.IK],
            zl         : gtscript.Field[F_TYPE, gtscript.IK],
            zl_0       : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(...):
        if mask[0,0] > 0:
            kpblx  = kpblx[0,-1]
            hpblx  = hpblx[0,-1]
            zl_0   = zl_0[0,-1]
            kpbl   = kpbl[0,-1]
            hpbl   = hpbl[0,-1]
            kpbl   = kpbl[0,-1]
            pblflg = pblflg[0,-1]
            crb    = crb[0,-1]
            rbup   = rbup[0,-1]
            rbdn   = rbdn[0,-1]
        
        rbint = 0.0

        if mask[0,0] == kpblx[0,0]:
            if kpblx[0,0] > 0:
                if rbdn[0,0] >= crb[0,0]:
                    rbint = 0.0
                elif rbup[0,0] <= crb[0,0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0,0] - rbdn[0,0])/ (rbup[0,0] - rbdn[0,0])
                hpblx = zl[0,-1] + rbint*(zl[0,0] - zl[0,-1])

                if hpblx[0,0] < zi[0,0]:
                    kpblx = kpblx[0,0] - 1
            else:
                hpblx = zl[0,0]
                kpblx = 0
            
            hpbl = hpblx[0,0]
            kpbl = kpblx[0,0]

            if kpbl[0,0] <= 0:
                pblflg = 0

    with computation(BACKWARD), interval(0,-1):
        kpblx  = kpblx[0,1]
        hpblx  = hpblx[0,1]
        kpbl   = kpbl[0,1]
        hpbl   = hpbl[0,1]
        pblflg = pblflg[0,1]
        
@gtscript.stencil(backend=backend)
def part3b(crb    : gtscript.Field[F_TYPE, gtscript.IK],
           evap   : gtscript.Field[F_TYPE, gtscript.IK],
           fh     : gtscript.Field[F_TYPE, gtscript.IK],
           flg    : gtscript.Field[B_TYPE, gtscript.IK],
           fm     : gtscript.Field[F_TYPE, gtscript.IK],
           gotvx  : gtscript.Field[F_TYPE, gtscript.IK],
           heat   : gtscript.Field[F_TYPE, gtscript.IK],
           hpbl   : gtscript.Field[F_TYPE, gtscript.IK],
           hpblx  : gtscript.Field[F_TYPE, gtscript.IK],
           kpbl   : gtscript.Field[F_TYPE, gtscript.IK],
           kpblx  : gtscript.Field[I_TYPE, gtscript.IK],
           phih   : gtscript.Field[F_TYPE, gtscript.IK],
           phim   : gtscript.Field[F_TYPE, gtscript.IK],
           pblflg : gtscript.Field[B_TYPE, gtscript.IK],
           pcnvflg: gtscript.Field[B_TYPE, gtscript.IK],
           rbdn   : gtscript.Field[F_TYPE, gtscript.IK],
           rbsoil : gtscript.Field[F_TYPE, gtscript.IK],
           rbup   : gtscript.Field[F_TYPE, gtscript.IK],
           sfcflg : gtscript.Field[B_TYPE, gtscript.IK],
           sflux  : gtscript.Field[F_TYPE, gtscript.IK],
           thermal: gtscript.Field[F_TYPE, gtscript.IK],
           theta  : gtscript.Field[F_TYPE, gtscript.IK],
           ustar  : gtscript.Field[F_TYPE, gtscript.IK],
           vpert  : gtscript.Field[F_TYPE, gtscript.IK],
           wscale : gtscript.Field[F_TYPE, gtscript.IK],
           zl     : gtscript.Field[F_TYPE, gtscript.IK],
           zol    : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           fv     : float):
    
    with computation(PARALLEL), interval(0,1):
        zol = max(rbsoil[0,0]*fm[0,0]*fm[0,0]/fh[0,0], rimin)
        if sfcflg[0,0]:
            zol = min(zol[0,0], -zfmin)
        else:
            zol = max(zol[0,0], zfmin)

        zol1 = zol[0,0] * sfcfrac * hpbl[0,0] / zl[0,0]
        tem = 1.0 / (1.0 - aphi16*zol1)
        
        if sfcflg[0,0]:
            phih = sqrt(tem)
            phim = sqrt(phih[0,0])
        else:
            phim = 1.0 + aphi5*zol1
            phih = phim[0,0]

        pcnvflg = pblflg[0,0] and (zol[0,0] < zolcru)

        wst3 = gotvx[0,0] * sflux[0,0] * hpbl[0,0]
        ust3 = ustar[0,0]**3.0

        tmp1 = (ust3 + wfac * vk * wst3 * sfcfrac)**h1
        tmp2 = ustar[0,0]/aphi5

        if pblflg[0,0]:
            wscale = max(tmp1,tmp2)

        hgamt = heat[0,0]/wscale[0,0]
        hgamq = evap[0,0]/wscale[0,0]

        if pcnvflg[0,0]:
            vpert = max(hgamt+hgamq*fv*theta[0,0],0.0)

        flg = 1
        if pcnvflg[0,0]:
            tmp1 = cfac*vpert[0,0]
            tmp1 = min(tmp1, gamcrt)
            thermal = thermal[0,0] + tmp1
            flg = 0
            rbup = rbsoil[0,0]

@gtscript.stencil(backend=backend)
def part3c(crb     : gtscript.Field[F_TYPE, gtscript.IK],
           flg     : gtscript.Field[B_TYPE, gtscript.IK],
           kpblx   : gtscript.Field[I_TYPE, gtscript.IK],
           mask    : gtscript.Field[I_TYPE, gtscript.IK],
           rbdn    : gtscript.Field[F_TYPE, gtscript.IK],
           rbup    : gtscript.Field[F_TYPE, gtscript.IK],
           thermal : gtscript.Field[F_TYPE, gtscript.IK],
           thlvx   : gtscript.Field[F_TYPE, gtscript.IK],
           thlvx_0 : gtscript.Field[F_TYPE, gtscript.IK],
           u1      : gtscript.Field[F_TYPE, gtscript.IK],
           v1      : gtscript.Field[F_TYPE, gtscript.IK],
           zl      : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           g       : float):

    with computation(FORWARD):
        with interval(0,1):
            thlvx_0 = thlvx[0,0]

        with interval(1,None):
            thlvx_0 = thlvx_0[0,-1]
            crb = crb[0,-1]
            thermal = thermal[0,-1]
            rbup    = rbup[0,-1]
            flg     = flg[0,-1]
            kpblx   = kpblx[0,-1]

    with computation(FORWARD): 
        with interval(1,2):
            if flg[0,0] == 0:
                tem  = u1[0,0]**2 + v1[0,0]**2
                spdk = max(tem,1.0)
                rbdn = rbup[0,0]
                rbup = (thlvx[0,0] - thermal[0,0]) * (g*zl[0,0]/thlvx_0[0,0])/spdk
                kpblx = mask[0,0]
                flg = rbup[0,0] > crb[0,0]

        with interval(2,None):            
            if flg[0,-1] == 0:
                spdk = max(u1[0,0]**2 + v1[0,0]**2,1.0)
                rbdn = rbup[0,-1]
                rbup = (thlvx[0,0] - thermal[0,0]) * (g*zl[0,0]/thlvx_0[0,0])/spdk
                kpblx = mask[0,0]
                flg = rbup[0,0] > crb[0,0]
            else:
                rbdn = rbdn[0,-1]
                rbup = rbup[0,-1]
                kpblx = kpblx[0,-1]
                flg = flg[0,-1]

    with computation(BACKWARD), interval(0,-1):
        rbdn = rbdn[0,1]
        rbup = rbup[0,1]
        kpblx = kpblx[0,1]
        flg = flg[0,1]

@gtscript.stencil(backend=backend)
def part3c1(crb          : gtscript.Field[F_TYPE, gtscript.IK],
            hpbl         : gtscript.Field[F_TYPE, gtscript.IK],
            kpbl         : gtscript.Field[I_TYPE, gtscript.IK],
            mask         : gtscript.Field[I_TYPE, gtscript.IK],
            pblflg       : gtscript.Field[B_TYPE, gtscript.IK],
            pcnvflg      : gtscript.Field[B_TYPE, gtscript.IK],
            rbdn         : gtscript.Field[F_TYPE, gtscript.IK],
            rbup         : gtscript.Field[F_TYPE, gtscript.IK],
            zi           : gtscript.Field[F_TYPE, gtscript.IK],
            zl           : gtscript.Field[F_TYPE, gtscript.IK]):
    
    with computation(FORWARD), interval(...):
        if mask[0,0] > 0:
            crb     = crb[0,-1]
            hpbl    = hpbl[0,-1]
            kpbl    = kpbl[0,-1]
            pblflg  = pblflg[0,-1]
            pcnvflg = pcnvflg[0,-1]
            rbdn    = rbdn[0,-1]
            rbup    = rbup[0,-1]
    
        rbint = 0.0

        if pcnvflg[0,0] and kpbl[0,0] == mask[0,0]:
            if rbdn[0,0] >= crb[0,0]:
                rbint = 0.0
            elif rbup[0,0] <= crb[0,0]:
                rbint = 1.0
            else:
                rbint = (crb[0,0] - rbdn[0,0]) / (rbup[0,0] - rbdn[0,0])

            hpbl[0,0] = zl[0,-1] + rbint * (zl[0,0] - zl[0,-1])

            if hpbl[0,0] < zi[0,0]:
                kpbl[0,0] = kpbl[0,0] - 1

            if kpbl[0,0] <= 0:
                pblflg[0,0]  = 0
                pcnvflg[0,0] = 0

    with computation(BACKWARD), interval(0,-1):
        hpbl    = hpbl[0,1]
        kpbl    = kpbl[0,1]
        pblflg  = pblflg[0,1]
        pcnvflg = pcnvflg[0,1]

@gtscript.stencil(backend=backend)
def part3d(flg    : gtscript.Field[B_TYPE, gtscript.IK],
           lcld   : gtscript.Field[I_TYPE, gtscript.IK],
           mask   : gtscript.Field[I_TYPE, gtscript.IK],
           scuflg : gtscript.Field[B_TYPE, gtscript.IK],
           zl     : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD):
        with interval(0,1):
            flg = scuflg[0,0]
            if flg[0,0] and zl[0,0] >= zstblmax:
                lcld = mask[0,0]
                flg = 0 
        with interval(1,None):
            lcld = lcld[0,-1]
            flg  = flg[0,-1]
            if flg[0,0] and zl[0,0] >= zstblmax:
                lcld = mask[0,0]
                flg = 0

    with computation(BACKWARD), interval(0,-1):
        lcld = lcld[0,1]
        flg  = flg[0,1]

@gtscript.stencil(backend=backend)
def part3e(flg    : gtscript.Field[B_TYPE, gtscript.IK],
           kcld   : gtscript.Field[I_TYPE, gtscript.IK],
           krad   : gtscript.Field[I_TYPE, gtscript.IK],
           lcld   : gtscript.Field[I_TYPE, gtscript.IK],
           mask   : gtscript.Field[I_TYPE, gtscript.IK], 
           radmin : gtscript.Field[F_TYPE, gtscript.IK],
           radx   : gtscript.Field[F_TYPE, gtscript.IK],
           qlx    : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg : gtscript.Field[B_TYPE, gtscript.IK],
           *,
           km1    : int):
    
    with computation(FORWARD):
        with interval(0,1):
            flg = scuflg[0,0]
        with interval(1,None):
            flg = flg[0,-1]
            lcld = lcld[0,-1]

    with computation(BACKWARD):
        with interval(-1,None):
            if flg[0,0] and mask[0,0] <= lcld[0,0] and qlx[0,0] >= qlcr:
                kcld = mask[0,0]
                flg  = 0

        with interval(0,-1):
            kcld[0,0] = kcld[0,1]
            flg[0,0]  = flg[0,1]
            if flg[0,0] and mask[0,0] <= lcld[0,0] and qlx[0,0] >= qlcr:
                kcld = mask[0,0]
                flg  = 0

    with computation(FORWARD):
        with interval(0,1):
            if scuflg[0,0] and kcld[0,0] == (km1-1):
                scuflg = 0
            flg = scuflg[0,0]

        with interval(1,None):
            flg = flg[0,-1]
            kcld = kcld[0,-1]

    with computation(BACKWARD):
        with interval(-1,None):
            if flg[0,0] and mask[0,0] <= kcld[0,0]:
                if qlx[0,0] >= qlcr:
                    if radx[0,0] < radmin[0,0]:
                        radmin = radx[0,0]
                        krad   = mask[0,0]
                else:
                    flg = 0

        with interval(0,-1):
            flg = flg[0,1]
            radmin = radmin[0,1]
            krad   = krad[0,1]
            if flg[0,0] and mask[0,0] <= kcld[0,0]:
                if qlx[0,0] >= qlcr:
                    if radx[0,0] < radmin[0,0]:
                        radmin = radx[0,0]
                        krad   = mask[0,0]
                else:
                    flg = 0
    
    with computation(PARALLEL), interval(0,1):
        if scuflg[0,0] and krad[0,0] <= 0:
            scuflg = 0
        if scuflg[0,0] and radmin[0,0] >= 0.0:
            scuflg = 0

@gtscript.stencil(backend=backend)
def part4(pcnvflg : gtscript.Field[B_TYPE, gtscript.IK],
          scuflg  : gtscript.Field[B_TYPE, gtscript.IK],
          t1      : gtscript.Field[F_TYPE, gtscript.IK],
          tcdo    : gtscript.Field[F_TYPE, gtscript.IK],
          tcko    : gtscript.Field[F_TYPE, gtscript.IK],
          u1      : gtscript.Field[F_TYPE, gtscript.IK],
          ucdo    : gtscript.Field[F_TYPE, gtscript.IK],
          ucko    : gtscript.Field[F_TYPE, gtscript.IK],
          v1      : gtscript.Field[F_TYPE, gtscript.IK],
          vcdo    : gtscript.Field[F_TYPE, gtscript.IK],
          vcko    : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        pcnvflg = pcnvflg[0,-1]
        scuflg  = scuflg[0,-1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0,0]:
            tcko = t1[0,0]
            ucko = u1[0,0]
            vcko = v1[0,0]
        if scuflg[0,0]:
            tcdo = t1[0,0]
            ucdo = u1[0,0]
            vcdo = v1[0,0]

@gtscript.stencil(backend=backend)
def part4a(pcnvflg_v2      : gtscript.Field[B_TYPE],
           q1              : gtscript.Field[F_TYPE],
           qcdo            : gtscript.Field[F_TYPE],
           qcko            : gtscript.Field[F_TYPE],
           scuflg_v2       : gtscript.Field[B_TYPE]):

    with computation(FORWARD), interval(1,None):
        pcnvflg_v2 = pcnvflg_v2[0,0,-1]
        scuflg_v2  = scuflg_v2[0,0,-1]

    with computation(PARALLEL), interval(...):
        if pcnvflg_v2[0,0,0]:
            qcko = q1[0,0,0]
        if scuflg_v2[0,0,0]:
            qcdo = q1[0,0,0]

@gtscript.stencil(backend=backend)
def part5(chz      : gtscript.Field[F_TYPE, gtscript.IK],
          ckz      : gtscript.Field[F_TYPE, gtscript.IK],
          hpbl     : gtscript.Field[F_TYPE, gtscript.IK],
          kpbl     : gtscript.Field[I_TYPE, gtscript.IK],
          mask     : gtscript.Field[I_TYPE, gtscript.IK],
          pcnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
          phih     : gtscript.Field[F_TYPE, gtscript.IK],
          phim     : gtscript.Field[F_TYPE, gtscript.IK],
          prn      : gtscript.Field[F_TYPE, gtscript.IK],
          zi       : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        phih = phih[0,-1]
        phim = phim[0,-1]

    with computation(PARALLEL), interval(...):
        tem = phih[0,0] / phim[0,0]
        tem1 = max(zi[0,1] - sfcfrac*hpbl[0,0],0.0)
        ptem = -3.0 * (tem1**2.0) / (hpbl[0,0]**2.0)


        if mask[0,0] < kpbl[0,0]:
            if pcnvflg[0,0]:
                prn = 1.0 + (tem-1.0) * exp(ptem)
            else:
                prn = tem

        prn_tmp = min(prn[0,0], prmax)
        prn_tmp = max(prn_tmp,    prmin)

        ckz_tmp = ck1 + (ck0-ck1)*exp(ptem)
        ckz_tmp = min(ckz_tmp, ck0)
        ckz_tmp = max(ckz_tmp, ck1)

        chz_tmp = ch1 + (ch0-ch1) * exp(ptem)
        chz_tmp = min(chz_tmp, ch0)
        chz_tmp = max(chz_tmp, ch1)

        if mask[0,0] < kpbl[0,0]:
            prn = prn_tmp
            ckz = ckz_tmp
            chz = chz_tmp

@gtscript.stencil(backend=backend)
def part6(bf      : gtscript.Field[F_TYPE, gtscript.IK],
          chz     : gtscript.Field[F_TYPE, gtscript.IK],
          ckz     : gtscript.Field[F_TYPE, gtscript.IK],
          dku     : gtscript.Field[F_TYPE, gtscript.IK],
          dkt     : gtscript.Field[F_TYPE, gtscript.IK],
          dkq     : gtscript.Field[F_TYPE, gtscript.IK],
          ele     : gtscript.Field[F_TYPE, gtscript.IK], 
          elm     : gtscript.Field[F_TYPE, gtscript.IK],
          gdx     : gtscript.Field[F_TYPE, gtscript.IK],
          kpbl    : gtscript.Field[I_TYPE, gtscript.IK],
          mask    : gtscript.Field[I_TYPE, gtscript.IK],
          mrad    : gtscript.Field[I_TYPE, gtscript.IK],
          krad    : gtscript.Field[I_TYPE, gtscript.IK],
          pblflg  : gtscript.Field[B_TYPE, gtscript.IK],
          prn     : gtscript.Field[F_TYPE, gtscript.IK],
          rlam    : gtscript.Field[F_TYPE, gtscript.IK],
          scuflg  : gtscript.Field[B_TYPE, gtscript.IK],
          shr2    : gtscript.Field[F_TYPE, gtscript.IK],
          tkeh    : gtscript.Field[F_TYPE, gtscript.IK],
          xkzo    : gtscript.Field[F_TYPE, gtscript.IK],
          xkzmo   : gtscript.Field[F_TYPE, gtscript.IK],
          zi      : gtscript.Field[F_TYPE, gtscript.IK],
          zl      : gtscript.Field[F_TYPE, gtscript.IK],
          zol     : gtscript.Field[F_TYPE, gtscript.IK]):
    
    with computation(FORWARD), interval(1,None):
        zol = zol[0,-1]
        pblflg = pblflg[0,-1]
        scuflg = scuflg[0,-1]

    with computation(PARALLEL):
        with interval(0,-1):
            tem = vk * zl[0,0]            
            zk = 0.0
            if zol[0,0] < 0.0:
                ptem1 = (1.0 - 100.0 * zol[0,0])**0.2
                zk = tem * ptem1
            elif zol[0,0] >= 1.0:
                zk = tem / 3.7
            else:
                zk = tem / (1.0 + 2.7 * zol[0,0])
            
            elm = zk * rlam[0,0]/(rlam[0,0] + zk)

            dz = zi[0,1] - zi[0,0]
            tem = max(gdx[0,0], dz)
            elm = min(elm[0,0], tem)
            ele = min(ele[0,0], tem)

        with interval(-1,None):
            elm = elm[0,-1]
            ele = ele[0,-1]

    with computation(PARALLEL), interval(0,-1):
        tem = 0.5 * (elm[0,0] + elm[0,1]) * sqrt(tkeh[0,0])
        ri = max(bf[0,0]/shr2[0,0],rimin)
        prnum = min(1.0 + 2.1*ri, prmax)
    
        if mask[0,0] < kpbl[0,0]:
            if pblflg[0,0]:
                dku = ckz[0,0] * tem
                dkt = dku[0,0] / prn[0,0]
            else:
                dkt = chz[0,0] * tem
                dku = dkt[0,0] * prn[0,0]
        else:
            if(ri < 0.0):
                dku = ck1 * tem
                dkt = rchck * dku[0,0]
            else:
                dkt = ch1 * tem
                dku = dkt[0,0] * prnum

        tem = ckz[0,0] * tem
        ptem = tem / prscu
        dku_tmp = max(dku[0,0],tem)
        dkt_tmp = max(dkt[0,0], ptem)

        if scuflg[0,0]:
            if mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                dku = dku_tmp
                dkt = dkt_tmp
        
        dkq = prtke * dkt[0,0]

        dkt = min(dkt[0,0],dkmax)
        dkt = max(dkt[0,0],xkzo[0,0])

        dkq = min(dkq[0,0],dkmax)
        dkq = max(dkq[0,0],xkzo[0,0])

        dku = min(dku[0,0],dkmax)
        dku = max(dku[0,0],xkzmo[0,0])

@gtscript.stencil(backend=backend)
def part6a(bf       : gtscript.Field[F_TYPE, gtscript.IK],
           dkq      : gtscript.Field[F_TYPE, gtscript.IK],
           dkt      : gtscript.Field[F_TYPE, gtscript.IK],
           dku      : gtscript.Field[F_TYPE, gtscript.IK],
           gotvx    : gtscript.Field[F_TYPE, gtscript.IK],
           krad     : gtscript.Field[I_TYPE, gtscript.IK],
           mask     : gtscript.Field[I_TYPE, gtscript.IK],
           radj     : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg   : gtscript.Field[B_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(...):
        if mask[0,0] == krad[0,0]:
            if scuflg[0,0]:
                tem = bf[0,0] / gotvx[0,0]
                tem1 = tem
                if tem < tdzmin:
                    tem1 = tdzmin
                ptem = radj[0,0] / tem1
                dkt = dkt[0,0] + ptem
                dku = dku[0,0] + ptem
                dkq = dkq[0,0] + ptem
                

@gtscript.stencil(backend=backend)
def part7(bf      : gtscript.Field[F_TYPE, gtscript.IK],
          buod    : gtscript.Field[F_TYPE, gtscript.IK],
          buou    : gtscript.Field[F_TYPE, gtscript.IK],
          ele     : gtscript.Field[F_TYPE, gtscript.IK],
          dkt     : gtscript.Field[F_TYPE, gtscript.IK],
          dku     : gtscript.Field[F_TYPE, gtscript.IK],
          gotvx   : gtscript.Field[F_TYPE, gtscript.IK],
          krad    : gtscript.Field[I_TYPE, gtscript.IK],
          kpbl    : gtscript.Field[I_TYPE, gtscript.IK],
          mask    : gtscript.Field[I_TYPE, gtscript.IK],
          mrad    : gtscript.Field[I_TYPE, gtscript.IK],
          pcnvflg : gtscript.Field[B_TYPE, gtscript.IK],
          phim    : gtscript.Field[F_TYPE, gtscript.IK],
          prod    : gtscript.Field[F_TYPE, gtscript.IK],
          rdzt    : gtscript.Field[F_TYPE, gtscript.IK],
          rle     : gtscript.Field[F_TYPE, gtscript.IK],
          scuflg  : gtscript.Field[B_TYPE, gtscript.IK],
          sflux   : gtscript.Field[F_TYPE, gtscript.IK],
          shr2    : gtscript.Field[F_TYPE, gtscript.IK],
          stress  : gtscript.Field[F_TYPE, gtscript.IK],
          u1      : gtscript.Field[F_TYPE, gtscript.IK],
          ucdo    : gtscript.Field[F_TYPE, gtscript.IK],
          ucko    : gtscript.Field[F_TYPE, gtscript.IK],
          ustar   : gtscript.Field[F_TYPE, gtscript.IK],
          v1      : gtscript.Field[F_TYPE, gtscript.IK],
          vcdo    : gtscript.Field[F_TYPE, gtscript.IK],
          vcko    : gtscript.Field[F_TYPE, gtscript.IK],
          xmf     : gtscript.Field[F_TYPE, gtscript.IK],
          xmfd    : gtscript.Field[F_TYPE, gtscript.IK],
          zl      : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        sflux = sflux[0,-1]
        ustar = ustar[0,-1]
        phim  = phim[0,-1]
        kpbl  = kpbl[0,-1]
        scuflg = scuflg[0,-1]
        pcnvflg = pcnvflg[0,-1]
        stress = stress[0,-1]
        mrad = mrad[0,-1]
        krad = krad[0,-1]

    with computation(PARALLEL):
        with interval(0,1):
            if scuflg[0,0] and mrad[0,0] == 0:
                ptem2 = xmfd[0,0] * buod[0,0]
            else:
                ptem2 = 0.0
            
            buop = 0.5 * (gotvx[0,0] * sflux[0,0] + (-dkt[0,0] * bf[0,0]) + ptem2)

            if scuflg[0,0] and mrad[0,0] == 0:
                ptem1 = ucdo[0,0] + ucdo[0,1] - u1[0,0] - u1[0,1]
                ptem1 = 0.5 * ((u1[0,1] - u1[0,0]) * rdzt[0,0]) * xmfd[0,0] * ptem1
            else:
                ptem1 = 0.0

            if scuflg[0,0] and mrad[0,0] == 0:
                ptem2 = vcdo[0,0] + vcdo[0,1] - v1[0,0] - v1[0,1]
                ptem2 = 0.5 * ((v1[0,1] - v1[0,0]) * rdzt[0,0]) * xmfd[0,0] * ptem2
            else:
                ptem2 = 0.0

            tem2 = stress[0,0] * ustar[0,0] * phim[0,0] / (vk*zl[0,0])
        
            shrp = 0.5 * ((dku[0,0] * shr2[0,0]) + ptem1 + ptem2 + tem2)

            prod = buop + shrp

        with interval(1,None):
            if pcnvflg[0,0] and mask[0,0] <= kpbl[0,0]:
                ptem1 = (0.5 * (xmf[0,-1] + xmf[0,0])) * buou[0,0]
            else:
                ptem1 = 0.0

            if scuflg[0,0]:
                if mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                    ptem2 = (0.5 * (xmfd[0,-1] + xmfd[0,0])) * buod[0,0]
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0

            buop = (0.5 * ((-dkt[0,-1] * bf[0,-1]) + (-dkt[0,0] * bf[0,0]))) + ptem1 + ptem2

            tem1 = (u1[0,1]-u1[0,0]) * rdzt[0,0]
            tem2 = (u1[0,0]-u1[0,-1]) * rdzt[0,-1]

            if pcnvflg[0,0] and mask[0,0] <= kpbl[0,0]:
                ptem = xmf[0,0] * tem1 + xmf[0,-1] * tem2
                ptem1 = 0.5 * ptem * (u1[0,0] - ucko[0,0])
            else:
                ptem1 = 0.0
            
            if scuflg[0,0]:
                if mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                    ptem = xmfd[0,0] * tem1 + xmfd[0,-1] * tem2
                    ptem2 = 0.5 * ptem * (ucdo[0,0] - u1[0,0])
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0
            
            shrp = 0.5 * (dku[0,-1] * shr2[0,-1] + dku[0,0] * shr2[0,0]) + ptem1 + ptem2
            tem1 = (v1[0,1] - v1[0,0]) * rdzt[0,0]
            tem2 = (v1[0,0] - v1[0,-1]) * rdzt[0,-1]

            if pcnvflg[0,0] and mask[0,0] <= kpbl[0,0]:
                ptem = xmf[0,0] * tem1 + xmf[0,-1] * tem2
                ptem1 = 0.5 * ptem * (v1[0,0] - vcko[0,0])
            else:
                ptem1 = 0.0
            
            if scuflg[0,0]:
                if mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                    ptem = xmfd[0,0] * tem1 + xmfd[0,-1] * tem2
                    ptem2 = 0.5 * ptem * (vcdo[0,0] - v1[0,0])
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0

            shrp = shrp + ptem1 + ptem2

            prod = buop + shrp

    with computation(PARALLEL), interval(...):
        rle = ce0 / ele[0,0]

@gtscript.stencil(backend=backend)
def part8(diss     : gtscript.Field[F_TYPE, gtscript.IK],
          prod     : gtscript.Field[F_TYPE, gtscript.IK],
          rle      : gtscript.Field[F_TYPE, gtscript.IK],
          tke      : gtscript.Field[F_TYPE, gtscript.IK],
          *,
          dtn      : float):
    with computation(PARALLEL), interval(...):
        diss = rle[0,0] * tke[0,0] * sqrt(tke[0,0])
        tem1 = prod[0,0] + tke[0,0] / dtn
        tem2 = min(diss[0,0], tem1)
        diss = max(tem2, 0.0)
        tem1 = tke[0,0] + dtn * (prod[0,0] - diss[0,0])
        tke  = max(tem1,tkmin)

@gtscript.stencil(backend=backend)
def part9(pcnvflg      : gtscript.Field[B_TYPE, gtscript.IK],
          qcdo_ntke    : gtscript.Field[F_TYPE, gtscript.IK],
          qcko_ntke    : gtscript.Field[F_TYPE, gtscript.IK],
          scuflg       : gtscript.Field[B_TYPE, gtscript.IK],
          tke          : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        scuflg = scuflg[0,-1]
        pcnvflg = pcnvflg[0,-1]
    
    with computation(PARALLEL), interval(...):
        if pcnvflg[0,0]:
            qcko_ntke = tke[0,0]
        if scuflg[0,0]:
            qcdo_ntke = tke[0,0]

@gtscript.stencil(backend=backend)
def part10(kpbl         : gtscript.Field[I_TYPE, gtscript.IK],
           mask         : gtscript.Field[I_TYPE, gtscript.IK],
           pcnvflg      : gtscript.Field[B_TYPE, gtscript.IK],
           qcko_ntke    : gtscript.Field[F_TYPE, gtscript.IK],
           tke          : gtscript.Field[F_TYPE, gtscript.IK],
           xlamue       : gtscript.Field[F_TYPE, gtscript.IK],
           zl           : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        dz = zl[0,0] - zl[0,-1]
        tem = 0.5 * xlamue[0,-1] * dz
        factor = 1.0 + tem
        if pcnvflg[0,0] and mask[0,0] <= kpbl[0,0]:
            qcko_ntke = ((1.0-tem)*qcko_ntke[0,-1] + tem*(tke[0,0]+tke[0,-1]))/factor

@gtscript.stencil(backend=backend)
def part11(ad          : gtscript.Field[F_TYPE, gtscript.IK],
           f1          : gtscript.Field[F_TYPE, gtscript.IK],
           krad        : gtscript.Field[I_TYPE, gtscript.IK],
           mask        : gtscript.Field[I_TYPE, gtscript.IK],
           mrad        : gtscript.Field[I_TYPE, gtscript.IK],
           qcdo_ntke   : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg      : gtscript.Field[B_TYPE, gtscript.IK],
           tke         : gtscript.Field[F_TYPE, gtscript.IK],
           xlamde      : gtscript.Field[F_TYPE, gtscript.IK],
           zl          : gtscript.Field[F_TYPE, gtscript.IK]):
    with computation(BACKWARD), interval(...):
        dz = zl[0,1] - zl[0,0]
        tem = 0.5 * xlamde[0,0] * dz
        factor = 1.0 + tem
        if scuflg[0,0] and mask[0,0] < krad[0,0] and mask[0,0] >= mrad[0,0]:
            qcdo_ntke = ((1.0-tem)*qcdo_ntke[0,1] + tem*(tke[0,0]+tke[0,1]))/factor

    with computation(PARALLEL), interval(0,1):
        ad = 1.0
        f1 = tke[0,0]

@gtscript.stencil(backend=backend)
def part12(ad        : gtscript.Field[F_TYPE, gtscript.IK],
           ad_p1     : gtscript.Field[F_TYPE, gtscript.IK],
           al        : gtscript.Field[F_TYPE, gtscript.IK],
           au        : gtscript.Field[F_TYPE, gtscript.IK],
           del_      : gtscript.Field[F_TYPE, gtscript.IK],
           dkq       : gtscript.Field[F_TYPE, gtscript.IK],
           f1        : gtscript.Field[F_TYPE, gtscript.IK],
           f1_p1     : gtscript.Field[F_TYPE, gtscript.IK],
           kpbl      : gtscript.Field[I_TYPE, gtscript.IK],
           krad      : gtscript.Field[I_TYPE, gtscript.IK],
           mask      : gtscript.Field[I_TYPE, gtscript.IK],
           mrad      : gtscript.Field[I_TYPE, gtscript.IK],
           pcnvflg   : gtscript.Field[B_TYPE, gtscript.IK],
           prsl      : gtscript.Field[F_TYPE, gtscript.IK],
           qcdo_ntke : gtscript.Field[F_TYPE, gtscript.IK],
           qcko_ntke : gtscript.Field[F_TYPE, gtscript.IK],
           rdzt      : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg    : gtscript.Field[B_TYPE, gtscript.IK],
           tke       : gtscript.Field[F_TYPE, gtscript.IK],
           xmf       : gtscript.Field[F_TYPE, gtscript.IK],
           xmfd      : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           dt2       : float):
    
    with computation(FORWARD):
        with interval(0,1):
            dtodsd = dt2/del_[0,0]
            dtodsu = dt2/del_[0,1]
            dsig = prsl[0,0] - prsl[0,1]
            rdz = rdzt[0,0]
            dsdz2 = dsig * dkq[0,0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0,0] - au[0,0]
            ad_p1 = 1.0 - al[0,0]
            tem2 = dsig * rdz

            if pcnvflg[0,0] and mask[0,0] < kpbl[0,0]:
                ptem = 0.5 * tem2 * xmf[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tke[0,0] + tke[0,1]
                ptem = qcko_ntke[0,0] + qcko_ntke[0,1]
                f1 = f1[0,0] - (ptem - tem) * ptem1
                f1_p1 = tke[0,1] + (ptem - tem) * ptem2
            else:
                f1_p1 = tke[0,1]
            
            if scuflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                ptem = 0.5 * tem2 * xmfd[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tke[0,0] + tke[0,1]
                ptem = qcdo_ntke[0,0] + qcdo_ntke[0,1]
                f1 = f1[0,0] + (ptem - tem) * ptem1
                f1_p1 = f1_p1[0,0] - (ptem - tem) * ptem2
        with interval(1,-1):
            ad = ad_p1[0,-1]
            f1 = f1_p1[0,-1]

            dtodsd = dt2/del_[0,0]
            dtodsu = dt2/del_[0,1]
            dsig = prsl[0,0] - prsl[0,1]
            rdz = rdzt[0,0]
            tem1 = dsig * dkq[0,0] * rdz
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0,0] - au[0,0]
            ad_p1 = 1.0 - al[0,0]
            tem2 = dsig * rdz

            if pcnvflg[0,0] and mask[0,0] < kpbl[0,0]:
                ptem = 0.5 * tem2 * xmf[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tke[0,0] + tke[0,1]
                ptem = qcko_ntke[0,0] + qcko_ntke[0,1]
                f1 = f1[0,0] - (ptem - tem) * ptem1
                f1_p1 = tke[0,1] + (ptem - tem) * ptem2
            else:
                f1_p1 = tke[0,1]
            
            if scuflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                ptem = 0.5 * tem2 * xmfd[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tke[0,0] + tke[0,1]
                ptem = qcdo_ntke[0,0] + qcdo_ntke[0,1]
                f1 = f1[0,0] + (ptem - tem) * ptem1
                f1_p1 = f1_p1[0,0] - (ptem - tem) * ptem2

        with interval(-1,None):
            ad = ad_p1[0,-1]
            f1 = f1_p1[0,-1]

@gtscript.stencil(backend=backend)
def part13(ad        : gtscript.Field[F_TYPE, gtscript.IK],
           ad_p1     : gtscript.Field[F_TYPE, gtscript.IK],
           al        : gtscript.Field[F_TYPE, gtscript.IK],
           au        : gtscript.Field[F_TYPE, gtscript.IK],
           del_      : gtscript.Field[F_TYPE, gtscript.IK],
           dkt       : gtscript.Field[F_TYPE, gtscript.IK],
           f1        : gtscript.Field[F_TYPE, gtscript.IK],
           f1_p1     : gtscript.Field[F_TYPE, gtscript.IK],
           f2        : gtscript.Field[F_TYPE, gtscript.IK],
           f2_p1     : gtscript.Field[F_TYPE, gtscript.IK],
           kpbl      : gtscript.Field[I_TYPE, gtscript.IK],
           krad      : gtscript.Field[I_TYPE, gtscript.IK],
           mask      : gtscript.Field[I_TYPE, gtscript.IK],
           mrad      : gtscript.Field[I_TYPE, gtscript.IK],
           pcnvflg   : gtscript.Field[B_TYPE, gtscript.IK],
           prsl      : gtscript.Field[F_TYPE, gtscript.IK],
           q1        : gtscript.Field[F_TYPE, gtscript.IK],  # q1(:,:,1)
           qcdo      : gtscript.Field[F_TYPE, gtscript.IK],  # qcdo(:,:,1)
           qcko      : gtscript.Field[F_TYPE, gtscript.IK],  # qcko(:,:,1)
           rdzt      : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg    : gtscript.Field[B_TYPE, gtscript.IK],
           tcdo      : gtscript.Field[F_TYPE, gtscript.IK],
           tcko      : gtscript.Field[F_TYPE, gtscript.IK],
           t1        : gtscript.Field[F_TYPE, gtscript.IK],
           xmf       : gtscript.Field[F_TYPE, gtscript.IK],
           xmfd      : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           dt2       : float,
           gocp      : float):

    with computation(FORWARD):
        with interval(0,-1):
            if mask[0,0] > 0:
                f1 = f1_p1[0,-1]
                f2 = f2_p1[0,-1]
                ad = ad_p1[0,-1]

            dtodsd = dt2/del_[0,0]
            dtodsu = dt2/del_[0,1]
            dsig   = prsl[0,0] - prsl[0,1]
            rdz    = rdzt[0,0]
            tem1   = dsig * dkt[0,0] * rdz
            dsdzt  = tem1 * gocp
            dsdz2  = tem1 * rdz
            au     = -dtodsd*dsdz2
            al     = -dtodsu*dsdz2
            ad     = ad[0,0] - au[0,0]
            ad_p1  = 1.0 - al[0,0]
            tem2   = dsig * rdz
            
            if pcnvflg[0,0] and mask[0,0] < kpbl[0,0]:
                ptem  = 0.5 * tem2 * xmf[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem   = t1[0,0] + t1[0,1]
                ptem  = tcko[0,0] + tcko[0,1]
                f1    = f1[0,0] + dtodsd * dsdzt - (ptem - tem) * ptem1
                f1_p1 = t1[0,1] - dtodsu * dsdzt + (ptem - tem) * ptem2
                tem   = q1[0,0] + q1[0,1]
                ptem  = qcko[0,0] + qcko[0,1]
                f2    = f2[0,0] - (ptem - tem) * ptem1
                f2_p1 = q1[0,1] + (ptem - tem) * ptem2
            else:
                f1    = f1[0,0] + dtodsd * dsdzt
                f1_p1 = t1[0,1] - dtodsu * dsdzt
                f2_p1 = q1[0,1]

            if scuflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                ptem  = 0.5 * tem2 * xmfd[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                ptem  = tcdo[0,0] + tcdo[0,1]
                tem   = t1[0,0] + t1[0,1]
                f1    = f1[0,0] + (ptem - tem) * ptem1
                f1_p1 = f1_p1[0,0] - (ptem - tem) * ptem2
                tem   = q1[0,0] + q1[0,1]
                ptem  = qcdo[0,0] + qcdo[0,1]
                f2    = f2[0,0] + (ptem - tem) * ptem1
                f2_p1 = f2_p1[0,0] - (ptem - tem) * ptem2
        with interval(-1,None):
            f1 = f1_p1[0,-1]
            f2 = f2_p1[0,-1]
            ad = ad_p1[0,-1]

@gtscript.stencil(backend=backend)
def part14(ad       : gtscript.Field[F_TYPE, gtscript.IK],
           ad_p1    : gtscript.Field[F_TYPE, gtscript.IK],
           al       : gtscript.Field[F_TYPE, gtscript.IK],
           au       : gtscript.Field[F_TYPE, gtscript.IK],
           del_     : gtscript.Field[F_TYPE, gtscript.IK],
           diss     : gtscript.Field[F_TYPE, gtscript.IK],
           dku      : gtscript.Field[F_TYPE, gtscript.IK],
           dtdz1    : gtscript.Field[F_TYPE, gtscript.IK],
           f1       : gtscript.Field[F_TYPE, gtscript.IK],
           f1_p1    : gtscript.Field[F_TYPE, gtscript.IK],
           f2       : gtscript.Field[F_TYPE, gtscript.IK],
           f2_p1    : gtscript.Field[F_TYPE, gtscript.IK],
           kpbl     : gtscript.Field[I_TYPE, gtscript.IK],
           krad     : gtscript.Field[I_TYPE, gtscript.IK],
           mask     : gtscript.Field[I_TYPE, gtscript.IK],
           mrad     : gtscript.Field[I_TYPE, gtscript.IK],
           pcnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
           prsl     : gtscript.Field[F_TYPE, gtscript.IK],
           rdzt     : gtscript.Field[F_TYPE, gtscript.IK],
           scuflg   : gtscript.Field[B_TYPE, gtscript.IK],
           spd1     : gtscript.Field[F_TYPE, gtscript.IK],
           stress   : gtscript.Field[F_TYPE, gtscript.IK],
           tdt      : gtscript.Field[F_TYPE, gtscript.IK],
           u1       : gtscript.Field[F_TYPE, gtscript.IK],
           ucdo     : gtscript.Field[F_TYPE, gtscript.IK],
           ucko     : gtscript.Field[F_TYPE, gtscript.IK],
           v1       : gtscript.Field[F_TYPE, gtscript.IK],
           vcdo     : gtscript.Field[F_TYPE, gtscript.IK],
           vcko     : gtscript.Field[F_TYPE, gtscript.IK],
           xmf      : gtscript.Field[F_TYPE, gtscript.IK],
           xmfd     : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           dspheat  : bool,
           dt2      : float):

    with computation(PARALLEL):
        with interval(0,-1):
            if dspheat:
                tdt   = tdt[0,0] + dspfac * (diss[0,0]/cp)

        with interval(0,1):
            ad = 1.0 + dtdz1[0,0] * stress[0,0] / spd1[0,0]
            f1 = u1[0,0]
            f2 = v1[0,0]

    with computation(FORWARD):
        with interval(0,-1):
            if mask[0,0] > 0:
                f1 = f1_p1[0,-1]
                f2 = f2_p1[0,-1]
                ad = ad_p1[0,-1]
            
            dtodsd = dt2/del_[0,0]
            dtodsu = dt2/del_[0,1]
            dsig   = prsl[0,0] - prsl[0,1]
            rdz    = rdzt[0,0]
            tem1   = dsig * dku[0,0] * rdz
            dsdz2  = tem1 * rdz
            au     = -dtodsd * dsdz2
            al     = -dtodsu * dsdz2
            ad     = ad[0,0] - au[0,0]
            ad_p1  = 1.0 - al[0,0]
            tem2   = dsig * rdz

            if pcnvflg[0,0] and mask[0,0] < kpbl[0,0]:
                ptem  = 0.5 * tem2 * xmf[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem   = u1[0,0] + u1[0,1]
                ptem  = ucko[0,0] + ucko[0,1]
                f1    = f1[0,0] - (ptem - tem) * ptem1
                f1_p1 = u1[0,1] + (ptem - tem) * ptem2
                tem   = v1[0,0] + v1[0,1]
                ptem  = vcko[0,0] + vcko[0,1]
                f2    = f2[0,0] - (ptem - tem) * ptem1
                f2_p1 = v1[0,1] + (ptem - tem) * ptem2
            else:
                f1_p1 = u1[0,1]
                f2_p1 = v1[0,1]

            if scuflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                ptem  = 0.5 * tem2 * xmfd[0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem   = u1[0,0] + u1[0,1]
                ptem  = ucdo[0,0] + ucdo[0,1]
                f1    = f1[0,0] + (ptem - tem) * ptem1
                f1_p1 = f1_p1[0,0] - (ptem - tem) * ptem2
                tem   = v1[0,0] + v1[0,1]
                ptem  = vcdo[0,0] + vcdo[0,1]
                f2    = f2[0,0] + (ptem - tem) * ptem1
                f2_p1 = f2_p1[0,0] - (ptem - tem) * ptem2

        with interval(-1,None):
            f1 = f1_p1[0,-1]
            f2 = f2_p1[0,-1]
            ad = ad_p1[0,-1]

@gtscript.stencil(backend=backend)
def part15(del_     : gtscript.Field[F_TYPE, gtscript.IK],
           du       : gtscript.Field[F_TYPE, gtscript.IK],
           dusfc    : gtscript.Field[F_TYPE, gtscript.IK],
           dv       : gtscript.Field[F_TYPE, gtscript.IK],
           dvsfc    : gtscript.Field[F_TYPE, gtscript.IK],
           f1       : gtscript.Field[F_TYPE, gtscript.IK],
           f2       : gtscript.Field[F_TYPE, gtscript.IK],
           hpbl     : gtscript.Field[F_TYPE, gtscript.IK],
           hpblx    : gtscript.Field[F_TYPE, gtscript.IK],
           kpbl     : gtscript.Field[I_TYPE, gtscript.IK],
           kpblx    : gtscript.Field[I_TYPE, gtscript.IK],
           mask     : gtscript.Field[I_TYPE, gtscript.IK],
           u1       : gtscript.Field[F_TYPE, gtscript.IK],
           v1       : gtscript.Field[F_TYPE, gtscript.IK],
           *,
           conw     : float,
           rdt      : float):

    with computation(PARALLEL), interval(...):
        utend = (f1[0,0] - u1[0,0]) * rdt
        vtend = (f2[0,0] - v1[0,0]) * rdt
        du    = du[0,0] + utend
        dv    = dv[0,0] + vtend
        dusfc = dusfc[0,0] + conw * del_[0,0] * utend
        dvsfc = dvsfc[0,0] + conw * del_[0,0] * vtend

    with computation(BACKWARD), interval(0,-1):
        dusfc = dusfc[0,0] + dusfc[0,1]
        dvsfc = dvsfc[0,0] + dvsfc[0,1]

    with computation(PARALLEL), interval(0,1):
        hpbl = hpblx[0,0]
        kpbl = kpblx[0,0]

def mfpblt(im,ix,km,kmpbl,ntcw,ntrac1,delt, 
           cnvflg,zl,zm,q1,q1_0,q1_ntcw,t1,u1,v1,plyr,pix,thlx,thvx,
           gdx,hpbl,kpbl,vpert,buo,xmf, 
           tcko,qcko,ucko,vcko,xlamue,
           g, gocp, elocp, el2orc, mask,
           compare_dict):

    ce0 = 0.4
    cm = 1.0
    qmin = 1e-8
    qlmin = 1e-12
    alp = 1.0
    pgcon = 0.55
    a1 = 0.13
    b1 = 0.5
    f1 = 0.15
    fv = 4.6150e+2 / 2.8705e+2 - 1.0
    eps = 2.8705e+2 / 4.6150e+2
    epsm1 = eps - 1.0

    wu2        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qtu        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qtx        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    xlamuem    = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    thlu       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qtu        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    thlu       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    kpblx      = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    kpbly      = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    rbup       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    rbdn       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    flg        = gt_storage.zeros(backend=backend, dtype=B_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    hpblx      = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    xlamavg    = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    sumx       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    sigma      = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    scaldfunc  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcko_1     = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcko_ntcw  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcko_track = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))

    totflag = True

    for i in range(im):
        totflag = totflag and ~cnvflg[i,0]

    if totflag:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

    
    mfpblt_s0(alp=alp,
              buo=buo,
              cnvflg=cnvflg,
              g=g,
              hpbl=hpbl,
              kpbl=kpbl,
              q1_0=q1_0,
              q1_ntcw=q1_ntcw,
              qtu=qtu,
              qtx=qtx,
              thlu=thlu,
              thlx=thlx,
              thvx=thvx,
              vpert=vpert,
              wu2=wu2,
              domain=(im,1,km+1))

    mfpblt_s1(buo=buo,
              ce0=ce0,
              cm=cm,
              cnvflg=cnvflg,
              elocp=elocp,
              el2orc=el2orc,
              eps=eps,
              epsm1=epsm1,
              flg=flg,
              fv=fv,
              g=g,
              hpbl=hpbl,
              kpbl=kpbl,
              kpblx=kpblx,
              kpbly=kpbly,
              mask=mask,
              pix=pix,
              plyr=plyr,
              qtu=qtu,
              qtx=qtx,
              rbdn=rbdn,
              rbup=rbup,
              thlu=thlu,
              thlx=thlx,
              thvx=thvx,
              wu2=wu2,
              xlamue=xlamue,
              xlamuem=xlamuem,
              zl=zl,
              zm=zm,
              domain=(im,1,kmpbl))

    mfpblt_s1a(cnvflg=cnvflg,
               hpblx=hpblx,
               kpblx=kpblx,
               mask=mask,
               rbdn=rbdn,
               rbup=rbup,
               zm=zm,
               domain=(im,1,km))

    mfpblt_s2(a1=a1,
              ce0=ce0,
              cm=cm,
              cnvflg=cnvflg,
              dt2=delt,
              el2orc=el2orc,
              elocp=elocp,
              eps=eps,
              epsm1=epsm1,
              gdx=gdx,
              hpbl=hpbl,
              hpblx=hpblx,
              kpbl=kpbl,
              kpblx=kpblx,
              kpbly=kpbly,
              mask=mask,
              pgcon=pgcon,
              pix=pix,
              plyr=plyr,
              qcko_1=qcko_1,
              qcko_ntcw=qcko_ntcw,
              qcko_track=qcko_track,
              qtu=qtu,
              qtx=qtx,
              scaldfunc=scaldfunc,
              sigma=sigma,
              sumx=sumx,
              tcko=tcko,
              thlu=thlu,
              thlx=thlx,
              u1=u1,
              ucko=ucko,
              v1=v1,
              vcko=vcko,
              xlamue=xlamue,
              xlamuem=xlamuem,
              xlamavg=xlamavg,
              xmf=xmf,
              wu2=wu2,
              zl=zl,
              zm=zm,
              domain=(im,1,kmpbl))

    for k in range(1,kmpbl):
        for i in range(im):
            if qcko_track[i,k] == 1:
                qcko[i,k,0] = qcko_1[i,k]
                qcko[i,k,ntcw-1] = qcko_ntcw[i,k]

    if ntcw > 2:
        for n in range(1,ntcw-1):
            for k in range(1,kmpbl):
                for i in range(im):
                    if cnvflg[i,0] and k <= kpbl[i,0]:
                        dz = zl[i,k] - zl[i,k-1]
                        tem = 0.5 * xlamue[i,k-1] * dz
                        factor = 1.0 + tem
                        qcko[i,k,n] = ((1.0-tem) * qcko[i,k-1,n] + tem*(q1[i,k,n]+q1[i,k-1,n]))/factor

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw,ntrac1):
            for k in range(1,kmpbl):
                for i in range(im):
                    if cnvflg[i,0] and k <= kpbl[i,0]:
                        dz = zl[i,k] - zl[i,k-1]
                        tem = 0.5 * xlamue[i,k-1]*dz
                        factor = 1.0 + tem

                        qcko[i,k,n] = ((1.0-tem) * qcko[i,k-1,n] + tem*(q1[i,k,n]+q1[i,k-1,n]))/factor

    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

@gtscript.stencil(backend=backend)
def mfpblt_s0(buo     : gtscript.Field[F_TYPE, gtscript.IK],
              cnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
              hpbl    : gtscript.Field[F_TYPE, gtscript.IK],
              kpbl    : gtscript.Field[I_TYPE, gtscript.IK],
              q1_0    : gtscript.Field[F_TYPE, gtscript.IK],
              q1_ntcw : gtscript.Field[F_TYPE, gtscript.IK],
              qtu     : gtscript.Field[F_TYPE, gtscript.IK],
              qtx     : gtscript.Field[F_TYPE, gtscript.IK],
              thlu    : gtscript.Field[F_TYPE, gtscript.IK],
              thlx    : gtscript.Field[F_TYPE, gtscript.IK],
              thvx    : gtscript.Field[F_TYPE, gtscript.IK],
              vpert   : gtscript.Field[F_TYPE, gtscript.IK],
              wu2     : gtscript.Field[F_TYPE, gtscript.IK],
              *,
              alp     : float,
              g       : float
              ):

    with computation(PARALLEL), interval(0,-1):
        if cnvflg[0,0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1_0[0,0] + q1_ntcw[0,0]

    with computation(PARALLEL), interval(0,1):
        ptem = alp * vpert[0,0]
        ptem = min(ptem,3.0)
        if cnvflg[0,0]:
            thlu = thlx[0,0] + ptem
            qtu  = qtx[0,0]
            buo  = g * ptem / thvx[0,0]

    # CK : This may not be needed later if stencils previous to this one update
    #       hpbl and kpbl over its entire range
    with computation(FORWARD), interval(1,None):
        hpbl = hpbl[0,-1]
        kpbl = kpbl[0,-1]

@gtscript.stencil(backend=backend)
def mfpblt_s1(buo     : gtscript.Field[F_TYPE, gtscript.IK],
              cnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
              flg     : gtscript.Field[B_TYPE, gtscript.IK],
              hpbl    : gtscript.Field[F_TYPE, gtscript.IK],
              kpbl    : gtscript.Field[I_TYPE, gtscript.IK],
              kpblx   : gtscript.Field[I_TYPE, gtscript.IK],
              kpbly   : gtscript.Field[I_TYPE, gtscript.IK],
              mask    : gtscript.Field[I_TYPE, gtscript.IK],
              pix     : gtscript.Field[F_TYPE, gtscript.IK],
              plyr    : gtscript.Field[F_TYPE, gtscript.IK],
              qtu     : gtscript.Field[F_TYPE, gtscript.IK],
              qtx     : gtscript.Field[F_TYPE, gtscript.IK],
              rbdn    : gtscript.Field[F_TYPE, gtscript.IK],
              rbup    : gtscript.Field[F_TYPE, gtscript.IK],
              thlu    : gtscript.Field[F_TYPE, gtscript.IK],
              thlx    : gtscript.Field[F_TYPE, gtscript.IK],
              thvx    : gtscript.Field[F_TYPE, gtscript.IK],
              wu2     : gtscript.Field[F_TYPE, gtscript.IK],
              xlamue  : gtscript.Field[F_TYPE, gtscript.IK],
              xlamuem : gtscript.Field[F_TYPE, gtscript.IK],
              zl      : gtscript.Field[F_TYPE, gtscript.IK],
              zm      : gtscript.Field[F_TYPE, gtscript.IK],
              *,
              ce0     : float,
              cm      : float,
              el2orc  : float,
              elocp   : float,
              eps     : float,
              epsm1   : float,
              fv      : float,
              g       : float):
    with computation(PARALLEL), interval(...):
        dz = zl[0,1] - zl[0,0]
        ptem = 1.0/(zm[0,0] + dz)
        tem = max(hpbl[0,0]-zm[0,0]+dz, dz)
        ptem1 = 1.0/tem
        if cnvflg[0,0]:
            if mask[0,0] < kpbl[0,0]:
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0,0]

    with computation(FORWARD):
        with interval(0,1):
            tem1 = 4.0 * buo[0,0] * zm[0,0]
            ptem1 = 1.0 + (0.5 * 2.0 * xlamue[0,0] * zm[0,0])
            if cnvflg[0,0]:
                wu2 = tem1 / ptem1

        with interval(1,None):
            dz = zl[0,0] - zl[0,-1]
            tem = 0.5 * xlamue[0,-1] * dz
            factor = 1.0 + tem
            if cnvflg[0,0]:
                thlu = ((1.0 - tem) * thlu[0,-1] + tem * (thlx[0,-1] + thlx[0,0]))/factor
                qtu  = ((1.0 - tem) * qtu[0,-1]  + tem * (qtx[0,-1] + qtx[0,0])) / factor

            tlu = thlu[0,0] / pix[0,0]
            es  = 0.01 * fpvs(tlu)
            qs = max(qmin, eps * es / (plyr[0,0]+ epsm1*es))
            dq = qtu[0,0] - qs
            gamma = el2orc * qs / (tlu**2)
            qlu = dq / (1.0 + gamma)
            thvu = 0.0
            tem1 = 0.0
            if cnvflg[0,0]:
                if dq > 0.0:
                    qtu = qs + qlu
                    tem1 = 1.0 + fv * qs - qlu
                    thvu = (thlu[0,0] + pix[0,0] * elocp * qlu) * tem1
                else:
                    tem1 = 1.0 + fv * qtu[0,0]
                    thvu = thlu[0,0] * tem1
                buo = g * (thvu / thvx[0,0] - 1.0)

    with computation(FORWARD):
        with interval(0,1):
            flg = 1
            kpbly = kpbl[0,0]
            if cnvflg[0,0]:
                flg = 0
                rbup = wu2[0,0]
        with interval(1,None):
            dz = zm[0,0] - zm[0,-1]
            tem = 0.25 * 2.0 * (xlamue[0,0] + xlamue[0,-1]) * dz
            tem1 = 4.0 * buo[0,0] * dz
            ptem = (1.0 - tem) * wu2[0,-1]
            ptem1 = 1.0 + tem
            if cnvflg[0,0]:
                wu2 = (ptem + tem1) / ptem1

    with computation(FORWARD), interval(1,None):
        kpblx = kpblx[0,-1]
        flg = flg[0,-1]
        rbup = rbup[0,-1]
        rbdn = rbdn[0,-1]
        if flg[0,0] == 0:
            rbdn = rbup[0,0]
            rbup = wu2[0,0]
            kpblx = mask[0,0]
            flg  = rbup[0,0] <= 0.0
    
    with computation(BACKWARD), interval(0,-1):
        rbup = rbup[0,1]
        rbdn = rbdn[0,1]
        kpblx = kpblx[0,1]
        flg   = flg[0,1]

@gtscript.stencil(backend=backend)
def mfpblt_s1a(cnvflg      : gtscript.Field[B_TYPE, gtscript.IK],
               hpblx       : gtscript.Field[F_TYPE, gtscript.IK],
               kpblx       : gtscript.Field[I_TYPE, gtscript.IK],
               mask        : gtscript.Field[I_TYPE, gtscript.IK],
               rbdn        : gtscript.Field[F_TYPE, gtscript.IK],
               rbup        : gtscript.Field[F_TYPE, gtscript.IK],
               zm          : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(...):
        if mask[0,0] > 0:
            hpblx = hpblx[0,-1]
            rbdn  = rbdn[0,-1]
            rbup  = rbup[0,-1]
            cnvflg = cnvflg[0,-1]

        rbint = 0.0

        if mask[0,0] == kpblx[0,0]:
            if cnvflg[0,0]:
                if rbdn[0,0] <= 0.0:
                    rbint = 0.0
                elif rbup[0,0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0,0] / (rbdn[0,0] - rbup[0,0])

                hpblx = zm[0,-1] + rbint * (zm[0,0] - zm[0,-1])

    with computation(BACKWARD), interval(0,-1):
        hpblx = hpblx[0,1]

@gtscript.stencil(backend=backend)
def mfpblt_s2(cnvflg : gtscript.Field[B_TYPE, gtscript.IK],
              gdx    : gtscript.Field[F_TYPE, gtscript.IK],
              hpbl   : gtscript.Field[F_TYPE, gtscript.IK],
              hpblx  : gtscript.Field[F_TYPE, gtscript.IK],
              kpbl   : gtscript.Field[I_TYPE, gtscript.IK],
              kpblx  : gtscript.Field[I_TYPE, gtscript.IK],
              kpbly  : gtscript.Field[I_TYPE, gtscript.IK],
              mask   : gtscript.Field[I_TYPE, gtscript.IK],
              pix    : gtscript.Field[F_TYPE, gtscript.IK],
              plyr   : gtscript.Field[F_TYPE, gtscript.IK],
              qcko_1 : gtscript.Field[F_TYPE, gtscript.IK],
              qcko_ntcw: gtscript.Field[F_TYPE, gtscript.IK],
              qcko_track: gtscript.Field[I_TYPE, gtscript.IK],
              qtu    : gtscript.Field[F_TYPE, gtscript.IK],
              qtx    : gtscript.Field[F_TYPE, gtscript.IK],
              scaldfunc: gtscript.Field[F_TYPE, gtscript.IK],
              sigma  : gtscript.Field[F_TYPE, gtscript.IK],
              sumx   : gtscript.Field[F_TYPE, gtscript.IK],
              tcko   : gtscript.Field[F_TYPE, gtscript.IK],
              thlu   : gtscript.Field[F_TYPE, gtscript.IK],
              thlx   : gtscript.Field[F_TYPE, gtscript.IK],
              u1     : gtscript.Field[F_TYPE, gtscript.IK],
              ucko   : gtscript.Field[F_TYPE, gtscript.IK],
              v1     : gtscript.Field[F_TYPE, gtscript.IK],
              vcko   : gtscript.Field[F_TYPE, gtscript.IK],
              xmf    : gtscript.Field[F_TYPE, gtscript.IK],
              xlamavg: gtscript.Field[F_TYPE, gtscript.IK],
              xlamue : gtscript.Field[F_TYPE, gtscript.IK],
              xlamuem: gtscript.Field[F_TYPE, gtscript.IK],
              wu2    : gtscript.Field[F_TYPE, gtscript.IK],
              zl     : gtscript.Field[F_TYPE, gtscript.IK],
              zm     : gtscript.Field[F_TYPE, gtscript.IK],
              *,
              a1     : float,
              dt2    : float,
              ce0    : float,
              cm     : float,
              el2orc : float,
              elocp  : float,
              eps    : float,
              epsm1  : float,
              pgcon  : float):

    with computation(FORWARD):
        with interval(0,1):
            if cnvflg[0,0]:
                if kpbl[0,0] > kpblx[0,0]:
                    kpbl = kpblx[0,0]
                    hpbl = hpblx[0,0]
        with interval(1,None):
            kpbly = kpbly[0,-1]
            kpbl  = kpbl[0,-1]
            hpbl  = hpbl[0,-1]

    with computation(PARALLEL), interval(...):
        dz = zl[0,1] - zl[0,0]
        ptem = 1 / (zm[0,0]+dz)
        tem = max(hpbl[0,0]-zm[0,0]+dz, dz)
        ptem1 = 1/tem
        if cnvflg[0,0] and (kpbly[0,0] > kpblx[0,0]):            
            if mask[0,0] < kpbl[0,0]:
                xlamue = ce0 * (ptem+ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0,0]

    with computation(FORWARD):
        with interval(0,1):
            dz = zl[0,1] - zl[0,0]
            if cnvflg[0,0] and (mask[0,0] < kpbl[0,0]):
                xlamavg = xlamavg[0,0] + xlamue[0,0] * dz
                sumx = sumx[0,0] + dz
        with interval(1,None):
            xlamavg = xlamavg[0,-1]
            sumx    = sumx[0,-1]
            dz = zl[0,1] - zl[0,0]
            if cnvflg[0,0] and (mask[0,0] < kpbl[0,0]):
                xlamavg = xlamavg[0,0] + xlamue[0,0] * dz
                sumx = sumx[0,0] + dz

    with computation(BACKWARD), interval(0,-1):
        xlamavg = xlamavg[0,1]
        sumx    = sumx[0,1]
            
    with computation(PARALLEL), interval(0,1):
        if cnvflg[0,0]:
            xlamavg = xlamavg[0,0] / sumx[0,0]

    with computation(PARALLEL), interval(...):
        tem = sqrt(wu2[0,0])
        if cnvflg[0,0] and (mask[0,0] < kpbl[0,0]):
            if wu2[0,0] > 0.0:
                xmf = a1 * tem
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0,1):
            tem = 0.2 / xlamavg[0,0]
            tem1 = 3.14 * tem * tem
            tem1 = tem1 / (gdx[0,0] * gdx[0,0])
            tem1 = max(tem1, 0.001)
            tem1 = min(tem1, 0.999)
            if cnvflg[0,0]:
                sigma = tem1
            tem1 = (1.0 - sigma[0,0]) * (1.0 - sigma[0,0])
            tem1 = min(tem1, 1.0)
            tem1 = max(tem1, 0.0)
            if cnvflg[0,0]:
                if sigma[0,0] > a1:
                    scaldfunc = tem1
                else:
                    scaldfunc = 1.0
        with interval(1,None):
            scaldfunc = scaldfunc[0,-1]

    with computation(PARALLEL), interval(...):
        tem1 = scaldfunc[0,0] * xmf[0,0]
        xmmx = (zl[0,1] - zl[0,0]) / dt2
        tem1 = min(tem1, xmmx)
        if cnvflg[0,0] and (mask[0,0] < kpbl[0,0]):
            xmf = tem1
    
    with computation(FORWARD): 
        with interval(0,1):
            if cnvflg[0,0]:
                thlu = thlx[0,0]
        with interval(1,None):
            dz = zl[0,0] - zl[0,-1]
            tem = 0.5 * xlamue[0,-1] * dz
            factor = 1.0 + tem

            if cnvflg[0,0] and (mask[0,0] <= kpbl[0,0]):
                thlu = ((1.0-tem) * thlu[0,-1] + tem*(thlx[0,-1]+thlx[0,0]))/factor
                qtu  = ((1.0-tem) * qtu[0,-1]  + tem*(qtx[0,-1]+qtx[0,0]))/factor
            
            tlu = thlu[0,0] / pix[0,0]
            es = 0.01 * fpvs(tlu)
            qs = max(qmin, eps * es / (plyr[0,0]+epsm1*es))
            dq = qtu[0,0] - qs
            qlu = dq / (1.0 + (el2orc * qs / (tlu**2)))

            if cnvflg[0,0] and (mask[0,0] <= kpbl[0,0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko_1 = qs
                    qcko_ntcw = qlu
                    tcko = tlu + elocp * qlu
                    qcko_track = 1
                else:
                    qcko_1 = qtu[0,0]
                    qcko_ntcw = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0,-1] * dz
            factor = 1.0 + tem
            ptem = tem + pgcon
            ptem1 = tem - pgcon

            if cnvflg[0,0] and (mask[0,0] <= kpbl[0,0]):
                ucko = ((1.0-tem) * ucko[0,-1] + ptem*u1[0,0] + ptem1*u1[0,-1])/factor
                vcko = ((1.0-tem) * vcko[0,-1] + ptem*v1[0,0] + ptem1*v1[0,-1])/factor

def mfscu(im,ix,km,kmscu,ntcw,ntrac1,delt,
          cnvflg,zl,zm,q1,q1_1, q1_ntcw,t1,u1,v1,plyr,pix,
          thlx,thvx,thlvx,gdx,thetae,radj,
          krad,mrad,radmin,buo,xmfd,
          tcdo,qcdo,ucdo,vcdo,xlamde,
          g, gocp, elocp, el2orc, mask,
          compare_dict):
    
    ce0 = 0.4
    cm = 1.0
    pgcon = 0.55
    qmin = 1e-8
    qlmin = 1e-12
    b1 = 0.45
    f1 = 0.15
    a1 = 0.12
    a2 = 0.50
    a11 = 0.2
    a22 = 1.0
    cldtime = 500.0
    actei = 0.7
    hvap = 2.5000e+6
    cp = 1.0046e+3
    eps = 2.8705e+2/4.6150e+2
    epsm1 = eps - 1.0
    fv = 4.6150e+2 / 2.8705e+2 - 1.0

    wd2        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qtx        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    hrad       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    krad1      = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    thld       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qtd        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    thlvd      = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    ra1        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    ra2        = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    radj       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    flg        = gt_storage.zeros(backend=backend, dtype=B_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    xlamdem    = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    mradx      = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    mrady      = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    sumx       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    xlamavg    = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    xmfd       = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    sigma      = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    scaldfunc  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcdo_1     = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcdo_ntcw  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))
    qcdo_track = gt_storage.zeros(backend=backend, dtype=I_TYPE, shape=(im,km+1), default_origin=(0,0),mask=(True,False,True))

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i,0]
    
    if totflg:
        return
    
    mfscu_s0(buo=buo,
             cnvflg=cnvflg,
             q1_1=q1_1,
             q1_ntcw=q1_ntcw,
             qtx=qtx,
             wd2=wd2,
             domain=(im,1,km))

    mfscu_s0a(buo=buo,
              cnvflg=cnvflg,
              flg=flg,
              hrad=hrad,
              krad=krad,
              krad1=krad1,
              mask=mask,
              mrad=mrad,
              qtd=qtd,
              qtx=qtx,
              ra1=ra1,
              ra2=ra2,
              radmin=radmin,
              radj=radj,
              thetae=thetae,
              thld=thld,
              thlvd=thlvd,
              thlvx=thlvx,
              thlx=thlx,
              thvx=thvx,
              zm=zm,
              a1=a1,
              a11=a11,
              a2=a2,
              a22=a22,
              actei=actei,
              cldtime=cldtime,
              cp=cp,
              hvap=hvap,
              g=g,
              domain=(im,1,km))

    mfscu_s0b(cnvflg=cnvflg,
              flg=flg,
              krad=krad,
              mask=mask,
              mrad=mrad,
              thlvd=thlvd,
              thlvx=thlvx,
              domain=(im,1,kmscu))

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i,0]

    if totflg:
        return

    for k in range(kmscu):
        for i in range(im):
            if cnvflg[i,0]:
                dz = zl[i,k+1] - zl[i,k]
                if (k >= mrad[i,0]) and (k < krad[i,0]):
                    if mrad[i,0] == 0:
                       ptem = 1.0 / (zm[i,k] + dz)
                    else:
                        ptem = 1.0 / (zm[i,k] - zm[i,mrad[i,0]-1] + dz)
                    tem = max(hrad[i,0] - zm[i,k] + dz, dz)
                    ptem1 = 1.0/tem
                    xlamde[i,k] = ce0 * (ptem+ptem1)
                else:
                    xlamde[i,k] = ce0 / dz
                xlamdem[i,k] = cm * xlamde[i,k]

    mfscu_s1(buo=buo,
             cnvflg=cnvflg,
             el2orc=el2orc,
             elocp=elocp,
             eps=eps,
             epsm1=epsm1,
             fv=fv,
             g=g,
             krad=krad,
             mask=mask,
             pix=pix,
             plyr=plyr,
             radmin=radmin,
             thld=thld,
             thlx=thlx,
             thvx=thvx,
             qtd=qtd,
             qtx=qtx,
             xlamde=xlamde,
             zl=zl,
             domain=(im,1,kmscu))

    bb1 = 2.0
    bb2 = 4.0

    mfscu_s1a(buo=buo,
              cnvflg=cnvflg,
              krad1=krad1,
              mask=mask,
              wd2=wd2,
              xlamde=xlamde,
              zm=zm,
              bb1=bb1,
              bb2=bb2,
              domain=(im,1,km))
    
    mfscu_s2(buo=buo,
             cnvflg=cnvflg,
             flg=flg,
             krad=krad,
             krad1=krad1,
             mask=mask,
             mrad=mrad,
             mradx=mradx,
             mrady=mrady,
             xlamde=xlamde,
             wd2=wd2,
             zm=zm,
             domain=(im,1,kmscu))

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i,0]

    if totflg:
        return

    for k in range(kmscu):
        for i in range(im):
            if cnvflg[i,0] and mrady[i,0] < mradx[i,0]:
                dz = zl[i,k+1] - zl[i,k]
                if (k >= mrad[i,0]) and (k < krad[i,0]):
                    if mrad[i,0] == 0:
                       ptem = 1.0 / (zm[i,k] + dz)
                    else:
                        ptem = 1.0 / (zm[i,k] - zm[i,mrad[i,0]-1] + dz)
                    tem = max(hrad[i,0] - zm[i,k] + dz, dz)
                    ptem1 = 1.0/tem
                    xlamde[i,k] = ce0 * (ptem+ptem1)
                else:
                    xlamde[i,k] = ce0 / dz
                xlamdem[i,k] = cm * xlamde[i,k]    


    mfscu_s3(cnvflg=cnvflg,
             dt2=delt,
             gdx=gdx,
             krad=krad,
             mask=mask,
             mrad=mrad,
             ra1=ra1,
             sigma=sigma,
             scaldfunc=scaldfunc,
             sumx=sumx,
             wd2=wd2,
             xlamde=xlamde,
             xlamavg=xlamavg,
             xmfd=xmfd,
             zl=zl,
             domain=(im,1,kmscu))
    
    mfscu_s3a(cnvflg=cnvflg,
              krad=krad,
              mask=mask,
              thld=thld,
              thlx=thlx,
              domain=(im,1,km))

    qcdo_1[:,:] = qcdo[:,:,0].reshape((im,km+1))
    qcdo_ntcw[:,:] = qcdo[:,:,ntcw-1].reshape((im,km+1))

    mfscu_s4(cnvflg=cnvflg,
             el2orc=el2orc,
             elocp=elocp,
             eps=eps,
             epsm1=epsm1,
             krad=krad,
             mask=mask,
             mrad=mrad,
             pgcon=pgcon,
             pix=pix,
             plyr=plyr,
             qcdo_1=qcdo_1,
             qcdo_ntcw=qcdo_ntcw,
             qcdo_track=qcdo_track,
             qtd=qtd,
             qtx=qtx,
             tcdo=tcdo,
             thld=thld,
             thlx=thlx,
             u1=u1,
             ucdo=ucdo,
             v1=v1,
             vcdo=vcdo,
             xlamde=xlamde,
             xlamdem=xlamdem,
             zl=zl,
             domain=(im,1,kmscu))

    for k in range(kmscu):
        for i in range(im):
            if qcdo_track[i,k] == 1:
                qcdo[i,k,0] = qcdo_1[i,k]
                qcdo[i,k,ntcw-1] = qcdo_ntcw[i,k]

    if ntcw > 2:
        for n in range(1,ntcw-1):
            for k in range(kmscu-1,-1,-1):
                for i in range(im):
                    if cnvflg[i,0,0] and k < krad[i,0] and k >= mrad[i,0]:
                        dz = zl[i,k+1] - zl[i,k]
                        tem = 0.5 * xlamde[i,k] * dz
                        factor = 1.0 + tem
                        qcdo[i,k,n] = ((1.0-tem) * qcdo[i,k+1,n] + tem*(q1[i,k,n]+q1[i,k+1,n]))/factor

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw,ntrac1):
            for k in range(kmscu-1,-1,-1):
                for i in range(im):
                    if cnvflg[i,0] and k < krad[i,0] and k >= mrad[i,0]:
                        dz = zl[i,k+1] - zl[i,k]
                        tem = 0.5 * xlamde[i,k]*dz
                        factor = 1.0 + tem

                        qcdo[i,k,n] = ((1.0-tem) * qcdo[i,k+1,n] + tem*(q1[i,k,n]+q1[i,k+1,n]))/factor

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

@gtscript.stencil(backend=backend)
def mfscu_s0(buo      : gtscript.Field[F_TYPE, gtscript.IK],
             cnvflg   : gtscript.Field[B_TYPE, gtscript.IK],
             q1_1     : gtscript.Field[F_TYPE, gtscript.IK],
             q1_ntcw  : gtscript.Field[F_TYPE, gtscript.IK],
             qtx      : gtscript.Field[F_TYPE, gtscript.IK],
             wd2      : gtscript.Field[F_TYPE, gtscript.IK]):
    
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1_1[0,0] + q1_ntcw[0,0]

@gtscript.stencil(backend=backend)
def mfscu_s0a(buo      : gtscript.Field[F_TYPE, gtscript.IK],
              cnvflg   : gtscript.Field[B_TYPE, gtscript.IK],
              flg      : gtscript.Field[B_TYPE, gtscript.IK],
              hrad     : gtscript.Field[F_TYPE, gtscript.IK],
              krad     : gtscript.Field[I_TYPE, gtscript.IK],
              krad1    : gtscript.Field[I_TYPE, gtscript.IK],
              mask     : gtscript.Field[I_TYPE, gtscript.IK],
              mrad     : gtscript.Field[I_TYPE, gtscript.IK],
              qtd      : gtscript.Field[F_TYPE, gtscript.IK],
              qtx      : gtscript.Field[F_TYPE, gtscript.IK],
              ra1      : gtscript.Field[F_TYPE, gtscript.IK],
              ra2      : gtscript.Field[F_TYPE, gtscript.IK],
              radmin   : gtscript.Field[F_TYPE, gtscript.IK],
              radj     : gtscript.Field[F_TYPE, gtscript.IK],
              thetae   : gtscript.Field[F_TYPE, gtscript.IK],
              thld     : gtscript.Field[F_TYPE, gtscript.IK],
              thlvd    : gtscript.Field[F_TYPE, gtscript.IK],
              thlvx    : gtscript.Field[F_TYPE, gtscript.IK],
              thlx     : gtscript.Field[F_TYPE, gtscript.IK],
              thvx     : gtscript.Field[F_TYPE, gtscript.IK],
              zm       : gtscript.Field[F_TYPE, gtscript.IK],
              *,
              a1       : float,
              a11      : float,
              a2       : float,
              a22      : float,
              actei    : float,
              cldtime  : float,
              cp       : float,
              hvap     : float,
              g        : float):
    
    with computation(FORWARD), interval(...):
        if mask[0,0] > 0:
            hrad   = hrad[0,-1]
            krad   =  krad[0,-1]
            krad1  = krad1[0,-1]
            ra1    = ra1[0,-1]
            ra2    = ra2[0,-1]
            radj   = radj[0,-1]
            cnvflg = cnvflg[0,-1]
            radmin = radmin[0,-1]
            thlvd  = thlvd[0,-1]

        if krad[0,0] == mask[0,0]:
            if cnvflg[0,0]:
                hrad = zm[0,0]
                krad1 = mask[0,0] - 1

                tem1 = cldtime * radmin[0,0]/(zm[0,1] - zm[0,0])
                tem1 = max(tem1, -3.0)
                thld = thlx[0,0] + tem1
                qtd = qtx[0,0]
                thlvd = thlvx[0,0] + tem1
                buo   = -g * tem1 / thvx[0,0]

                ra1 = a1
                ra2 = a11

                tem = thetae[0,0] - thetae[0,1]
                tem1 = qtx[0,0] - qtx[0,1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = cp * tem / (hvap * tem1)
                    if cteit > actei:
                        ra1 = a2
                        ra2 = a22

                radj = -ra2[0,0] * radmin[0,0]
            
    with computation(PARALLEL), interval(0,1):
        flg = cnvflg[0,0]
        mrad = krad[0,0]

    with computation(BACKWARD), interval(0,-1):
        thlvd = thlvd[0,1]
        radj  = radj[0,1]
        ra1   = ra1[0,1]
        ra2   = ra2[0,1]
        krad1 = krad1[0,1]
        hrad  = hrad[0,1]

@gtscript.stencil(backend=backend)
def mfscu_s0b(cnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
              flg     : gtscript.Field[B_TYPE, gtscript.IK],
              krad    : gtscript.Field[I_TYPE, gtscript.IK],
              mask    : gtscript.Field[I_TYPE, gtscript.IK],
              mrad    : gtscript.Field[I_TYPE, gtscript.IK],
              thlvd   : gtscript.Field[F_TYPE, gtscript.IK],
              thlvx   : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        flg = flg[0,-1]
        mrad = mrad[0,-1]

    with computation(BACKWARD):
        with interval(-1,None):
            if flg[0,0] and mask[0,0] < krad[0,0]:
                if thlvd[0,0] <= thlvx[0,0]:
                    mrad[0,0] = mask[0,0]
                else:
                    flg[0,0] = 0
        with interval(0,-1):
            mrad = mrad[0,1]
            flg  = flg[0,1]

            if flg[0,0] and mask[0,0] < krad[0,0]:
                if thlvd[0,0] <= thlvx[0,0]:
                    mrad[0,0] = mask[0,0]
                else:
                    flg[0,0] = 0

    
    with computation(FORWARD), interval(0,1):
        kk = krad[0,0] - mrad[0,0]
        if cnvflg[0,0]:
            if kk < 1:
                cnvflg[0,0] = 0


@gtscript.stencil(backend=backend)
def mfscu_s1(buo     : gtscript.Field[F_TYPE, gtscript.IK],
             cnvflg  : gtscript.Field[B_TYPE, gtscript.IK],
             krad    : gtscript.Field[I_TYPE, gtscript.IK],
             mask    : gtscript.Field[I_TYPE, gtscript.IK],
             pix     : gtscript.Field[F_TYPE, gtscript.IK],
             plyr    : gtscript.Field[F_TYPE, gtscript.IK],
             radmin  : gtscript.Field[F_TYPE, gtscript.IK],
             thld    : gtscript.Field[F_TYPE, gtscript.IK],
             thlx    : gtscript.Field[F_TYPE, gtscript.IK],
             thvx    : gtscript.Field[F_TYPE, gtscript.IK],
             qtd     : gtscript.Field[F_TYPE, gtscript.IK],
             qtx     : gtscript.Field[F_TYPE, gtscript.IK],
             xlamde  : gtscript.Field[F_TYPE, gtscript.IK],
             zl      : gtscript.Field[F_TYPE, gtscript.IK],
             *,
             el2orc  : float,
             elocp   : float,
             eps     : float,
             epsm1   : float,
             fv      : float,
             g       : float):

        with computation(FORWARD), interval(1,None):
            cnvflg = cnvflg[0,-1]

        with computation(BACKWARD), interval(...):
            dz = zl[0,1] - zl[0,0]
            tem = 0.5 * xlamde[0,0] * dz
            factor = 1.0 + tem
            if cnvflg[0,0] and mask[0,0] < krad[0,0]:
                thld = ((1.0 - tem) * thld[0,1] + tem * (thlx[0,0] + thlx[0,1]))/factor
                qtd  = ((1.0 - tem) * qtd[0,1]  + tem * (qtx[0,0] + qtx[0,1])) / factor

            tld = thld[0,0] / pix[0,0]
            es  = 0.01 * fpvs(tld)
            qs = max(qmin, eps * es / (plyr[0,0]+ epsm1*es))
            dq = qtd[0,0] - qs
            gamma = el2orc * qs / (tld**2)
            qld = dq / (1.0 + gamma)

            if cnvflg[0,0] and mask[0,0] < krad[0,0]:
                if dq > 0.0:
                    qtd = qs + qld
                    thvd = (thld[0,0] + pix[0,0] * elocp * qld) * (1.0 + fv * qs - qld)
                else:
                    thvd = thld[0,0] * (1.0 + fv * qtd[0,0])
                buo = g * (1.0 - thvd / thvx[0,0])

@gtscript.stencil(backend=backend)
def mfscu_s1a(buo      : gtscript.Field[F_TYPE, gtscript.IK],
              cnvflg   : gtscript.Field[B_TYPE, gtscript.IK],
              krad1    : gtscript.Field[I_TYPE, gtscript.IK],
              mask     : gtscript.Field[I_TYPE, gtscript.IK],
              wd2      : gtscript.Field[F_TYPE, gtscript.IK],
              xlamde   : gtscript.Field[F_TYPE, gtscript.IK],
              zm       : gtscript.Field[F_TYPE, gtscript.IK],
              *,
              bb1      : float,
              bb2      : float):
    
    with computation(FORWARD), interval(...):
        if mask[0,0] == krad1[0,0]:
            if cnvflg[0,0]:
                dz = zm[0,1] - zm[0,0]
                ptem1 = 1.0 + (0.5 * bb1 * xlamde[0,0] * dz)
                wd2 = (bb2 * buo[0,1] * dz) / ptem1

@gtscript.stencil(backend=backend)
def mfscu_s2(buo       : gtscript.Field[F_TYPE, gtscript.IK],
             cnvflg    : gtscript.Field[B_TYPE, gtscript.IK],
             flg       : gtscript.Field[B_TYPE, gtscript.IK],
             krad      : gtscript.Field[I_TYPE, gtscript.IK],
             krad1     : gtscript.Field[I_TYPE, gtscript.IK],
             mask      : gtscript.Field[I_TYPE, gtscript.IK],
             mrad      : gtscript.Field[I_TYPE, gtscript.IK],
             mradx     : gtscript.Field[I_TYPE, gtscript.IK],
             mrady     : gtscript.Field[I_TYPE, gtscript.IK],
             xlamde    : gtscript.Field[F_TYPE, gtscript.IK],
             wd2       : gtscript.Field[F_TYPE, gtscript.IK],
             zm        : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(1,None):
        krad1 = krad1[0,-1]
        mrad  = mrad[0,-1]
        krad  = krad[0,-1]

    with computation(BACKWARD), interval(...):
        dz = zm[0,1] - zm[0,0]
        tem = 0.25 * 2.0 * (xlamde[0,0] + xlamde[0,1]) * dz
        ptem = (1.0 - tem) * wd2[0,1]
        ptem1 = 1.0 + tem
        if cnvflg[0,0] and mask[0,0] < krad1[0,0]:
            wd2 = (ptem + (4.0 * buo[0,1] * dz)) / ptem1

    with computation(FORWARD):
        with interval(0,1):
            flg = cnvflg[0,0]
            mrady = mrad[0,0]
            if flg[0,0]:
                mradx = krad[0,0]
        with interval(1,None):
            flg = flg[0,-1]
            mradx = mradx[0,-1]

    with computation(BACKWARD):
        with interval(-1,None):
            if flg[0,0] and mask[0,0] < krad[0,0]:
                if wd2[0,0] > 0.0:
                    mradx = mask[0,0]
                else:
                    flg = 0
        with interval(0,-1):
            flg = flg[0,1]
            mradx = mradx[0,1]
            if flg[0,0] and mask[0,0] < krad[0,0]:
                if wd2[0,0] > 0.0:
                    mradx = mask[0,0]
                else:
                    flg = 0

    with computation(PARALLEL), interval(0,1):
        kk = 0.0
        if cnvflg[0,0]:
            if mrad[0,0] < mradx[0,0]:
                mrad = mradx[0,0]

            kk = krad[0,0] - mrad[0,0]
            if kk < 1:
                cnvflg = 0
            
@gtscript.stencil(backend=backend)
def mfscu_s3(cnvflg    : gtscript.Field[B_TYPE, gtscript.IK],
             gdx       : gtscript.Field[F_TYPE, gtscript.IK],
             krad      : gtscript.Field[I_TYPE, gtscript.IK],
             mask      : gtscript.Field[I_TYPE, gtscript.IK],
             mrad      : gtscript.Field[I_TYPE, gtscript.IK],
             ra1       : gtscript.Field[F_TYPE, gtscript.IK],
             scaldfunc : gtscript.Field[F_TYPE, gtscript.IK],
             sigma     : gtscript.Field[F_TYPE, gtscript.IK],
             sumx      : gtscript.Field[F_TYPE, gtscript.IK],
             wd2       : gtscript.Field[F_TYPE, gtscript.IK],
             xlamde    : gtscript.Field[F_TYPE, gtscript.IK],
             xlamavg   : gtscript.Field[F_TYPE, gtscript.IK],
             xmfd      : gtscript.Field[F_TYPE, gtscript.IK],
             zl        : gtscript.Field[F_TYPE, gtscript.IK],
             *,
             dt2       : float):

    with computation(FORWARD), interval(1,None):
        mrad = mrad[0,-1]
        ra1  = ra1[0,-1]
        cnvflg = cnvflg[0,-1]

    with computation(BACKWARD):
        with interval(-1,None):
            dz = zl[0,1] - zl[0,0]
            if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                xlamavg = xlamavg[0,0] + xlamde[0,0] * dz
                sumx    = sumx[0,0] + dz
        with interval(0,-1):
            xlamavg = xlamavg[0,1]
            sumx    = sumx[0,1]
            dz = zl[0,1] - zl[0,0]
            if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
                xlamavg = xlamavg[0,0] + xlamde[0,0] * dz
                sumx    = sumx[0,0] + dz

    with computation(PARALLEL), interval(0,1):
        if cnvflg[0,0]:
            xlamavg = xlamavg[0,0] / sumx[0,0]

    with computation(BACKWARD), interval(...):
        tem = sqrt(wd2[0,0])
        if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
            if wd2[0,0] > 0:
                xmfd = ra1[0,0] * tem
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0,1):
            if cnvflg[0,0]:
                tem = 0.2 / xlamavg[0,0]
                tem1 = 3.14 * tem * tem
                tem1 = tem1 / (gdx[0,0] * gdx[0,0])
                tem1 = max(tem1, 0.001)
                tem1 = min(tem1, 0.999)
                sigma = tem1

                if sigma[0,0] > ra1[0,0]:
                    tem1 = (1.0-sigma[0,0]) * (1.0 - sigma[0,0])
                    tem1 = min(tem1,1.0)
                    tem1 = max(tem1,0.0)
                    scaldfunc = tem1
                else:
                    scaldfunc = 1.0
        with interval(1,None):
            scaldfunc = scaldfunc[0,-1]

    with computation(BACKWARD), interval(...):
        if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
            xmmx = (zl[0,1] - zl[0,0]) / dt2
            tmp  = scaldfunc[0,0] * xmfd[0,0]
            xmfd = min(tmp,xmmx)

@gtscript.stencil(backend=backend)
def mfscu_s3a(cnvflg    : gtscript.Field[B_TYPE, gtscript.IK],
              krad      : gtscript.Field[I_TYPE, gtscript.IK],
              mask      : gtscript.Field[I_TYPE, gtscript.IK],
              thld      : gtscript.Field[F_TYPE, gtscript.IK],
              thlx      : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(FORWARD), interval(...):
        if krad[0,0] == mask[0,0] and cnvflg[0,0]:
            thld = thlx[0,0]

@gtscript.stencil(backend=backend)
def mfscu_s4(cnvflg    : gtscript.Field[B_TYPE, gtscript.IK],
             krad      : gtscript.Field[I_TYPE, gtscript.IK],
             mask      : gtscript.Field[I_TYPE, gtscript.IK],
             mrad      : gtscript.Field[I_TYPE, gtscript.IK],
             pix       : gtscript.Field[F_TYPE, gtscript.IK],
             plyr      : gtscript.Field[F_TYPE, gtscript.IK],
             qcdo_1    : gtscript.Field[F_TYPE, gtscript.IK],
             qcdo_ntcw : gtscript.Field[F_TYPE, gtscript.IK],
             qcdo_track: gtscript.Field[I_TYPE, gtscript.IK],
             qtd       : gtscript.Field[F_TYPE, gtscript.IK],
             qtx       : gtscript.Field[F_TYPE, gtscript.IK],
             tcdo      : gtscript.Field[F_TYPE, gtscript.IK],
             thld      : gtscript.Field[F_TYPE, gtscript.IK],
             thlx      : gtscript.Field[F_TYPE, gtscript.IK],
             u1        : gtscript.Field[F_TYPE, gtscript.IK],
             ucdo      : gtscript.Field[F_TYPE, gtscript.IK],
             v1        : gtscript.Field[F_TYPE, gtscript.IK],
             vcdo      : gtscript.Field[F_TYPE, gtscript.IK],
             xlamde    : gtscript.Field[F_TYPE, gtscript.IK],
             xlamdem   : gtscript.Field[F_TYPE, gtscript.IK],
             zl        : gtscript.Field[F_TYPE, gtscript.IK],
             *,
             el2orc    : float,
             elocp     : float,
             eps       : float,
             epsm1     : float,
             pgcon     : float):
    
    with computation(BACKWARD), interval(...):
        dz = zl[0,1] - zl[0,0]
        tem = 0.5 * xlamde[0,0] * dz
        factor = 1.0 + tem
        
        if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
            thld = ((1.0 - tem) * thld[0,1] + tem * (thlx[0,0] + thlx[0,1]))/factor
            qtd  = ((1.0 - tem) * qtd[0,1]  + tem * (qtx[0,0] + qtx[0,1])) / factor
        
        tld = thld[0,0] / pix[0,0]
        es  = 0.01 * fpvs(tld)
        qs = max(qmin, eps * es / (plyr[0,0]+ epsm1*es))
        dq = qtd[0,0] - qs
        gamma = el2orc * qs / (tld**2)
        qld = dq / (1.0 + gamma)
        
        if cnvflg[0,0] and mask[0,0] >= mrad[0,0] and mask[0,0] < krad[0,0]:
            qcdo_track = 1
            if dq > 0.0:
                qtd = qs + qld
                qcdo_1 = qs
                qcdo_ntcw = qld
                tcdo = tld + elocp * qld
            else:
                qcdo_1 = qtd[0,0]
                qcdo_ntcw = 0.0
                tcdo = tld
    
        tem = 0.5 * xlamdem[0,0] * dz
        factor = 1.0 + tem

        if cnvflg[0,0] and mask[0,0] < krad[0,0] and mask[0,0] >= mrad[0,0]:
            ptem = tem - pgcon
            ptem1 = tem + pgcon
            ucdo = ((1.0 - tem) * ucdo[0,1] + ptem * u1[0,1] + ptem1*u1[0,0])/factor
            vcdo = ((1.0 - tem) * vcdo[0,1] + ptem * v1[0,1] + ptem1*v1[0,0])/factor

def tridit(l,n,nt,cl,cm,cu,rt,au,at,compare_dict):

    fk  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(l,n+1), default_origin=(0,0),mask=(True,False,True))
    fkk = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(l,n+1), default_origin=(0,0),mask=(True,False,True))

    tridit_s0(au=au,
              cm=cm,
              cl=cl,
              cu=cu,
              fk=fk,
              fkk=fkk,
              domain=(l,1,n))

    for k in range(nt):
        is_ = k * n
        at[:l, is_] = fk[:l,0] * rt[:l,is_]

    for kk in range(nt):
        is_ = kk * n
        for k in range(1,n-1):
            at[:l,k+is_] = fkk[:l,k] * (rt[:l,k+is_] - cl[:l,k-1] * at[:l,k+is_-1])

    tridit_s1(au=au,
              cm=cm,
              cl=cl,
              cu=cu,
              fk=fk,
              domain=(l,1,n))

    for k in range(nt):
        is_ = k * n
        at[:l,n+is_-1] = fk[:l,n-1] * (rt[:l,n+is_-1] - cl[:l,n-2] * at[:l,n+is_-2])

    for kk in range(nt):
        is_ = kk * n
        for k in range(n-2,-1,-1):
            for i in range(l):
                at[i,k+is_] = at[i, k+is_] - au[i,k] * at[i,k+is_+1]

    return au, at

@gtscript.stencil(backend=backend)
def tridit_s0(au    : gtscript.Field[F_TYPE, gtscript.IK],
              cm    : gtscript.Field[F_TYPE, gtscript.IK],
              cl    : gtscript.Field[F_TYPE, gtscript.IK],
              cu    : gtscript.Field[F_TYPE, gtscript.IK],
              fk    : gtscript.Field[F_TYPE, gtscript.IK],
              fkk   : gtscript.Field[F_TYPE, gtscript.IK],
              ):
    with computation(PARALLEL), interval(0,1):
        fk = 1.0/cm[0,0]
        au = fk[0,0] * cu[0,0]

    with computation(FORWARD), interval(1,-1):
        fkk = 1.0 / (cm[0,0] - cl[0,-1] * au[0,-1])
        au = fkk[0,0] * cu[0,0]

@gtscript.stencil(backend=backend)
def tridit_s1(au      : gtscript.Field[F_TYPE, gtscript.IK],
              cm      : gtscript.Field[F_TYPE, gtscript.IK],
              cl      : gtscript.Field[F_TYPE, gtscript.IK],
              cu      : gtscript.Field[F_TYPE, gtscript.IK],
              fk      : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(PARALLEL), interval(-1,None):
        fk = 1.0 / (cm[0,0] - cl[0,-1] * au[0,-1])

def tridin(l,n,nt,cl,cm,cu,r1,r2,au,a1,a2, compare_dict):
    fk  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(l,n+1), default_origin=(0,0),mask=(True,False,True))
    fkk = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(l,n+1), default_origin=(0,0),mask=(True,False,True))

    fk[:,0] = 1.0 / cm[:,0]
    au[:,0] = fk[:,0] * cu[:,0]
    a1[:,0] = fk[:,0] * r1[:,0]

    for k in range(nt):
        is_ = k * n
        a2[:,0,is_] = fk[:,0] * r2[:,0,is_]

    for k in range(1,n-1):
        fkk[:,k] = 1.0 / (cm[:,k] - cl[:,k-1]* au[:,k-1])
        au[:,k]  = fkk[:,k] * cu[:,k]
        a1[:,k]  = fkk[:,k] * (r1[:,k] - cl[:,k-1] * a1[:,k-1])

    for kk in range(nt):
        is_ = kk * n
        for k in range(1,n-1):
            a2[:,0,k+is_] = fkk[:,k] * (r2[:,0,k+is_] - cl[:,k-1] * a2[:,0,k+is_-1])

    fk[:,0] = 1 / (cm[:,n-1] - cl[:,n-2] * au[:,n-2])
    a1[:,n-1] = fk[:,0] * (r1[:,n-1] - cl[:,n-2] * a1[:,n-2])

    for k in range(nt):
        is_ = k * n
        a2[:,0,n+is_-1] = fk[:,0] * (r2[:,0,n+is_-1] - cl[:,n-2]*a2[:,0,n+is_-2])

    for k in range(n-2,-1,-1):
        a1[:,k] = a1[:,k] - au[:,k] * a1[:,k+1]

    for kk in range(nt):
        is_ = kk * n
        for k in range(n-2,-1,-1):
            a2[:,0,k+is_] = a2[:,0,k+is_] - au[:,k]*a2[:,0,k+is_+1]

    return au, a1, a2

# tridi2(im,km,al,ad,au,f1,f2,au,f1,f2...)
def tridi2(l,n,cl,cm,cu,r1,r2,au,a1,a2, compare_dict):

    tridi2_s0(a1=a1,
              a2=a2,
              au=au,
              cl=cl,
              cm=cm,
              cu=cu,
              r1=r1,
              r2=r2,
              domain=(l,1,n))

    return 0

@gtscript.stencil(backend=backend)
def tridi2_s0(a1      : gtscript.Field[F_TYPE, gtscript.IK],
              a2      : gtscript.Field[F_TYPE, gtscript.IK],
              au      : gtscript.Field[F_TYPE, gtscript.IK],
              cl      : gtscript.Field[F_TYPE, gtscript.IK],
              cm      : gtscript.Field[F_TYPE, gtscript.IK],
              cu      : gtscript.Field[F_TYPE, gtscript.IK],
              r1      : gtscript.Field[F_TYPE, gtscript.IK],
              r2      : gtscript.Field[F_TYPE, gtscript.IK]):

    with computation(PARALLEL), interval(0,1):
        fk = 1 / cm[0,0]
        au = fk * cu[0,0]
        a1 = fk * r1[0,0]
        a2 = fk * r2[0,0]

    with computation(FORWARD):
        with interval(1,-1):
            fk = 1.0 / (cm[0,0] - cl[0,-1] * au[0,-1])
            au = fk * cu[0,0]
            a1 = fk * (r1[0,0] - cl[0,-1] * a1[0,-1])
            a2 = fk * (r2[0,0] - cl[0,-1] * a2[0,-1])
        with interval(-1, None):
            fk = 1.0 / (cm[0,0] - cl[0,-1] * au[0,-1])
            a1 = fk *  (r1[0,0] - cl[0,-1] * a1[0,-1])
            a2 = fk *  (r2[0,0] - cl[0,-1] * a2[0,-1])

    with computation(BACKWARD), interval(0,-1):
        a1 = a1[0,0] - au[0,0] * a1[0,1]
        a2 = a2[0,0] - au[0,0] * a2[0,1]