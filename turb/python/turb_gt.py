#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import math
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import timeit

from config import *
from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    PARALLEL,
    FORWARD,
    computation,
    interval,
    exp,
    floor,
    sqrt,
)

backend = BACKEND


def run(in_dict, timings):
    """run function"""

    if len(timings) == 0:
        timings = {"elapsed_time": 0, "run_time": 0}

    tic = timeit.default_timer()

    dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl = satmedmfvdif_gt(
        in_dict["im"],
        in_dict["ix"],
        in_dict["km"],
        in_dict["ntrac"],
        in_dict["ntcw"],
        in_dict["ntiw"],
        in_dict["ntke"],
        in_dict["dv"],
        in_dict["du"],
        in_dict["tdt"],
        in_dict["rtg"],
        in_dict["u1"],
        in_dict["v1"],
        in_dict["t1"],
        in_dict["q1"],
        in_dict["swh"],
        in_dict["hlw"],
        in_dict["xmu"],
        in_dict["garea"],
        in_dict["psk"],
        in_dict["rbsoil"],
        in_dict["zorl"],
        in_dict["u10m"],
        in_dict["v10m"],
        in_dict["fm"],
        in_dict["fh"],
        in_dict["tsea"],
        in_dict["heat"],
        in_dict["evap"],
        in_dict["stress"],
        in_dict["spd1"],
        in_dict["kpbl"],
        in_dict["prsi"],
        in_dict["del"],
        in_dict["prsl"],
        in_dict["prslk"],
        in_dict["phii"],
        in_dict["phil"],
        in_dict["delt"],
        in_dict["dspheat"],
        in_dict["dusfc"],
        in_dict["dvsfc"],
        in_dict["dtsfc"],
        in_dict["dqsfc"],
        in_dict["hpbl"],
        in_dict["kinver"],
        in_dict["xkzm_m"],
        in_dict["xkzm_h"],
        in_dict["xkzm_s"],
    )

    toc = timeit.default_timer()
    timings["elapsed_time"] += toc - tic
    # timings["run_time"] += exec_info["run_end_time"] - exec_info["run_start_time"]

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


def satmedmfvdif_gt(
    im,
    ix,
    km,
    ntrac,
    ntcw,
    ntiw,
    ntke,
    dv,
    du,
    tdt,
    rtg,
    u1,
    v1,
    t1,
    q1,
    swh,
    hlw,
    xmu,
    garea,
    psk,
    rbsoil,
    zorl,
    u10m,
    v10m,
    fm,
    fh,
    tsea,
    heat,
    evap,
    stress,
    spd1,
    kpbl,
    prsi,
    del_,
    prsl,
    prslk,
    phii,
    phil,
    delt,
    dspheat,
    dusfc,
    dvsfc,
    dtsfc,
    dqsfc,
    hpbl,
    kinver,
    xkzm_m,
    xkzm_h,
    xkzm_s,
):

    fv = rv / rd - 1.0
    eps = rd / rv
    epsm1 = rd / rv - 1.0

    gravi = 1.0 / grav
    g = grav
    gocp = g / cp
    cont = cp / g
    conq = hvap / g
    conw = 1.0 / g
    elocp = hvap / cp
    el2orc = hvap * hvap / (rv * cp)

    dt2 = delt
    rdt = 1.0 / dt2
    ntrac1 = ntrac - 1
    km1 = km - 1
    kmpbl = int(km / 2)
    kmscu = int(km / 2)

    # 3D GT storage
    qcko = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT, (ntrac,)),
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    qcdo = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(ntrac,)),
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    f2 = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(ntrac-1,)),
        shape=(im, 1, km),
        default_origin=(0, 0, 0),
    )

    q1_gt = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(ntrac,)),
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    rtg_gt = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(ntrac,)),
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    # 2D GT storages extended into 3D
    # Note : I'm setting 2D GT storages to be size (im, 1, km+1) since this represents
    #        the largest "2D" array that will be examined.  There is a 1 in the 2nd dimension
    #        since GT4py establishes update policies that iterate over the "j" or "z" dimension
    zi = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    zl = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    zm = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ckz = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    chz = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    tke = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    rdzt = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    prn = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xkzo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xkzmo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    pix = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    theta = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qlx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    slx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thvx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlvx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlvx_0 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    svx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thetae = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    gotvx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    plyr = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    cfly = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    bf = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    dku = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    dkt = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    dkq = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    radx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    shr2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    tcko = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    tcdo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ucko = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ucdo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    vcko = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    vcdo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    buou = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xmf = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xlamue = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    rhly = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qstl = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    buod = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xmfd = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xlamde = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    rlam = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ele = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    elm = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    prod = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    rle = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    diss = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ad = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ad_p1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    f1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    f1_p1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    al = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    au = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    f2_p1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    # 1D GT storages extended into 3D
    gdx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xkzm_hx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xkzm_mx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    kx1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    z0 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    kpblx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    hpblx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    pblflg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    sfcflg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    pcnvflg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    scuflg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    radmin = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    mrad = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    krad = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    lcld = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kcld = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    flg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    rbup = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    rbdn = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    sflux = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    thermal = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    crb = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    dtdz1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    ustar = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    zol = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    phim = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    phih = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    wscale = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    vpert = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    radj = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    # Mask/Index Array
    mask = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )

    garea = numpy_to_gt4py_storage_1D(garea, backend, km + 1)
    tx1 = numpy_to_gt4py_storage_1D(1.0 / prsi[:, 0], backend, km + 1)
    tx2 = numpy_to_gt4py_storage_1D(1.0 / prsi[:, 0], backend, km + 1)
    kinver = numpy_to_gt4py_storage_1D(kinver, backend, km + 1)
    zorl = numpy_to_gt4py_storage_1D(zorl, backend, km + 1)
    dusfc = numpy_to_gt4py_storage_1D(dusfc, backend, km + 1)
    dvsfc = numpy_to_gt4py_storage_1D(dvsfc, backend, km + 1)
    dtsfc = numpy_to_gt4py_storage_1D(dtsfc, backend, km + 1)
    dqsfc = numpy_to_gt4py_storage_1D(dqsfc, backend, km + 1)
    kpbl = numpy_to_gt4py_storage_1D(kpbl, backend, km + 1)
    hpbl = numpy_to_gt4py_storage_1D(hpbl, backend, km + 1)
    rbsoil = numpy_to_gt4py_storage_1D(rbsoil, backend, km + 1)
    evap = numpy_to_gt4py_storage_1D(evap, backend, km + 1)
    heat = numpy_to_gt4py_storage_1D(heat, backend, km + 1)
    # Note, psk has length "ix", but in the test example, ix = im, so I
    # can maintain that all the arrays are equal length in the x-direction.
    # This may NOT be the case in general that ix = im
    psk = numpy_to_gt4py_storage_1D(psk, backend, km + 1)
    xmu = numpy_to_gt4py_storage_1D(xmu, backend, km + 1)
    tsea = numpy_to_gt4py_storage_1D(tsea, backend, km + 1)
    u10m = numpy_to_gt4py_storage_1D(u10m, backend, km + 1)
    v10m = numpy_to_gt4py_storage_1D(v10m, backend, km + 1)
    stress = numpy_to_gt4py_storage_1D(stress, backend, km + 1)
    fm = numpy_to_gt4py_storage_1D(fm, backend, km + 1)
    fh = numpy_to_gt4py_storage_1D(fh, backend, km + 1)
    spd1 = numpy_to_gt4py_storage_1D(spd1, backend, km + 1)

    phii = numpy_to_gt4py_storage_2D(phii, backend, km + 1)
    phil = numpy_to_gt4py_storage_2D(phil, backend, km + 1)
    prsi = numpy_to_gt4py_storage_2D(prsi, backend, km + 1)
    swh = numpy_to_gt4py_storage_2D(swh, backend, km + 1)
    hlw = numpy_to_gt4py_storage_2D(hlw, backend, km + 1)
    u1 = numpy_to_gt4py_storage_2D(u1, backend, km + 1)
    v1 = numpy_to_gt4py_storage_2D(v1, backend, km + 1)
    del_ = numpy_to_gt4py_storage_2D(del_, backend, km + 1)
    du = numpy_to_gt4py_storage_2D(du, backend, km + 1)
    dv = numpy_to_gt4py_storage_2D(dv, backend, km + 1)
    tdt = numpy_to_gt4py_storage_2D(tdt, backend, km + 1)

    # Note: prslk has dimensions (ix,km)
    prslk = numpy_to_gt4py_storage_2D(prslk, backend, km + 1)
    # Note : t1 has dimensions (ix,km)
    t1 = numpy_to_gt4py_storage_2D(t1, backend, km + 1)
    # Note : prsl has dimension (ix,km)
    prsl = numpy_to_gt4py_storage_2D(prsl, backend, km + 1)

    mask_init(mask=mask)

    for I in range(ntrac):
        q1_gt[:, 0, :-1, I] = q1[:, :, I]
        rtg_gt[:,0,:-1, I] = rtg[:,:,I]

    init(
        bf=bf,
        cfly=cfly,
        chz=chz,
        ckz=ckz,
        crb=crb,
        dqsfc=dqsfc,
        dt2=dt2,
        dtdz1=dtdz1,
        dtsfc=dtsfc,
        dusfc=dusfc,
        dvsfc=dvsfc,
        elocp=elocp,
        el2orc=el2orc,
        eps=eps,
        evap=evap,
        fv=fv,
        garea=garea,
        gdx=gdx,
        heat=heat,
        hlw=hlw,
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
        g=g,
        gotvx=gotvx,
        gravi=gravi,
        mask=mask,
        mrad=mrad,
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
        q1=q1_gt,
        qlx=qlx,
        qstl=qstl,
        qtx=qtx,
        radmin=radmin,
        radx=radx,
        rbsoil=rbsoil,
        rbup=rbup,
        rdzt=rdzt,
        rhly=rhly,
        sfcflg=sfcflg,
        sflux=sflux,
        scuflg=scuflg,
        shr2=shr2,
        slx=slx,
        stress=stress,
        svx=svx,
        swh=swh,
        t1=t1,
        thermal=thermal,
        theta=theta,
        thetae=thetae,
        thlvx=thlvx,
        thlx=thlx,
        thvx=thvx,
        tke=tke,
        tkmin=tkmin,
        tsea=tsea,
        tx1=tx1,
        tx2=tx2,
        u10m=u10m,
        ustar=ustar,
        u1=u1,
        v1=v1,
        v10m=v10m,
        xkzm_h=xkzm_h,
        xkzm_m=xkzm_m,
        xkzm_s=xkzm_s,
        xkzm_hx=xkzm_hx,
        xkzm_mx=xkzm_mx,
        xkzmo=xkzmo,
        xkzo=xkzo,
        xmu=xmu,
        z0=z0,
        zi=zi,
        zl=zl,
        zm=zm,
        zorl=zorl,
        ntke=ntke-1,
        ntcw=ntcw-1,
        ntiw=ntiw-1,
        domain=(im, 1, km + 1),
    )

    part3a(
        crb=crb,
        flg=flg,
        g=g,
        kpblx=kpblx,
        mask=mask,
        rbdn=rbdn,
        rbup=rbup,
        thermal=thermal,
        thlvx=thlvx,
        thlvx_0=thlvx_0,
        u1=u1,
        v1=v1,
        zl=zl,
        domain=(im, 1, kmpbl),
    )

    part3a1(
        crb=crb,
        evap=evap,
        fh=fh,
        flg=flg,
        fm=fm,
        gotvx=gotvx,
        heat=heat,
        hpbl=hpbl,
        hpblx=hpblx,
        kpbl=kpbl,
        kpblx=kpblx,
        mask=mask,
        pblflg=pblflg,
        pcnvflg=pcnvflg,
        phih=phih,
        phim=phim,
        rbdn=rbdn,
        rbup=rbup,
        rbsoil=rbsoil,
        sfcflg=sfcflg,
        sflux=sflux,
        thermal=thermal,
        theta=theta,
        ustar=ustar,
        vpert=vpert,
        wscale=wscale,
        zi=zi,
        zl=zl,
        zol=zol,
        fv=fv,
        domain=(im, 1, km),
    )

    part3c(
        crb=crb,
        flg=flg,
        g=g,
        kpbl=kpbl,
        mask=mask,
        rbdn=rbdn,
        rbup=rbup,
        thermal=thermal,
        thlvx=thlvx,
        thlvx_0=thlvx_0,
        u1=u1,
        v1=v1,
        zl=zl,
        domain=(im, 1, kmpbl),
    )

    part3c1(
        crb=crb,
        flg=flg,
        hpbl=hpbl,
        kpbl=kpbl,
        lcld=lcld,
        mask=mask,
        pblflg=pblflg,
        pcnvflg=pcnvflg,
        rbdn=rbdn,
        rbup=rbup,
        scuflg=scuflg,
        zi=zi,
        zl=zl,
        domain=(im, 1, km),
    )

    part3e(
        flg=flg,
        kcld=kcld,
        krad=krad,
        lcld=lcld,
        km1=km1,
        mask=mask,
        radmin=radmin,
        radx=radx,
        qlx=qlx,
        scuflg=scuflg,
        domain=(im, 1, kmscu),
    )

    part4(
        pcnvflg=pcnvflg,
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
        qcdo=qcdo,
    )

    part4a(
        pcnvflg=pcnvflg,
        q1=q1_gt,
        qcdo=qcdo,
        qcko=qcko,
        scuflg=scuflg,
        domain=(im, 1, km),
    )

    kpbl, hpbl, buou, xmf, tcko, qcko, ucko, vcko, xlamue = mfpblt(
        im,
        ix,
        km,
        kmpbl,
        ntcw,
        ntrac1,
        dt2,
        pcnvflg,
        zl,
        zm,
        q1_gt,
        t1,
        u1,
        v1,
        plyr,
        pix,
        thlx,
        thvx,
        gdx,
        hpbl,
        kpbl,
        vpert,
        buou,
        xmf,
        tcko,
        qcko,
        ucko,
        vcko,
        xlamue,
        g,
        gocp,
        elocp,
        el2orc,
        mask,
    )

    radj, mrad, buod, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde = mfscu(
        im,
        ix,
        km,
        kmscu,
        ntcw,
        ntrac1,
        dt2,
        scuflg,
        zl,
        zm,
        q1_gt,
        t1,
        u1,
        v1,
        plyr,
        pix,
        thlx,
        thvx,
        thlvx,
        gdx,
        thetae,
        radj,
        krad,
        mrad,
        radmin,
        buod,
        xmfd,
        tcdo,
        qcdo,
        ucdo,
        vcdo,
        xlamde,
        g,
        gocp,
        elocp,
        el2orc,
        mask,
    )

    part5(
        chz=chz,
        ckz=ckz,
        hpbl=hpbl,
        kpbl=kpbl,
        mask=mask,
        pcnvflg=pcnvflg,
        phih=phih,
        phim=phim,
        prn=prn,
        zi=zi,
        domain=(im, 1, kmpbl),
    )

    thvx_n = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(km+1,)),
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    zl_n = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(km+1,)),
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    gotvx_n = gt_storage.zeros(
        backend=backend,
        dtype=(DTYPE_FLT,(km+1,)),
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    for k in range(km+1):
        for i in range(im):
            thvx_n[i,0,k] = thvx[i,0,k]
            gotvx_n[i,0,k] = gotvx[i,0,k]
            zl_n[i,0,k] = zl[i,0,k]

    # Compute asymtotic mixing length
    for k in range(km1):
        for i in range(im):
            zlup = 0.0
            bsum = 0.0
            mlenflg = True
            for n in range(k, km1):
                if mlenflg:
                    dz = zl[i, 0, n + 1] - zl[i, 0, n]
                    ptem = gotvx[i, 0, n] * (thvx[i, 0, n + 1] - thvx[i, 0, k]) * dz
                    bsum = bsum + ptem
                    zlup = zlup + dz
                    if bsum >= tke[i, 0, k]:
                        if ptem >= 0.0:
                            tem2 = max(ptem, zfmin)
                        else:
                            tem2 = min(ptem, -zfmin)
                        ptem1 = (bsum - tke[i, 0, k]) / tem2
                        zlup = zlup - ptem1 * dz
                        zlup = max(zlup, 0.0)
                        mlenflg = False
            zldn = 0.0
            bsum = 0.0
            mlenflg = True
            for n in range(k, -1, -1):
                if mlenflg:
                    if n == 0:
                        dz = zl[i, 0, 0]
                        tem1 = tsea[i, 0] * (1.0 + fv * max(q1_gt[i, 0, 0, 0], qmin))
                    else:
                        dz = zl[i, 0, n] - zl[i, 0, n - 1]
                        tem1 = thvx[i, 0, n - 1]
                    ptem = gotvx[i, 0, n] * (thvx[i, 0, k] - tem1) * dz
                    bsum = bsum + ptem
                    zldn = zldn + dz
                    if bsum >= tke[i, 0, k]:
                        if ptem >= 0.0:
                            tem2 = max(ptem, zfmin)
                        else:
                            tem2 = min(ptem, -zfmin)
                        ptem1 = (bsum - tke[i, 0, k]) / tem2
                        zldn = zldn - ptem1 * dz
                        zldn = max(zldn, 0.0)
                        mlenflg = False

            tem = 0.5 * (zi[i, 0, k + 1] - zi[i, 0, k])
            tem1 = min(tem, rlmn)

            ptem2 = min(zlup, zldn)
            rlam[i, 0, k] = elmfac * ptem2
            rlam[i, 0, k] = max(rlam[i, 0, k], tem1)
            rlam[i, 0, k] = min(rlam[i, 0, k], rlmx)

            ptem2 = math.sqrt(zlup * zldn)
            ele[i, 0, k] = elefac * ptem2
            ele[i, 0, k] = max(ele[i, 0, k], tem1)
            ele[i, 0, k] = min(ele[i, 0, k], elmx)

    part6(
        bf=bf,
        buod=buod,
        buou=buou,
        chz=chz,
        ckz=ckz,
        dku=dku,
        dkt=dkt,
        dkq=dkq,
        ele=ele,
        elm=elm,
        gdx=gdx,
        gotvx=gotvx,
        kpbl=kpbl,
        mask=mask,
        mrad=mrad,
        krad=krad,
        pblflg=pblflg,
        pcnvflg=pcnvflg,
        phim=phim,
        prn=prn,
        prod=prod,
        radj=radj,
        rdzt=rdzt,
        rlam=rlam,
        rle=rle,
        scuflg=scuflg,
        sflux=sflux,
        shr2=shr2,
        stress=stress,
        tke=tke,
        u1=u1,
        ucdo=ucdo,
        ucko=ucko,
        ustar=ustar,
        v1=v1,
        vcdo=vcdo,
        vcko=vcko,
        xkzo=xkzo,
        xkzmo=xkzmo,
        xmf=xmf,
        xmfd=xmfd,
        zi=zi,
        zl=zl,
        zol=zol,
        domain=(im, 1, km),
    )

    kk = max(round(dt2 / cdtn), 1)
    dtn = dt2 / kk

    # for n in range(kk):
    part8(diss=diss, prod=prod, rle=rle, tke=tke, dtn=dtn, kk=kk, domain=(im, 1, km1))

    part9(
        pcnvflg=pcnvflg,
        qcdo=qcdo,
        qcko=qcko,
        scuflg=scuflg,
        tke=tke,
        domain=(im, 1, km),
    )

    part10(
        kpbl=kpbl,
        mask=mask,
        pcnvflg=pcnvflg,
        qcko=qcko,
        tke=tke,
        xlamue=xlamue,
        zl=zl,
        domain=(im, 1, kmpbl),
    )

    part11(
        ad=ad,
        f1=f1,
        krad=krad,
        mask=mask,
        mrad=mrad,
        qcdo=qcdo,
        scuflg=scuflg,
        tke=tke,
        xlamde=xlamde,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    part12(
        ad=ad,
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
        mask=mask,
        mrad=mrad,
        pcnvflg=pcnvflg,
        prsl=prsl,
        qcdo=qcdo,
        qcko=qcko,
        rdzt=rdzt,
        scuflg=scuflg,
        tke=tke,
        xmf=xmf,
        xmfd=xmfd,
        domain=(im, 1, km),
    )

    tridit(au=au, cm=ad, cl=al, f1=f1, domain=(im, 1, km))

    part12a(
            rtg=rtg_gt,
            f1=f1,
            q1=q1_gt,
            ad=ad,
            f2=f2,
            dtdz1=dtdz1,
            evap=evap,
            heat=heat,
            t1=t1,
            rdt=rdt,
            ntrac1=ntrac1,
            ntke=ntke,
            domain=(im,1,km),
    )

    # for k in range(km):
    #     for i in range(im):
    #         rtg_gt[i, 0, k, ntke - 1] = (
    #             rtg_gt[i, 0, k, ntke - 1] + (f1[i, 0, k] - q1_gt[i, 0, k, ntke - 1]) * rdt
    #         )

    # for i in range(im):
    #     ad[i, 0, 0] = 1.0
    #     f1[i, 0, 0] = t1[i, 0, 0] + dtdz1[i, 0, 0] * heat[i, 0]
    #     f2[i, 0, 0, 0] = q1_gt[i, 0, 0, 0] + dtdz1[i, 0, 0] * evap[i, 0]

    # if ntrac1 >= 2:
    #     for kk in range(1, ntrac1):
    #         for i in range(im):
    #             f2[i, 0, 0, kk] = q1_gt[i, 0, 0, kk]

    part13(
        ad=ad,
        ad_p1=ad_p1,
        al=al,
        au=au,
        del_=del_,
        dkt=dkt,
        f1=f1,
        f1_p1=f1_p1,
        f2=f2,
        f2_p1=f2_p1,
        kpbl=kpbl,
        krad=krad,
        mask=mask,
        mrad=mrad,
        pcnvflg=pcnvflg,
        prsl=prsl,
        q1=q1_gt,
        qcdo=qcdo,
        qcko=qcko,
        rdzt=rdzt,
        scuflg=scuflg,
        tcdo=tcdo,
        tcko=tcko,
        t1=t1,
        xmf=xmf,
        xmfd=xmfd,
        dt2=dt2,
        gocp=gocp,
        domain=(im, 1, km),
    )

    if ntrac1 >= 2:
        part13a(pcnvflg=pcnvflg,
                mask=mask,
                kpbl=kpbl,
                del_=del_,
                prsl=prsl,
                rdzt=rdzt,
                xmf=xmf,
                qcko=qcko,
                q1=q1_gt,
                f2=f2,
                scuflg=scuflg,
                mrad=mrad,
                krad=krad,
                xmfd=xmfd,
                qcdo=qcdo,
                ntrac1=ntrac1,
                dt2=dt2,
                domain=(im,1,km)
                )
        # for kk in range(1, ntrac1):
        #     for k in range(km1):
        #         for i in range(im):
        #             if pcnvflg[i, 0] and k < kpbl[i, 0]:
        #                 dtodsd = dt2 / del_[i, 0, k]
        #                 dtodsu = dt2 / del_[i, 0, k + 1]
        #                 dsig = prsl[i, 0, k] - prsl[i, 0, k + 1]
        #                 tem = dsig * rdzt[i, 0, k]
        #                 ptem = 0.5 * tem * xmf[i, 0, k]
        #                 ptem1 = dtodsd * ptem
        #                 ptem2 = dtodsu * ptem
        #                 tem1 = qcko[i, 0, k, kk] + qcko[i, 0, k + 1, kk]
        #                 tem2 = q1_gt[i, 0, k, kk] + q1_gt[i, 0, k + 1, kk]
        #                 f2[i, 0, k, kk] = f2[i, 0, k, kk] - (tem1 - tem2) * ptem1
        #                 f2[i, 0, k + 1, kk] = q1_gt[i, 0, k + 1, kk] + (tem1 - tem2) * ptem2
        #             else:
        #                 f2[i, 0, k + 1, kk] = q1_gt[i, 0, k + 1, kk]

        #             if scuflg[i, 0] and k >= mrad[i, 0] and k < krad[i, 0]:
        #                 dtodsd = dt2 / del_[i, 0, k]
        #                 dtodsu = dt2 / del_[i, 0, k + 1]
        #                 dsig = prsl[i, 0, k] - prsl[i, 0, k + 1]
        #                 tem = dsig * rdzt[i, 0, k]
        #                 ptem = 0.5 * tem * xmfd[i, 0, k]
        #                 ptem1 = dtodsd * ptem
        #                 ptem2 = dtodsu * ptem
        #                 tem1 = qcdo[i, 0, k, kk] + qcdo[i, 0, k + 1, kk]
        #                 tem2 = q1_gt[i, 0, k, kk] + q1_gt[i, 0, k + 1, kk]
        #                 f2[i, 0, k, kk] = f2[i, 0, k, kk] + (tem1 - tem2) * ptem1
        #                 f2[i, 0, k + 1, kk] = (
        #                     f2[i, 0, k + 1, kk] - (tem1 - tem2) * ptem2
        #                 )

    # au, f1, f2 = tridin(im, km, ntrac1, al, ad, au, f1, f2, au, f1, f2)
    tridin(cl = al,
           cm = ad,
           cu = au,
           r1 = f1,
           r2 = f2,
           au = au,
           a1 = f1,
           a2 = f2,
           nt = ntrac1,
           domain =(im, 1, km))

    part13b(f1=f1,
            t1=t1,
            f2=f2,
            q1=q1_gt,
            tdt=tdt,
            rtg=rtg_gt,
            dtsfc=dtsfc,
            del_=del_,
            dqsfc=dqsfc,
            conq=conq,
            cont=cont,
            rdt=rdt,
            ntrac1=ntrac1,
            domain=(im,1,km)
            )

    # for k in range(km):
    #     for i in range(im):
    #         ttend = (f1[i, 0, k] - t1[i, 0, k]) * rdt
    #         qtend = (f2[i, 0, k, 0] - q1_gt[i, 0, k, 0]) * rdt
    #         tdt[i, 0, k] = tdt[i, 0, k] + ttend
    #         rtg_gt[i, 0, k, 0] = rtg_gt[i, 0, k, 0] + qtend
    #         dtsfc[i, 0] = dtsfc[i, 0] + cont * del_[i, 0, k] * ttend
    #         dqsfc[i, 0] = dqsfc[i, 0] + conq * del_[i, 0, k] * qtend

    # if ntrac1 >= 2:
    #     for kk in range(1, ntrac1):
    #         # is_ = kk * km
    #         for k in range(km):
    #             for i in range(im):
    #                 rtg_gt[i, 0, k, kk] = rtg_gt[i, 0, k, kk] + (
    #                     (f2[i, 0, k, kk] - q1_gt[i, 0, k, kk]) * rdt
    #                 )

    

    part14(
        ad=ad,
        ad_p1=ad_p1,
        al=al,
        au=au,
        del_=del_,
        diss=diss,
        dku=dku,
        dtdz1=dtdz1,
        f1=f1,
        f1_p1=f1_p1,
        f2=f2,
        f2_p1=f2_p1,
        kpbl=kpbl,
        krad=krad,
        mask=mask,
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
        domain=(im, 1, km),
    )

    tridi2(
        a1=f1, a2=f2, au=au, cl=al, cm=ad, cu=au, r1=f1, r2=f2, domain=(im, 1, km)
    )

    part15(
        del_=del_,
        du=du,
        dusfc=dusfc,
        dv=dv,
        dvsfc=dvsfc,
        f1=f1,
        f2=f2,
        hpbl=hpbl,
        hpblx=hpblx,
        kpbl=kpbl,
        kpblx=kpblx,
        mask=mask,
        u1=u1,
        v1=v1,
        conw=conw,
        rdt=rdt,
        domain=(im, 1, km),
    )

    dv = storage_to_numpy(dv, (im, km))
    du = storage_to_numpy(du, (im, km))
    tdt = storage_to_numpy(tdt, (im, km))
    kpbl = storage_to_numpy(kpbl, im)
    dusfc = storage_to_numpy(dusfc, im)
    dvsfc = storage_to_numpy(dvsfc, im)
    dtsfc = storage_to_numpy(dtsfc, im)
    dqsfc = storage_to_numpy(dqsfc, im)
    hpbl = storage_to_numpy(hpbl, im)

    kpbl[:] = kpbl + 1

    for I in range(ntrac):
        rtg[:,:, I] = rtg_gt[:,0, :-1, I]

    return dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl


def numpy_to_gt4py_storage_2D(arr, backend, k_depth):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    # Enforce that arrays are at least of length k_depth in the "k" direction
    if arr.shape[1] < k_depth:
        Z = np.zeros((arr.shape[0], 1, k_depth - arr.shape[1]))
        data = np.dstack((data, Z))
    return gt_storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def numpy_to_gt4py_storage_1D(arr, backend, k_depth):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    # Replicate 2D array in z direction "k_depth" number of times
    # data = np.repeat(data[:, :, np.newaxis], k_depth, axis=2)
    return gt_storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def storage_to_numpy(gt_storage, array_dim):
    if isinstance(array_dim, tuple):
        np_tmp = np.zeros(array_dim)
        np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 0 : array_dim[1]]
    else:
        np_tmp = np.zeros(array_dim)
        if len(gt_storage.shape) == 1:
            np_tmp[:] = gt_storage[0:array_dim]
        elif len(gt_storage.shape) == 2:
            np_tmp[:] = gt_storage[0:array_dim, 0]
        else:
            np_tmp[:] = gt_storage[0:array_dim, 0, 0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp


def storage_to_numpy_and_assert_equal(gt_storage, numpy_array):
    numpy_array[np.isnan(numpy_array)] = 0.0
    if numpy_array.ndim == 1:
        temp = gt_storage[0 : numpy_array.shape[0], 0, 0]
        temp2 = np.zeros(temp.shape)
        temp2[:] = temp
        # np.testing.assert_allclose(temp2,numpy_array, rtol=1e-13,atol=0)
        np.testing.assert_array_equal(temp2, numpy_array)
    elif numpy_array.ndim == 2:
        temp = gt_storage.reshape(gt_storage.shape[0], gt_storage.shape[2])
        temp = temp[0 : numpy_array.shape[0], 0 : numpy_array.shape[1]]
        temp2 = np.zeros(temp.shape)
        temp2[:, :] = temp
        # np.testing.assert_allclose(temp2,numpy_array,rtol=1e-13,atol=0)
        np.testing.assert_array_equal(temp2, numpy_array)
    else:
        temp = gt_storage
        temp = temp[
            0 : numpy_array.shape[0], 0 : numpy_array.shape[1], 0 : numpy_array.shape[2]
        ]
        temp2 = np.zeros(temp.shape)
        temp2[:, :, :] = temp
        # np.testing.assert_allclose(temp2,numpy_array,rtol=1e-13,atol=0)
        np.testing.assert_array_equal(temp2, numpy_array)


@gtscript.function
def fpvsx(t):
    con_ttp = 2.7316e2
    con_cvap = 1.8460e3
    con_cliq = 4.1855e3
    con_hvap = 2.5000e6
    con_rv = 4.6150e2
    con_csol = 2.1060e3
    con_hfus = 3.3358e5
    con_psat = 6.1078e2

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

    tr = con_ttp / t

    fpvsx = 0.0
    if t > tliq:
        fpvsx = con_psat * tr ** xponal * exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = con_psat * tr ** xponai * exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = con_psat * (tr ** xponal) * exp(xponbl * (1.0 - tr))
        pvi = con_psat * (tr ** xponai) * exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx


@gtscript.function
def fpvs(t):
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = min(max(c1xpvs + c2xpvs * t[0, 0, 0], 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    jx = floor(jx)

    # Convert jx to "x"
    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

    return fpvs


@gtscript.stencil(backend=backend)
def mask_init(mask: FIELD_INT):
    with computation(FORWARD), interval(1, None):
        mask = mask[0, 0, -1] + 1


@gtscript.stencil(backend=backend)
def init(
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    phii: FIELD_FLT,
    phil: FIELD_FLT,
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    garea: FIELD_FLT_IJ,
    gdx: FIELD_FLT,
    tke: FIELD_FLT,
    q1: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    prn: FIELD_FLT,
    kx1: FIELD_INT,
    prsi: FIELD_FLT,
    xkzm_hx: FIELD_FLT,
    xkzm_mx: FIELD_FLT,
    mask: FIELD_INT,
    kinver: FIELD_INT_IJ,
    tx1: FIELD_FLT_IJ,
    tx2: FIELD_FLT_IJ,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    z0: FIELD_FLT,
    kpblx: FIELD_INT_IJ,
    hpblx: FIELD_FLT_IJ,
    pblflg: FIELD_BOOL_IJ,
    sfcflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    scuflg: FIELD_BOOL_IJ,
    zorl: FIELD_FLT_IJ,
    dusfc: FIELD_FLT_IJ,
    dvsfc: FIELD_FLT_IJ,
    dtsfc: FIELD_FLT_IJ,
    dqsfc: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    hpbl: FIELD_FLT_IJ,
    rbsoil: FIELD_FLT_IJ,
    radmin: FIELD_FLT_IJ,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    kcld: FIELD_INT_IJ,
    theta: FIELD_FLT,
    prslk: FIELD_FLT,
    psk: FIELD_FLT_IJ,
    t1: FIELD_FLT,
    pix: FIELD_FLT,
    qlx: FIELD_FLT,
    slx: FIELD_FLT,
    thvx: FIELD_FLT,
    qtx: FIELD_FLT,
    thlx: FIELD_FLT,
    thlvx: FIELD_FLT,
    svx: FIELD_FLT,
    thetae: FIELD_FLT,
    gotvx: FIELD_FLT,
    prsl: FIELD_FLT,
    plyr: FIELD_FLT,
    rhly: FIELD_FLT,
    qstl: FIELD_FLT,
    bf: FIELD_FLT,
    cfly: FIELD_FLT,
    crb: FIELD_FLT_IJ,
    dtdz1: FIELD_FLT,
    evap: FIELD_FLT_IJ,
    heat: FIELD_FLT_IJ,
    hlw: FIELD_FLT,
    radx: FIELD_FLT,
    rbup: FIELD_FLT_IJ,
    sflux: FIELD_FLT_IJ,
    shr2: FIELD_FLT,
    stress: FIELD_FLT_IJ,
    swh: FIELD_FLT,
    thermal: FIELD_FLT_IJ,
    tsea: FIELD_FLT_IJ,
    u10m: FIELD_FLT_IJ,
    ustar: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    v10m: FIELD_FLT_IJ,
    xmu: FIELD_FLT_IJ,
    gravi: float,
    dt2: float,
    el2orc: float,
    tkmin: float,
    xkzm_h: float,
    xkzm_m: float,
    xkzm_s: float,
    km1: int,
    ntiw: int,
    fv: float,
    elocp: float,
    g: float,
    eps: float,
    ntke: int,
    ntcw: int,
):

    with computation(FORWARD), interval(0,1):
        pcnvflg = 0
        scuflg = 1
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl = 1
        hpbl = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = 1
        lcld = km1 - 1
        kcld = km1 - 1
        mrad = km1
        krad = 0
        radmin = 0.0
        sfcflg = 1
        if rbsoil[0, 0] > 0.0:
            sfcflg = 0

    with computation(PARALLEL), interval(...):
        zi = phii[0, 0, 0] * gravi
        zl = phil[0, 0, 0] * gravi
        tke = max(q1[0, 0, 0][ntke], tkmin)
    with computation(PARALLEL), interval(0, -1):
        ckz = ck1
        chz = ch1
        gdx = sqrt(garea[0, 0])
        prn = 1.0
        kx1 = 0.0
        zm = zi[0, 0, 1]
        rdzt = 1.0 / (zl[0, 0, 1] - zl[0, 0, 0])

        if gdx[0, 0, 0] >= xkgdx:
            xkzm_hx = xkzm_h
            xkzm_mx = xkzm_m
        else:
            xkzm_hx = 0.01 + ((xkzm_h - 0.01) * (1.0 / (xkgdx - 5.0))) * (
                gdx[0, 0, 0] - 5.0
            )
            xkzm_mx = 0.01 + ((xkzm_m - 0.01) * (1.0 / (xkgdx - 5.0))) * (
                gdx[0, 0, 0] - 5.0
            )

        # if mask[0, 0, 0] == kx1[0, 0, 0] and mask[0, 0, 0] > 0:
        #     tx2 = 1.0 / prsi[0, 0, 0]

        if mask[0, 0, 0] < kinver[0, 0]:
            ptem = prsi[0, 0, 1] * tx1[0, 0]
            xkzo = xkzm_hx[0, 0, 0] * min(
                1.0, exp(-((1.0 - ptem) * (1.0 - ptem) * 10.0))
            )

            if ptem >= xkzm_s:
                xkzmo = xkzm_mx[0, 0, 0]
                kx1 = mask[0, 0, 0] + 1
            else:
                tem1 = min(
                    1.0,
                    exp(
                        -(
                            (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * 5.0
                        )
                    ),
                )
                xkzmo = xkzm_mx[0, 0, 0] * tem1

        z0 = 0.01 * zorl[0, 0]
        # dusfc = 0.0
        # dvsfc = 0.0
        # dtsfc = 0.0
        # dqsfc = 0.0
        # kpbl = 1
        # hpbl = 0.0
        # kpblx = 1
        # hpblx = 0.0
        # pblflg = 1
        # sfcflg = 1
        # if rbsoil[0, 0] > 0.0:
        #     sfcflg = 0
        # pcnvflg = 0
        # scuflg = 1
        # radmin = 0.0
        # mrad = km1
        # krad = 0
        # lcld = km1 - 1
        # kcld = km1 - 1

        pix = psk[0, 0] / prslk[0, 0, 0]
        theta = t1[0, 0, 0] * pix[0, 0, 0]
        if (ntiw+1) > 0:
            tem = max(q1[0, 0, 0][ntcw], qlmin)
            tem1 = max(q1[0, 0, 0][ntiw], qlmin)
            ptem = hvap * tem + (hvap + hfus) * tem1
            qlx = tem + tem1
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - ptem
        else:
            qlx = max(q1[0, 0, 0][ntcw], qlmin)
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - hvap * qlx[0, 0, 0]

        tem = 1.0 + fv * max(q1[0, 0, 0][0], qmin) - qlx[0, 0, 0]
        thvx = theta[0, 0, 0] * tem
        qtx = max(q1[0, 0, 0][0], qmin) + qlx[0, 0, 0]
        thlx = theta[0, 0, 0] - pix[0, 0, 0] * elocp * qlx[0, 0, 0]
        thlvx = thlx[0, 0, 0] * (1.0 + fv * qtx[0, 0, 0])
        svx = cp * t1[0, 0, 0] * tem
        thetae = theta[0, 0, 0] + elocp * pix[0, 0, 0] * max(q1[0, 0, 0][0], qmin)
        gotvx = g / (t1[0, 0, 0] * tem)

        tem = (t1[0, 0, 1] - t1[0, 0, 0]) * tem * rdzt[0, 0, 0]
        if tem > 1.0e-5:
            xkzo = min(xkzo[0, 0, 0], xkzinv)
            xkzmo = min(xkzmo[0, 0, 0], xkzinv)

        plyr = 0.01 * prsl[0, 0, 0]
        es = 0.01 * fpvs(t1)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + (eps - 1) * es))
        rhly = max(0.0, min(1.0, max(qmin, q1[0, 0, 0][0]) / qs))
        qstl = qs

    with computation(FORWARD), interval(...):
        cfly = 0.0
        clwt = 1.0e-6 * (plyr[0, 0, 0] * 0.001)
        if qlx[0, 0, 0] > clwt:
            onemrh = max(1.0e-10, 1.0 - rhly[0, 0, 0])
            tem1 = cql / min(max((onemrh * qstl[0, 0, 0]) ** 0.49, 0.0001), 1.0)
            val = max(min(tem1 * qlx[0, 0, 0], 50.0), 0.0)
            cfly = min(max(sqrt(sqrt(rhly[0, 0, 0])) * (1.0 - exp(-val)), 0.0), 1.0)

    with computation(PARALLEL), interval(0, -2):
        tem1 = 0.5 * (t1[0, 0, 0] + t1[0, 0, 1])
        cfh = min(cfly[0, 0, 1], 0.5 * (cfly[0, 0, 0] + cfly[0, 0, 1]))
        alp = g / (0.5 * (svx[0, 0, 0] + svx[0, 0, 1]))
        gamma = el2orc * (0.5 * (qstl[0, 0, 0] + qstl[0, 0, 1])) / (tem1 ** 2)
        epsi = tem1 / elocp
        beta = (1.0 + gamma * epsi * (1.0 + fv)) / (1.0 + gamma)
        chx = cfh * alp * beta + (1.0 - cfh) * alp
        cqx = cfh * alp * hvap * (beta - epsi)
        cqx = cqx + (1.0 - cfh) * fv * g
        bf = chx * ((slx[0, 0, 1] - slx[0, 0, 0]) * rdzt[0, 0, 0]) + cqx * (
            (qtx[0, 0, 1] - qtx[0, 0, 0]) * rdzt[0, 0, 0]
        )
        radx = (zi[0, 0, 1] - zi[0, 0, 0]) * (
            swh[0, 0, 0] * xmu[0, 0] + hlw[0, 0, 0]
        )

    with computation(FORWARD):
        with interval(0, 1):
            sflux = heat[0, 0] + evap[0, 0] * fv * theta[0, 0, 0]

            if sfcflg[0, 0] == 0 or sflux[0, 0] <= 0.0:
                pblflg = 0

            if pblflg[0, 0]:
                thermal = thlvx[0, 0, 0]
                crb = rbcr
            else:
                tem1 = 1e-7 * (
                    max(sqrt(u10m[0, 0] ** 2 + v10m[0, 0] ** 2), 1.0)
                    / (f0 * z0[0, 0, 0])
                )
                thermal = tsea[0, 0] * (1.0 + fv * max(q1[0, 0, 0][0], qmin))
                crb = max(min(0.16 * (tem1 ** (-0.18)), crbmax), crbmin)

            dtdz1 = dt2 / (zi[0, 0, 1] - zi[0, 0, 0])
            ustar = sqrt(stress[0, 0])

    with computation(PARALLEL):
        with interval(0, -2):
            dw2 = (u1[0, 0, 0] - u1[0, 0, 1]) ** 2 + (v1[0, 0, 0] - v1[0, 0, 1]) ** 2
            shr2 = max(dw2, dw2min) * rdzt[0, 0, 0] * rdzt[0, 0, 0]

    with computation(FORWARD):
        with interval(0,1):
            rbup = rbsoil[0, 0]


# Possible stencil name : mrf_pbl_scheme_part1
@gtscript.stencil(backend=backend)
def part3a(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    g: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]

            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(1, None):
            # thlvx_0 = thlvx_0[0, 0, -1]
            # crb = crb[0, 0, -1]
            # thermal = thermal[0, 0, -1]

            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]
            # else:
                # rbdn = rbdn[0, 0, -1]
                # rbup = rbup[0, 0, -1]
                # kpblx = kpblx[0, 0, -1]
                # flg = flg[0, 0, -1]

    # with computation(BACKWARD), interval(0, -1):
        # rbdn = rbdn[0, 0, 1]
        # rbup = rbup[0, 0, 1]
        # kpblx = kpblx[0, 0, 1]
        # flg = flg[0, 0, 1]


# Possible stencil name : mrf_pbl_2_thermal_1
@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def part3a1(
    crb: FIELD_FLT_IJ,
    evap: FIELD_FLT_IJ,
    fh: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    fm: FIELD_FLT_IJ,
    gotvx: FIELD_FLT,
    heat: FIELD_FLT_IJ,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    phih: FIELD_FLT_IJ,
    phim: FIELD_FLT_IJ,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    rbsoil: FIELD_FLT_IJ,
    sfcflg: FIELD_BOOL_IJ,
    sflux: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    theta: FIELD_FLT,
    ustar: FIELD_FLT_IJ,
    vpert: FIELD_FLT,
    wscale: FIELD_FLT,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT_IJ,
    fv: float,
):

    with computation(FORWARD), interval(...):
        # if mask[0, 0, 0] > 0:
            # kpblx = kpblx[0, 0, -1]
            # hpblx = hpblx[0, 0, -1]
            # zl_0 = zl_0[0, 0, -1]
            # kpbl = kpbl[0, 0, -1]
            # hpbl = hpbl[0, 0, -1]
            # pblflg = pblflg[0, 0, -1]
            # crb = crb[0, 0, -1]
            # rbup = rbup[0, 0, -1]
            # rbdn = rbdn[0, 0, -1]

        if mask[0, 0, 0] == kpblx[0, 0]:
            if kpblx[0, 0] > 0:
                if rbdn[0, 0] >= crb[0, 0]:
                    rbint = 0.0
                elif rbup[0, 0] <= crb[0, 0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0, 0] - rbdn[0, 0]) / (
                        rbup[0, 0] - rbdn[0, 0]
                    )
                hpblx = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

                if hpblx[0, 0] < zi[0, 0, 0]:
                    kpblx = kpblx[0, 0] - 1
            else:
                hpblx = zl[0, 0, 0]
                kpblx = 0

            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]

            if kpbl[0, 0] <= 0:
                pblflg = 0

    # with computation(BACKWARD), interval(0, -1):
        # kpblx = kpblx[0, 0, 1]
        # hpblx = hpblx[0, 0, 1]
        # kpbl = kpbl[0, 0, 1]
        # hpbl = hpbl[0, 0, 1]
        # pblflg = pblflg[0, 0, 1]

    with computation(FORWARD), interval(0, 1):
        zol = max(rbsoil[0, 0] * fm[0, 0] * fm[0, 0] / fh[0, 0], rimin)
        if sfcflg[0, 0]:
            zol = min(zol[0, 0], -zfmin)
        else:
            zol = max(zol[0, 0], zfmin)

        zol1 = zol[0, 0] * sfcfrac * hpbl[0, 0] / zl[0, 0, 0]

        if sfcflg[0, 0]:
            phih = sqrt(1.0 / (1.0 - aphi16 * zol1))
            phim = sqrt(phih[0, 0])
        else:
            phim = 1.0 + aphi5 * zol1
            phih = phim[0, 0]

        pcnvflg = pblflg[0, 0] and (zol[0, 0] < zolcru)

        wst3 = gotvx[0, 0, 0] * sflux[0, 0] * hpbl[0, 0]
        ust3 = ustar[0, 0] ** 3.0

        if pblflg[0, 0]:
            wscale = max(
                (ust3 + wfac * vk * wst3 * sfcfrac) ** h1, ustar[0, 0] / aphi5
            )

        flg = 1

        if pcnvflg[0, 0]:
            hgamt = heat[0, 0] / wscale[0, 0, 0]
            hgamq = evap[0, 0] / wscale[0, 0, 0]
            vpert = max(hgamt + hgamq * fv * theta[0, 0, 0], 0.0)
            thermal = thermal[0, 0] + min(cfac * vpert[0, 0, 0], gamcrt)
            flg = 0
            rbup = rbsoil[0, 0]


# Possible stencil name : thermal_2
@gtscript.stencil(backend=backend)
def part3c(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    g: float,
):

    with computation(FORWARD):
        # with interval(0, 1):
        #     thlvx_0 = thlvx[0, 0, 0]

        with interval(1, 2):
            thlvx_0 = thlvx[0,0,-1]
            # thlvx_0 = thlvx_0[0, 0, -1]
            # crb = crb[0, 0, -1]
            # thermal = thermal[0, 0, -1]
            # rbup = rbup[0, 0, -1]
            # flg = flg[0, 0, -1]
            # kpblx = kpblx[0, 0, -1]
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(2, None):
            # thlvx_0 = thlvx_0[0, 0, -1]
            # crb = crb[0, 0, -1]
            # thermal = thermal[0, 0, -1]
            # rbup = rbup[0, 0, -1]
            # flg = flg[0, 0, -1]
            # kpblx = kpblx[0, 0, -1]
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]
            # else:
                # rbdn = rbdn[0, 0, -1]
                # rbup = rbup[0, 0, -1]
                # kpblx = kpblx[0, 0, -1]
                # flg = flg[0, 0, -1]

    # with computation(BACKWARD), interval(0, -1):
        # rbdn = rbdn[0, 0, 1]
        # rbup = rbup[0, 0, 1]
        # kpblx = kpblx[0, 0, 1]
        # flg = flg[0, 0, 1]


# Possible stencil name : pbl_height_enhance
@gtscript.stencil(backend=backend)
def part3c1(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    scuflg: FIELD_BOOL_IJ,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        # if mask[0, 0, 0] > 0:
            # crb = crb[0, 0, -1]
            # hpbl = hpbl[0, 0, -1]
            # kpbl = kpbl[0, 0, -1]
            # pblflg = pblflg[0, 0, -1]
            # pcnvflg = pcnvflg[0, 0, -1]
            # rbdn = rbdn[0, 0, -1]
            # rbup = rbup[0, 0, -1]

        if pcnvflg[0, 0] and kpbl[0, 0] == mask[0, 0, 0]:
            if rbdn[0, 0] >= crb[0, 0]:
                rbint = 0.0
            elif rbup[0, 0] <= crb[0, 0]:
                rbint = 1.0
            else:
                rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])

            hpbl[0, 0] = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

            if hpbl[0, 0] < zi[0, 0, 0]:
                kpbl[0, 0] = kpbl[0, 0] - 1

            if kpbl[0, 0] <= 0:
                pblflg[0, 0] = 0
                pcnvflg[0, 0] = 0

    # with computation(BACKWARD), interval(0, -1):
        # hpbl = hpbl[0, 0, 1]
        # kpbl = kpbl[0, 0, 1]
        # pblflg = pblflg[0, 0, 1]
        # pcnvflg = pcnvflg[0, 0, 1]

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]
            if flg[0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0
        with interval(1, -1):
            # lcld = lcld[0, 0, -1]
            # flg = flg[0, 0, -1]
            if flg[0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0

    # with computation(BACKWARD), interval(0, -2):
        # lcld = lcld[0, 0, 1]
        # flg = flg[0, 0, 1]


# Possible stencil name : stratocumulus
@gtscript.stencil(backend=backend)
def part3e(
    flg: FIELD_BOOL_IJ,
    kcld: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    mask: FIELD_INT,
    radmin: FIELD_FLT_IJ,
    radx: FIELD_FLT,
    qlx: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    km1: int,
):

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]
        # with interval(1, None):
            # flg = flg[0, 0, -1]
            # lcld = lcld[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                flg[0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= qlcr)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

        with interval(0, -1):
            # kcld[0, 0, 0] = kcld[0, 0, 1]
            # flg[0, 0, 0] = flg[0, 0, 1]
            if (
                flg[0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= qlcr)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

    with computation(FORWARD):
        with interval(0, 1):
            if scuflg[0, 0] and (kcld[0, 0] == (km1 - 1)):
                scuflg = 0
            flg = scuflg[0, 0]

        # with interval(1, None):
            # flg = flg[0, 0, -1]
            # kcld = kcld[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

        with interval(0, -1):
            # flg = flg[0, 0, 1]
            # radmin = radmin[0, 0, 1]
            # krad = krad[0, 0, 1]
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if scuflg[0, 0] and krad[0, 0] <= 0:
            scuflg = 0
        if scuflg[0, 0] and radmin[0, 0] >= 0.0:
            scuflg = 0


# Possible stencil name : mass_flux_comp_1
@gtscript.stencil(backend=backend)
def part4(
    pcnvflg: FIELD_BOOL_IJ,
    qcdo: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    t1: FIELD_FLT,
    tcdo: FIELD_FLT,
    tcko: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
):

    # with computation(FORWARD), interval(1, None):
    #     pcnvflg = pcnvflg[0, 0, -1]
    #     scuflg = scuflg[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            tcko = t1[0, 0, 0]
            ucko = u1[0, 0, 0]
            vcko = v1[0, 0, 0]
        if scuflg[0, 0]:
            tcdo = t1[0, 0, 0]
            ucdo = u1[0, 0, 0]
            vcdo = v1[0, 0, 0]


# # Possible stencil name : mass_flux_comp_2
@gtscript.stencil(backend=backend)
def part4a(
    pcnvflg: FIELD_BOOL_IJ,
    q1: FIELD_FLT_8,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
):

    # with computation(FORWARD), interval(1, None):
    #     pcnvflg_v2 = pcnvflg_v2[0, 0, -1]
    #     scuflg_v2 = scuflg_v2[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            for ii in range(8): 
                qcko[0, 0, 0][ii] = q1[0, 0, 0][ii]
        if scuflg[0, 0]:
            for i2 in range(8):
                qcdo[0, 0, 0][i2] = q1[0, 0, 0][i2]


# Possible stencil name : prandtl_comp_exchg_coeff
@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def part5(
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    pcnvflg: FIELD_BOOL_IJ,
    phih: FIELD_FLT_IJ,
    phim: FIELD_FLT_IJ,
    prn: FIELD_FLT,
    zi: FIELD_FLT,
):

    # with computation(FORWARD), interval(1, None):
        # phih = phih[0, 0, -1]
        # phim = phim[0, 0, -1]

    with computation(PARALLEL), interval(...):
        tem1 = max(zi[0, 0, 1] - sfcfrac * hpbl[0, 0], 0.0)
        ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0] ** 2.0)
        if mask[0, 0, 0] < kpbl[0, 0]:
            if pcnvflg[0, 0]:
                prn = 1.0 + ((phih[0, 0] / phim[0, 0]) - 1.0) * exp(ptem)
            else:
                prn = phih[0, 0] / phim[0, 0]

        if mask[0, 0, 0] < kpbl[0, 0]:
            prn = max(min(prn[0, 0, 0], prmax), prmin)
            ckz = max(min(ck1 + (ck0 - ck1) * exp(ptem), ck0), ck1)
            chz = max(min(ch1 + (ch0 - ch1) * exp(ptem), ch0), ch1)


# Possible stencil name : compute_eddy_buoy_shear
@gtscript.stencil(backend=backend)
def part6(
    bf: FIELD_FLT,
    buod: FIELD_FLT,
    buou: FIELD_FLT,
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    dku: FIELD_FLT,
    dkt: FIELD_FLT,
    dkq: FIELD_FLT,
    ele: FIELD_FLT,
    elm: FIELD_FLT,
    gdx: FIELD_FLT,
    gotvx: FIELD_FLT,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    phim: FIELD_FLT_IJ,
    prn: FIELD_FLT,
    prod: FIELD_FLT,
    radj: FIELD_FLT_IJ,
    rdzt: FIELD_FLT,
    rlam: FIELD_FLT,
    rle: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    sflux: FIELD_FLT_IJ,
    shr2: FIELD_FLT,
    stress: FIELD_FLT_IJ,
    tke: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    ustar: FIELD_FLT_IJ,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT_IJ,
):

    # with computation(FORWARD), interval(1, None):
        # zol = zol[0, 0, -1]
        # pblflg = pblflg[0, 0, -1]
        # scuflg = scuflg[0, 0, -1]

    with computation(FORWARD):
        with interval(0, -1):
            if zol[0, 0] < 0.0:
                zk = vk * zl[0, 0, 0] * (1.0 - 100.0 * zol[0, 0]) ** 0.2
            elif zol[0, 0] >= 1.0:
                zk = vk * zl[0, 0, 0] / 3.7
            else:
                zk = vk * zl[0, 0, 0] / (1.0 + 2.7 * zol[0, 0])

            elm = zk * rlam[0, 0, 0] / (rlam[0, 0, 0] + zk)
            dz = zi[0, 0, 1] - zi[0, 0, 0]
            tem = max(gdx[0, 0, 0], dz)
            elm = min(elm[0, 0, 0], tem)
            ele = min(ele[0, 0, 0], tem)

        with interval(-1, None):
            elm = elm[0, 0, -1]
            ele = ele[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        tem = (
            0.5
            * (elm[0, 0, 0] + elm[0, 0, 1])
            * sqrt(0.5 * (tke[0, 0, 0] + tke[0, 0, 1]))
        )
        ri = max(bf[0, 0, 0] / shr2[0, 0, 0], rimin)

        if mask[0, 0, 0] < kpbl[0, 0]:
            if pblflg[0, 0]:
                dku = ckz[0, 0, 0] * tem
                dkt = dku[0, 0, 0] / prn[0, 0, 0]
            else:
                dkt = chz[0, 0, 0] * tem
                dku = dkt[0, 0, 0] * prn[0, 0, 0]
        else:
            if ri < 0.0:
                dku = ck1 * tem
                dkt = rchck * dku[0, 0, 0]
            else:
                dkt = ch1 * tem
                dku = dkt[0, 0, 0] * min(1.0 + 2.1 * ri, prmax)

        tem = ckz[0, 0, 0] * tem
        dku_tmp = max(dku[0, 0, 0], tem)
        dkt_tmp = max(dkt[0, 0, 0], tem / prscu)

        if scuflg[0, 0]:
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                dku = dku_tmp
                dkt = dkt_tmp

        dkq = prtke * dkt[0, 0, 0]

        dkt = max(min(dkt[0, 0, 0], dkmax), xkzo[0, 0, 0])

        dkq = max(min(dkq[0, 0, 0], dkmax), xkzo[0, 0, 0])

        dku = max(min(dku[0, 0, 0], dkmax), xkzmo[0, 0, 0])

    with computation(PARALLEL), interval(...):
        if mask[0, 0, 0] == krad[0, 0]:
            if scuflg[0, 0]:
                tem1 = bf[0, 0, 0] / gotvx[0, 0, 0]
                if tem1 < tdzmin:
                    tem1 = tdzmin
                ptem = radj[0, 0] / tem1
                dkt = dkt[0, 0, 0] + ptem
                dku = dku[0, 0, 0] + ptem
                dkq = dkq[0, 0, 0] + ptem

    # with computation(FORWARD), interval(1, -1):
        # sflux = sflux[0, 0, -1]
        # ustar = ustar[0, 0, -1]
        # phim = phim[0, 0, -1]
        # kpbl = kpbl[0, 0, -1]
        # scuflg = scuflg[0, 0, -1]
        # pcnvflg = pcnvflg[0, 0, -1]
        # stress = stress[0, 0, -1]
        # mrad = mrad[0, 0, -1]
        # krad = krad[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, 1):
            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem = xmfd[0, 0, 0] * buod[0, 0, 0]
            else:
                ptem = 0.0

            buop = 0.5 * (
                gotvx[0, 0, 0] * sflux[0, 0] + (-dkt[0, 0, 0] * bf[0, 0, 0] + ptem)
            )

            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem1 = (
                    0.5
                    * (u1[0, 0, 1] - u1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (ucdo[0, 0, 0] + ucdo[0, 0, 1] - u1[0, 0, 0] - u1[0, 0, 1])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem2 = (
                    0.5
                    * (v1[0, 0, 1] - v1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (vcdo[0, 0, 0] + vcdo[0, 0, 1] - v1[0, 0, 0] - v1[0, 0, 1])
                )
            else:
                ptem2 = 0.0

            shrp = 0.5 * (
                dku[0, 0, 0] * shr2[0, 0, 0]
                + ptem1
                + ptem2
                + (
                    stress[0, 0]
                    * ustar[0, 0]
                    * phim[0, 0]
                    / (vk * zl[0, 0, 0])
                )
            )

            prod = buop + shrp

        with interval(1, -1):
            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1 = 0.5 * (xmf[0, 0, -1] + xmf[0, 0, 0]) * buou[0, 0, 0]
            else:
                ptem1 = 0.0

            if scuflg[0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                    ptem2 = 0.5 * (xmfd[0, 0, -1] + xmfd[0, 0, 0]) * buod[0, 0, 0]
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0

            buop = (
                0.5 * ((-dkt[0, 0, -1] * bf[0, 0, -1]) + (-dkt[0, 0, 0] * bf[0, 0, 0]))
                + ptem1
                + ptem2
            )

            tem1 = (u1[0, 0, 1] - u1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2 = (u1[0, 0, 0] - u1[0, 0, -1]) * rdzt[0, 0, -1]

            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1 + xmf[0, 0, -1] * tem2)
                    * (u1[0, 0, 0] - ucko[0, 0, 0])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                    ptem2 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1 + xmfd[0, 0, -1] * tem2)
                        * (ucdo[0, 0, 0] - u1[0, 0, 0])
                    )
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0

            shrp = (
                0.5
                * ((dku[0, 0, -1] * shr2[0, 0, -1]) + (dku[0, 0, 0] * shr2[0, 0, 0]))
                + ptem1
                + ptem2
            )
            tem1 = (v1[0, 0, 1] - v1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2 = (v1[0, 0, 0] - v1[0, 0, -1]) * rdzt[0, 0, -1]

            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1 + xmf[0, 0, -1] * tem2)
                    * (v1[0, 0, 0] - vcko[0, 0, 0])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                    ptem2 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1 + xmfd[0, 0, -1] * tem2)
                        * (vcdo[0, 0, 0] - v1[0, 0, 0])
                    )
                else:
                    ptem2 = 0.0
            else:
                ptem2 = 0.0

            shrp = shrp + ptem1 + ptem2

            prod = buop + shrp

    with computation(PARALLEL), interval(0, -1):
        rle = ce0 / ele[0, 0, 0]


# Possible stencil name : predict_tke
@gtscript.stencil(backend=backend)
def part8(diss: FIELD_FLT, prod: FIELD_FLT, rle: FIELD_FLT, tke: FIELD_FLT, dtn: float, kk : int,):
    with computation(PARALLEL), interval(...):
        for n in range(kk):
            diss = max(
                min(
                    rle[0, 0, 0] * tke[0, 0, 0] * sqrt(tke[0, 0, 0]),
                    prod[0, 0, 0] + tke[0, 0, 0] / dtn,
                ),
                0.0,
            )
            tke = max(tke[0, 0, 0] + dtn * (prod[0, 0, 0] - diss[0, 0, 0]), tkmin)


# Possible stencil name : tke_up_down_prop_1
@gtscript.stencil(backend=backend)
def part9(
    pcnvflg: FIELD_BOOL_IJ,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
    tke: FIELD_FLT,
):

    # with computation(FORWARD), interval(1, None):
    #     scuflg = scuflg[0, 0, -1]
    #     pcnvflg = pcnvflg[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            qcko[0,0,0][7] = tke[0, 0, 0]
        if scuflg[0, 0]:
            qcdo[0,0,0][7] = tke[0, 0, 0]


# Possible stencil name : tke_up_down_prop_2
@gtscript.stencil(backend=backend)
def part10(
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    pcnvflg: FIELD_BOOL_IJ,
    qcko: FIELD_FLT_8,
    tke: FIELD_FLT,
    xlamue: FIELD_FLT,
    zl: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
        if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
            qcko[0,0,0][7] = (
                (1.0 - tem) * qcko[0, 0, -1][7] + tem * (tke[0, 0, 0] + tke[0, 0, -1])
            ) / (1.0 + tem)


# Possible stencil name : tke_up_down_prop_3
@gtscript.stencil(backend=backend)
def part11(
    ad: FIELD_FLT,
    f1: FIELD_FLT,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    qcdo: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
    tke: FIELD_FLT,
    xlamde: FIELD_FLT,
    zl: FIELD_FLT,
):
    with computation(BACKWARD), interval(...):
        tem = 0.5 * xlamde[0, 0, 0] * (zl[0, 0, 1] - zl[0, 0, 0])
        if (
            scuflg[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
        ):
            qcdo[0,0,0][7] = (
                (1.0 - tem) * qcdo[0, 0, 1][7] + tem * (tke[0, 0, 0] + tke[0, 0, 1])
            ) / (1.0 + tem)

    with computation(PARALLEL), interval(0, 1):
        ad = 1.0
        f1 = tke[0, 0, 0]


# Possible stencil name : tke_tridiag_matrix_ele_comp
@gtscript.stencil(backend=backend)
def part12(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkq: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    tke: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dt2: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]
            tem2 = dsig * rdz

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7]
                    + qcko[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * tem2 * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * tem2 * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7]
                    + qcdo[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * tem2 * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * tem2 * xmfd[0, 0, 0]
        with interval(1, -1):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7]
                    + qcko[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * dsig * rdz * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * dsig * rdz * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7]
                    + qcdo[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * dsig * rdz * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * dsig * rdz * xmfd[0, 0, 0]

        with interval(-1, None):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]


@gtscript.stencil(backend=backend)
def part12a(
            rtg : FIELD_FLT_8,
            f1 : FIELD_FLT,
            q1 : FIELD_FLT_8,
            ad : FIELD_FLT,
            f2 : FIELD_FLT_8,
            dtdz1 : FIELD_FLT,
            evap : FIELD_FLT_IJ,
            heat : FIELD_FLT_IJ,
            t1 : FIELD_FLT,
            rdt : float,
            ntrac1 : int,
            ntke : int,
):
    with computation(PARALLEL), interval(...):
        rtg[0,0,0][ntke-1] = (rtg[0,0,0][ntke-1] + (f1[0,0,0] - q1[0,0,0][ntke-1]) * rdt)

    with computation(FORWARD), interval(0,1):
        ad = 1.0
        f1 = t1[0,0,0] + dtdz1[0,0,0] * heat[0,0]
        f2[0,0,0][0] = q1[0,0,0][0] + dtdz1[0,0,0] * evap[0,0]
    
    with computation(FORWARD), interval(0,1):
        if ntrac1 >= 2:
            for kk in range(1,ntrac1):
                f2[0,0,0][kk] = q1[0,0,0][kk]

    

# Possible stencil name : heat_moist_tridiag_mat_ele_comp
@gtscript.stencil(backend=backend)
def part13(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkt: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    f2: FIELD_FLT_7,
    f2_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    q1: FIELD_FLT_8,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    tcdo: FIELD_FLT,
    tcko: FIELD_FLT,
    t1: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dt2: float,
    gocp: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = qcko[0, 0, 0][0] + qcko[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = qcdo[0, 0, 0][0] + qcdo[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            # if mask[0, 0, 0] > 0:
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = qcko[0, 0, 0][0] + qcko[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = qcdo[0, 0, 0][0] + qcdo[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


@gtscript.stencil(backend=backend)
def part13a(
    pcnvflg : FIELD_BOOL_IJ,
    mask : FIELD_INT,
    kpbl : FIELD_INT_IJ,
    del_ : FIELD_FLT,
    prsl : FIELD_FLT,
    rdzt : FIELD_FLT,
    xmf : FIELD_FLT,
    qcko : FIELD_FLT_8,
    q1 : FIELD_FLT_8,
    f2 : FIELD_FLT_8,
    scuflg : FIELD_BOOL_IJ,
    mrad : FIELD_INT_IJ,
    krad : FIELD_INT_IJ,
    xmfd : FIELD_FLT,
    qcdo : FIELD_FLT_8,
    ntrac1 : int,
    dt2 : float,
):
    with computation(FORWARD), interval(0,-1):
        for kk in range(1, ntrac1):
            if mask[0,0,0] > 0:
                if pcnvflg[0, 0] and mask[0,0,-1] < kpbl[0,0]:
                    dtodsu = dt2 / del_[0,0,0]
                    dsig = prsl[0,0,-1] - prsl[0,0,0]
                    tem = dsig * rdzt[0,0,-1]
                    ptem = 0.5 * tem * xmf[0,0,-1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcko[0,0,-1][kk] + qcko[0,0,0][kk]
                    tem2 = q1[0,0,-1][kk] + q1[0,0,0][kk]
                    f2[0,0,0][kk] = q1[0,0,0][kk] + (tem1 - tem2) * ptem2
                else:
                    f2[0,0,0][kk] = q1[0,0,0][kk]

                if scuflg[0,0] and mask[0,0,-1] >= mrad[0,0] and mask[0,0,-1] < krad[0,0]:
                    dtodsu = dt2 / del_[0,0,0]
                    dsig = prsl[0,0,-1] - prsl[0,0,0]
                    tem = dsig * rdzt[0,0,-1]
                    ptem = 0.5 * tem * xmfd[0,0,-1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcdo[0,0,-1][kk] + qcdo[0,0,0][kk]
                    tem2 = q1[0,0,-1][kk] + q1[0,0,0][kk]
                    f2[0,0,0][kk] = f2[0,0,0][kk] - (tem1 - tem2) * ptem2


            if pcnvflg[0,0] and mask[0,0,0] < kpbl[0,0]:
                dtodsd = dt2 / del_[0,0,0]
                dtodsu = dt2 / del_[0,0,1]
                dsig = prsl[0,0,0] - prsl[0,0,1]
                tem = dsig * rdzt[0,0,0]
                ptem = 0.5 * tem * xmf[0,0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcko[0,0,0][kk] + qcko[0,0,1][kk]
                tem2 = q1[0,0,0][kk] + q1[0,0,1][kk]
                f2[0,0,0][kk] = f2[0,0,0][kk] - (tem1 - tem2) * ptem1

            
            if scuflg[0,0] and mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                dtodsd = dt2 / del_[0,0,0]
                dtodsu = dt2 / del_[0,0,1]
                dsig = prsl[0,0,0] - prsl[0,0,1]
                tem = dsig * rdzt[0,0,0]
                ptem = 0.5 * tem * xmfd[0,0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcdo[0,0,0][kk] + qcdo[0,0,1][kk]
                tem2 = q1[0,0,0][kk] + q1[0,0,1][kk]
                f2[0,0,0][kk] = f2[0,0,0][kk] + (tem1 - tem2) * ptem1
    
    with computation(FORWARD), interval(-1,None):
        for kk2 in range(1, ntrac1):
            if pcnvflg[0, 0] and mask[0,0,-1] < kpbl[0,0]:
                dtodsu = dt2 / del_[0,0,0]
                dsig = prsl[0,0,-1] - prsl[0,0,0]
                tem = dsig * rdzt[0,0,-1]
                ptem = 0.5 * tem * xmf[0,0,-1]
                ptem2 = dtodsu * ptem
                tem1 = qcko[0,0,-1][kk2] + qcko[0,0,0][kk2]
                tem2 = q1[0,0,-1][kk2] + q1[0,0,0][kk2]
                f2[0,0,0][kk2] = q1[0,0,0][kk2] + (tem1 - tem2) * ptem2
            else:
                f2[0,0,0][kk2] = q1[0,0,0][kk2]


@gtscript.stencil(backend=backend)
def part13b(
            f1 : FIELD_FLT,
            t1 : FIELD_FLT,
            f2 : FIELD_FLT_8,
            q1 : FIELD_FLT_8,
            tdt : FIELD_FLT,
            rtg : FIELD_FLT_8,
            dtsfc : FIELD_FLT_IJ,
            del_ : FIELD_FLT,
            dqsfc : FIELD_FLT_IJ,
            conq : float,
            cont : float,
            rdt : float,
            ntrac1 : int,
           ):
    with computation(PARALLEL), interval(...):
        ttend = (f1[0,0,0] - t1[0,0,0]) * rdt
        qtend = (f2[0,0,0][0] - q1[0,0,0][0]) * rdt
        tdt = tdt[0,0,0] + ttend
        rtg[0,0,0][0] = rtg[0,0,0][0] + qtend

    with computation(FORWARD), interval(...):
        dtsfc = dtsfc[0,0] + cont * del_[0,0,0] * ((f1[0,0,0] - t1[0,0,0]) * rdt)
        dqsfc = dqsfc[0,0] + conq * del_[0,0,0] * ((f2[0,0,0][0] - q1[0,0,0][0]) * rdt)

    with computation(PARALLEL), interval(...):
        if ntrac1 >= 2:
            for kk in range(1, ntrac1):
                rtg[0,0,0][kk] = rtg[0,0,0][kk] + ((f2[0,0,0][kk] - q1[0,0,0][kk]) * rdt)

# Possible stencil name : moment_tridiag_mat_ele_comp
@gtscript.stencil(backend=backend)
def part14(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    diss: FIELD_FLT,
    dku: FIELD_FLT,
    dtdz1: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    f2: FIELD_FLT_7,
    f2_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    spd1: FIELD_FLT_IJ,
    stress: FIELD_FLT_IJ,
    tdt: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dspheat: bool,
    dt2: float,
):

    with computation(PARALLEL):
        with interval(0, -1):
            if dspheat:
                tdt = tdt[0, 0, 0] + dspfac * (diss[0, 0, 0] / cp)

    with computation(PARALLEL):
        with interval(0, 1):
            ad = 1.0 + dtdz1[0, 0, 0] * stress[0, 0] / spd1[0, 0]
            f1 = u1[0, 0, 0]
            f2[0,0,0][0] = v1[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0]- tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            # if mask[0, 0, 0] > 0:
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2

        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


# Possible stencil name : moment_recover
@gtscript.stencil(backend=backend)
def part15(
    del_: FIELD_FLT,
    du: FIELD_FLT,
    dusfc: FIELD_FLT_IJ,
    dv: FIELD_FLT,
    dvsfc: FIELD_FLT_IJ,
    f1: FIELD_FLT,
    f2: FIELD_FLT_7,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    conw: float,
    rdt: float,
):

    with computation(FORWARD), interval(...):
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0][0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0] + conw * del_[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0] + conw * del_[0, 0, 0] * vtend

    # with computation(BACKWARD), interval(0, -1):
    #     dusfc = dusfc[0, 0, 0] + dusfc[0, 0, 1]
    #     dvsfc = dvsfc[0, 0, 0] + dvsfc[0, 0, 1]

    with computation(FORWARD), interval(0, 1):
        hpbl = hpblx[0, 0]
        kpbl = kpblx[0, 0]


def mfpblt(
    im,
    ix,
    km,
    kmpbl,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1_gt,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    gdx,
    hpbl,
    kpbl,
    vpert,
    buo,
    xmf,
    tcko,
    qcko,
    ucko,
    vcko,
    xlamue,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
):

    ce0 = 0.4
    cm = 1.0
    qmin = 1e-8
    qlmin = 1e-12
    alp = 1.0
    pgcon = 0.55
    a1 = 0.13
    b1 = 0.5
    f1 = 0.15
    fv = 4.6150e2 / 2.8705e2 - 1.0
    eps = 2.8705e2 / 4.6150e2
    epsm1 = eps - 1.0

    wu2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtu = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    xlamuem = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlu = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtu = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlu = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    kpblx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kpbly = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    rbup = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    rbdn = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    flg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    hpblx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    xlamavg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    sumx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    # sigma = gt_storage.zeros(
    #     backend=backend,
    #     dtype=DTYPE_FLT,
    #     shape=(im, 1, km + 1),
    #     default_origin=(0, 0, 0),
    # )
    scaldfunc = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    totflag = True

    for i in range(im):
        totflag = totflag and ~cnvflg[i, 0]

    if totflag:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

    mfpblt_s0(
        alp=alp,
        buo=buo,
        cnvflg=cnvflg,
        g=g,
        hpbl=hpbl,
        kpbl=kpbl,
        q1=q1_gt,
        qtu=qtu,
        qtx=qtx,
        thlu=thlu,
        thlx=thlx,
        thvx=thvx,
        vpert=vpert,
        wu2=wu2,
        ntcw=ntcw-1,
    )

    mfpblt_s1(
        buo=buo,
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
        domain=(im, 1, kmpbl),
    )

    mfpblt_s1a(
        cnvflg=cnvflg,
        hpblx=hpblx,
        kpblx=kpblx,
        mask=mask,
        rbdn=rbdn,
        rbup=rbup,
        zm=zm,
        domain=(im, 1, km),
    )

    mfpblt_s2(
        a1=a1,
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
        qcko=qcko,
        qtu=qtu,
        qtx=qtx,
        scaldfunc=scaldfunc,
        # sigma=sigma,
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
        domain=(im, 1, kmpbl),
    )

    # if ntcw > 2:
    #     for n in range(1, ntcw - 1):
    #         for k in range(1, kmpbl):
    #             for i in range(im):
    #                 if cnvflg[i, 0] and k <= kpbl[i, 0]:
    #                     dz = zl[i, 0, k] - zl[i, 0, k - 1]
    #                     tem = 0.5 * xlamue[i, 0, k - 1] * dz
    #                     factor = 1.0 + tem
    #                     qcko[i, 0, k, n] = (
    #                         (1.0 - tem) * qcko[i, 0, k - 1, n]
    #                         + tem * (q1_gt[i, 0, k, n] + q1_gt[i, 0, k - 1, n])
    #                     ) / factor
    # ndc = ntrac1 - ntcw

    # if ndc > 0:
    #     for n in range(ntcw, ntrac1):
    #         for k in range(1, kmpbl):
    #             for i in range(im):
    #                 if cnvflg[i, 0] and k <= kpbl[i, 0]:
    #                     dz = zl[i, 0, k] - zl[i, 0, k - 1]
    #                     tem = 0.5 * xlamue[i, 0, k - 1] * dz
    #                     factor = 1.0 + tem

    #                     qcko[i, 0, k, n] = (
    #                         (1.0 - tem) * qcko[i, 0, k - 1, n]
    #                         + tem * (q1_gt[i, 0, k, n] + q1_gt[i, 0, k - 1, n])
    #                     ) / factor

    mfpblt_leftover(cnvflg = cnvflg,
                kpbl = kpbl,
                mask = mask,
                xlamue = xlamue,
                qcko = qcko,
                q1_gt = q1_gt,
                zl = zl,
                kmpbl = kmpbl,
                ntcw = ntcw,
                ntrac1 = ntrac1,
                domain=(im, 1, kmpbl)
    )

    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue


@gtscript.stencil(backend=backend)
def mfpblt_leftover(cnvflg : FIELD_BOOL_IJ,
                    kpbl : FIELD_INT_IJ,
                    mask : FIELD_INT,
                    xlamue : FIELD_FLT,
                    qcko : FIELD_FLT_8,
                    q1_gt : FIELD_FLT_8,
                    zl : FIELD_FLT,
                    kmpbl : int,
                    ntcw  : int,
                    ntrac1 : int,
                   ):
    with computation(FORWARD), interval(1,None):
        if ntcw > 2:
            for n in range(ntcw, ntcw - 1):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0,0,0][n] = (
                        (1.0 - tem) * qcko[0,0,-1][n]
                        + tem * (q1_gt[0,0,0][n] + q1_gt[0,0,-1][n])
                    ) / factor

    with computation(FORWARD), interval(1,None):
        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n2 in range(ntcw, ntrac1):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0,0,0][n2] = (
                        (1.0 - tem) * qcko[0,0,-1][n2]
                        + tem * (q1_gt[0,0,0][n2] + q1_gt[0,0,-1][n2])
                    ) / factor

@gtscript.stencil(backend=backend)
def mfpblt_s0(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    q1: FIELD_FLT_8,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    vpert: FIELD_FLT,
    wu2: FIELD_FLT,
    alp: float,
    g: float,
    ntcw: int,
):

    with computation(PARALLEL), interval(0, -1):
        if cnvflg[0, 0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw]

    with computation(PARALLEL), interval(0, 1):
        if cnvflg[0, 0]:
            ptem = min(alp * vpert[0, 0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = g * ptem / thvx[0, 0, 0]

    # CK : This may not be needed later if stencils previous to this one update
    #       hpbl and kpbl over its entire range
    # with computation(FORWARD), interval(1, None):
    #     hpbl = hpbl[0, 0, -1]
    #     kpbl = kpbl[0, 0, -1]


@gtscript.stencil(backend=backend)
def mfpblt_s1(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    kpbly: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wu2: FIELD_FLT,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                xlamue = ce0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(1, None):
            if cnvflg[0, 0]:
                tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
                factor = 1.0 + tem
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

                tlu = thlu[0, 0, 0] / pix[0, 0, 0]
                es = 0.01 * fpvs(tlu)
                qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
                dq = qtu[0, 0, 0] - qs

                if dq > 0.0:
                    gamma = el2orc * qs / (tlu ** 2)
                    qlu = dq / (1.0 + gamma)
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * elocp * qlu) * (
                        1.0 + fv * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + fv * qtu[0, 0, 0])
                buo = g * (thvu / thvx[0, 0, 0] - 1.0)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                wu2 = (4.0 * buo[0, 0, 0] * zm[0, 0, 0]) / (
                    1.0 + (0.5 * 2.0 * xlamue[0, 0, 0] * zm[0, 0, 0])
                )
        with interval(1, None):
            if cnvflg[0, 0]:
                dz = zm[0, 0, 0] - zm[0, 0, -1]
                tem = 0.25 * 2.0 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                wu2 = (((1.0 - tem) * wu2[0, 0, -1]) + (4.0 * buo[0, 0, 0] * dz)) / (
                    1.0 + tem
                )

    with computation(FORWARD), interval(0, 1):
        flg = True
        kpbly = kpbl[0, 0]
        if cnvflg[0, 0]:
            flg = False
            rbup = wu2[0, 0, 0]

    with computation(FORWARD), interval(1, None):
        # kpblx = kpblx[0, 0, -1]
        # flg = flg[0, 0, -1]
        # rbup = rbup[0, 0, -1]
        # rbdn = rbdn[0, 0, -1]
        if flg[0, 0] == False:
            rbdn = rbup[0, 0]
            rbup = wu2[0, 0, 0]
            kpblx = mask[0, 0, 0]
            flg = rbup[0, 0] <= 0.0

    # with computation(BACKWARD), interval(0, -1):
        # rbup = rbup[0, 0, 1]
        # rbdn = rbdn[0, 0, 1]
        # kpblx = kpblx[0, 0, 1]
        # flg = flg[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfpblt_s1a(
    cnvflg: FIELD_BOOL_IJ,
    hpblx: FIELD_FLT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    zm: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        # if mask[0, 0, 0] > 0:
            # hpblx = hpblx[0, 0, -1]
            # rbdn = rbdn[0, 0, -1]
            # rbup = rbup[0, 0, -1]
            # cnvflg = cnvflg[0, 0, -1]

        rbint = 0.0

        if mask[0, 0, 0] == kpblx[0, 0]:
            if cnvflg[0, 0]:
                if rbdn[0, 0] <= 0.0:
                    rbint = 0.0
                elif rbup[0, 0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0, 0] / (rbdn[0, 0] - rbup[0, 0])

                hpblx = zm[0, 0, -1] + rbint * (zm[0, 0, 0] - zm[0, 0, -1])

    # with computation(BACKWARD), interval(0, -1):
    #     hpblx = hpblx[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfpblt_s2(
    cnvflg: FIELD_BOOL_IJ,
    gdx: FIELD_FLT,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    kpbly: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcko: FIELD_FLT_8,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    scaldfunc: FIELD_FLT_IJ,
    # sigma: FIELD_FLT,
    sumx: FIELD_FLT_IJ,
    tcko: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    u1: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xlamavg: FIELD_FLT_IJ,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    wu2: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    a1: float,
    dt2: float,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                if kpbl[0, 0] > kpblx[0, 0]:
                    kpbl = kpblx[0, 0]
                    hpbl = hpblx[0, 0]
        # with interval(1, None):
            # kpbly = kpbly[0, 0, -1]
            # kpbl = kpbl[0, 0, -1]
            # hpbl = hpbl[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (kpbly[0, 0] > kpblx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        with interval(1, None):
            # xlamavg = xlamavg[0, 0, -1]
            # sumx = sumx[0, 0, -1]
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    # with computation(BACKWARD), interval(0, -1):
        # xlamavg = xlamavg[0, 0, 1]
        # sumx = sumx[0, 0, 1]

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = a1 * sqrt(wu2[0, 0, 0])
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem = 0.2 / xlamavg[0, 0]
                sigma = min(
                    max((3.14 * tem * tem) / (gdx[0, 0, 0] * gdx[0, 0, 0]), 0.001),
                    0.999,
                )

                if sigma > a1:
                    scaldfunc = max(
                        min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0
        # with interval(1, None):
        #     scaldfunc = scaldfunc[0, 0, -1]

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            xmf = min(scaldfunc[0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

            tlu = thlu[0, 0, 0] / pix[0, 0, 0]
            es = 0.01 * fpvs(tlu)
            qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
            dq = qtu[0, 0, 0] - qs
            qlu = dq / (1.0 + (el2orc * qs / (tlu ** 2)))

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko[0,0,0][0] = qs
                    qcko[0,0,0][1] = qlu
                    tcko = tlu + elocp * qlu
                else:
                    qcko[0,0,0][0] = qtu[0, 0, 0]
                    qcko[0,0,0][1] = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1]
                    + (tem + pgcon) * u1[0, 0, 0]
                    + (tem - pgcon) * u1[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1]
                    + (tem + pgcon) * v1[0, 0, 0]
                    + (tem - pgcon) * v1[0, 0, -1]
                ) / factor


def mfscu(
    im,
    ix,
    km,
    kmscu,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    thlvx,
    gdx,
    thetae,
    radj,
    krad,
    mrad,
    radmin,
    buo,
    xmfd,
    tcdo,
    qcdo,
    ucdo,
    vcdo,
    xlamde,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
):

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
    hvap = 2.5000e6
    cp = 1.0046e3
    eps = 2.8705e2 / 4.6150e2
    epsm1 = eps - 1.0
    fv = 4.6150e2 / 2.8705e2 - 1.0

    wd2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    hrad = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    krad1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    thld = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    qtd = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    thlvd = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    ra1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    ra2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    flg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    xlamdem = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    mradx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    mrady = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    sumx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    xlamavg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    xmfd = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    # sigma = gt_storage.zeros(
    #     backend=backend,
    #     dtype=DTYPE_FLT,
    #     shape=(im, 1, km + 1),
    #     default_origin=(0, 0, 0),
    # )
    scaldfunc = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    mfscu_s0a(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        hrad=hrad,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        q1=q1,
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
        wd2=wd2,
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
        ntcw=ntcw,
        domain=(im, 1, km),
    )

    mfscu_s0b(
        cnvflg=cnvflg,
        flg=flg,
        krad=krad,
        mask=mask,
        mrad=mrad,
        thlvd=thlvd,
        thlvx=thlvx,
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    zm_mrad = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    for i in range(im):
        zm_mrad[i,0] = zm[i, 0, mrad[i,0]-1]

    mfscu_s0c(
              zl=zl,
              mask=mask,
              mrad=mrad,
              krad=krad,
              zm=zm,
              zm_mrad=zm_mrad,
              xlamde=xlamde,
              xlamdem=xlamdem,
              hrad=hrad,
              cnvflg=cnvflg,
              ce0=ce0,
              cm=cm,
              domain=(im,1,kmscu),
    )

    # for k in range(kmscu):
    #     for i in range(im):
    #         if cnvflg[i, 0]:
    #             dz = zl[i, 0, k + 1] - zl[i, 0, k]
    #             if (k >= mrad[i, 0]) and (k < krad[i, 0]):
    #                 if mrad[i, 0] == 0:
    #                     ptem = 1.0 / (zm[i, 0, k] + dz)
    #                 else:
    #                     # ptem = 1.0 / (zm[i, 0, k] - zm[i, 0, mrad[i, 0] - 1] + dz)
    #                     ptem = 1.0 / (zm[i, 0, k] - zm_mrad[i, 0] + dz)

    #                 xlamde[i, 0, k] = ce0 * (
    #                     ptem + 1.0 / max(hrad[i, 0] - zm[i, 0, k] + dz, dz)
    #                 )
    #             else:
    #                 xlamde[i, 0, k] = ce0 / dz
    #             xlamdem[i, 0, k] = cm * xlamde[i, 0, k]

    mfscu_s1(
        buo=buo,
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
        thld=thld,
        thlx=thlx,
        thvx=thvx,
        qtd=qtd,
        qtx=qtx,
        xlamde=xlamde,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    bb1 = 2.0
    bb2 = 4.0

    mfscu_s1a(
        buo=buo,
        cnvflg=cnvflg,
        krad1=krad1,
        mask=mask,
        wd2=wd2,
        xlamde=xlamde,
        zm=zm,
        bb1=bb1,
        bb2=bb2,
        domain=(im, 1, km),
    )

    mfscu_s2(
        buo=buo,
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
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    for i in range(im):
        zm_mrad[i,0] = zm[i, 0, mrad[i,0]-1]

    mfscu_s0c2(
              zl=zl,
              mask=mask,
              mrad=mrad,
              krad=krad,
              zm=zm,
              zm_mrad=zm_mrad,
              xlamde=xlamde,
              xlamdem=xlamdem,
              hrad=hrad,
              cnvflg=cnvflg,
              mrady=mrady,
              mradx=mradx,
              ce0=ce0,
              cm=cm,
              domain=(im,1,kmscu),
    )

    # for k in range(kmscu):
    #     for i in range(im):
    #         if cnvflg[i, 0] and mrady[i, 0] < mradx[i, 0]:
    #             dz = zl[i, 0, k + 1] - zl[i, 0, k]
    #             if (k >= mrad[i, 0]) and (k < krad[i, 0]):
    #                 if mrad[i, 0] == 0:
    #                     ptem = 1.0 / (zm[i, 0, k] + dz)
    #                 else:
    #                     ptem = 1.0 / (zm[i, 0, k] - zm[i, 0, mrad[i, 0] - 1] + dz)
    #                 xlamde[i, 0, k] = ce0 * (
    #                     ptem + 1.0 / max(hrad[i, 0] - zm[i, 0, k] + dz, dz)
    #                 )
    #             else:
    #                 xlamde[i, 0, k] = ce0 / dz
    #             xlamdem[i, 0, k] = cm * xlamde[i, 0, k]

    

    mfscu_s3(
        cnvflg=cnvflg,
        dt2=delt,
        gdx=gdx,
        krad=krad,
        mask=mask,
        mrad=mrad,
        ra1=ra1,
        # sigma=sigma,
        scaldfunc=scaldfunc,
        sumx=sumx,
        wd2=wd2,
        xlamde=xlamde,
        xlamavg=xlamavg,
        xmfd=xmfd,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    mfscu_s3a(
        cnvflg=cnvflg, krad=krad, mask=mask, thld=thld, thlx=thlx, domain=(im, 1, km)
    )

    mfscu_s4(
        cnvflg=cnvflg,
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
        qcdo=qcdo,
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
        ntcw=ntcw,
        domain=(im, 1, kmscu),
    )

    # if ntcw > 2:
    #     for n in range(1, ntcw - 1):
    #         for k in range(kmscu - 1, -1, -1):
    #             for i in range(im):
    #                 if cnvflg[i, 0] and k < krad[i, 0] and k >= mrad[i, 0]:
    #                     dz = zl[i, 0, k + 1] - zl[i, 0, k]
    #                     tem = 0.5 * xlamde[i, 0, k] * dz
    #                     factor = 1.0 + tem
    #                     qcdo[i, 0, k, n] = (
    #                         (1.0 - tem) * qcdo[i, 0, k + 1, n]
    #                         + tem * (q1[i, 0, k, n] + q1[i, 0, k + 1, n])
    #                     ) / factor

    # ndc = ntrac1 - ntcw

    # if ndc > 0:
    #     for n in range(ntcw, ntrac1):
    #         for k in range(kmscu - 1, -1, -1):
    #             for i in range(im):
    #                 if cnvflg[i, 0] and k < krad[i, 0] and k >= mrad[i, 0]:
    #                     dz = zl[i, 0, k + 1] - zl[i, 0, k]
    #                     tem = 0.5 * xlamde[i, 0, k] * dz
    #                     factor = 1.0 + tem

    #                     qcdo[i, 0, k, n] = (
    #                         (1.0 - tem) * qcdo[i, 0, k + 1, n]
    #                         + tem * (q1[i, 0, k, n] + q1[i, 0, k + 1, n])
    #                     ) / factor

    mfscu_remainder(cnvflg = cnvflg,
                    krad = krad,
                    mrad = mrad,
                    mask = mask,
                    zl = zl,
                    xlamde = xlamde,
                    qcdo = qcdo,
                    q1 = q1,
                    ntcw = ntcw,
                    kmscu = kmscu,
                    ntrac1 = ntrac1,
                    domain = (im, 1, kmscu),
                   )

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

@gtscript.stencil(backend=backend)
def mfscu_s0c(
    zl : FIELD_FLT,
    mask : FIELD_INT,
    mrad : FIELD_INT_IJ,
    krad : FIELD_INT_IJ,
    zm : FIELD_FLT,
    zm_mrad : FIELD_FLT_IJ,
    xlamde : FIELD_FLT,
    xlamdem : FIELD_FLT,
    hrad : FIELD_FLT_IJ,
    cnvflg: FIELD_BOOL_IJ,
    ce0 : float,
    cm : float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0]:
            dz = zl[0,0,1] - zl[0,0,0]
            if mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                if mrad[0,0] == 0:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] - zm_mrad[0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0,0,0]

@gtscript.stencil(backend=backend)
def mfscu_s0c2(
    zl : FIELD_FLT,
    mask : FIELD_INT,
    mrad : FIELD_INT_IJ,
    krad : FIELD_INT_IJ,
    zm : FIELD_FLT,
    zm_mrad : FIELD_FLT_IJ,
    xlamde : FIELD_FLT,
    xlamdem : FIELD_FLT,
    hrad : FIELD_FLT_IJ,
    cnvflg: FIELD_BOOL_IJ,
    mrady : FIELD_INT_IJ,
    mradx : FIELD_INT_IJ,
    ce0 : float,
    cm : float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0] and (mrady[0,0] < mradx[0,0]):
            dz = zl[0,0,1] - zl[0,0,0]
            if mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                if mrad[0,0] == 0:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] - zm_mrad[0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0,0,0]

@gtscript.stencil(backend=backend)
def mfscu_remainder(cnvflg : FIELD_BOOL_IJ,
                    krad   : FIELD_INT_IJ,
                    mrad   : FIELD_INT_IJ,
                    mask   : FIELD_INT,
                    zl     : FIELD_FLT,
                    xlamde : FIELD_FLT,
                    qcdo   : FIELD_FLT_8,
                    q1     : FIELD_FLT_8,
                    ntcw   : int,
                    kmscu  : int,
                    ntrac1 : int,

                   ):
    with computation(BACKWARD), interval(...):
        if ntcw > 2:
            for n in range(1, ntcw-1):
                if cnvflg[0,0] and mask[0,0,0] < krad[0,0] and mask[0,0,0] >= mrad[0,0]:
                    dz = zl[0,0,1] - zl[0,0,0]
                    tem = 0.5 * xlamde[0,0,0] * dz
                    factor = 1.0 + tem
                    qcdo[0,0,0][n] = (
                        (1.0 - tem) * qcdo[0,0,1][n]
                        + tem * (q1[0,0,0][n] + q1[0,0,1][n])
                    ) / factor
            

    with computation(BACKWARD), interval(...):
        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n1 in range(ntcw, ntrac1):
                if cnvflg[0,0] and mask[0,0,0] < krad[0,0] and mask[0,0,0] >= mrad[0,0]:
                    dz = zl[0,0,1] - zl[0,0,0]
                    tem = 0.5 * xlamde[0,0,0] * dz
                    factor = 1.0 + tem
                    qcdo[0,0,0][n1] = (
                        (1.0 - tem) * qcdo[0,0,1][n1]
                        + tem * (q1[0,0,0][n1] + q1[0,0,1][n1])
                    ) / factor

@gtscript.stencil(backend=backend)
def mfscu_s0a(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    hrad: FIELD_FLT_IJ,
    krad: FIELD_INT_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    q1: FIELD_FLT_8,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    ra1: FIELD_FLT_IJ,
    ra2: FIELD_FLT_IJ,
    radmin: FIELD_FLT_IJ,
    radj: FIELD_FLT_IJ,
    thetae: FIELD_FLT,
    thld: FIELD_FLT,
    thlvd: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
    a1: float,
    a11: float,
    a2: float,
    a22: float,
    actei: float,
    cldtime: float,
    cp: float,
    hvap: float,
    g: float,
    ntcw: int,
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw-1]

    with computation(FORWARD), interval(...):
    #     if mask[0, 0, 0] > 0:
            # hrad = hrad[0, 0, -1]
            # krad = krad[0, 0, -1]
            # krad1 = krad1[0, 0, -1]
            # ra1 = ra1[0, 0, -1]
            # ra2 = ra2[0, 0, -1]
            # radj = radj[0, 0, -1]
            # cnvflg = cnvflg[0, 0, -1]
            # radmin = radmin[0, 0, -1]
            # thlvd = thlvd[0, 0, -1]

        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = mask[0, 0, 0] - 1
                tem1 = max(
                    cldtime * radmin[0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0
                )
                thld = thlx[0, 0, 0] + tem1
                qtd = qtx[0, 0, 0]
                thlvd = thlvx[0, 0, 0] + tem1
                buo = -g * tem1 / thvx[0, 0, 0]

                ra1 = a1
                ra2 = a11

                tem = thetae[0, 0, 0] - thetae[0, 0, 1]
                tem1 = qtx[0, 0, 0] - qtx[0, 0, 1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = cp * tem / (hvap * tem1)
                    if cteit > actei:
                        ra1 = a2
                        ra2 = a22

                radj = -ra2[0, 0] * radmin[0, 0]

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrad = krad[0, 0]

    # with computation(BACKWARD), interval(0, -1):
        # thlvd = thlvd[0, 0, 1]
        # radj = radj[0, 0, 1]
        # ra1 = ra1[0, 0, 1]
        # ra2 = ra2[0, 0, 1]
        # krad1 = krad1[0, 0, 1]
        # hrad = hrad[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfscu_s0b(
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    thlvd: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
):

    # with computation(FORWARD), interval(1, None):
    #     flg = flg[0, 0, -1]
    #     mrad = mrad[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0
        with interval(0, -1):
            # mrad = mrad[0, 0, 1]
            # flg = flg[0, 0, 1]

            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0

    with computation(FORWARD), interval(0, 1):
        kk = krad[0, 0] - mrad[0, 0]
        if cnvflg[0, 0]:
            if kk < 1:
                cnvflg[0, 0] = 0


@gtscript.stencil(backend=backend)
def mfscu_s1(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    xlamde: FIELD_FLT,
    zl: FIELD_FLT,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float,
):

    # with computation(FORWARD), interval(1, None):
    #     cnvflg = cnvflg[0, 0, -1]

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + fv * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * elocp * qld) * tem1
            else:
                tem1 = 1.0 + fv * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = g * (1.0 - thvd / thvx[0, 0, 0])


@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def mfscu_s1a(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    zm: FIELD_FLT,
    bb1: float,
    bb2: float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == krad1[0, 0]:
            if cnvflg[0, 0]:
                dz = zm[0, 0, 1] - zm[0, 0, 0]
                wd2 = (bb2 * buo[0, 0, 1] * dz) / (
                    1.0 + (0.5 * bb1 * xlamde[0, 0, 0] * dz)
                )


@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def mfscu_s2(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    mradx: FIELD_INT_IJ,
    mrady: FIELD_INT_IJ,
    xlamde: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
):

    # with computation(FORWARD), interval(1, None):
        # krad1 = krad1[0, 0, -1]
        # mrad = mrad[0, 0, -1]
        # krad = krad[0, 0, -1]

    with computation(BACKWARD), interval(...):
        dz = zm[0, 0, 1] - zm[0, 0, 0]
        tem = 0.25 * 2.0 * (xlamde[0, 0, 0] + xlamde[0, 0, 1]) * dz
        ptem1 = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad1[0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD):
        with interval(0, 1):
            flg = cnvflg[0, 0]
            mrady = mrad[0, 0]
            if flg[0, 0]:
                mradx = krad[0, 0]
        # with interval(1, None):
            # flg = flg[0, 0, -1]
            # mradx = mradx[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            # flg = flg[0, 0, 1]
            # mradx = mradx[0, 0, 1]
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            if mrad[0, 0] < mradx[0, 0]:
                mrad = mradx[0, 0]
            if (krad[0, 0] - mrad[0, 0]) < 1:
                cnvflg = 0


@gtscript.stencil(backend=backend)
def mfscu_s3(
    cnvflg: FIELD_BOOL_IJ,
    gdx: FIELD_FLT,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    ra1: FIELD_FLT_IJ,
    scaldfunc: FIELD_FLT_IJ,
    # sigma: FIELD_FLT,
    sumx: FIELD_FLT_IJ,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    xlamavg: FIELD_FLT_IJ,
    xmfd: FIELD_FLT,
    zl: FIELD_FLT,
    dt2: float,
):

    # with computation(FORWARD), interval(1, None):
        # mrad = mrad[0, 0, -1]
        # ra1 = ra1[0, 0, -1]
        # cnvflg = cnvflg[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                cnvflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                dz = zl[0, 0, 1] - zl[0, 0, 0]
                xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        # with interval(0, -1):
        #     xlamavg = xlamavg[0, 0, 1]
        #     sumx = sumx[0, 0, 1]
            if (
                cnvflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                dz = zl[0, 0, 1] - zl[0, 0, 0]
                xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            if wd2[0, 0, 0] > 0:
                xmfd = ra1[0, 0] * sqrt(wd2[0, 0, 0])
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem1 = (3.14 * (0.2 / xlamavg[0, 0]) * (0.2 / xlamavg[0, 0])) / (
                    gdx[0, 0, 0] * gdx[0, 0, 0]
                )
                sigma = min(max(tem1, 0.001), 0.999)

            if cnvflg[0, 0]:
                if sigma > ra1[0, 0]:
                    scaldfunc = max(
                        min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0
        # with interval(1, None):
        #     scaldfunc = scaldfunc[0, 0, -1]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0] * xmfd[0, 0, 0], xmmx)


@gtscript.stencil(backend=backend)
def mfscu_s3a(
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                thld = thlx[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s4(
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    tcdo: FIELD_FLT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    xlamde: FIELD_FLT,
    xlamdem: FIELD_FLT,
    zl: FIELD_FLT,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float,
    ntcw: int,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            tem = 0.5 * xlamde[0, 0, 0] * dz
            factor = 1.0 + tem
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)

        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            if dq > 0.0:
                qtd = qs + qld
                qcdo[0,0,0][0] = qs
                qcdo[0,0,0][ntcw-1] = qld
                tcdo = tld + elocp * qld
            else:
                qcdo[0,0,0] = qtd[0, 0, 0]
                qcdo[0,0,0][ntcw-1] = 0.0
                tcdo = tld

        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
        ):
            tem = 0.5 * xlamdem[0, 0, 0] * dz
            factor = 1.0 + tem
            ptem = tem - pgcon
            ptem1 = tem + pgcon
            ucdo = (
                (1.0 - tem) * ucdo[0, 0, 1] + ptem * u1[0, 0, 1] + ptem1 * u1[0, 0, 0]
            ) / factor
            vcdo = (
                (1.0 - tem) * vcdo[0, 0, 1] + ptem * v1[0, 0, 1] + ptem1 * v1[0, 0, 0]
            ) / factor


@gtscript.stencil(backend=backend)
def tridit(
    au: FIELD_FLT,
    cm: FIELD_FLT,
    cl: FIELD_FLT,
    f1: FIELD_FLT,
):
    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * au[0, 0, 0]
            f1 = fk * f1[0, 0, 0]
        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * au[0, 0, 0]
            f1 = fkk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])

    with computation(BACKWARD):
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            f1 = fk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])
        with interval(0, -1):
            f1 = f1[0, 0, 0] - au[0, 0, 0] * f1[0, 0, 1]


# def tridin(l, n, nt, cl, cm, cu, r1, r2, au, a1, a2):
@gtscript.stencil(backend=backend)
def tridin(
           cl : FIELD_FLT, 
           cm : FIELD_FLT,
           cu : FIELD_FLT, 
           r1 : FIELD_FLT, 
           r2 : FIELD_FLT_7, 
           au : FIELD_FLT, 
           a1 : FIELD_FLT, 
           a2 : FIELD_FLT_7,
           nt : int,
           ):
    # fk = gt_storage.zeros(
    #     backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    # )
    # fkk = gt_storage.zeros(
    #     backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    # )

    # for i in range(l):
    with computation(FORWARD):
        with interval(0,1):
            fk = 1.0 / cm[0,0,0]
            au = fk * cu[0,0,0]
            a1 = fk * r1[0,0,0]
            for n0 in range(nt):
                a2[0,0,0][n0] = fk * r2[0,0,0][n0]

        with interval(1,-1):
            fkk = 1.0 / (cm[0,0,0] - cl[0,0,-1] * au[0,0,-1])
            au = fkk * cu[0,0,0]
            a1 = fkk * (r1[0,0,0] - cl[0,0,-1] * a1[0,0,-1])

            for n1 in range(nt):
                a2[0,0,0][n1] = fkk * (r2[0,0,0][n1] - cl[0,0,-1] * a2[0,0,-1][n1])

        with interval(-1,None):
            fk = 1.0 / (cm[0,0,0] - cl[0,0,-1] * au[0,0,-1])
            a1 = fk * (r1[0,0,0] - cl[0,0,-1] * a1[0,0,-1])

            for n2 in range(nt):
                a2[0,0,0][n2] = fk * (r2[0,0,0][n2] - cl[0,0,-1] * a2[0,0,-1][n2])

    with computation(BACKWARD):
        with interval(0,-1):
            a1 = a1[0,0,0] - au[0,0,0] * a1[0,0,1]
            for n3 in range(nt):
                a2[0,0,0][n3] = a2[0,0,0][n3] - au[0,0,0] * a2[0,0,1][n3]

    # for k in range(nt):
    #     # is_ = k * n
    #     for i in range(l):
    #         a2[i, 0, 0, k] = fk[i, 0, 0] * r2[i, 0, 0, k]

    # for k in range(1, n - 1):
    #     for i in range(l):
    #         fkk[i, 0, k] = 1.0 / (cm[i, 0, k] - cl[i, 0, k - 1] * au[i, 0, k - 1])
    #         au[i, 0, k] = fkk[i, 0, k] * cu[i, 0, k]
    #         a1[i, 0, k] = fkk[i, 0, k] * (
    #             r1[i, 0, k] - cl[i, 0, k - 1] * a1[i, 0, k - 1]
    #         )

    # for kk in range(nt):
    #     # is_ = kk * n
    #     for k in range(1, n - 1):
    #         for i in range(l):
    #             a2[i, 0, k, kk] = fkk[i, 0, k] * (
    #                 r2[i, 0, k, kk] - cl[i, 0, k - 1] * a2[i, 0, k-1, kk]
    #             )

    # for i in range(l):
    #     fk[i, 0, 0] = 1 / (cm[i, 0, n - 1] - cl[i, 0, n - 2] * au[i, 0, n - 2])
    #     a1[i, 0, n - 1] = fk[i, 0, 0] * (
    #         r1[i, 0, n - 1] - cl[i, 0, n - 2] * a1[i, 0, n - 2]
    #     )

    # for k in range(nt):
    #     for i in range(l):
    #         a2[i, 0, n-1, k] = fk[i, 0, 0] * (
    #             r2[i, 0, n-1, k] - cl[i, 0, n - 2] * a2[i, 0, n-2, k]
    #         )

    # for k in range(n - 2, -1, -1):
    #     for i in range(l):
    #         a1[i, 0, k] = a1[i, 0, k] - au[i, 0, k] * a1[i, 0, k + 1]

    # for kk in range(nt):
    #     for k in range(n - 2, -1, -1):
    #         for i in range(l):
    #             a2[i, 0, k, kk] = (
    #                 a2[i, 0, k, kk] - au[i, 0, k] * a2[i, 0, k+1, kk]
    #             )

    # return au, a1, a2


@gtscript.stencil(backend=backend)
def tridi2(
    a1: FIELD_FLT,
    a2: FIELD_FLT_7,
    au: FIELD_FLT,
    cl: FIELD_FLT,
    cm: FIELD_FLT,
    cu: FIELD_FLT,
    r1: FIELD_FLT,
    r2: FIELD_FLT_7,
):

    with computation(PARALLEL), interval(0, 1):
        fk = 1 / cm[0, 0, 0]
        au = fk * cu[0, 0, 0]
        a1 = fk * r1[0, 0, 0]
        a2[0, 0, 0][0] = fk * r2[0, 0, 0][0]

    with computation(FORWARD):
        with interval(1, -1):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fk * cu[0, 0, 0]
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])

    with computation(BACKWARD), interval(0, -1):
        a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
        a2[0, 0, 0][0] = a2[0, 0, 0][0] - au[0, 0, 0] * a2[0, 0, 1][0]

# @gtscript.stencil(backend=backend)
# def comp_asym_mix_len(
#     mask : FIELD_INT,
#     gotvx : FIELD_FLT,
#     thvx : FIELD_FLT,
#     thvx_n : gtscript.Field[gtscript.IJ, (DTYPE_FLT,(80,))],
#     tke : FIELD_FLT,
#     zl_n : gtscript.Field[gtscript.IJ, (DTYPE_FLT,(80,))],
#     tsea : FIELD_FLT_IJ,
#     q1 : FIELD_FLT_8,
#     gotvx_n : gtscript.Field[gtscript.IJ, (DTYPE_FLT,(80,))],
#     zi : FIELD_FLT,
#     rlam : FIELD_FLT,
#     ele : FIELD_FLT,
#     elmfac : float,
#     elmx : float,
#     rlmn : float,
#     rlmx : float,
#     qmin : float,
#     zfmin : float,
#     km1 : int,
# ):  
#     with computation(FORWARD), interval(...):
#         zlup = 0.0
#         bsum = 0.0
#         mlenflg = True
#         k_start = mask[0,0,0]
#         for n in range(k_start, km1):
#             if mlenflg:
#                 dz = zl_n[0,0][n+1] - zl_n[0,0][n]
#                 ptem = gotvx_n[0,0][n] * (thvx_n[0,0][n+1] - thvx[0,0,0]) * dz
#                 bsum = bsum + ptem
#                 zlup = zlup + dz
#                 if bsum >= tke[0,0,0]:
#                     tem2 = 0.0
#                     if ptem >= 0.0:
#                         tem2 = max(ptem, zfmin)
#                     else:
#                         tem2 = min(ptem, -zfmin)
#                     ptem1 = (bsum - tke[0,0,0]) / tem2
#                     zlup = zlup - ptem1 * dz
#                     zlup = max(zlup, 0.0)
#                     mlenflg = False