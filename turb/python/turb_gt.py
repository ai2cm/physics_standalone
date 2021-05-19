#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
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
    PARALLEL,
    computation,
    interval,
)

backend = "gtx86"
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
    #timings["run_time"] += exec_info["run_end_time"] - exec_info["run_start_time"]

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
        dtype=DTYPE_FLT,
        shape=(im, km + 1, ntrac),
        default_origin=(0, 0, 0),
    )
    qcdo = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, km + 1, ntrac),
        default_origin=(0, 0, 0),
    )
    f2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km * (ntrac - 1)),
        default_origin=(0, 0, 0),
    )
    pcnvflg_v2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, km + 1, ntrac),
        default_origin=(0, 0, 0),
    )
    scuflg_v2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, km + 1, ntrac),
        default_origin=(0, 0, 0),
    )
    q1_gt = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, km + 1, ntrac),
        default_origin=(0, 0, 0),
    )

    # 2D GT storages extended into 3D
    # Note : I'm setting 2D GT storages to be size (im, 1, km+1) since this represents
    #        the largest "2D" array that will be examined.  There is a 1 in the 2nd dimension
    #        since GT4py establishes update policies that iterate over the "j" or "z" dimension
    zi = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    zl = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    zm = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ckz = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    chz = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    tke = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rdzt = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    prn = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xkzo = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xkzmo = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    pix = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    theta = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qlx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    slx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thvx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlvx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlvx_0 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    svx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thetae = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    gotvx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    plyr = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    cfly = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    bf = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    dku = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    dkt = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    dkq = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    radx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    shr2 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    tcko = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    tcdo = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ucko = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ucdo = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    vcko = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    vcdo = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcko_0 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcko_ntke = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcdo_0 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcdo_ntke = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    buou = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xmf = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamue = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rhly = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qstl = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    buod = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xmfd = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamde = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rlam = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ele = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    elm = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    prod = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rle = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    diss = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ad = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ad_p1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    f1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    f1_p1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    al = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    au = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    f2_km = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    f2_p1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )

    # 1D GT storages extended into 3D
    gdx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xkzm_hx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xkzm_mx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    kx1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    z0 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    kpblx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    hpblx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    pblflg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sfcflg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    pcnvflg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    scuflg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    radmin = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    mrad = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    krad = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    lcld = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    kcld = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    flg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rbup = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rbdn = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sflux = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thermal = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    crb = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    dtdz1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ustar = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    zol = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    phim = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    phih = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    vpert = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    radj = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )

    # Mask/Index Array
    mask = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
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
    q1_ntke = numpy_to_gt4py_storage_2D(q1[:, :, ntke - 1], backend, km + 1)
    q1_ntcw = numpy_to_gt4py_storage_2D(q1[:, :, ntcw - 1], backend, km + 1)
    q1_ntiw = numpy_to_gt4py_storage_2D(q1[:, :, ntiw - 1], backend, km + 1)
    q1_0 = numpy_to_gt4py_storage_2D(q1[:, :, 0], backend, km + 1)
    prsi = numpy_to_gt4py_storage_2D(prsi, backend, km + 1)
    swh = numpy_to_gt4py_storage_2D(swh, backend, km + 1)
    hlw = numpy_to_gt4py_storage_2D(hlw, backend, km + 1)
    u1 = numpy_to_gt4py_storage_2D(u1, backend, km + 1)
    v1 = numpy_to_gt4py_storage_2D(v1, backend, km + 1)
    del_ = numpy_to_gt4py_storage_2D(del_, backend, km + 1)
    du = numpy_to_gt4py_storage_2D(du, backend, km + 1)
    dv = numpy_to_gt4py_storage_2D(dv, backend, km + 1)

    # Note: prslk has dimensions (ix,km)
    prslk = numpy_to_gt4py_storage_2D(prslk, backend, km + 1)
    # Note : t1 has dimensions (ix,km)
    t1 = numpy_to_gt4py_storage_2D(t1, backend, km + 1)
    # Note : prsl has dimension (ix,km)
    prsl = numpy_to_gt4py_storage_2D(prsl, backend, km + 1)

    mask_init(mask=mask)

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
        kpblx=kpbl,
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

    q1_gt[:, :-1, :] = q1[:, :, :]

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
        q1=q1_gt,
        qcdo=qcdo,
        qcko=qcko,
    )

    pcnvflg_v2[:, :, 0] = pcnvflg[:, 0, :]
    scuflg_v2[:, :, 0] = scuflg[:, 0, :]

    part4a(
        pcnvflg_v2=pcnvflg_v2,
        q1=q1_gt,
        qcdo=qcdo,
        qcko=qcko,
        scuflg_v2=scuflg_v2,
        domain=(im, km, ntrac1),
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
        q1,
        q1_0,
        q1_ntcw,
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
        q1,
        q1_0,
        q1_ntcw,
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
                        tem1 = tsea[i, 0, 0] * (1.0 + fv * max(q1[i, 0, 0], qmin))
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

    for n in range(kk):
        part8(diss=diss, prod=prod, rle=rle, tke=tke, dtn=dtn, domain=(im, 1, km1))

    qcko_ntke[:, :, :] = qcko[:, :, ntke - 1].reshape((im, 1, km + 1))
    qcdo_ntke[:, :, :] = qcdo[:, :, ntke - 1].reshape((im, 1, km + 1))

    part9(
        pcnvflg=pcnvflg,
        qcdo_ntke=qcdo_ntke,
        qcko_ntke=qcko_ntke,
        scuflg=scuflg,
        tke=tke,
        domain=(im, 1, km),
    )

    part10(
        kpbl=kpbl,
        mask=mask,
        pcnvflg=pcnvflg,
        qcko_ntke=qcko_ntke,
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
        qcdo_ntke=qcdo_ntke,
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
        qcdo_ntke=qcdo_ntke,
        qcko_ntke=qcko_ntke,
        rdzt=rdzt,
        scuflg=scuflg,
        tke=tke,
        xmf=xmf,
        xmfd=xmfd,
        domain=(im, 1, km),
    )

    au, f1 = tridit(im, km, 1, al, ad, au, f1, au, f1)

    qtend = (f1[:, 0, :-1] - q1[:, :, ntke - 1]) * rdt
    rtg[:, :, ntke - 1] = rtg[:, :, ntke - 1] + qtend

    for i in range(im):
        ad[i, 0, 0] = 1.0
        f1[i, 0, 0] = t1[i, 0, 0] + dtdz1[i, 0, 0] * heat[i, 0, 0]
        f2[i, 0, 0] = q1[i, 0, 0] + dtdz1[i, 0, 0] * evap[i, 0, 0]

    if ntrac1 >= 2:
        for kk in range(1, ntrac1):
            is_ = kk * km
            for i in range(im):
                f2[i, 0, is_] = q1[i, 0, kk]

    f2_km[:, :, :-1] = f2[:, 0, 0:km].reshape((im, 1, km))
    qcdo_0[:, :, :] = qcdo[:, :, 0].reshape((im, 1, km + 1))
    qcko_0[:, :, :] = qcko[:, :, 0].reshape((im, 1, km + 1))

    part13(
        ad=ad,
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
        mask=mask,
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
        domain=(im, 1, km),
    )

    f2[:, 0, 0:km] = f2_km[:, 0, 0:km]

    if ntrac1 >= 2:
        for kk in range(1, ntrac1):
            is_ = kk * km
            for k in range(km1):
                for i in range(im):
                    if pcnvflg[i, 0, 0] and k < kpbl[i, 0, 0]:
                        dtodsd = dt2 / del_[i, 0, k]
                        dtodsu = dt2 / del_[i, 0, k + 1]
                        dsig = prsl[i, 0, k] - prsl[i, 0, k + 1]
                        tem = dsig * rdzt[i, 0, k]
                        ptem = 0.5 * tem * xmf[i, 0, k]
                        ptem1 = dtodsd * ptem
                        ptem2 = dtodsu * ptem
                        tem1 = qcko[i, k, kk] + qcko[i, k + 1, kk]
                        tem2 = q1[i, k, kk] + q1[i, k + 1, kk]
                        f2[i, 0, k + is_] = f2[i, 0, k + is_] - (tem1 - tem2) * ptem1
                        f2[i, 0, k + 1 + is_] = q1[i, k + 1, kk] + (tem1 - tem2) * ptem2
                    else:
                        f2[i, 0, k + 1 + is_] = q1[i, k + 1, kk]

                    if scuflg[i, 0, 0] and k >= mrad[i, 0, 0] and k < krad[i, 0, 0]:
                        dtodsd = dt2 / del_[i, 0, k]
                        dtodsu = dt2 / del_[i, 0, k + 1]
                        dsig = prsl[i, 0, k] - prsl[i, 0, k + 1]
                        tem = dsig * rdzt[i, 0, k]
                        ptem = 0.5 * tem * xmfd[i, 0, k]
                        ptem1 = dtodsd * ptem
                        ptem2 = dtodsu * ptem
                        tem1 = qcdo[i, k, kk] + qcdo[i, k + 1, kk]
                        tem2 = q1[i, k, kk] + q1[i, k + 1, kk]
                        f2[i, 0, k + is_] = f2[i, 0, k + is_] + (tem1 - tem2) * ptem1
                        f2[i, 0, k + 1 + is_] = (
                            f2[i, 0, k + 1 + is_] - (tem1 - tem2) * ptem2
                        )

    au, f1, f2 = tridin(im, km, ntrac1, al, ad, au, f1, f2, au, f1, f2)

    for k in range(km):
        ttend = (f1[:, 0, k] - t1[:, 0, k]) * rdt
        qtend = (f2[:, 0, k] - q1[:, k, 0]) * rdt
        tdt[:, k] = tdt[:, k] + ttend
        rtg[:, k, 0] = rtg[:, k, 0] + qtend
        dtsfc[:, 0, 0] = dtsfc[:, 0, 0] + cont * del_[:, 0, k] * ttend
        dqsfc[:, 0, 0] = dqsfc[:, 0, 0] + conq * del_[:, 0, k] * qtend

    if ntrac1 >= 2:
        for kk in range(1, ntrac1):
            is_ = kk * km
            for k in range(km):
                rtg[:, k, kk] = rtg[:, k, kk] + (
                    (f2[:, 0, k + is_] - q1[:, k, kk]) * rdt
                )

    tdt = numpy_to_gt4py_storage_2D(tdt, backend, km + 1)
    f2_km[:, :, :-1] = f2[:, 0, 0:km].reshape((im, 1, km))

    part14(
        ad=ad,
        ad_p1=ad_p1,
        al=al,
        au=au,
        del_=del_,
        diss=diss,
        dku=dku,
        dtdz1=dtdz1,
        du=du,
        dusfc=dusfc,
        dv=dv,
        dvsfc=dvsfc,
        f1=f1,
        f1_p1=f1_p1,
        f2=f2_km,
        f2_p1=f2_p1,
        hpbl=hpbl,
        hpblx=hpblx,
        kpbl=kpbl,
        kpblx=kpblx,
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
        conw=conw,
        dspheat=dspheat,
        dt2=dt2,
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
    data = np.repeat(data[:, :, np.newaxis], k_depth, axis=2)
    return gt_storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def storage_to_numpy(gt_storage, array_dim):
    if isinstance(array_dim, tuple):
        np_tmp = np.zeros(array_dim)
        np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 0 : array_dim[1]]
    else:
        np_tmp = np.zeros(array_dim)
        np_tmp[:] = gt_storage[0:array_dim, 0, 0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp


def storage_to_numpy_and_assert_equal(gt_storage, numpy_array):
    if numpy_array.ndim == 1:
        temp = gt_storage[0 : numpy_array.shape[0], 0, 0]
        temp2 = np.zeros(temp.shape)
        temp2[:] = temp
        np.testing.assert_allclose(temp2, numpy_array, rtol=1e-14, atol=0)
        # np.testing.assert_array_equal(temp2,numpy_array)
    elif numpy_array.ndim == 2:
        temp = gt_storage.reshape(gt_storage.shape[0], gt_storage.shape[2])
        temp = temp[0 : numpy_array.shape[0], 0 : numpy_array.shape[1]]
        temp2 = np.zeros(temp.shape)
        temp2[:, :] = temp
        np.testing.assert_allclose(temp2, numpy_array, rtol=1e-14, atol=0)
        # np.testing.assert_array_equal(temp2,numpy_array)


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
    garea: FIELD_FLT,
    gdx: FIELD_FLT,
    tke: FIELD_FLT,
    q1_ntke: FIELD_FLT,
    q1_0: FIELD_FLT,
    rdzt: FIELD_FLT,
    prn: FIELD_FLT,
    kx1: FIELD_INT,
    prsi: FIELD_FLT,
    xkzm_hx: FIELD_FLT,
    xkzm_mx: FIELD_FLT,
    mask: FIELD_INT,
    kinver: FIELD_INT,
    tx1: FIELD_FLT,
    tx2: FIELD_FLT,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    z0: FIELD_FLT,
    kpblx: FIELD_INT,
    hpblx: FIELD_FLT,
    pblflg: FIELD_BOOL,
    sfcflg: FIELD_BOOL,
    pcnvflg: FIELD_BOOL,
    scuflg: FIELD_BOOL,
    zorl: FIELD_FLT,
    dusfc: FIELD_FLT,
    dvsfc: FIELD_FLT,
    dtsfc: FIELD_FLT,
    dqsfc: FIELD_FLT,
    kpbl: FIELD_INT,
    hpbl: FIELD_FLT,
    rbsoil: FIELD_FLT,
    radmin: FIELD_FLT,
    mrad: FIELD_INT,
    krad: FIELD_INT,
    lcld: FIELD_INT,
    kcld: FIELD_INT,
    theta: FIELD_FLT,
    prslk: FIELD_FLT,
    psk: FIELD_FLT,
    t1: FIELD_FLT,
    pix: FIELD_FLT,
    q1_ntcw: FIELD_FLT,
    q1_ntiw: FIELD_FLT,
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
    crb: FIELD_FLT,
    dtdz1: FIELD_FLT,
    evap: FIELD_FLT,
    heat: FIELD_FLT,
    hlw: FIELD_FLT,
    radx: FIELD_FLT,
    rbup: FIELD_FLT,
    sflux: FIELD_FLT,
    shr2: FIELD_FLT,
    stress: FIELD_FLT,
    swh: FIELD_FLT,
    thermal: FIELD_FLT,
    tsea: FIELD_FLT,
    u10m: FIELD_FLT,
    ustar: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    v10m: FIELD_FLT,
    xmu: FIELD_FLT,
    *,
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
    eps: float
):

    with computation(PARALLEL), interval(0, -1):
        zi = phii[0, 0, 0] * gravi
        zl = phil[0, 0, 0] * gravi
        tke = max(q1_ntke[0, 0, 0], tkmin)

    with computation(PARALLEL), interval(0, -1):
        ckz = ck1
        chz = ch1
        gdx = sqrt(garea[0, 0, 0])
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

        if mask[0, 0, 0] == kx1[0, 0, 0] and mask[0, 0, 0] > 0:
            tx2 = 1.0 / prsi[0, 0, 0]

        if mask[0, 0, 0] < kinver[0, 0, 0]:
            ptem = prsi[0, 0, 1] * tx1[0, 0, 0]
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
                            (1.0 - prsi[0, 0, 1] * tx2[0, 0, 0])
                            * (1.0 - prsi[0, 0, 1] * tx2[0, 0, 0])
                            * 5.0
                        )
                    ),
                )
                xkzmo = xkzm_mx[0, 0, 0] * tem1

        z0 = 0.01 * zorl[0, 0, 0]
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl = 1
        hpbl = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = 1
        sfcflg = 1
        if rbsoil[0, 0, 0] > 0.0:
            sfcflg = 0
        pcnvflg = 0
        scuflg = 1
        radmin = 0.0
        mrad = km1
        krad = 0
        lcld = km1 - 1
        kcld = km1 - 1

        pix = psk[0, 0, 0] / prslk[0, 0, 0]
        theta = t1[0, 0, 0] * pix[0, 0, 0]
        if ntiw > 0:
            tem = max(q1_ntcw[0, 0, 0], qlmin)
            tem1 = max(q1_ntiw[0, 0, 0], qlmin)
            ptem = hvap * tem + (hvap + hfus) * tem1
            qlx = tem + tem1
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - ptem
        else:
            qlx = max(q1_ntcw[0, 0, 0], qlmin)
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - hvap * qlx[0, 0, 0]

        tem = 1.0 + fv * max(q1_0[0, 0, 0], qmin) - qlx[0, 0, 0]
        thvx = theta[0, 0, 0] * tem
        qtx = max(q1_0[0, 0, 0], qmin) + qlx[0, 0, 0]
        thlx = theta[0, 0, 0] - pix[0, 0, 0] * elocp * qlx[0, 0, 0]
        thlvx = thlx[0, 0, 0] * (1.0 + fv * qtx[0, 0, 0])
        svx = cp * t1[0, 0, 0] * tem
        thetae = theta[0, 0, 0] + elocp * pix[0, 0, 0] * max(q1_0[0, 0, 0], qmin)
        gotvx = g / (t1[0, 0, 0] * tem)

        tem = (t1[0, 0, 1] - t1[0, 0, 0]) * tem * rdzt[0, 0, 0]
        if tem > 1.0e-5:
            xkzo = min(xkzo[0, 0, 0], xkzinv)
            xkzmo = min(xkzmo[0, 0, 0], xkzinv)

        plyr = 0.01 * prsl[0, 0, 0]
        es = 0.01 * fpvs(t1)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + (eps - 1) * es))
        rhly = max(0.0, min(1.0, max(qmin, q1_0[0, 0, 0]) / qs))
        qstl = qs

    with computation(PARALLEL):
        with interval(...):
            cfly = 0.0
            clwt = 1.0e-6 * (plyr[0, 0, 0] * 0.001)
            if qlx[0, 0, 0] > clwt:
                onemrh = max(1.0e-10, 1.0 - rhly[0, 0, 0])
                tem1 = cql / min(max((onemrh * qstl[0, 0, 0]) ** 0.49, 0.0001), 1.0)
                val = max(min(tem1 * qlx[0, 0, 0], 50.0), 0.0)
                cfly = min(max(sqrt(sqrt(rhly[0, 0, 0])) * (1.0 - exp(-val)), 0.0), 1.0)

        with interval(0, -2):
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
                swh[0, 0, 0] * xmu[0, 0, 0] + hlw[0, 0, 0]
            )

        with interval(0, 1):
            sflux = heat[0, 0, 0] + evap[0, 0, 0] * fv * theta[0, 0, 0]

            if sfcflg[0, 0, 0] == 0 or sflux[0, 0, 0] <= 0.0:
                pblflg = 0

            if pblflg[0, 0, 0]:
                thermal = thlvx[0, 0, 0]
                crb = rbcr
            else:
                tem1 = 1e-7 * (
                    max(sqrt(u10m[0, 0, 0] ** 2 + v10m[0, 0, 0] ** 2), 1.0)
                    / (f0 * z0[0, 0, 0])
                )
                thermal = tsea[0, 0, 0] * (1.0 + fv * max(q1_0[0, 0, 0], qmin))
                crb = max(min(0.16 * (tem1 ** (-0.18)), crbmax), crbmin)

            dtdz1 = dt2 / (zi[0, 0, 1] - zi[0, 0, 0])
            ustar = sqrt(stress[0, 0, 0])

        with interval(0, -2):
            dw2 = (u1[0, 0, 0] - u1[0, 0, 1]) ** 2 + (v1[0, 0, 0] - v1[0, 0, 1]) ** 2
            shr2 = max(dw2, dw2min) * rdzt[0, 0, 0] * rdzt[0, 0, 0]

        with interval(...):
            rbup = rbsoil[0, 0, 0]

@gtscript.stencil(backend=backend)
def part3a(
    crb: FIELD_FLT,
    flg: FIELD_BOOL,
    kpblx: FIELD_INT,
    mask: FIELD_INT,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    thermal: FIELD_FLT,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    *,
    g: float
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]

            if flg[0, 0, 0] == 0:
                rbdn = rbup[0, 0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0, 0] > crb[0, 0, 0]

        with interval(1, None):
            thlvx_0 = thlvx_0[0, 0, -1]
            crb = crb[0, 0, -1]
            thermal = thermal[0, 0, -1]

            if flg[0, 0, -1] == 0:
                rbdn = rbup[0, 0, -1]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0, 0] > crb[0, 0, 0]
            else:
                rbdn = rbdn[0, 0, -1]
                rbup = rbup[0, 0, -1]
                kpblx = kpblx[0, 0, -1]
                flg = flg[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        rbdn = rbdn[0, 0, 1]
        rbup = rbup[0, 0, 1]
        kpblx = kpblx[0, 0, 1]
        flg = flg[0, 0, 1]


@gtscript.stencil(backend=backend)
def part3a1(
    crb: FIELD_FLT,
    evap: FIELD_FLT,
    fh: FIELD_FLT,
    flg: FIELD_BOOL,
    fm: FIELD_FLT,
    gotvx: FIELD_FLT,
    heat: FIELD_FLT,
    hpbl: FIELD_FLT,
    hpblx: FIELD_FLT,
    kpbl: FIELD_INT,
    kpblx: FIELD_INT,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL,
    pcnvflg: FIELD_BOOL,
    phih: FIELD_FLT,
    phim: FIELD_FLT,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    rbsoil: FIELD_FLT,
    sfcflg: FIELD_BOOL,
    sflux: FIELD_FLT,
    thermal: FIELD_FLT,
    theta: FIELD_FLT,
    ustar: FIELD_FLT,
    vpert: FIELD_FLT,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT,
    *,
    fv: float
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] > 0:
            kpblx = kpblx[0, 0, -1]
            hpblx = hpblx[0, 0, -1]
            kpbl = kpbl[0, 0, -1]
            hpbl = hpbl[0, 0, -1]
            kpbl = kpbl[0, 0, -1]
            pblflg = pblflg[0, 0, -1]
            crb = crb[0, 0, -1]
            rbup = rbup[0, 0, -1]
            rbdn = rbdn[0, 0, -1]

        if mask[0, 0, 0] == kpblx[0, 0, 0]:
            if kpblx[0, 0, 0] > 0:
                if rbdn[0, 0, 0] >= crb[0, 0, 0]:
                    rbint = 0.0
                elif rbup[0, 0, 0] <= crb[0, 0, 0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0, 0, 0] - rbdn[0, 0, 0]) / (
                        rbup[0, 0, 0] - rbdn[0, 0, 0]
                    )
                hpblx = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

                if hpblx[0, 0, 0] < zi[0, 0, 0]:
                    kpblx = kpblx[0, 0, 0] - 1
            else:
                hpblx = zl[0, 0, 0]
                kpblx = 0

            hpbl = hpblx[0, 0, 0]
            kpbl = kpblx[0, 0, 0]

            if kpbl[0, 0, 0] <= 0:
                pblflg = 0

    with computation(BACKWARD), interval(0, -1):
        kpblx = kpblx[0, 0, 1]
        hpblx = hpblx[0, 0, 1]
        kpbl = kpbl[0, 0, 1]
        hpbl = hpbl[0, 0, 1]
        pblflg = pblflg[0, 0, 1]

    with computation(PARALLEL), interval(0, 1):
        zol = max(rbsoil[0, 0, 0] * fm[0, 0, 0] * fm[0, 0, 0] / fh[0, 0, 0], rimin)
        if sfcflg[0, 0, 0]:
            zol = min(zol[0, 0, 0], -zfmin)
        else:
            zol = max(zol[0, 0, 0], zfmin)

        zol1 = zol[0, 0, 0] * sfcfrac * hpbl[0, 0, 0] / zl[0, 0, 0]

        if sfcflg[0, 0, 0]:
            phih = sqrt(1.0 / (1.0 - aphi16 * zol1))
            phim = sqrt(phih[0, 0, 0])
        else:
            phim = 1.0 + aphi5 * zol1
            phih = phim[0, 0, 0]

        pcnvflg = pblflg[0, 0, 0] and (zol[0, 0, 0] < zolcru)

        wst3 = gotvx[0, 0, 0] * sflux[0, 0, 0] * hpbl[0, 0, 0]
        ust3 = ustar[0, 0, 0] ** 3.0

        wscale = 0.0
        if pblflg[0, 0, 0]:
            wscale = max(
                (ust3 + wfac * vk * wst3 * sfcfrac) ** h1, ustar[0, 0, 0] / aphi5
            )

        flg = 1
        if pcnvflg[0, 0, 0]:
            hgamt = heat[0, 0, 0] / wscale
            hgamq = evap[0, 0, 0] / wscale
            vpert = max(hgamt + hgamq * fv * theta[0, 0, 0], 0.0)
            thermal = thermal[0, 0, 0] + min(cfac * vpert[0, 0, 0], gamcrt)
            flg = 0
            rbup = rbsoil[0, 0, 0]

@gtscript.stencil(backend=backend)
def part3c(
    crb: FIELD_FLT,
    flg: FIELD_BOOL,
    kpblx: FIELD_INT,
    mask: FIELD_INT,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    thermal: FIELD_FLT,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    *,
    g: float
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]

        with interval(1, 2):
            thlvx_0 = thlvx_0[0, 0, -1]
            crb = crb[0, 0, -1]
            thermal = thermal[0, 0, -1]
            rbup = rbup[0, 0, -1]
            flg = flg[0, 0, -1]
            kpblx = kpblx[0, 0, -1]
            if flg[0, 0, 0] == 0:
                rbdn = rbup[0, 0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0, 0] > crb[0, 0, 0]

        with interval(2, None):
            thlvx_0 = thlvx_0[0, 0, -1]
            crb = crb[0, 0, -1]
            thermal = thermal[0, 0, -1]
            rbup = rbup[0, 0, -1]
            flg = flg[0, 0, -1]
            kpblx = kpblx[0, 0, -1]
            if flg[0, 0, -1] == 0:
                rbdn = rbup[0, 0, -1]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0, 0] > crb[0, 0, 0]
            else:
                rbdn = rbdn[0, 0, -1]
                rbup = rbup[0, 0, -1]
                kpblx = kpblx[0, 0, -1]
                flg = flg[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        rbdn = rbdn[0, 0, 1]
        rbup = rbup[0, 0, 1]
        kpblx = kpblx[0, 0, 1]
        flg = flg[0, 0, 1]


@gtscript.stencil(backend=backend)
def part3c1(
    crb: FIELD_FLT,
    flg: FIELD_BOOL,
    hpbl: FIELD_FLT,
    kpbl: FIELD_INT,
    lcld: FIELD_INT,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL,
    pcnvflg: FIELD_BOOL,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    scuflg: FIELD_BOOL,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] > 0:
            crb = crb[0, 0, -1]
            hpbl = hpbl[0, 0, -1]
            kpbl = kpbl[0, 0, -1]
            pblflg = pblflg[0, 0, -1]
            pcnvflg = pcnvflg[0, 0, -1]
            rbdn = rbdn[0, 0, -1]
            rbup = rbup[0, 0, -1]

        if pcnvflg[0, 0, 0] and kpbl[0, 0, 0] == mask[0, 0, 0]:
            if rbdn[0, 0, 0] >= crb[0, 0, 0]:
                rbint = 0.0
            elif rbup[0, 0, 0] <= crb[0, 0, 0]:
                rbint = 1.0
            else:
                rbint = (crb[0, 0, 0] - rbdn[0, 0, 0]) / (rbup[0, 0, 0] - rbdn[0, 0, 0])

            hpbl[0, 0, 0] = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

            if hpbl[0, 0, 0] < zi[0, 0, 0]:
                kpbl[0, 0, 0] = kpbl[0, 0, 0] - 1

            if kpbl[0, 0, 0] <= 0:
                pblflg[0, 0, 0] = 0
                pcnvflg[0, 0, 0] = 0

    with computation(BACKWARD), interval(0, -1):
        hpbl = hpbl[0, 0, 1]
        kpbl = kpbl[0, 0, 1]
        pblflg = pblflg[0, 0, 1]
        pcnvflg = pcnvflg[0, 0, 1]

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0, 0]
            if flg[0, 0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0
        with interval(1, -1):
            lcld = lcld[0, 0, -1]
            flg = flg[0, 0, -1]
            if flg[0, 0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0

    with computation(BACKWARD), interval(0, -2):
        lcld = lcld[0, 0, 1]
        flg = flg[0, 0, 1]

@gtscript.stencil(backend=backend)
def part3e(
    flg: FIELD_BOOL,
    kcld: FIELD_INT,
    krad: FIELD_INT,
    lcld: FIELD_INT,
    mask: FIELD_INT,
    radmin: FIELD_FLT,
    radx: FIELD_FLT,
    qlx: FIELD_FLT,
    scuflg: FIELD_BOOL,
    *,
    km1: int
):

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0, 0]
        with interval(1, None):
            flg = flg[0, 0, -1]
            lcld = lcld[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                flg[0, 0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0, 0])
                and (qlx[0, 0, 0] >= qlcr)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

        with interval(0, -1):
            kcld[0, 0, 0] = kcld[0, 0, 1]
            flg[0, 0, 0] = flg[0, 0, 1]
            if (
                flg[0, 0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0, 0])
                and (qlx[0, 0, 0] >= qlcr)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

    with computation(FORWARD):
        with interval(0, 1):
            if scuflg[0, 0, 0] and (kcld[0, 0, 0] == (km1 - 1)):
                scuflg = 0
            flg = scuflg[0, 0, 0]

        with interval(1, None):
            flg = flg[0, 0, -1]
            kcld = kcld[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0, 0] and (mask[0, 0, 0] <= kcld[0, 0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

        with interval(0, -1):
            flg = flg[0, 0, 1]
            radmin = radmin[0, 0, 1]
            krad = krad[0, 0, 1]
            if flg[0, 0, 0] and (mask[0, 0, 0] <= kcld[0, 0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(PARALLEL), interval(0, 1):
        if scuflg[0, 0, 0] and krad[0, 0, 0] <= 0:
            scuflg = 0
        if scuflg[0, 0, 0] and radmin[0, 0, 0] >= 0.0:
            scuflg = 0


@gtscript.stencil(backend=backend)
def part4(
    pcnvflg: FIELD_BOOL,
    q1: FIELD_FLT,
    qcko: FIELD_FLT,
    qcdo: FIELD_FLT,
    scuflg: FIELD_BOOL,
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

    with computation(FORWARD), interval(1, None):
        pcnvflg = pcnvflg[0, 0, -1]
        scuflg = scuflg[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0, 0]:
            tcko = t1[0, 0, 0]
            ucko = u1[0, 0, 0]
            vcko = v1[0, 0, 0]
        if scuflg[0, 0, 0]:
            tcdo = t1[0, 0, 0]
            ucdo = u1[0, 0, 0]
            vcdo = v1[0, 0, 0]


@gtscript.stencil(backend=backend)
def part4a(
    pcnvflg_v2: FIELD_BOOL,
    q1: FIELD_FLT,
    qcdo: FIELD_FLT,
    qcko: FIELD_FLT,
    scuflg_v2: FIELD_BOOL,
):

    with computation(FORWARD), interval(1, None):
        pcnvflg_v2 = pcnvflg_v2[0, 0, -1]
        scuflg_v2 = scuflg_v2[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg_v2[0, 0, 0]:
            qcko = q1[0, 0, 0]
        if scuflg_v2[0, 0, 0]:
            qcdo = q1[0, 0, 0]


@gtscript.stencil(backend=backend)
def part5(
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    hpbl: FIELD_FLT,
    kpbl: FIELD_INT,
    mask: FIELD_INT,
    pcnvflg: FIELD_BOOL,
    phih: FIELD_FLT,
    phim: FIELD_FLT,
    prn: FIELD_FLT,
    zi: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        phih = phih[0, 0, -1]
        phim = phim[0, 0, -1]

    with computation(PARALLEL), interval(...):
        tem1 = max(zi[0, 0, 1] - sfcfrac * hpbl[0, 0, 0], 0.0)
        ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0, 0] ** 2.0)
        if mask[0, 0, 0] < kpbl[0, 0, 0]:
            if pcnvflg[0, 0, 0]:
                prn = 1.0 + ((phih / phim) - 1.0) * exp(ptem)
            else:
                prn = phih / phim

        if mask[0, 0, 0] < kpbl[0, 0, 0]:
            prn = max(min(prn[0, 0, 0], prmax), prmin)
            ckz = max(min(ck1 + (ck0 - ck1) * exp(ptem), ck0), ck1)
            chz = max(min(ch1 + (ch0 - ch1) * exp(ptem), ch0), ch1)


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
    kpbl: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    krad: FIELD_INT,
    pblflg: FIELD_BOOL,
    pcnvflg: FIELD_BOOL,
    phim: FIELD_FLT,
    prn: FIELD_FLT,
    prod: FIELD_FLT,
    radj: FIELD_FLT,
    rdzt: FIELD_FLT,
    rlam: FIELD_FLT,
    rle: FIELD_FLT,
    scuflg: FIELD_BOOL,
    sflux: FIELD_FLT,
    shr2: FIELD_FLT,
    stress: FIELD_FLT,
    tke: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    ustar: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        zol = zol[0, 0, -1]
        pblflg = pblflg[0, 0, -1]
        scuflg = scuflg[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, -1):
            if zol[0, 0, 0] < 0.0:
                zk = vk * zl[0, 0, 0] * (1.0 - 100.0 * zol[0, 0, 0]) ** 0.2
            elif zol[0, 0, 0] >= 1.0:
                zk = vk * zl[0, 0, 0] / 3.7
            else:
                zk = vk * zl[0, 0, 0] / (1.0 + 2.7 * zol[0, 0, 0])

            elm = zk * rlam[0, 0, 0] / (rlam[0, 0, 0] + zk)

            dz = zi[0, 0, 1] - zi[0, 0, 0]
            tem = max(gdx[0, 0, 0], dz)
            elm = min(elm[0, 0, 0], tem)
            ele = min(ele[0, 0, 0], tem)

            tem = (
                0.5
                * (elm[0, 0, 0] + elm[0, 0, 1])
                * sqrt(0.5 * (tke[0, 0, 0] + tke[0, 0, 1]))
            )
            ri = max(bf[0, 0, 0] / shr2[0, 0, 0], rimin)

            if mask[0, 0, 0] < kpbl[0, 0, 0]:
                if pblflg[0, 0, 0]:
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

            if scuflg[0, 0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
                    dku = dku_tmp
                    dkt = dkt_tmp

            dkq = prtke * dkt[0, 0, 0]

            dkt = max(min(dkt[0, 0, 0], dkmax), xkzo[0, 0, 0])

            dkq = max(min(dkq[0, 0, 0], dkmax), xkzo[0, 0, 0])

            dku = max(min(dku[0, 0, 0], dkmax), xkzmo[0, 0, 0])

        with interval(-1, None):
            elm = elm[0, 0, -1]
            ele = ele[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if mask[0, 0, 0] == krad[0, 0, 0]:
            if scuflg[0, 0, 0]:
                tem1 = bf[0, 0, 0] / gotvx[0, 0, 0]
                if tem1 < tdzmin:
                    tem1 = tdzmin
                ptem = radj[0, 0, 0] / tem1
                dkt = dkt[0, 0, 0] + ptem
                dku = dku[0, 0, 0] + ptem
                dkq = dkq[0, 0, 0] + ptem

    with computation(FORWARD), interval(1, -1):
        sflux = sflux[0, 0, -1]
        ustar = ustar[0, 0, -1]
        phim = phim[0, 0, -1]
        kpbl = kpbl[0, 0, -1]
        scuflg = scuflg[0, 0, -1]
        pcnvflg = pcnvflg[0, 0, -1]
        stress = stress[0, 0, -1]
        mrad = mrad[0, 0, -1]
        krad = krad[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, 1):
            if scuflg[0, 0, 0] and mrad[0, 0, 0] == 0:
                ptem = xmfd[0, 0, 0] * buod[0, 0, 0]
            else:
                ptem = 0.0

            buop = 0.5 * (
                gotvx[0, 0, 0] * sflux[0, 0, 0] + (-dkt[0, 0, 0] * bf[0, 0, 0] + ptem)
            )

            if scuflg[0, 0, 0] and mrad[0, 0, 0] == 0:
                ptem1 = (
                    0.5
                    * (u1[0, 0, 1] - u1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (ucdo[0, 0, 0] + ucdo[0, 0, 1] - u1[0, 0, 0] - u1[0, 0, 1])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0, 0] and mrad[0, 0, 0] == 0:
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
                    stress[0, 0, 0]
                    * ustar[0, 0, 0]
                    * phim[0, 0, 0]
                    / (vk * zl[0, 0, 0])
                )
            )

            prod = buop + shrp

        with interval(1, -1):
            if pcnvflg[0, 0, 0] and mask[0, 0, 0] <= kpbl[0, 0, 0]:
                ptem1 = 0.5 * (xmf[0, 0, -1] + xmf[0, 0, 0]) * buou[0, 0, 0]
            else:
                ptem1 = 0.0

            if scuflg[0, 0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
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

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] <= kpbl[0, 0, 0]:
                ptem1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1 + xmf[0, 0, -1] * tem2)
                    * (u1[0, 0, 0] - ucko[0, 0, 0])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
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

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] <= kpbl[0, 0, 0]:
                ptem1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1 + xmf[0, 0, -1] * tem2)
                    * (v1[0, 0, 0] - vcko[0, 0, 0])
                )
            else:
                ptem1 = 0.0

            if scuflg[0, 0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
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

@gtscript.stencil(backend=backend)
def part8(
    diss: FIELD_FLT,
    prod: FIELD_FLT,
    rle: FIELD_FLT,
    tke: FIELD_FLT,
    *,
    dtn: float
):
    with computation(PARALLEL), interval(...):
        diss = max(
            min(
                rle[0, 0, 0] * tke[0, 0, 0] * sqrt(tke[0, 0, 0]),
                prod[0, 0, 0] + tke[0, 0, 0] / dtn,
            ),
            0.0,
        )
        tke = max(tke[0, 0, 0] + dtn * (prod[0, 0, 0] - diss[0, 0, 0]), tkmin)


@gtscript.stencil(backend=backend)
def part9(
    pcnvflg: FIELD_BOOL,
    qcdo_ntke: FIELD_FLT,
    qcko_ntke: FIELD_FLT,
    scuflg: FIELD_BOOL,
    tke: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        scuflg = scuflg[0, 0, -1]
        pcnvflg = pcnvflg[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0, 0]:
            qcko_ntke = tke[0, 0, 0]
        if scuflg[0, 0, 0]:
            qcdo_ntke = tke[0, 0, 0]


@gtscript.stencil(backend=backend)
def part10(
    kpbl: FIELD_INT,
    mask: FIELD_INT,
    pcnvflg: FIELD_BOOL,
    qcko_ntke: FIELD_FLT,
    tke: FIELD_FLT,
    xlamue: FIELD_FLT,
    zl: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
        if pcnvflg[0, 0, 0] and mask[0, 0, 0] <= kpbl[0, 0, 0]:
            qcko_ntke = (
                (1.0 - tem) * qcko_ntke[0, 0, -1] + tem * (tke[0, 0, 0] + tke[0, 0, -1])
            ) / (1.0 + tem)


@gtscript.stencil(backend=backend)
def part11(
    ad: FIELD_FLT,
    f1: FIELD_FLT,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    qcdo_ntke: FIELD_FLT,
    scuflg: FIELD_BOOL,
    tke: FIELD_FLT,
    xlamde: FIELD_FLT,
    zl: FIELD_FLT,
):
    with computation(BACKWARD), interval(...):
        tem = 0.5 * xlamde[0, 0, 0] * (zl[0, 0, 1] - zl[0, 0, 0])
        if (
            scuflg[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
        ):
            qcdo_ntke = (
                (1.0 - tem) * qcdo_ntke[0, 0, 1] + tem * (tke[0, 0, 0] + tke[0, 0, 1])
            ) / (1.0 + tem)

    with computation(PARALLEL), interval(0, 1):
        ad = 1.0
        f1 = tke[0, 0, 0]


@gtscript.stencil(backend=backend)
def part12(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkq: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT,
    kpbl: FIELD_INT,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    pcnvflg: FIELD_BOOL,
    prsl: FIELD_FLT,
    qcdo_ntke: FIELD_FLT,
    qcko_ntke: FIELD_FLT,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL,
    tke: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    *,
    dt2: float
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

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] < kpbl[0, 0, 0]:
                tem = (
                    qcko_ntke[0, 0, 0]
                    + qcko_ntke[0, 0, 1]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * tem2 * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * tem2 * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                tem = (
                    qcdo_ntke[0, 0, 0]
                    + qcdo_ntke[0, 0, 1]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * tem2 * xmfd[0, 0, 0]
                f1_p1 = f1_p1[0, 0, 0] - tem * dtodsu * 0.5 * tem2 * xmfd[0, 0, 0]
        with interval(1, -1):
            ad = ad_p1[0, 0, -1]
            f1 = f1_p1[0, 0, -1]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] < kpbl[0, 0, 0]:
                tem = (
                    qcko_ntke[0, 0, 0]
                    + qcko_ntke[0, 0, 1]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * dsig * rdz * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * dsig * rdz * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                tem = (
                    qcdo_ntke[0, 0, 0]
                    + qcdo_ntke[0, 0, 1]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * dsig * rdz * xmfd[0, 0, 0]
                f1_p1 = f1_p1[0, 0, 0] - tem * dtodsu * 0.5 * dsig * rdz * xmfd[0, 0, 0]

        with interval(-1, None):
            ad = ad_p1[0, 0, -1]
            f1 = f1_p1[0, 0, -1]


@gtscript.stencil(backend=backend)
def part13(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkt: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT,
    f2: FIELD_FLT,
    f2_p1: FIELD_FLT,
    kpbl: FIELD_INT,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    pcnvflg: FIELD_BOOL,
    prsl: FIELD_FLT,
    q1: FIELD_FLT,  # q1(:,:,1)
    qcdo: FIELD_FLT,  # qcdo(:,:,1)
    qcko: FIELD_FLT,  # qcko(:,:,1)
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL,
    tcdo: FIELD_FLT,
    tcko: FIELD_FLT,
    t1: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    *,
    dt2: float,
    gocp: float
):

    with computation(FORWARD):
        with interval(0, -1):
            if mask[0, 0, 0] > 0:
                f1 = f1_p1[0, 0, -1]
                f2 = f2_p1[0, 0, -1]
                ad = ad_p1[0, 0, -1]

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

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] < kpbl[0, 0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = qcko[0, 0, 0] + qcko[0, 0, 1] - (q1[0, 0, 0] + q1[0, 0, 1])
                f2 = f2[0, 0, 0] - tem * ptem1
                f2_p1 = q1[0, 0, 1] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1]

            if (
                scuflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0, 0] - tem * ptem2
                tem = qcdo[0, 0, 0] + qcdo[0, 0, 1] - (q1[0, 0, 0] + q1[0, 0, 1])
                f2 = f2[0, 0, 0] + tem * ptem1
                f2_p1 = f2_p1[0, 0, 0] - tem * ptem2
        with interval(-1, None):
            f1 = f1_p1[0, 0, -1]
            f2 = f2_p1[0, 0, -1]
            ad = ad_p1[0, 0, -1]


@gtscript.stencil(backend=backend)
def part14(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    diss: FIELD_FLT,
    dku: FIELD_FLT,
    dtdz1: FIELD_FLT,

    du: FIELD_FLT,

    dusfc: FIELD_FLT,

    dv: FIELD_FLT,

    dvsfc: FIELD_FLT,

    f1: FIELD_FLT,
    f1_p1: FIELD_FLT,
    f2: FIELD_FLT,
    f2_p1: FIELD_FLT,

    hpbl: FIELD_FLT,
    
    hpblx: FIELD_FLT,

    kpbl: FIELD_INT,

    kpblx: FIELD_INT,

    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    pcnvflg: FIELD_BOOL,
    prsl: FIELD_FLT,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL,
    spd1: FIELD_FLT,
    stress: FIELD_FLT,
    tdt: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    *,
    conw: float,

    dspheat: bool,
    dt2: float,

    rdt: float
):

    with computation(PARALLEL):
        with interval(0, -1):
            if dspheat:
                tdt = tdt[0, 0, 0] + dspfac * (diss[0, 0, 0] / cp)

        with interval(0, 1):
            ad = 1.0 + dtdz1[0, 0, 0] * stress[0, 0, 0] / spd1[0, 0, 0]
            f1 = u1[0, 0, 0]
            f2 = v1[0, 0, 0]

    with computation(FORWARD):
        with interval(0, -1):
            if mask[0, 0, 0] > 0:
                f1 = f1_p1[0, 0, -1]
                f2 = f2_p1[0, 0, -1]
                ad = ad_p1[0, 0, -1]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0, 0] and mask[0, 0, 0] < kpbl[0, 0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2 = f2[0, 0, 0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2 = f2[0, 0, 0] + tem * ptem1
                f2_p1 = f2_p1[0, 0, 0] - tem * ptem2

        with interval(-1, None):
            f1 = f1_p1[0, 0, -1]
            f2 = f2_p1[0, 0, -1]
            ad = ad_p1[0, 0, -1]

    with computation(PARALLEL), interval(0, 1):
        fk = 1 / ad[0, 0, 0]
        au = fk * au[0, 0, 0]
        f1 = fk * f1[0, 0, 0]
        f2 = fk * f2[0, 0, 0]

    with computation(FORWARD):
        with interval(1, -1):
            fk = 1.0 / (ad[0, 0, 0] - al[0, 0, -1] * au[0, 0, -1])
            au = fk * au[0, 0, 0]
            f1 = fk * (f1[0, 0, 0] - al[0, 0, -1] * f1[0, 0, -1])
            f2 = fk * (f2[0, 0, 0] - al[0, 0, -1] * f2[0, 0, -1])
        with interval(-1, None):
            fk = 1.0 / (ad[0, 0, 0] - al[0, 0, -1] * au[0, 0, -1])
            f1 = fk * (f1[0, 0, 0] - al[0, 0, -1] * f1[0, 0, -1])
            f2 = fk * (f2[0, 0, 0] - al[0, 0, -1] * f2[0, 0, -1])

    with computation(BACKWARD), interval(0, -1):
        f1 = f1[0, 0, 0] - au[0, 0, 0] * f1[0, 0, 1]
        f2 = f2[0, 0, 0] - au[0, 0, 0] * f2[0, 0, 1]

    with computation(PARALLEL), interval(...):
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0, 0] + conw * del_[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0, 0] + conw * del_[0, 0, 0] * vtend

    with computation(BACKWARD), interval(0, -1):
        dusfc = dusfc[0, 0, 0] + dusfc[0, 0, 1]
        dvsfc = dvsfc[0, 0, 0] + dvsfc[0, 0, 1]

    with computation(PARALLEL), interval(0, 1):
        hpbl = hpblx[0, 0, 0]
        kpbl = kpblx[0, 0, 0]


@gtscript.stencil(backend=backend)
def part15(
    del_: FIELD_FLT,
    du: FIELD_FLT,
    dusfc: FIELD_FLT,
    dv: FIELD_FLT,
    dvsfc: FIELD_FLT,
    f1: FIELD_FLT,
    f2: FIELD_FLT,
    hpbl: FIELD_FLT,
    hpblx: FIELD_FLT,
    kpbl: FIELD_INT,
    kpblx: FIELD_INT,
    mask: FIELD_INT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    *,
    conw: float,
    rdt: float
):

    with computation(PARALLEL), interval(...):
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0, 0] + conw * del_[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0, 0] + conw * del_[0, 0, 0] * vtend

    with computation(BACKWARD), interval(0, -1):
        dusfc = dusfc[0, 0, 0] + dusfc[0, 0, 1]
        dvsfc = dvsfc[0, 0, 0] + dvsfc[0, 0, 1]

    with computation(PARALLEL), interval(0, 1):
        hpbl = hpblx[0, 0, 0]
        kpbl = kpblx[0, 0, 0]


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
    q1,
    q1_0,
    q1_ntcw,
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
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtu = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamuem = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlu = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtu = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlu = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    kpblx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    kpbly = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rbup = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    rbdn = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    flg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    hpblx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamavg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sumx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sigma = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    scaldfunc = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )

    qcko_1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcko_ntcw = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcko_track = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )

    totflag = True

    for i in range(im):
        totflag = totflag and ~cnvflg[i, 0, 0]

    if totflag:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

    mfpblt_s0(
        alp=alp,
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
        domain=(im, 1, kmpbl),
    )

    for k in range(1, kmpbl):
        for i in range(im):
            if qcko_track[i, 0, k] == 1:
                qcko[i, k, 0] = qcko_1[i, 0, k]
                qcko[i, k, ntcw - 1] = qcko_ntcw[i, 0, k]

    if ntcw > 2:
        for n in range(1, ntcw - 1):
            for k in range(1, kmpbl):
                for i in range(im):
                    if cnvflg[i, 0, 0] and k <= kpbl[i, 0, 0]:
                        dz = zl[i, 0, k] - zl[i, 0, k - 1]
                        tem = 0.5 * xlamue[i, 0, k - 1] * dz
                        factor = 1.0 + tem
                        qcko[i, k, n] = (
                            (1.0 - tem) * qcko[i, k - 1, n]
                            + tem * (q1[i, k, n] + q1[i, k - 1, n])
                        ) / factor

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw, ntrac1):
            for k in range(1, kmpbl):
                for i in range(im):
                    if cnvflg[i, 0, 0] and k <= kpbl[i, 0, 0]:
                        dz = zl[i, 0, k] - zl[i, 0, k - 1]
                        tem = 0.5 * xlamue[i, 0, k - 1] * dz
                        factor = 1.0 + tem

                        qcko[i, k, n] = (
                            (1.0 - tem) * qcko[i, k - 1, n]
                            + tem * (q1[i, k, n] + q1[i, k - 1, n])
                        ) / factor

    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue


@gtscript.stencil(backend=backend)
def mfpblt_s0(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    hpbl: FIELD_FLT,
    kpbl: FIELD_INT,
    q1_0: FIELD_FLT,
    q1_ntcw: FIELD_FLT,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    vpert: FIELD_FLT,
    wu2: FIELD_FLT,
    *,
    alp: float,
    g: float
):

    with computation(PARALLEL), interval(0, -1):
        if cnvflg[0, 0, 0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1_0[0, 0, 0] + q1_ntcw[0, 0, 0]

    with computation(PARALLEL), interval(0, 1):
        if cnvflg[0, 0, 0]:
            ptem = min(alp * vpert[0, 0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = g * ptem / thvx[0, 0, 0]

    # CK : This may not be needed later if stencils previous to this one update
    #       hpbl and kpbl over its entire range
    with computation(FORWARD), interval(1, None):
        hpbl = hpbl[0, 0, -1]
        kpbl = kpbl[0, 0, -1]


@gtscript.stencil(backend=backend)
def mfpblt_s1(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    flg: FIELD_BOOL,
    hpbl: FIELD_FLT,
    kpbl: FIELD_INT,
    kpblx: FIELD_INT,
    kpbly: FIELD_INT,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wu2: FIELD_FLT,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    *,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0, 0]:
                xlamue = ce0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0, 0]:
                wu2 = (4.0 * buo[0, 0, 0] * zm[0, 0, 0]) / (
                    1.0 + (0.5 * 2.0 * xlamue[0, 0, 0] * zm[0, 0, 0])
                )

        with interval(1, None):
            if cnvflg[0, 0, 0]:
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
            gamma = el2orc * qs / (tlu ** 2)
            qlu = dq / (1.0 + gamma)
            thvu = 0.0
            if cnvflg[0, 0, 0]:
                if dq > 0.0:
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * elocp * qlu) * (
                        1.0 + fv * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + fv * qtu[0, 0, 0])
                buo = g * (thvu / thvx[0, 0, 0] - 1.0)

    with computation(FORWARD):
        with interval(0, 1):
            flg = 1
            kpbly = kpbl[0, 0, 0]
            if cnvflg[0, 0, 0]:
                flg = 0
                rbup = wu2[0, 0, 0]
        with interval(1, None):
            if cnvflg[0, 0, 0]:
                dz = zm[0, 0, 0] - zm[0, 0, -1]
                tem = 0.25 * 2.0 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                wu2 = (((1.0 - tem) * wu2[0, 0, -1]) + (4.0 * buo[0, 0, 0] * dz)) / (
                    1.0 + tem
                )

    with computation(FORWARD), interval(1, None):
        kpblx = kpblx[0, 0, -1]
        flg = flg[0, 0, -1]
        rbup = rbup[0, 0, -1]
        rbdn = rbdn[0, 0, -1]
        if flg[0, 0, 0] == 0:
            rbdn = rbup[0, 0, 0]
            rbup = wu2[0, 0, 0]
            kpblx = mask[0, 0, 0]
            flg = rbup[0, 0, 0] < 0.0

    with computation(BACKWARD), interval(0, -1):
        rbup = rbup[0, 0, 1]
        rbdn = rbdn[0, 0, 1]
        kpblx = kpblx[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfpblt_s1a(
    cnvflg: FIELD_BOOL,
    hpblx: FIELD_FLT,
    kpblx: FIELD_INT,
    mask: FIELD_INT,
    rbdn: FIELD_FLT,
    rbup: FIELD_FLT,
    zm: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] > 0:
            hpblx = hpblx[0, 0, -1]
            rbdn = rbdn[0, 0, -1]
            rbup = rbup[0, 0, -1]
            cnvflg = cnvflg[0, 0, -1]

        rbint = 0.0

        if mask[0, 0, 0] == kpblx[0, 0, 0]:
            if cnvflg[0, 0, 0]:
                if rbdn[0, 0, 0] <= 0.0:
                    rbint = 0.0
                elif rbup[0, 0, 0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0, 0, 0] / (rbdn[0, 0, 0] - rbup[0, 0, 0])

                hpblx = zm[0, 0, -1] + rbint * (zm[0, 0, 0] - zm[0, 0, -1])

    with computation(BACKWARD), interval(0, -1):
        hpblx = hpblx[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfpblt_s2(
    cnvflg: FIELD_BOOL,
    gdx: FIELD_FLT,
    hpbl: FIELD_FLT,
    hpblx: FIELD_FLT,
    kpbl: FIELD_INT,
    kpblx: FIELD_INT,
    kpbly: FIELD_INT,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcko_1: FIELD_FLT,
    qcko_ntcw: FIELD_FLT,
    qcko_track: FIELD_INT,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    scaldfunc: FIELD_FLT,
    sigma: FIELD_FLT,
    sumx: FIELD_FLT,
    tcko: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    u1: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xlamavg: FIELD_FLT,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    wu2: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    *,
    a1: float,
    dt2: float,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float
):

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0, 0]:
                if kpbl[0, 0, 0] > kpblx[0, 0, 0]:
                    kpbl = kpblx[0, 0, 0]
                    hpbl = hpblx[0, 0, 0]
        with interval(1, None):
            kpbly = kpbly[0, 0, -1]
            kpbl = kpbl[0, 0, -1]
            hpbl = hpbl[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0, 0] and (kpbly[0, 0, 0] > kpblx[0, 0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0, 0] and (mask[0, 0, 0] < kpbl[0, 0, 0]):
                xlamavg = xlamavg[0, 0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0, 0] + dz
        with interval(1, None):
            xlamavg = xlamavg[0, 0, -1]
            sumx = sumx[0, 0, -1]
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0, 0] and (mask[0, 0, 0] < kpbl[0, 0, 0]):
                xlamavg = xlamavg[0, 0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0, 0] + dz

    with computation(BACKWARD), interval(0, -1):
        xlamavg = xlamavg[0, 0, 1]
        sumx = sumx[0, 0, 1]

    with computation(PARALLEL), interval(0, 1):
        if cnvflg[0, 0, 0]:
            xlamavg = xlamavg[0, 0, 0] / sumx[0, 0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0, 0] and (mask[0, 0, 0] < kpbl[0, 0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = a1 * sqrt(wu2[0, 0, 0])
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0, 0]:
                tem = 0.2 / xlamavg[0, 0, 0]
                sigma = min(
                    max((3.14 * tem * tem) / (gdx[0, 0, 0] * gdx[0, 0, 0]), 0.001),
                    0.999,
                )

                if sigma[0, 0, 0] > a1:
                    scaldfunc = max(
                        min((1.0 - sigma[0, 0, 0]) * (1.0 - sigma[0, 0, 0]), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0
        with interval(1, None):
            scaldfunc = scaldfunc[0, 0, -1]

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0, 0] and (mask[0, 0, 0] < kpbl[0, 0, 0]):
            xmf = min(scaldfunc[0, 0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0, 0] and (mask[0, 0, 0] <= kpbl[0, 0, 0]):
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

            if cnvflg[0, 0, 0] and (mask[0, 0, 0] <= kpbl[0, 0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko_1 = qs
                    qcko_ntcw = qlu
                    tcko = tlu + elocp * qlu
                    qcko_track = 1
                else:
                    qcko_1 = qtu[0, 0, 0]
                    qcko_ntcw = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0, 0] and (mask[0, 0, 0] <= kpbl[0, 0, 0]):
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
    q1_1,
    q1_ntcw,
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
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    hrad = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    krad1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thld = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qtd = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    thlvd = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ra1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    ra2 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    flg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_BOOL, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamdem = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    mradx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    mrady = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sumx = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xlamavg = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    xmfd = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    sigma = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    scaldfunc = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcdo_1 = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcdo_ntcw = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )
    qcdo_track = gt_storage.zeros(
        backend=backend, dtype=DTYPE_INT, shape=(im, 1, km + 1), default_origin=(0, 0, 0)
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0, 0]

    if totflg:
        return

    # mfscu_s0(buo=buo,
    #          cnvflg=cnvflg,
    #          q1_1=q1_1,
    #          q1_ntcw=q1_ntcw,
    #          qtx=qtx,
    #          wd2=wd2,
    #          domain=(im,1,km))

    mfscu_s0a(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        hrad=hrad,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        q1_1=q1_1,
        q1_ntcw=q1_ntcw,
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
        totflg = totflg and ~cnvflg[i, 0, 0]

    if totflg:
        return

    for k in range(kmscu):
        for i in range(im):
            if cnvflg[i, 0, 0]:
                dz = zl[i, 0, k + 1] - zl[i, 0, k]
                if (k >= mrad[i, 0, 0]) and (k < krad[i, 0, 0]):
                    if mrad[i, 0, 0] == 0:
                        ptem = 1.0 / (zm[i, 0, k] + dz)
                    else:
                        ptem = 1.0 / (zm[i, 0, k] - zm[i, 0, mrad[i, 0, 0] - 1] + dz)

                    xlamde[i, 0, k] = ce0 * (
                        ptem + 1.0 / max(hrad[i, 0, 0] - zm[i, 0, k] + dz, dz)
                    )
                else:
                    xlamde[i, 0, k] = ce0 / dz
                xlamdem[i, 0, k] = cm * xlamde[i, 0, k]

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
        radmin=radmin,
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
        totflg = totflg and ~cnvflg[i, 0, 0]

    if totflg:
        return

    for k in range(kmscu):
        for i in range(im):
            if cnvflg[i, 0, 0] and mrady[i, 0, 0] < mradx[i, 0, 0]:
                dz = zl[i, 0, k + 1] - zl[i, 0, k]
                if (k >= mrad[i, 0, 0]) and (k < krad[i, 0, 0]):
                    if mrad[i, 0, 0] == 0:
                        ptem = 1.0 / (zm[i, 0, k] + dz)
                    else:
                        ptem = 1.0 / (zm[i, 0, k] - zm[i, 0, mrad[i, 0, 0] - 1] + dz)
                    xlamde[i, 0, k] = ce0 * (
                        ptem + (1.0 / max(hrad[i, 0, 0] - zm[i, 0, k] + dz, dz))
                    )
                else:
                    xlamde[i, 0, k] = ce0 / dz
                xlamdem[i, 0, k] = cm * xlamde[i, 0, k]

    mfscu_s3(
        cnvflg=cnvflg,
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
        domain=(im, 1, kmscu),
    )

    mfscu_s3a(
        cnvflg=cnvflg, krad=krad, mask=mask, thld=thld, thlx=thlx, domain=(im, 1, km)
    )

    qcdo_1[:, :, :] = qcdo[:, :, 0].reshape((im, 1, km + 1))
    qcdo_ntcw[:, :, :] = qcdo[:, :, ntcw - 1].reshape((im, 1, km + 1))

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
        domain=(im, 1, kmscu),
    )

    for k in range(kmscu):
        for i in range(im):
            if qcdo_track[i, 0, k] == 1:
                qcdo[i, k, 0] = qcdo_1[i, 0, k]
                qcdo[i, k, ntcw - 1] = qcdo_ntcw[i, 0, k]

    if ntcw > 2:
        for n in range(1, ntcw - 1):
            for k in range(kmscu - 1, -1, -1):
                for i in range(im):
                    if cnvflg[i, 0, 0] and k < krad[i, 0, 0] and k >= mrad[i, 0, 0]:
                        dz = zl[i, 0, k + 1] - zl[i, 0, k]
                        tem = 0.5 * xlamde[i, 0, k] * dz
                        factor = 1.0 + tem
                        qcdo[i, k, n] = (
                            (1.0 - tem) * qcdo[i, k + 1, n]
                            + tem * (q1[i, k, n] + q1[i, k + 1, n])
                        ) / factor

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw, ntrac1):
            for k in range(kmscu - 1, -1, -1):
                for i in range(im):
                    if cnvflg[i, 0, 0] and k < krad[i, 0, 0] and k >= mrad[i, 0, 0]:
                        dz = zl[i, 0, k + 1] - zl[i, 0, k]
                        tem = 0.5 * xlamde[i, 0, k] * dz
                        factor = 1.0 + tem

                        qcdo[i, k, n] = (
                            (1.0 - tem) * qcdo[i, k + 1, n]
                            + tem * (q1[i, k, n] + q1[i, k + 1, n])
                        ) / factor

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde


@gtscript.stencil(backend=backend)
def mfscu_s0(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    q1_1: FIELD_FLT,
    q1_ntcw: FIELD_FLT,
    qtx: FIELD_FLT,
    wd2: FIELD_FLT,
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1_1[0, 0, 0] + q1_ntcw[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s0a(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    flg: FIELD_BOOL,
    hrad: FIELD_FLT,
    krad: FIELD_INT,
    krad1: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    q1_1: FIELD_FLT,
    q1_ntcw: FIELD_FLT,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    ra1: FIELD_FLT,
    ra2: FIELD_FLT,
    radmin: FIELD_FLT,
    radj: FIELD_FLT,
    thetae: FIELD_FLT,
    thld: FIELD_FLT,
    thlvd: FIELD_FLT,
    thlvx: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
    *,
    a1: float,
    a11: float,
    a2: float,
    a22: float,
    actei: float,
    cldtime: float,
    cp: float,
    hvap: float,
    g: float
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1_1[0, 0, 0] + q1_ntcw[0, 0, 0]

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] > 0:
            hrad = hrad[0, 0, -1]
            krad = krad[0, 0, -1]
            krad1 = krad1[0, 0, -1]
            ra1 = ra1[0, 0, -1]
            ra2 = ra2[0, 0, -1]
            radj = radj[0, 0, -1]
            cnvflg = cnvflg[0, 0, -1]
            radmin = radmin[0, 0, -1]
            thlvd = thlvd[0, 0, -1]

        if krad[0, 0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = mask[0, 0, 0] - 1
                tem1 = max(
                    cldtime * radmin[0, 0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0
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

                radj = -ra2[0, 0, 0] * radmin[0, 0, 0]

    with computation(PARALLEL), interval(0, 1):
        flg = cnvflg[0, 0, 0]
        mrad = krad[0, 0, 0]

    with computation(BACKWARD), interval(0, -1):
        thlvd = thlvd[0, 0, 1]
        radj = radj[0, 0, 1]
        ra1 = ra1[0, 0, 1]
        ra2 = ra2[0, 0, 1]
        krad1 = krad1[0, 0, 1]
        hrad = hrad[0, 0, 1]


@gtscript.stencil(backend=backend)
def mfscu_s0b(
    cnvflg: FIELD_BOOL,
    flg: FIELD_BOOL,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    thlvd: FIELD_FLT,
    thlvx: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        flg = flg[0, 0, -1]
        mrad = mrad[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
                if thlvd[0, 0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0, 0] = 0
        with interval(0, -1):
            mrad = mrad[0, 0, 1]
            flg = flg[0, 0, 1]

            if flg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
                if thlvd[0, 0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0, 0] = 0

    with computation(FORWARD), interval(0, 1):
        kk = krad[0, 0, 0] - mrad[0, 0, 0]
        if cnvflg[0, 0, 0]:
            if kk < 1:
                cnvflg[0, 0, 0] = 0


@gtscript.stencil(backend=backend)
def mfscu_s1(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    krad: FIELD_INT,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    radmin: FIELD_FLT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    xlamde: FIELD_FLT,
    zl: FIELD_FLT,
    *,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float
):

    with computation(FORWARD), interval(1, None):
        cnvflg = cnvflg[0, 0, -1]

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
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
        if cnvflg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + fv * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * elocp * qld) * tem1
            else:
                tem1 = 1.0 + fv * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = g * (1.0 - thvd / thvx[0, 0, 0])


@gtscript.stencil(backend=backend)
def mfscu_s1a(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    krad1: FIELD_INT,
    mask: FIELD_INT,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    zm: FIELD_FLT,
    *,
    bb1: float,
    bb2: float
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == krad1[0, 0, 0]:
            if cnvflg[0, 0, 0]:
                dz = zm[0, 0, 1] - zm[0, 0, 0]
                wd2 = (bb2 * buo[0, 0, 1] * dz) / (
                    1.0 + (0.5 * bb1 * xlamde[0, 0, 0] * dz)
                )


@gtscript.stencil(backend=backend)
def mfscu_s2(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL,
    flg: FIELD_BOOL,
    krad: FIELD_INT,
    krad1: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    mradx: FIELD_INT,
    mrady: FIELD_INT,
    xlamde: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
):

    with computation(FORWARD), interval(1, None):
        krad1 = krad1[0, 0, -1]
        mrad = mrad[0, 0, -1]
        krad = krad[0, 0, -1]

    with computation(BACKWARD), interval(...):
        dz = zm[0, 0, 1] - zm[0, 0, 0]
        tem = 0.25 * 2.0 * (xlamde[0, 0, 0] + xlamde[0, 0, 1]) * dz
        ptem1 = 1.0 + tem
        if cnvflg[0, 0, 0] and mask[0, 0, 0] < krad1[0, 0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD):
        with interval(0, 1):
            flg = cnvflg[0, 0, 0]
            mrady = mrad[0, 0, 0]
            if flg[0, 0, 0]:
                mradx = krad[0, 0, 0]
        with interval(1, None):
            flg = flg[0, 0, -1]
            mradx = mradx[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            flg = flg[0, 0, 1]
            mradx = mradx[0, 0, 1]
            if flg[0, 0, 0] and mask[0, 0, 0] < krad[0, 0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(PARALLEL), interval(0, 1):
        if cnvflg[0, 0, 0]:
            if mrad[0, 0, 0] < mradx[0, 0, 0]:
                mrad = mradx[0, 0, 0]
            if (krad[0, 0, 0] - mrad[0, 0, 0]) < 1:
                cnvflg = 0


@gtscript.stencil(backend=backend)
def mfscu_s3(
    cnvflg: FIELD_BOOL,
    gdx: FIELD_FLT,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    ra1: FIELD_FLT,
    scaldfunc: FIELD_FLT,
    sigma: FIELD_FLT,
    sumx: FIELD_FLT,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    xlamavg: FIELD_FLT,
    xmfd: FIELD_FLT,
    zl: FIELD_FLT,
    *,
    dt2: float
):

    with computation(FORWARD), interval(1, None):
        mrad = mrad[0, 0, -1]
        ra1 = ra1[0, 0, -1]
        cnvflg = cnvflg[0, 0, -1]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                cnvflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                dz = zl[0, 0, 1] - zl[0, 0, 0]
                xlamavg = xlamavg[0, 0, 0] + xlamde[0, 0, 0] * dz
                sumx = sumx[0, 0, 0] + dz
        with interval(0, -1):
            xlamavg = xlamavg[0, 0, 1]
            sumx = sumx[0, 0, 1]
            if (
                cnvflg[0, 0, 0]
                and mask[0, 0, 0] >= mrad[0, 0, 0]
                and mask[0, 0, 0] < krad[0, 0, 0]
            ):
                dz = zl[0, 0, 1] - zl[0, 0, 0]
                xlamavg = xlamavg[0, 0, 0] + xlamde[0, 0, 0] * dz
                sumx = sumx[0, 0, 0] + dz

    with computation(PARALLEL), interval(0, 1):
        if cnvflg[0, 0, 0]:
            xlamavg = xlamavg[0, 0, 0] / sumx[0, 0, 0]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
        ):
            if wd2[0, 0, 0] > 0:
                xmfd = ra1[0, 0, 0] * sqrt(wd2[0, 0, 0])
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0, 0]:
                tem1 = (3.14 * (0.2 / xlamavg[0, 0, 0]) * (0.2 / xlamavg[0, 0, 0])) / (
                    gdx[0, 0, 0] * gdx[0, 0, 0]
                )
                sigma = min(max(tem1, 0.001), 0.999)

            if cnvflg[0, 0, 0]:
                if sigma[0, 0, 0] > ra1[0, 0, 0]:
                    scaldfunc = max(
                        min((1.0 - sigma[0, 0, 0]) * (1.0 - sigma[0, 0, 0]), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0
        with interval(1, None):
            scaldfunc = scaldfunc[0, 0, -1]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
        ):
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0, 0] * xmfd[0, 0, 0], xmmx)


@gtscript.stencil(backend=backend)
def mfscu_s3a(
    cnvflg: FIELD_BOOL,
    krad: FIELD_INT,
    mask: FIELD_INT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        if krad[0, 0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0, 0]:
                thld = thlx[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s4(
    cnvflg: FIELD_BOOL,
    krad: FIELD_INT,
    mask: FIELD_INT,
    mrad: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcdo_1: FIELD_FLT,
    qcdo_ntcw: FIELD_FLT,
    qcdo_track: FIELD_INT,
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
    *,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if (
            cnvflg[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
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
            cnvflg[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
        ):
            qcdo_track = 1
            if dq > 0.0:
                qtd = qs + qld
                qcdo_1 = qs
                qcdo_ntcw = qld
                tcdo = tld + elocp * qld
            else:
                qcdo_1 = qtd[0, 0, 0]
                qcdo_ntcw = 0.0
                tcdo = tld

        if (
            cnvflg[0, 0, 0]
            and mask[0, 0, 0] < krad[0, 0, 0]
            and mask[0, 0, 0] >= mrad[0, 0, 0]
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


def tridit(l, n, nt, cl, cm, cu, rt, au, at):

    fk = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    )
    fkk = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    )

    tridit_s0(au=au, cm=cm, cl=cl, cu=cu, fk=fk, fkk=fkk, domain=(l, 1, n))

    for k in range(nt):
        is_ = k * n
        at[:l, 0, is_] = fk[:l, 0, 0] * rt[:l, 0, is_]

    for kk in range(nt):
        is_ = kk * n
        for k in range(1, n - 1):
            at[:l, 0, k + is_] = fkk[:l, 0, k] * (
                rt[:l, 0, k + is_] - cl[:l, 0, k - 1] * at[:l, 0, k + is_ - 1]
            )

    tridit_s1(au=au, cm=cm, cl=cl, cu=cu, fk=fk, domain=(l, 1, n))

    for k in range(nt):
        is_ = k * n
        at[:l, 0, n + is_ - 1] = fk[:l, 0, n - 1] * (
            rt[:l, 0, n + is_ - 1] - cl[:l, 0, n - 2] * at[:l, 0, n + is_ - 2]
        )

    for kk in range(nt):
        is_ = kk * n
        for k in range(n - 2, -1, -1):
            for i in range(l):
                at[i, 0, k + is_] = (
                    at[i, 0, k + is_] - au[i, 0, k] * at[i, 0, k + is_ + 1]
                )

    return au, at


@gtscript.stencil(backend=backend)
def tridit_s0(
    au: FIELD_FLT,
    cm: FIELD_FLT,
    cl: FIELD_FLT,
    cu: FIELD_FLT,
    fk: FIELD_FLT,
    fkk: FIELD_FLT,
):
    with computation(PARALLEL), interval(0, 1):
        fk = 1.0 / cm[0, 0, 0]
        au = fk[0, 0, 0] * cu[0, 0, 0]

    with computation(FORWARD), interval(1, -1):
        fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
        au = fkk[0, 0, 0] * cu[0, 0, 0]


@gtscript.stencil(backend=backend)
def tridit_s1(
    au: FIELD_FLT,
    cm: FIELD_FLT,
    cl: FIELD_FLT,
    cu: FIELD_FLT,
    fk: FIELD_FLT,
):

    with computation(PARALLEL), interval(-1, None):
        fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])


def tridin(l, n, nt, cl, cm, cu, r1, r2, au, a1, a2):
    fk = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    )
    fkk = gt_storage.zeros(
        backend=backend, dtype=DTYPE_FLT, shape=(l, 1, n + 1), default_origin=(0, 0, 0)
    )

    fk[:, 0, 0] = 1.0 / cm[:, 0, 0]
    au[:, 0, 0] = fk[:, 0, 0] * cu[:, 0, 0]
    a1[:, 0, 0] = fk[:, 0, 0] * r1[:, 0, 0]

    for k in range(nt):
        is_ = k * n
        a2[:, 0, is_] = fk[:, 0, 0] * r2[:, 0, is_]

    for k in range(1, n - 1):
        fkk[:, 0, k] = 1.0 / (cm[:, 0, k] - cl[:, 0, k - 1] * au[:, 0, k - 1])
        au[:, 0, k] = fkk[:, 0, k] * cu[:, 0, k]
        a1[:, 0, k] = fkk[:, 0, k] * (r1[:, 0, k] - cl[:, 0, k - 1] * a1[:, 0, k - 1])

    for kk in range(nt):
        is_ = kk * n
        for k in range(1, n - 1):
            a2[:, 0, k + is_] = fkk[:, 0, k] * (
                r2[:, 0, k + is_] - cl[:, 0, k - 1] * a2[:, 0, k + is_ - 1]
            )

    fk[:, 0, 0] = 1 / (cm[:, 0, n - 1] - cl[:, 0, n - 2] * au[:, 0, n - 2])
    a1[:, 0, n - 1] = fk[:, 0, 0] * (
        r1[:, 0, n - 1] - cl[:, 0, n - 2] * a1[:, 0, n - 2]
    )

    for k in range(nt):
        is_ = k * n
        a2[:, 0, n + is_ - 1] = fk[:, 0, 0] * (
            r2[:, 0, n + is_ - 1] - cl[:, 0, n - 2] * a2[:, 0, n + is_ - 2]
        )

    for k in range(n - 2, -1, -1):
        a1[:, 0, k] = a1[:, 0, k] - au[:, 0, k] * a1[:, 0, k + 1]

    for kk in range(nt):
        is_ = kk * n
        for k in range(n - 2, -1, -1):
            a2[:, 0, k + is_] = a2[:, 0, k + is_] - au[:, 0, k] * a2[:, 0, k + is_ + 1]

    return au, a1, a2
