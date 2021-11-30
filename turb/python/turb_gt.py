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

from stencils.turb_stencils import *

backend = BACKEND

class Turbulence:
    """
    Object for Turbulence Physics calcuations
    """

    def __init__(
        self,
        im,
        km,
        ntrac,
    ):

        # *** Multidimensional Storages ***
        self.qcko = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT, (ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

        self.qcdo = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

        self.f2 = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac-1,)),
            shape=(im, 1, km),
            default_origin=(0, 0, 0),
        )

        self.q1_gt = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

        self.rtg_gt = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
    
        # *** 3D Storage where the IJ data is shoved into the I dimension ***
        self.zi = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
        self.zl = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.zm = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ckz = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.chz = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.tke = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.rdzt = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.prn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xkzo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xkzmo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.pix = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.theta = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.qlx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.slx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.thvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.qtx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.thlx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.thlvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.thlvx_0 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.svx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.thetae = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.gotvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.plyr = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.cfly = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.bf = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.dku = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.dkt = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.dkq = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.radx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.shr2 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.tcko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.tcdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ucko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ucdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.vcko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.vcdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.buou = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xmf = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xlamue = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.rhly = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.qstl = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.buod = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xmfd = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.xlamde = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.rlam = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ele = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.elm = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.prod = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.rle = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.diss = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ad_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.f1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.f1_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.al = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.au = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

        self.f2_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )

        # 1D GT storages extended into 2D
        self.gdx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.kx1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        ) # kx1 could be taken out later, but leaving in for now
        self.kpblx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.hpblx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.pblflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.sfcflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.pcnvflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.scuflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.radmin = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.mrad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.krad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.lcld = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.kcld = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.flg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.rbup = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.rbdn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.sflux = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.thermal = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.crb = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.dtdz1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.ustar = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.zol = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.phim = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.phih = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.vpert = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self.radj = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.zlup = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self.zldn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )

        self.bsum = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )

        self.mlenflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )

        self.thvx_k = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )

        self.tke_k = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        # Mask/Index Array
        self.mask = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

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

    wd2 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
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
    xlamdem = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
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
    # 1D GT storages extended into 3D
    gdx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kx1 = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1, km + 1),
        default_origin=(0, 0, 0),
    )
    kpblx = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kpblx_mfp = gt_storage.zeros(
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
    hpblx_mfp = gt_storage.zeros(
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
    vpert = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    radj = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    zlup = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    zldn = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    bsum = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    mlenflg = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_BOOL,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    thvx_k = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )

    tke_k = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
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
    scaldfunc = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_FLT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kpbly = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    kpbly_mfp = gt_storage.zeros(
        backend=backend,
        dtype=DTYPE_INT,
        shape=(im, 1),
        default_origin=(0, 0, 0),
    )
    zm_mrad = gt_storage.zeros(
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
        xkzmo=xkzmo,
        xkzo=xkzo,
        xmu=xmu,
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
        qtx,
        wu2,
        qtu,
        xlamuem,
        thlu,
        kpblx_mfp,
        kpbly_mfp,
        rbup,
        rbdn,
        flg,
        hpblx_mfp,
        xlamavg,
        sumx,
        scaldfunc,
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
        qtx,
        wd2,
        hrad,
        krad1,
        thld,
        qtd,
        thlvd,
        ra1,
        ra2,
        flg,
        xlamdem,
        mradx,
        mrady,
        sumx,
        xlamavg,
        scaldfunc,
        zm_mrad,
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

    # Compute asymtotic mixing length
    # for k in range(km1):
    #     for i in range(im):
    #         zlup = 0.0
    #         bsum = 0.0
    #         mlenflg = True
    #         for n in range(k, km1):
    #             if mlenflg:
    #                 dz = zl[i, 0, n + 1] - zl[i, 0, n]
    #                 ptem = gotvx[i, 0, n] * (thvx[i, 0, n + 1] - thvx[i, 0, k]) * dz
    #                 bsum = bsum + ptem
    #                 zlup = zlup + dz
    #                 if bsum >= tke[i, 0, k]:
    #                     if ptem >= 0.0:
    #                         tem2 = max(ptem, zfmin)
    #                     else:
    #                         tem2 = min(ptem, -zfmin)
    #                     ptem1 = (bsum - tke[i, 0, k]) / tem2
    #                     zlup = zlup - ptem1 * dz
    #                     zlup = max(zlup, 0.0)
    #                     mlenflg = False
    #         zldn = 0.0
    #         bsum = 0.0
    #         mlenflg = True
    #         for n in range(k, -1, -1):
    #             if mlenflg:
    #                 if n == 0:
    #                     dz = zl[i, 0, 0]
    #                     tem1 = tsea[i, 0] * (1.0 + fv * max(q1_gt[i, 0, 0, 0], qmin))
    #                 else:
    #                     dz = zl[i, 0, n] - zl[i, 0, n - 1]
    #                     tem1 = thvx[i, 0, n - 1]
    #                 ptem = gotvx[i, 0, n] * (thvx[i, 0, k] - tem1) * dz
    #                 bsum = bsum + ptem
    #                 zldn = zldn + dz
    #                 if bsum >= tke[i, 0, k]:
    #                     if ptem >= 0.0:
    #                         tem2 = max(ptem, zfmin)
    #                     else:
    #                         tem2 = min(ptem, -zfmin)
    #                     ptem1 = (bsum - tke[i, 0, k]) / tem2
    #                     zldn = zldn - ptem1 * dz
    #                     zldn = max(zldn, 0.0)
    #                     mlenflg = False

    #         tem = 0.5 * (zi[i, 0, k + 1] - zi[i, 0, k])
    #         tem1 = min(tem, rlmn)

    #         ptem2 = min(zlup, zldn)
    #         rlam[i, 0, k] = elmfac * ptem2
    #         rlam[i, 0, k] = max(rlam[i, 0, k], tem1)
    #         rlam[i, 0, k] = min(rlam[i, 0, k], rlmx)

    #         ptem2 = math.sqrt(zlup * zldn)
    #         ele[i, 0, k] = elefac * ptem2
    #         ele[i, 0, k] = max(ele[i, 0, k], tem1)
    #         ele[i, 0, k] = min(ele[i, 0, k], elmx)

    for k in range(km1):
        # for n in range(k,km1):            
        #     for i in range(im):
        #         if n == k:
        #             mlenflg[i,0] = True
        #             bsum[i,0] = 0.0
        #             zlup_tmp[i,0] = 0.0
        #             thvx_k[i,0] = thvx[i, 0 , n]
        #             tke_k[i,0] = tke[i, 0, n]
        #         if mlenflg[i,0] == True:
        #             dz = zl[i, 0, n + 1] - zl[i, 0, n]
        #             ptem = gotvx[i, 0, n] * (thvx[i, 0, n + 1] - thvx_k[i, 0]) * dz
        #             bsum[i,0] = bsum[i,0] + ptem
        #             zlup_tmp[i,0] = zlup_tmp[i,0] + dz
        #             if bsum[i,0] >= tke_k[i, 0]:
        #                 if ptem >= 0.0:
        #                         tem2 = max(ptem, zfmin)
        #                 else:
        #                     tem2 = min(ptem, -zfmin)
        #                 ptem1 = (bsum[i,0] - tke_k[i, 0]) / tem2
        #                 zlup_tmp[i,0] = zlup_tmp[i,0] - ptem1 * dz
        #                 zlup_tmp[i,0] = max(zlup_tmp[i,0], 0.0)
        #                 mlenflg[i,0] = False

        comp_asym_mix_up(mask=mask,
                         mlenflg=mlenflg,
                         bsum=bsum,
                         zlup=zlup,
                         thvx_k=thvx_k,
                         tke_k=tke_k,
                         thvx=thvx,
                         tke=tke,
                         gotvx=gotvx,
                         zl=zl,
                         zfmin=zfmin,
                         k=k,
                         domain=(im,1,km1-k),
                         origin=(0,0,k)
                         )

        # print("BREAK!")

        # for n in range(k, -1, -1):
        #     for i in range(im):
        #         if n == k:
        #             mlenflg[i,0] = True
        #             bsum[i,0] = 0.0
        #             zldn[i,0] = 0.0
        #             thvx_k[i,0] = thvx[i, 0 , n]
        #             tke_k[i,0] = tke[i, 0, n]

        #         if mlenflg[i,0] == True:
        #             if n == 0:
        #                 dz = zl[i, 0, 0]
        #                 tem1 = tsea[i, 0] * (1.0 + fv * max(q1_gt[i, 0, 0, 0], qmin))
        #             else:
        #                 dz = zl[i, 0, n] - zl[i, 0, n - 1]
        #                 tem1 = thvx[i, 0, n - 1]
        #             ptem = gotvx[i, 0, n] * (thvx_k[i, 0] - tem1) * dz
        #             bsum[i,0] = bsum[i,0] + ptem
        #             zldn[i,0] = zldn[i,0] + dz
        #             if bsum[i,0] >= tke_k[i, 0]:
        #                 if ptem >= 0.0:
        #                     tem2 = max(ptem, zfmin)
        #                 else:
        #                     tem2 = min(ptem, -zfmin)
        #                 ptem1 = (bsum[i,0] - tke_k[i, 0]) / tem2
        #                 zldn[i,0] = zldn[i,0] - ptem1 * dz
        #                 zldn[i,0] = max(zldn[i,0], 0.0)
        #                 mlenflg[i,0] = False

        comp_asym_mix_dn(mask=mask,
                         mlenflg=mlenflg,
                         bsum=bsum,
                         zldn=zldn,
                         thvx_k=thvx_k,
                         tke_k=tke_k,
                         thvx=thvx,
                         tke=tke,
                         gotvx=gotvx,
                         zl=zl,
                         tsea=tsea,
                         q1_gt=q1_gt,
                         zfmin=zfmin,
                         fv=fv,
                         k=k,
                         domain=(im,1,k+1),
                         origin=(0,0,0),
                         )

        # print("BREAK")

        # for i in range(im):
        #     tem = 0.5 * (zi[i, 0, k + 1] - zi[i, 0, k])
        #     tem1 = min(tem, rlmn)

        #     ptem2 = min(zlup_tmp[i,0], zldn[i,0])
        #     rlam[i, 0, k] = elmfac * ptem2
        #     rlam[i, 0, k] = max(rlam[i, 0, k], tem1)
        #     rlam[i, 0, k] = min(rlam[i, 0, k], rlmx)

        #     ptem2 = math.sqrt(zlup_tmp[i,0] * zldn[i,0])
        #     ele[i, 0, k] = elefac * ptem2
        #     ele[i, 0, k] = max(ele[i, 0, k], tem1)
        #     ele[i, 0, k] = min(ele[i, 0, k], elmx)

        comp_asym_rlam_ele(zi=zi,
                           rlam=rlam,
                           ele=ele,
                           zlup=zlup,
                           zldn=zldn,
                           rlmn=rlmn,
                           rlmx=rlmx,
                           elmfac=elmfac,
                           elmx=elmx,
                           domain=(im,1,1),
                           origin=(0,0,k),
        )

        # print("BREAK")

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
