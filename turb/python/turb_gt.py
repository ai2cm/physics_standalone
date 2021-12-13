#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np
import gt4py
import gt4py.storage as gt_storage
import time
import timeit

from config import *
from stencils.turb_stencils import *

backend = BACKEND

class Turbulence:
    """
    Object for Turbulence Physics calculations
    """

    def __init__(
        self,
        im,
        ix,
        km,
        ntrac,
        ntcw,
        ntiw,
        ntke,
        delt,
        dspheat,
        xkzm_m,
        xkzm_h,
        xkzm_s,
    ):

       # *** Multi-dimensional storages ***
        self._qcko = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT, (ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qcdo = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._f2 = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac-1,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._q1 = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._rtg_gt = gt_storage.zeros(
            backend=backend,
            dtype=(DTYPE_FLT,(ntrac,)),
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        # *** 3D storages ***
        self._zi = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._zl = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._zm = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ckz = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._chz = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._tke = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._rdzt = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._prn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xkzo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xkzmo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._pix = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._theta = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qlx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._slx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qtx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thlx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thlvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._svx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thetae = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._gotvx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._plyr = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._cfly = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._bf = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._dku = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._dkt = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._dkq = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._radx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._shr2 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._tcko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._tcdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ucko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ucdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._vcko = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._vcdo = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._buou = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xmf = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xlamue = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._rhly = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qstl = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._buod = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xmfd = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xlamde = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._rlam = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ele = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._elm = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._prod = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._rle = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._diss = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._f1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._al = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._au = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._wd2 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thld = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qtd = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xlamdem = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._wu2 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._qtu = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._xlamuem = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._thlu = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        # 1D GT storages extended into 2D
        self._f1_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._f2_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._ad_p1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._thlvx_0 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._gdx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._kx1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._kpblx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._kpblx_mfp = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._hpblx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._hpblx_mfp = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._pblflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._sfcflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._pcnvflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._scuflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._radmin = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._mrad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._krad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._lcld = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._kcld = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._flg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._rbup = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._rbdn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._sflux = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._thermal = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._crb = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._dtdz1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )
        self._ustar = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._zol = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._phim = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._phih = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._vpert = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._radj = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._zlup = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._zldn = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._bsum = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._mlenflg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_BOOL,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._thvx_k = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._tke_k = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._hrad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._krad1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._thlvd = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._ra1 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._ra2 = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._mradx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._mrady = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._sumx = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._xlamavg = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._scaldfunc = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._kpbly = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._kpbly_mfp = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        self._zm_mrad = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_FLT,
            shape=(im, 1),
            default_origin=(0, 0, 0),
        )
        # Mask/Index Array
        self._mask = gt_storage.zeros(
            backend=backend,
            dtype=DTYPE_INT,
            shape=(im, 1, km + 1),
            default_origin=(0, 0, 0),
        )

        mask_init(mask=self._mask)

        # Constants
        self._fv = rv / rd - 1.0
        self._eps = rd / rv
        self._epsm1 = rd / rv - 1.0

        self._gravi = 1.0 / grav
        self._g = grav
        self._gocp = self._g / cp
        self._cont = cp / self._g
        self._conq = hvap / self._g
        self._conw = 1.0 / self._g
        self._elocp = hvap / cp
        self._el2orc = hvap * hvap / (rv * cp)

        self._dt2 = delt
        self._rdt = 1.0 / self._dt2
        self._ntrac = ntrac
        self._ntrac1 = ntrac - 1
        self._km1 = km - 1
        self._kmpbl = int(km / 2)
        self._kmscu = int(km / 2)
        self._km = km
        self._im = im
        self._ix = ix
        self._ntcw = ntcw
        self._ntiw = ntiw
        self._ntke = ntke
        self._dspheat = dspheat
        self._xkzm_m = xkzm_m
        self._xkzm_h = xkzm_h
        self._xkzm_s = xkzm_s
        self._tkmin = tkmin
        self._zfmin = zfmin
        self._rlmn = rlmn
        self._rlmx = rlmx
        self._elmfac = elmfac
        self._elmx = elmx
        self._cdtn = cdtn

        self._kk = max(round(self._dt2 / self._cdtn), 1)
        self._dtn = self._dt2 / self._kk

        self._ce0 = ce0
        self._cm = 1.0
        self._qmin = qmin
        self._qlmin = qlmin
        self._alp = 1.0
        self._pgcon = 0.55

        self._a1_mfpblt = 0.13
        self._a1_mfscu = 0.12

        self._b1_mfpblt = 0.5
        self._b1_mfscu = 0.45

        self._a2 = 0.50
        self._a11 = 0.2
        self._a22 = 1.0
        self._cldtime = 500.0
        self._actei = 0.7
        self._hvap = hvap
        self._cp = cp
        self._f1_const = 0.15

    def turbInit(
        self,
        garea,
        prsi,
        kinver,
        zorl,
        dusfc,
        dvsfc,
        dtsfc,
        dqsfc,
        kpbl,
        hpbl,
        rbsoil,
        evap,
        heat,
        psk,
        xmu,
        tsea,
        u10m,
        v10m,
        stress,
        fm,
        fh,
        spd1,
        phii,
        phil,
        swh,
        hlw,
        u1,
        v1,
        del_,
        du,
        dv,
        tdt,
        prslk,
        t1,
        prsl,
        q1,
        rtg,
    ):

        # *** Note : These storage declarations below may become regular storage allocations
        # ***        in __init__.  For now, this will use turbInit to take input 
        # ***        for the standalone test.
        self._garea = numpy_to_gt4py_storage_1D(garea, backend, self._km + 1)
        self._tx1 = numpy_to_gt4py_storage_1D(1.0 / prsi[:, 0], backend, self._km + 1)
        self._tx2 = numpy_to_gt4py_storage_1D(1.0 / prsi[:, 0], backend, self._km + 1)
        self._kinver = numpy_to_gt4py_storage_1D(kinver, backend, self._km + 1)
        self._zorl = numpy_to_gt4py_storage_1D(zorl, backend, self._km + 1)
        self._dusfc = numpy_to_gt4py_storage_1D(dusfc, backend, self._km + 1)
        self._dvsfc = numpy_to_gt4py_storage_1D(dvsfc, backend, self._km + 1)
        self._dtsfc = numpy_to_gt4py_storage_1D(dtsfc, backend, self._km + 1)
        self._dqsfc = numpy_to_gt4py_storage_1D(dqsfc, backend, self._km + 1)
        self._kpbl = numpy_to_gt4py_storage_1D(kpbl, backend, self._km + 1)
        self._hpbl = numpy_to_gt4py_storage_1D(hpbl, backend, self._km + 1)
        self._rbsoil = numpy_to_gt4py_storage_1D(rbsoil, backend, self._km + 1)
        self._evap = numpy_to_gt4py_storage_1D(evap, backend, self._km + 1)
        self._heat = numpy_to_gt4py_storage_1D(heat, backend, self._km + 1)
        self._psk = numpy_to_gt4py_storage_1D(psk, backend, self._km + 1)
        self._xmu = numpy_to_gt4py_storage_1D(xmu, backend, self._km + 1)
        self._tsea = numpy_to_gt4py_storage_1D(tsea, backend, self._km + 1)
        self._u10m = numpy_to_gt4py_storage_1D(u10m, backend, self._km + 1)
        self._v10m = numpy_to_gt4py_storage_1D(v10m, backend, self._km + 1)
        self._stress = numpy_to_gt4py_storage_1D(stress, backend, self._km + 1)
        self._fm = numpy_to_gt4py_storage_1D(fm, backend, self._km + 1)
        self._fh = numpy_to_gt4py_storage_1D(fh, backend, self._km + 1)
        self._spd1 = numpy_to_gt4py_storage_1D(spd1, backend, self._km + 1)

        self._phii = numpy_to_gt4py_storage_2D(phii, backend, self._km + 1)
        self._phil = numpy_to_gt4py_storage_2D(phil, backend, self._km + 1)
        self._prsi = numpy_to_gt4py_storage_2D(prsi, backend, self._km + 1)
        self._swh = numpy_to_gt4py_storage_2D(swh, backend, self._km + 1)
        self._hlw = numpy_to_gt4py_storage_2D(hlw, backend, self._km + 1)
        self._u1 = numpy_to_gt4py_storage_2D(u1, backend, self._km + 1)
        self._v1 = numpy_to_gt4py_storage_2D(v1, backend, self._km + 1)
        self._del_ = numpy_to_gt4py_storage_2D(del_, backend, self._km + 1)
        self._du = numpy_to_gt4py_storage_2D(du, backend, self._km + 1)
        self._dv = numpy_to_gt4py_storage_2D(dv, backend, self._km + 1)
        self._tdt = numpy_to_gt4py_storage_2D(tdt, backend, self._km + 1)
        self._prslk = numpy_to_gt4py_storage_2D(prslk, backend, self._km + 1)
        self._t1 = numpy_to_gt4py_storage_2D(t1, backend, self._km + 1)
        self._prsl = numpy_to_gt4py_storage_2D(prsl, backend, self._km + 1)

        for I in range(self._ntrac):
            self._q1[:, 0, :-1, I] = q1[:, :, I]
            self._rtg_gt[:,0,:-1, I] = rtg[:,:,I]

        
    def run_turb(
        self,
    ):
        init(
            bf=self._bf,
            cfly=self._cfly,
            chz=self._chz,
            ckz=self._ckz,
            crb=self._crb,
            dqsfc=self._dqsfc,
            dt2=self._dt2,
            dtdz1=self._dtdz1,
            dtsfc=self._dtsfc,
            dusfc=self._dusfc,
            dvsfc=self._dvsfc,
            elocp=self._elocp,
            el2orc=self._el2orc,
            eps=self._eps,
            evap=self._evap,
            fv=self._fv,
            garea=self._garea,
            gdx=self._gdx,
            heat=self._heat,
            hlw=self._hlw,
            hpbl=self._hpbl,
            hpblx=self._hpblx,
            kpbl=self._kpbl,
            kcld=self._kcld,
            kinver=self._kinver,
            km1=self._km1,
            kpblx=self._kpblx,
            krad=self._krad,
            kx1=self._kx1,
            lcld=self._lcld,
            g=self._g,
            gotvx=self._gotvx,
            gravi=self._gravi,
            mask=self._mask,
            mrad=self._mrad,
            pblflg=self._pblflg,
            pcnvflg=self._pcnvflg,
            phii=self._phii,
            phil=self._phil,
            pix=self._pix,
            plyr=self._plyr,
            prn=self._prn,
            prsi=self._prsi,
            prsl=self._prsl,
            prslk=self._prslk,
            psk=self._psk,
            q1=self._q1,
            qlx=self._qlx,
            qstl=self._qstl,
            qtx=self._qtx,
            radmin=self._radmin,
            radx=self._radx,
            rbsoil=self._rbsoil,
            rbup=self._rbup,
            rdzt=self._rdzt,
            rhly=self._rhly,
            sfcflg=self._sfcflg,
            sflux=self._sflux,
            scuflg=self._scuflg,
            shr2=self._shr2,
            slx=self._slx,
            stress=self._stress,
            svx=self._svx,
            swh=self._swh,
            t1=self._t1,
            thermal=self._thermal,
            theta=self._theta,
            thetae=self._thetae,
            thlvx=self._thlvx,
            thlx=self._thlx,
            thvx=self._thvx,
            tke=self._tke,
            tkmin=self._tkmin,
            tsea=self._tsea,
            tx1=self._tx1,
            tx2=self._tx2,
            u10m=self._u10m,
            ustar=self._ustar,
            u1=self._u1,
            v1=self._v1,
            v10m=self._v10m,
            xkzm_h=self._xkzm_h,
            xkzm_m=self._xkzm_m,
            xkzm_s=self._xkzm_s,
            xkzmo=self._xkzmo,
            xkzo=self._xkzo,
            xmu=self._xmu,
            zi=self._zi,
            zl=self._zl,
            zm=self._zm,
            zorl=self._zorl,
            ntke=self._ntke-1,
            ntcw=self._ntcw-1,
            ntiw=self._ntiw-1,
            domain=(self._im, 1, self._km + 1),
        )

        mrf_pbl_scheme_part1(
            crb=self._crb,
            flg=self._flg,
            g=self._g,
            kpblx=self._kpblx,
            mask=self._mask,
            rbdn=self._rbdn,
            rbup=self._rbup,
            thermal=self._thermal,
            thlvx=self._thlvx,
            thlvx_0=self._thlvx_0,
            u1=self._u1,
            v1=self._v1,
            zl=self._zl,
            domain=(self._im, 1, self._kmpbl),
        )

        mrf_pbl_2_thermal_1(
            crb=self._crb,
            evap=self._evap,
            fh=self._fh,
            flg=self._flg,
            fm=self._fm,
            gotvx=self._gotvx,
            heat=self._heat,
            hpbl=self._hpbl,
            hpblx=self._hpblx,
            kpbl=self._kpbl,
            kpblx=self._kpblx,
            mask=self._mask,
            pblflg=self._pblflg,
            pcnvflg=self._pcnvflg,
            phih=self._phih,
            phim=self._phim,
            rbdn=self._rbdn,
            rbup=self._rbup,
            rbsoil=self._rbsoil,
            sfcflg=self._sfcflg,
            sflux=self._sflux,
            thermal=self._thermal,
            theta=self._theta,
            ustar=self._ustar,
            vpert=self._vpert,
            zi=self._zi,
            zl=self._zl,
            zol=self._zol,
            fv=self._fv,
            domain=(self._im, 1, self._km),
        )

        thermal_2(
            crb=self._crb,
            flg=self._flg,
            g=self._g,
            kpbl=self._kpbl,
            mask=self._mask,
            rbdn=self._rbdn,
            rbup=self._rbup,
            thermal=self._thermal,
            thlvx=self._thlvx,
            thlvx_0=self._thlvx_0,
            u1=self._u1,
            v1=self._v1,
            zl=self._zl,
            domain=(self._im, 1, self._kmpbl),
        )

        pbl_height_enhance(
            crb=self._crb,
            flg=self._flg,
            hpbl=self._hpbl,
            kpbl=self._kpbl,
            lcld=self._lcld,
            mask=self._mask,
            pblflg=self._pblflg,
            pcnvflg=self._pcnvflg,
            rbdn=self._rbdn,
            rbup=self._rbup,
            scuflg=self._scuflg,
            zi=self._zi,
            zl=self._zl,
            domain=(self._im, 1, self._km),
        )

        stratocumulus(
            flg=self._flg,
            kcld=self._kcld,
            krad=self._krad,
            lcld=self._lcld,
            km1=self._km1,
            mask=self._mask,
            radmin=self._radmin,
            radx=self._radx,
            qlx=self._qlx,
            scuflg=self._scuflg,
            domain=(self._im, 1, self._kmscu),
        )

        mass_flux_comp(
            pcnvflg=self._pcnvflg,
            q1=self._q1,
            scuflg=self._scuflg,
            t1=self._t1,
            tcdo=self._tcdo,
            tcko=self._tcko,
            u1=self._u1,
            ucdo=self._ucdo,
            ucko=self._ucko,
            v1=self._v1,
            vcdo=self._vcdo,
            vcko=self._vcko,
            qcdo=self._qcdo,
            qcko=self._qcko,
        )

        self._kpbl, self._hpbl, self._buou, self._xmf, \
        self._tcko, self._qcko, self._ucko, self._vcko, \
        self._xlamue = mfpblt(
            self._im,
            self._ix,
            self._km,
            self._kmpbl,
            self._ntcw,
            self._ntrac1,
            self._dt2,
            self._pcnvflg,
            self._zl,
            self._zm,
            self._q1,
            self._t1,
            self._u1,
            self._v1,
            self._plyr,
            self._pix,
            self._thlx,
            self._thvx,
            self._gdx,
            self._hpbl,
            self._kpbl,
            self._vpert,
            self._buou,
            self._xmf,
            self._tcko,
            self._qcko,
            self._ucko,
            self._vcko,
            self._xlamue,
            self._g,
            self._gocp,
            self._elocp,
            self._el2orc,
            self._mask,
            self._qtx,
            self._wu2,
            self._qtu,
            self._xlamuem,
            self._thlu,
            self._kpblx_mfp,
            self._kpbly_mfp,
            self._rbup,
            self._rbdn,
            self._flg,
            self._hpblx_mfp,
            self._xlamavg,
            self._sumx,
            self._scaldfunc,
            self._ce0,
            self._cm,
            self._qmin,
            self._qlmin,
            self._alp,
            self._pgcon,
            self._a1_mfpblt,
            self._b1_mfpblt,
            self._f1_const,
            self._fv,
            self._eps,
            self._epsm1,
        )

        self._radj, self._mrad, self._buod, self._xmfd, \
        self._tcdo, self._qcdo, self._ucdo, self._vcdo, self._xlamde = mfscu(
            self._im,
            self._ix,
            self._km,
            self._kmscu,
            self._ntcw,
            self._ntrac1,
            self._dt2,
            self._scuflg,
            self._zl,
            self._zm,
            self._q1,
            self._t1,
            self._u1,
            self._v1,
            self._plyr,
            self._pix,
            self._thlx,
            self._thvx,
            self._thlvx,
            self._gdx,
            self._thetae,
            self._radj,
            self._krad,
            self._mrad,
            self._radmin,
            self._buod,
            self._xmfd,
            self._tcdo,
            self._qcdo,
            self._ucdo,
            self._vcdo,
            self._xlamde,
            self._g,
            self._gocp,
            self._elocp,
            self._el2orc,
            self._mask,
            self._qtx,
            self._wd2,
            self._hrad,
            self._krad1,
            self._thld,
            self._qtd,
            self._thlvd,
            self._ra1,
            self._ra2,
            self._flg,
            self._xlamdem,
            self._mradx,
            self._mrady,
            self._sumx,
            self._xlamavg,
            self._scaldfunc,
            self._zm_mrad,
            self._ce0,
            self._cm,
            self._pgcon,
            self._qmin,
            self._qlmin,
            self._b1_mfscu,
            self._f1_const,
            self._a1_mfscu,
            self._a2,
            self._a11,
            self._a22,
            self._cldtime,
            self._actei,
            self._hvap,
            self._cp,
            self._eps,
            self._epsm1,
            self._fv,
        )

        prandtl_comp_exchg_coeff(
            chz=self._chz,
            ckz=self._ckz,
            hpbl=self._hpbl,
            kpbl=self._kpbl,
            mask=self._mask,
            pcnvflg=self._pcnvflg,
            phih=self._phih,
            phim=self._phim,
            prn=self._prn,
            zi=self._zi,
            domain=(self._im, 1, self._kmpbl),
        )

        for k in range(self._km1):
            comp_asym_mix_up(
                mask=self._mask,
                mlenflg=self._mlenflg,
                bsum=self._bsum,
                zlup=self._zlup,
                thvx_k=self._thvx_k,
                tke_k=self._tke_k,
                thvx=self._thvx,
                tke=self._tke,
                gotvx=self._gotvx,
                zl=self._zl,
                zfmin=self._zfmin,
                k=k,
                domain=(self._im,1,self._km1-k),
                origin=(0,0,k)
            )
            comp_asym_mix_dn(
                mask=self._mask,
                mlenflg=self._mlenflg,
                bsum=self._bsum,
                zldn=self._zldn,
                thvx_k=self._thvx_k,
                tke_k=self._tke_k,
                thvx=self._thvx,
                tke=self._tke,
                gotvx=self._gotvx,
                zl=self._zl,
                tsea=self._tsea,
                q1_gt=self._q1,
                zfmin=self._zfmin,
                fv=self._fv,
                k=k,
                domain=(self._im,1,k+1),
                origin=(0,0,0),
            )
            comp_asym_rlam_ele(
                zi=self._zi,
                rlam=self._rlam,
                ele=self._ele,
                zlup=self._zlup,
                zldn=self._zldn,
                rlmn=self._rlmn,
                rlmx=self._rlmx,
                elmfac=self._elmfac,
                elmx=self._elmx,
                domain=(self._im,1,1),
                origin=(0,0,k),
            )

        compute_eddy_buoy_shear(
            bf=self._bf,
            buod=self._buod,
            buou=self._buou,
            chz=self._chz,
            ckz=self._ckz,
            dku=self._dku,
            dkt=self._dkt,
            dkq=self._dkq,
            ele=self._ele,
            elm=self._elm,
            gdx=self._gdx,
            gotvx=self._gotvx,
            kpbl=self._kpbl,
            mask=self._mask,
            mrad=self._mrad,
            krad=self._krad,
            pblflg=self._pblflg,
            pcnvflg=self._pcnvflg,
            phim=self._phim,
            prn=self._prn,
            prod=self._prod,
            radj=self._radj,
            rdzt=self._rdzt,
            rlam=self._rlam,
            rle=self._rle,
            scuflg=self._scuflg,
            sflux=self._sflux,
            shr2=self._shr2,
            stress=self._stress,
            tke=self._tke,
            u1=self._u1,
            ucdo=self._ucdo,
            ucko=self._ucko,
            ustar=self._ustar,
            v1=self._v1,
            vcdo=self._vcdo,
            vcko=self._vcko,
            xkzo=self._xkzo,
            xkzmo=self._xkzmo,
            xmf=self._xmf,
            xmfd=self._xmfd,
            zi=self._zi,
            zl=self._zl,
            zol=self._zol,
            domain=(self._im, 1, self._km),
        )

        predict_tke(diss=self._diss, 
              prod=self._prod, 
              rle=self._rle, 
              tke=self._tke, 
              dtn=self._dtn, 
              kk=self._kk, 
              domain=(self._im, 1, self._km1))

        tke_up_down_prop(
            pcnvflg=self._pcnvflg,
            qcdo=self._qcdo,
            qcko=self._qcko,
            scuflg=self._scuflg,
            tke=self._tke,
            kpbl=self._kpbl,
            mask=self._mask,
            xlamue=self._xlamue,
            zl=self._zl,
            ad=self._ad,
            f1=self._f1,
            krad=self._krad,
            mrad=self._mrad,
            xlamde=self._xlamde,
            kmpbl=self._kmpbl,
            kmscu=self._kmscu,
            domain=(self._im, 1, self._km),
        )

        tke_tridiag_matrix_ele_comp(
            ad=self._ad,
            ad_p1=self._ad_p1,
            al=self._al,
            au=self._au,
            del_=self._del_,
            dkq=self._dkq,
            dt2=self._dt2,
            f1=self._f1,
            f1_p1=self._f1_p1,
            kpbl=self._kpbl,
            krad=self._krad,
            mask=self._mask,
            mrad=self._mrad,
            pcnvflg=self._pcnvflg,
            prsl=self._prsl,
            qcdo=self._qcdo,
            qcko=self._qcko,
            rdzt=self._rdzt,
            scuflg=self._scuflg,
            tke=self._tke,
            xmf=self._xmf,
            xmfd=self._xmfd,
            domain=(self._im, 1, self._km),
        )

        tridit(au=self._au, 
               cm=self._ad, 
               cl=self._al, 
               f1=self._f1, 
               domain=(self._im, 1, self._km))

        part12a(
            rtg=self._rtg_gt,
            f1=self._f1,
            q1=self._q1,
            ad=self._ad,
            f2=self._f2,
            dtdz1=self._dtdz1,
            evap=self._evap,
            heat=self._heat,
            t1=self._t1,
            rdt=self._rdt,
            ntrac1=self._ntrac1,
            ntke=self._ntke,
            domain=(self._im, 1, self._km),
        )

        heat_moist_tridiag_mat_ele_comp(
            ad=self._ad,
            ad_p1=self._ad_p1,
            al=self._al,
            au=self._au,
            del_=self._del_,
            dkt=self._dkt,
            f1=self._f1,
            f1_p1=self._f1_p1,
            f2=self._f2,
            f2_p1=self._f2_p1,
            kpbl=self._kpbl,
            krad=self._krad,
            mask=self._mask,
            mrad=self._mrad,
            pcnvflg=self._pcnvflg,
            prsl=self._prsl,
            q1=self._q1,
            qcdo=self._qcdo,
            qcko=self._qcko,
            rdzt=self._rdzt,
            scuflg=self._scuflg,
            tcdo=self._tcdo,
            tcko=self._tcko,
            t1=self._t1,
            xmf=self._xmf,
            xmfd=self._xmfd,
            dt2=self._dt2,
            gocp=self._gocp,
            domain=(self._im, 1, self._km),
        )

        if self._ntrac1 >= 2:
            part13a(
                pcnvflg=self._pcnvflg,
                mask=self._mask,
                kpbl=self._kpbl,
                del_=self._del_,
                prsl=self._prsl,
                rdzt=self._rdzt,
                xmf=self._xmf,
                qcko=self._qcko,
                q1=self._q1,
                f2=self._f2,
                scuflg=self._scuflg,
                mrad=self._mrad,
                krad=self._krad,
                xmfd=self._xmfd,
                qcdo=self._qcdo,
                ntrac1=self._ntrac1,
                dt2=self._dt2,
                domain=(self._im, 1, self._km)
            )
        tridin(cl=self._al,
           cm=self._ad,
           cu=self._au,
           r1=self._f1,
           r2=self._f2,
           au=self._au,
           a1=self._f1,
           a2=self._f2,
           nt=self._ntrac1,
           domain =(self._im, 1, self._km))

        part13b(
            f1=self._f1,
            t1=self._t1,
            f2=self._f2,
            q1=self._q1,
            tdt=self._tdt,
            rtg=self._rtg_gt,
            dtsfc=self._dtsfc,
            del_=self._del_,
            dqsfc=self._dqsfc,
            conq=self._conq,
            cont=self._cont,
            rdt=self._rdt,
            ntrac1=self._ntrac1,
            domain=(self._im,1,self._km)
        )

        moment_tridiag_mat_ele_comp(
            ad=self._ad,
            ad_p1=self._ad_p1,
            al=self._al,
            au=self._au,
            del_=self._del_,
            diss=self._diss,
            dku=self._dku,
            dtdz1=self._dtdz1,
            f1=self._f1,
            f1_p1=self._f1_p1,
            f2=self._f2,
            f2_p1=self._f2_p1,
            kpbl=self._kpbl,
            krad=self._krad,
            mask=self._mask,
            mrad=self._mrad,
            pcnvflg=self._pcnvflg,
            prsl=self._prsl,
            rdzt=self._rdzt,
            scuflg=self._scuflg,
            spd1=self._spd1,
            stress=self._stress,
            tdt=self._tdt,
            u1=self._u1,
            ucdo=self._ucdo,
            ucko=self._ucko,
            v1=self._v1,
            vcdo=self._vcdo,
            vcko=self._vcko,
            xmf=self._xmf,
            xmfd=self._xmfd,
            dspheat=self._dspheat,
            dt2=self._dt2,
            domain=(self._im, 1, self._km),
        )

        tridi2(
            a1=self._f1, 
            a2=self._f2, 
            au=self._au, 
            cl=self._al, 
            cm=self._ad, 
            cu=self._au, 
            r1=self._f1, 
            r2=self._f2, 
            domain=(self._im, 1, self._km)
        )

        moment_recover(
            del_=self._del_,
            du=self._du,
            dusfc=self._dusfc,
            dv=self._dv,
            dvsfc=self._dvsfc,
            f1=self._f1,
            f2=self._f2,
            hpbl=self._hpbl,
            hpblx=self._hpblx,
            kpbl=self._kpbl,
            kpblx=self._kpblx,
            mask=self._mask,
            u1=self._u1,
            v1=self._v1,
            conw=self._conw,
            rdt=self._rdt,
            domain=(self._im, 1, self._km),
        )

        dv = storage_to_numpy(self._dv, (self._im, self._km))
        du = storage_to_numpy(self._du, (self._im, self._km))
        tdt = storage_to_numpy(self._tdt, (self._im, self._km))
        kpbl = storage_to_numpy(self._kpbl, self._im)
        dusfc = storage_to_numpy(self._dusfc, self._im)
        dvsfc = storage_to_numpy(self._dvsfc, self._im)
        dtsfc = storage_to_numpy(self._dtsfc, self._im)
        dqsfc = storage_to_numpy(self._dqsfc, self._im)
        hpbl = storage_to_numpy(self._hpbl, self._im)

        kpbl[:] = kpbl + 1

        rtg = np.zeros((self._rtg_gt.shape[0], self._rtg_gt.shape[2]-1, self._ntrac))

        for I in range(self._ntrac):
            rtg[:,:, I] = self._rtg_gt[:,0, :-1, I]

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

def run(in_data, timings):

    turb_obj = Turbulence(in_data["im"], in_data["ix"], in_data["km"], in_data["ntrac"], 
                                       in_data["ntcw"], in_data["ntiw"], in_data["ntke"], in_data["delt"], 
                                       in_data["dspheat"], in_data["xkzm_m"], in_data["xkzm_h"], in_data["xkzm_s"])

    turb_obj.turbInit(in_data["garea"], in_data["prsi"], in_data["kinver"], in_data["zorl"], in_data["dusfc"], in_data["dvsfc"],
    in_data["dtsfc"], in_data["dqsfc"], in_data["kpbl"], in_data["hpbl"], in_data["rbsoil"], in_data["evap"], in_data["heat"], in_data["psk"],
    in_data["xmu"], in_data["tsea"], in_data["u10m"], in_data["v10m"], in_data["stress"], in_data["fm"], in_data["fh"], in_data["spd1"], in_data["phii"],
    in_data["phil"],in_data["swh"], in_data["hlw"], in_data["u1"], in_data["v1"], in_data["del"], in_data["du"], in_data["dv"], in_data["tdt"],
    in_data["prslk"], in_data["t1"], in_data["prsl"], in_data["q1"], in_data["rtg"])

    tic = timeit.default_timer()

    dv, du, tdt, rtg, kpbl, dusfc, dvsfc, dtsfc, dqsfc, hpbl = turb_obj.run_turb()

    toc = timeit.default_timer()

    # exec_info = {}
    timings["elapsed_time"] += toc - tic
    # timings["run_time"] += exec_info["run_end_time"] - exec_info["run_start_time"]

    out_data = {}
    for key in OUT_VARS:
        out_data[key] = np.zeros(1, dtype=np.float64)

    out_data["dv"] = dv
    out_data["du"] = du
    out_data["tdt"] = tdt
    out_data["rtg"] = rtg
    out_data["kpbl"] = kpbl
    out_data["dusfc"] = dusfc
    out_data["dvsfc"] = dvsfc
    out_data["dtsfc"] = dtsfc
    out_data["dqsfc"] = dqsfc
    out_data["hpbl"] = hpbl

    return out_data