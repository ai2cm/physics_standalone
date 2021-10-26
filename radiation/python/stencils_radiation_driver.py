import numpy as np
import xarray as xr
import os
import sys

from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    __externals__,
    PARALLEL,
    FORWARD,
    BACKWARD,
    min,
    max,
)

from config import *
from radphysparam import *
from phys_const import con_eps, con_epsm1, con_rocp, con_fvirt, con_rog, con_epsq
from funcphys_gt import fpvs

backend = "gtc:gt:cpu_ifirst"

# convert pressure unit from pa to mb
@stencil(backend=backend, 
        externals={"con_eps" : con_eps,
                   "con_epsm1" : con_epsm1,
                  })
def pressure_convert(plvl : FIELD_FLT,
                     prsi : FIELD_FLT,
                     plyr : FIELD_FLT,
                     prsl : FIELD_FLT,
                     tlyr : FIELD_FLT,
                     tgrs : FIELD_FLT,
                     prslk1 : FIELD_FLT,
                     prslk  : FIELD_FLT,
                     rhly : FIELD_FLT,
                     qgrs : Field[(DTYPE_FLT,(8,))],
                     qstl : FIELD_FLT,
                     tracer1 : Field[(DTYPE_FLT,(8,))],
                     QMIN : DTYPE_FLT,
                     NTRAC : DTYPE_INT,):
    from __externals__ import (con_eps, con_epsm1)
    with computation(PARALLEL), interval(0,-1):
            plvl = prsi[0,0,0] * 0.01
    with computation(PARALLEL), interval(1,None):
            plyr = prsl[0,0,0] * 0.01
            tlyr = tgrs[0,0,0]
            prslk1 = prslk

            es = min(prsl[0,0,0], fpvs(tgrs[0,0,0]))
            qs = max(QMIN, con_eps * es / (prsl[0,0,0] + con_epsm1 * es))
            rhly = max(0.0, min(1.0, max(QMIN, qgrs[0,0,0][0])/qs))
            qstl = qs
            for i in range(NTRAC):
                tracer1[0,0,0][i] = max(0.0, qgrs[0,0,0][i])