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

backend = "gtc:gt:cpu_ifirst"

fv = rv / rd - 1.0


@gtscript.stencil(backend=backend, rebuild=True)
def test_for(
    out_field: gtscript.Field[np.float64], in_field: gtscript.Field[np.float64]
):
    with computation(FORWARD), interval(...):
        for i in range(0, index(K)):
            out_field += in_field[0, 0, i]


# @gtscript.stencil(backend=backend)
# def forloop_stencil(
#     zl: FIELD_FLT,
#     zi: FIELD_FLT,
#     gotvx: FIELD_FLT,
#     thvx: FIELD_FLT,
#     tke: FIELD_FLT,
#     tsea: FIELD_FLT,
#     q1_0: FIELD_FLT,
#     q1_0_k0: FIELD_FLT,
#     rlam: FIELD_FLT,
#     ele: FIELD_FLT,
# ):
#     with computation(FORWARD):
#         with interval(0, 1):  # could also change it to a 2D storage
#             q1_0_k0 = q1_0[0, 0, 0]
#         with interval(1, None):
#             q1_0_k0 = q1_0_k0[0, 0, -1]

#     with computation(FORWARD), interval(0, -2):
#         zlup = 0.0
#         bsum = 0.0
#         mlenflg = True
#         dz = 0.0
#         ptem = 0.0
#         tem2 = 0.0
#         ptem1 = 0.0
#         for n in range(index(K), K[-1]):
#             if mlenflg:
#                 dz = zl[0, 0, n + 1] - zl[0, 0, n]
#             ptem = gotvx[0, 0, n] * (thvx[0, 0, n + 1] - thvx[0, 0, 0]) * dz
#             bsum = bsum + ptem
#             zlup = zlup + dz
#             if bsum >= tke[0, 0, 0]:
#                 if ptem >= 0.0:
#                     tem2 = max(ptem, zfmin)
#                 else:
#                     tem2 = min(ptem, -zfmin)
#                 ptem1 = (bsum - tke[0, 0, 0]) / tem2
#                 zlup = zlup - ptem1 * dz
#                 zlup = max(zlup, 0.0)
#                 mlenflg = False
#         zldn = 0.0
#         bsum = 0.0
#         mlenflg = True
#         for n in range(0, index(K) + 1):
#             ni = 79 - n
#             if mlenflg:
#                 if ni == 0:
#                     dz = zl[0, 0, 0]
#                     tem1 = tsea[0, 0, 0] * (1.0 + fv * max(q1_0_k0[0, 0, 0], qmin))
#                 else:
#                     dz = zl[0, 0, ni] - zl[0, 0, ni - 1]
#                     tem1 = thvx[0, 0, ni - 1]
#                 ptem = gotvx[0, 0, ni] * (thvx[0, 0, 0] - tem1) * dz
#                 bsum = bsum + ptem
#                 zldn = zldn + dz
#                 if bsum >= tke[0, 0, 0]:
#                     if ptem >= 0.0:
#                         tem2 = max(ptem, zfmin)
#                     else:
#                         tem2 = min(ptem, -zfmin)
#                     ptem1 = (bsum - tke[0, 0, 0]) / tem2
#                     zldn = zldn - ptem1 * dz
#                     zldn = max(zldn, 0.0)
#                     mlenflg = False
#                     tem = 0.5 * (zi[0, 0, 1] - zi[0, 0, 0])

#         tem1 = min(tem, rlmn)

#         ptem2 = min(zlup, zldn)
#         rlam[0, 0, 0] = elmfac * ptem2
#         rlam[0, 0, 0] = max(rlam[0, 0, 0], tem1)
#         rlam[0, 0, 0] = min(rlam[0, 0, 0], rlmx)

#         ptem2 = sqrt(zlup * zldn)
#         ele[0, 0, 0] = elefac * ptem2
#         ele[0, 0, 0] = max(ele[0, 0, 0], tem1)
#         ele[0, 0, 0] = min(ele[0, 0, 0], elmx)
