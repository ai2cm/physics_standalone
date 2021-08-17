import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import stencil, computation, interval, PARALLEL, BACKWARD, mod

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")

from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radsw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

invars = [
    "zrefb",
    "zrefd",
    "ztrab",
    "ztrad",
    "zldbt",
    "ztdbt",
]

outvars = ["zfu", "zfd"]

locvars = ["zrupb", "zrupd", "zrdnd", "ztdn", "zden1"]

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-vrtqdr-input-000000"])
    indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))

indict_gt4py = dict()
for var in invars:
    indict_gt4py[var] = create_storage_from_array(
        indict[var], backend, shape_nlp1, DTYPE_FLT
    )

outdict_gt4py = dict()
locdict_gt4py = dict()

for var in outvars:
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

for var in locvars:
    locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)


@stencil(backend=backend, rebuild=rebuild)
def vrtqdr(
    zrefb: FIELD_FLT,
    zrefd: FIELD_FLT,
    ztrab: FIELD_FLT,
    ztrad: FIELD_FLT,
    zldbt: FIELD_FLT,
    ztdbt: FIELD_FLT,
    zfu: FIELD_FLT,
    zfd: FIELD_FLT,
    zrupb: FIELD_FLT,
    zrupd: FIELD_FLT,
    zrdnd: FIELD_FLT,
    ztdn: FIELD_FLT,
    zden1: FIELD_FLT,
):

    with computation(PARALLEL), interval(0, 1):
        # Link lowest layer with surface.
        zrupb = zrefb  # direct beam
        zrupd = zrefd  # diffused

    with computation(PARALLEL), interval(1, None):
        # Pass from bottom to top.
        zden1 = 1.0 / (1.0 - zrupd[0, 0, -1] * zrefd)
        zrupb = (
            zrefb
            + (ztrad * ((ztrab - zldbt) * zrupd[0, 0, -1] + zldbt * zrupb[0, 0, -1]))
            * zden1
        )
        zrupd = zrefd + ztrad * ztrad * zrupd[0, 0, -1] * zden1

    # Upper boundary conditions
    with computation(PARALLEL):
        with interval(-2, -1):
            ztdn = ztrab[0, 0, 1]
            zrdnd = zrefd[0, 0, 1]
        with interval(-1, None):
            ztdn = 1.0
            zrdnd = 0.0

    # Pass from top to bottom
    with computation(BACKWARD):
        with interval(-2, -1):
            zden1 = 1.0 / (1.0 - zrefd * zrdnd)
        with interval(0, -2):
            ztdn = (
                ztdbt[0, 0, 1] * ztrab[0, 0, 1]
                + (
                    ztrad[0, 0, 1]
                    * (
                        (ztdn[0, 0, 1] - ztdbt[0, 0, 1])
                        + ztdbt[0, 0, 1] * zrefb[0, 0, 1] * zrdnd[0, 0, 1]
                    )
                )
                * zden1[0, 0, 1]
            )
            zrdnd = (
                zrefd[0, 0, 1]
                + ztrad[0, 0, 1] * ztrad[0, 0, 1] * zrdnd[0, 0, 1] * zden1[0, 0, 1]
            )
            zden1 = 1.0 / (1.0 - zrefd * zrdnd)

    # Up and down-welling fluxes at levels.
    with computation(PARALLEL), interval(...):
        zden1 = 1.0 / (1.0 - zrdnd * zrupd)
        zfu = (ztdbt * zrupb + (ztdn - ztdbt) * zrupd) * zden1
        zfd = ztdbt + (ztdn - ztdbt + ztdbt * zrupb * zrdnd) * zden1


vrtqdr(
    indict_gt4py["zrefb"],
    indict_gt4py["zrefd"],
    indict_gt4py["ztrab"],
    indict_gt4py["ztrad"],
    indict_gt4py["zldbt"],
    indict_gt4py["ztdbt"],
    outdict_gt4py["zfu"],
    outdict_gt4py["zfd"],
    locdict_gt4py["zrupb"],
    locdict_gt4py["zrupd"],
    locdict_gt4py["zrdnd"],
    locdict_gt4py["ztdn"],
    locdict_gt4py["zden1"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)


outdict_np = dict()
valdict_np = dict()
for var in outvars:
    outdict_np[var] = outdict_gt4py[var][0, :, :].view(np.ndarray).squeeze()
    valdict_np[var] = serializer.read(
        var, serializer.savepoint["swrad-vrtqdr-output-000000"]
    )

compare_data(outdict_np, valdict_np)
