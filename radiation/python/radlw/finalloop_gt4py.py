import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
from gt4py.gtscript import stencil, computation, interval, PARALLEL, FORWARD

sys.path.insert(0, "..")
from util import (
    view_gt4py_storage,
    compare_data,
    create_storage_from_array,
    create_storage_zeros,
    create_storage_ones,
    loadlookupdata,
)
from config import *

import serialbox as ser

serializer = ser.Serializer(ser.OpenModeKind.Read, SERIALIZED_DIR, "Serialized_rank0")
savepoints = serializer.savepoint_list()

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = [
    "totuflux",
    "totuclfl",
    "totdflux",
    "totdclfl",
    "htr",
    "htrcl",
]

outvars = [
    "upfxc_t",
    "upfx0_t",
    "upfxc_s",
    "upfx0_s",
    "dnfxc_s",
    "dnfx0_s",
    "hlwc",
    "hlw0",
]

indict = dict()
indict_gt4py = dict()

for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-finalloop-input-000000"])
    if var == "htr" or var == "htrcl":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
    else:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))

    indict_gt4py[var] = create_storage_from_array(
        indict[var], backend, shape_nlp1, DTYPE_FLT
    )

outdict_gt4py = dict()
for var in outvars:
    if var == "hlwc" or var == "hlw0":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    else:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)

lhlw0 = True


@stencil(backend=backend, rebuild=rebuild, externals={"lhlw0": lhlw0})
def finalloop(
    totuflux: FIELD_FLT,
    totuclfl: FIELD_FLT,
    totdflux: FIELD_FLT,
    totdclfl: FIELD_FLT,
    htr: FIELD_FLT,
    htrcl: FIELD_FLT,
    upfxc_t: FIELD_2D,
    upfx0_t: FIELD_2D,
    upfxc_s: FIELD_2D,
    upfx0_s: FIELD_2D,
    dnfxc_s: FIELD_2D,
    dnfx0_s: FIELD_2D,
    hlwc: FIELD_FLT,
    hlw0: FIELD_FLT,
):
    from __externals__ import lhlw0

    with computation(FORWARD):
        with interval(0, 1):
            # Output surface fluxes
            upfxc_s = totuflux
            upfx0_s = totuclfl
            dnfxc_s = totdflux
            dnfx0_s = totdclfl
        with interval(-1, None):
            # Output TOA fluxes
            upfxc_t = totuflux
            upfx0_t = totuclfl

    with computation(PARALLEL):
        with interval(1, None):
            hlwc = htr

            if lhlw0:
                hlw0 = htrcl


finalloop(
    indict_gt4py["totuflux"],
    indict_gt4py["totuclfl"],
    indict_gt4py["totdflux"],
    indict_gt4py["totdclfl"],
    indict_gt4py["htr"],
    indict_gt4py["htrcl"],
    outdict_gt4py["upfxc_t"],
    outdict_gt4py["upfx0_t"],
    outdict_gt4py["upfxc_s"],
    outdict_gt4py["upfx0_s"],
    outdict_gt4py["dnfxc_s"],
    outdict_gt4py["dnfx0_s"],
    outdict_gt4py["hlwc"],
    outdict_gt4py["hlw0"],
)

valdict = dict()
outdict_np = dict()
for var in outvars:
    valdict[var] = serializer.read(
        var, serializer.savepoint["lwrad-finalloop-output-000000"]
    )
    if var == "hlwc" or var == "hlw0":
        outdict_np[var] = outdict_gt4py[var][:, :, 1:].view(np.ndarray).squeeze()
    else:
        outdict_np[var] = outdict_gt4py[var].view(np.ndarray).squeeze()

compare_data(valdict, outdict_np)
