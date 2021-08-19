import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radlw_param import a0, a1, a2, nbands, nrates, ipat, delwave
from util import (
    view_gt4py_storage,
    compare_data,
    create_storage_from_array,
    create_storage_zeros,
    create_storage_ones,
    loadlookupdata,
)
from config import *

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
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
    print(var)
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
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
