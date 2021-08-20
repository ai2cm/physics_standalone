import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import stencil, computation, interval, PARALLEL, FORWARD

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import (
    ngptsw,
    nblow,
    nbhgh,
    nspa,
    nspb,
    ng,
    ngs,
    oneminus,
    NG16,
    NG17,
    NG18,
    NG19,
    NG20,
    NS16,
    NS17,
    NS18,
    NS19,
    NS20,
)
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
    "colamt",
    "colmol",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "jp",
    "jt",
    "jt1",
    "laytrop",
    "forfac",
    "forfrac",
    "indfor",
    "selffac",
    "selffrac",
    "indself",
]

outvars = ["sfluxzen", "taug", "taur"]

locvars = ["fs", "speccomb", "specmult", "colm1", "colm2", "id0", "id1"]

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-taumol-input-000000"])
    if var == "colamt":
        indict[var] = np.tile(tmp[None, None, :, :], (npts, 1, 1, 1))
    elif var != "laytrop":
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    elif var == "laytrop":
        laytrop = np.zeros((npts, 1, nlay), dtype=bool)
        laytrop[:, :, : tmp[0]] = True
        indict[var] = laytrop

indict_gt4py = dict()

for var in invars:
    if var == "colamt":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, type_maxgas
        )
    elif var == "laytrop":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, DTYPE_BOOL
        )
    elif var in ["indfor", "indself", "jp", "jt", "jt1"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, DTYPE_INT
        )
    else:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, DTYPE_FLT
        )

outdict_gt4py = dict()

for var in outvars:
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_ngptsw)

locdict_gt4py = dict()

for var in locvars:
    if var in ["id0", "id1"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_nbandssw)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_INT)


@stencil(backend=backend, rebuild=rebuild, externals={"nbands": nbands})
def setuptaumol(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    id0: Field[type_nbandssw],
    id1: Field[type_nbandssw],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
):
    from __externals__ import nbands

    with computation(PARALLEL), interval(...):
        for jb in range(nbands):
            #  --- ...  indices for layer optical depth
            if laytrop:
                id0[0, 0, 0][jb] = ((jp - 1) * 5 + (jt - 1)) * nspa[0, 0, 0][jb] - 1
                id1[0, 0, 0][jb] = (jp * 5 + (jt1 - 1)) * nspa[0, 0, 0][jb] - 1
            else:
                id0[0, 0, 0][jb] = ((jp - 13) * 5 + (jt - 1)) * nspb[0, 0, 0][jb]
                id1[0, 0, 0][jb] = ((jp - 12) * 5 + (jt1 - 1)) * nspb[0, 0, 0][jb]

    with computation(PARALLEL), interval(0, 1):
        for jb2 in range(nbands):
            ibd = ibx[0, 0, 0][jb]
            njb = ng[0, 0, 0][jb]
            ns = ngs[0, 0, 0][jb]
