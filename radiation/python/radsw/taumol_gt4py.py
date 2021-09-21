import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import stencil, computation, interval, PARALLEL, FORWARD, mod

sys.path.insert(0, "..")
from radsw.radsw_param import (
    nbandssw,
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
    NG21,
    NG22,
    NG23,
    NG24,
    NG25,
    NG26,
    NG27,
    NG28,
    NG29,
    NS16,
    NS17,
    NS18,
    NS19,
    NS20,
    NS21,
    NS22,
    NS23,
    NS24,
    NS25,
    NS26,
    NS27,
    NS28,
    NS29,
)
from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"
ddir2 = "../../fortran/data/SW"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
serializer2 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Generator_rank1")
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
    "idxday",
]

outvars = ["sfluxzen", "taug", "taur"]

locvars = [
    "id0",
    "id1",
    "ind01",
    "ind02",
    "ind03",
    "ind04",
    "ind11",
    "ind12",
    "ind13",
    "ind14",
    "inds",
    "indsp",
    "indf",
    "indfp",
    "fs",
    "js",
    "jsa",
    "colm1",
    "colm2",
]

indict = dict()
for var in invars:
    if var == "idxday":
        tmp = serializer2.read(var, serializer2.savepoint["swrad-in-000000"])
    else:
        tmp = serializer.read(var, serializer.savepoint["swrad-taumol-input-000000"])

    if var == "colamt" or var == "id0" or var == "id1":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var != "laytrop" and var != "idxday":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
    elif var == "laytrop":
        laytrop = np.zeros((npts, 1, nlay), dtype=bool)
        for n in range(npts):
            laytrop[n, :, : tmp[n]] = True
        indict[var] = np.insert(laytrop, 0, False, axis=2)

        laytropind = np.tile(tmp[:, None], (1, 1))
        indict["laytropind"] = laytropind - 1
    elif var == "idxday":
        tmp2 = np.zeros(npts, dtype=bool)
        for n in range(npts):
            if tmp[n] > 1 and tmp[n] < 25:
                tmp2[tmp[n] - 1] = True

        indict[var] = np.tile(tmp2[:, None], (1, 1))

        laytropind = np.tile(tmp[:, None], (1, 1))
        indict["laytropind"] = laytropind - 1
    elif var == "idxday":
        tmp2 = np.zeros(npts, dtype=bool)
        for n in range(npts):
            if tmp[n] > 1 and tmp[n] < 25:
                tmp2[tmp[n] - 1] = True

        indict[var] = np.tile(tmp2[:, None], (1, 1))

indict_gt4py = dict()

for var in invars:
    if var == "colamt":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_maxgas
        )
    elif var == "laytrop":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_BOOL
        )
    elif var in ["indfor", "indself", "jp", "jt", "jt1"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_INT
        )
    elif var == "id0" or var == "id1":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_nbandssw_int
        )
    elif var == "idxday":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, DTYPE_BOOL
        )
    elif var == "idxday":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, DTYPE_BOOL
        )
    else:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )

indict_gt4py["laytropind"] = create_storage_from_array(
    indict["laytropind"], backend, shape_2D, DTYPE_INT
)

outdict_gt4py = dict()

for var in outvars:
    if var == "sfluxzen":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, type_ngptsw)
    else:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw)

locdict_gt4py = dict()

for var in locvars:
    if var in ["id0", "id1"]:
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, type_nbandssw_int
        )
    elif var in ["colm1", "colm2"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)

layind = np.arange(nlay, dtype=np.int32)
layind = np.insert(layind, 0, 0)
layind = np.tile(layind[None, None, :], (npts, 1, 1))
nspa = np.tile(np.array(nspa)[None, None, :], (npts, 1, 1))
nspb = np.tile(np.array(nspb)[None, None, :], (npts, 1, 1))
ngtmp = np.tile(np.array(ng)[None, None, :], (npts, 1, 1))
ngs = np.tile(np.array(ngs)[None, None, :], (npts, 1, 1))
locdict_gt4py["layind"] = create_storage_from_array(
    layind, backend, shape_nlp1, DTYPE_INT
)
locdict_gt4py["nspa"] = create_storage_from_array(
    nspa, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["nspb"] = create_storage_from_array(
    nspb, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ng"] = create_storage_from_array(
    ngtmp, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ngs"] = create_storage_from_array(
    ngs, backend, shape_2D, type_nbandssw_int
)

layind = np.arange(nlay, dtype=np.int32)
layind = np.tile(layind[None, None, :], (npts, 1, 1))
nspa = np.tile(np.array(nspa)[None, None, :], (npts, 1, 1))
nspb = np.tile(np.array(nspb)[None, None, :], (npts, 1, 1))
ngtmp = np.tile(np.array(ng)[None, None, :], (npts, 1, 1))
ngs = np.tile(np.array(ngs)[None, None, :], (npts, 1, 1))
locdict_gt4py["layind"] = create_storage_from_array(
    layind, backend, shape_nlay, DTYPE_INT
)
locdict_gt4py["nspa"] = create_storage_from_array(
    nspa, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["nspb"] = create_storage_from_array(
    nspb, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ng"] = create_storage_from_array(
    ngtmp, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ngs"] = create_storage_from_array(
    ngs, backend, shape_2D, type_nbandssw_int
)


def loadlookupdata(name):
    """
    Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables
    """
    ds = xr.open_dataset("../lookupdata/radsw_" + name + "_data.nc")

    lookupdict = dict()
    lookupdict_gt4py = dict()

    for var in ds.data_vars.keys():
        # print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :], (npts, 1, nlp1, 1)
            )
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1)
            )
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :, :], (npts, 1, nlp1, 1, 1, 1)
            )
        else:
            lookupdict[var] = float(ds[var].data)

        if len(ds.data_vars[var].shape) >= 1:
            lookupdict_gt4py[var] = create_storage_from_array(
                lookupdict[var], backend, shape_nlp1, (ds[var].dtype, ds[var].shape)
            )
        else:
            lookupdict_gt4py[var] = lookupdict[var]

    return lookupdict_gt4py


lookupdict_ref = loadlookupdata("sflux")
lookupdict16 = loadlookupdata("kgb16")
lookupdict17 = loadlookupdata("kgb17")
lookupdict18 = loadlookupdata("kgb18")
lookupdict19 = loadlookupdata("kgb19")
lookupdict20 = loadlookupdata("kgb20")
lookupdict21 = loadlookupdata("kgb21")
lookupdict22 = loadlookupdata("kgb22")
lookupdict23 = loadlookupdata("kgb23")
lookupdict24 = loadlookupdata("kgb24")
lookupdict25 = loadlookupdata("kgb25")
lookupdict26 = loadlookupdata("kgb26")
lookupdict27 = loadlookupdata("kgb27")
lookupdict28 = loadlookupdata("kgb28")
lookupdict29 = loadlookupdata("kgb29")

# Subtract one from indexing variables for Fortran -> Python conversion
lookupdict_ref["ix1"] = lookupdict_ref["ix1"] - 1
lookupdict_ref["ix2"] = lookupdict_ref["ix2"] - 1
lookupdict_ref["ibx"] = lookupdict_ref["ibx"] - 1


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbandssw": nbandssw,
        "NG16": NG16,
        "NG17": NG17,
        "NG18": NG18,
        "NG19": NG19,
        "NG20": NG20,
        "NG21": NG21,
        "NG22": NG22,
        "NG23": NG23,
        "NG24": NG24,
        "NG25": NG25,
        "NG26": NG26,
        "NG27": NG27,
        "NG28": NG28,
        "NG29": NG29,
        "scalekur": lookupdict_ref["scalekur"],
        "oneminus": oneminus,
    },
)
def taumolsetup(
    colamt: Field[type_maxgas],
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    laytrop: FIELD_BOOL,
    laytropind: FIELD_2DINT,
    idxday: FIELD_2DBOOL,
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    layind: FIELD_INT,
    nspa: Field[gtscript.IJ, type_nbandssw_int],
    nspb: Field[gtscript.IJ, type_nbandssw_int],
    ngs: Field[gtscript.IJ, type_nbandssw_int],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    js: FIELD_INT,
    jsa: FIELD_INT,
    colm1: FIELD_FLT,
    colm2: FIELD_FLT,
    sfluxref01: Field[(DTYPE_FLT, (16, 1, 7))],
    sfluxref02: Field[(DTYPE_FLT, (16, 5, 2))],
    sfluxref03: Field[(DTYPE_FLT, (16, 9, 5))],
    layreffr: Field[type_nbandssw_int],
    ix1: Field[type_nbandssw_int],
    ix2: Field[type_nbandssw_int],
    ibx: Field[type_nbandssw_int],
    strrat: Field[type_nbandssw_flt],
    specwt: Field[type_nbandssw_flt],
):
    from __externals__ import (
        nbandssw,
        NG16,
        NG17,
        NG18,
        NG19,
        NG20,
        NG21,
        NG22,
        NG23,
        NG24,
        NG25,
        NG26,
        NG27,
        NG28,
        NG29,
        scalekur,
        oneminus,
    )

    with computation(FORWARD), interval(1, None):
        if idxday:
            for jb in range(nbandssw):
                if laytrop:
                    id0[0, 0, 0][jb] = ((jp - 1) * 5 + (jt - 1)) * nspa[0, 0][jb]
                    id1[0, 0, 0][jb] = (jp * 5 + (jt1 - 1)) * nspa[0, 0][jb]
                else:
                    id0[0, 0, 0][jb] = ((jp - 13) * 5 + (jt - 1)) * nspb[0, 0][jb]
                    id1[0, 0, 0][jb] = ((jp - 12) * 5 + (jt1 - 1)) * nspb[0, 0][jb]

            for j in range(NG16):
                sfluxzen[0, 0][ngs[0, 0][0] + j] = sfluxref01[0, 0, 0][
                    j, 0, ibx[0, 0, 0][0]
                ]
            for j2 in range(NG20):
                sfluxzen[0, 0][ngs[0, 0][4] + j2] = sfluxref01[0, 0, 0][
                    j2, 0, ibx[0, 0, 0][4]
                ]
            for j3 in range(NG23):
                sfluxzen[0, 0][ngs[0, 0][7] + j3] = sfluxref01[0, 0, 0][
                    j3, 0, ibx[0, 0, 0][7]
                ]
            for j4 in range(NG25):
                sfluxzen[0, 0][ngs[0, 0][9] + j4] = sfluxref01[0, 0, 0][
                    j4, 0, ibx[0, 0, 0][9]
                ]
            for j5 in range(NG26):
                sfluxzen[0, 0][ngs[0, 0][10] + j5] = sfluxref01[0, 0, 0][
                    j5, 0, ibx[0, 0, 0][10]
                ]
            for j6 in range(NG29):
                sfluxzen[0, 0][ngs[0, 0][13] + j6] = sfluxref01[0, 0, 0][
                    j6, 0, ibx[0, 0, 0][13]
                ]

            for j7 in range(NG27):
                sfluxzen[0, 0][ngs[0, 0][11] + j7] = (
                    scalekur * sfluxref01[0, 0, 0][j7, 0, ibx[0, 0, 0][11]]
                )

    with computation(FORWARD), interval(1, -1):
        if idxday:
            if not laytrop:
                # case default
                cond = jp < layreffr[0, 0, 0][1] and jp[0, 0, 1] >= layreffr[0, 0, 0][1]

                if cond:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][1]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][1]]
                elif sfluxzen[0, 0][ngs[0, 0][1]] == 0.0 and layind == 61:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][1]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][1]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][1] * colm2
                    specmult = specwt[0, 0, 0][1] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj in range(NG17):
                        sfluxzen[0, 0][ngs[0, 0][1] + jj] = sfluxref02[0, 0, 0][
                            jj, js, ibx[0, 0, 0][1]
                        ] + fs * (
                            sfluxref02[0, 0, 0][jj, js + 1, ibx[0, 0, 0][1]]
                            - sfluxref02[0, 0, 0][jj, js, ibx[0, 0, 0][1]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][12] and jp[0, 0, 1] >= layreffr[0, 0, 0][12]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][12]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][12]]
                elif sfluxzen[0, 0][ngs[0, 0][12]] == 0.0 and layind == 61:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][12]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][12]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][12] * colm2
                    specmult = specwt[0, 0, 0][12] * min(oneminus, colm1 / speccomb)
                    jsa = specmult
                    fsa = mod(specmult, 1.0)

                    for jj2 in range(NG28):
                        sfluxzen[0, 0][ngs[0, 0][12] + jj2] = sfluxref02[0, 0, 0][
                            jj2, jsa, ibx[0, 0, 0][12]
                        ] + fsa * (
                            sfluxref02[0, 0, 0][jj2, jsa + 1, ibx[0, 0, 0][12]]
                            - sfluxref02[0, 0, 0][jj2, jsa, ibx[0, 0, 0][12]]
                        )

            if laytrop:
                if jp < layreffr[0, 0, 0][2] and jp[0, 0, 1] >= layreffr[0, 0, 0][2]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][2]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][2]]
                if layind == laytropind and sfluxzen[0, 0][ngs[0, 0][2]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][2]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][2]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][2] * colm2
                    specmult = specwt[0, 0, 0][2] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj3 in range(NG18):
                        sfluxzen[0, 0][ngs[0, 0][2] + jj3] = sfluxref03[0, 0, 0][
                            jj3, js, ibx[0, 0, 0][2]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj3, js + 1, ibx[0, 0, 0][2]]
                            - sfluxref03[0, 0, 0][jj3, js, ibx[0, 0, 0][2]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][3] and jp[0, 0, 1] >= layreffr[0, 0, 0][3]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][3]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][3]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][3]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][3]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][3]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][3] * colm2
                    specmult = specwt[0, 0, 0][3] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj4 in range(NG19):
                        sfluxzen[0, 0][ngs[0, 0][3] + jj4] = sfluxref03[0, 0, 0][
                            jj4, js, ibx[0, 0, 0][3]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj4, js + 1, ibx[0, 0, 0][3]]
                            - sfluxref03[0, 0, 0][jj4, js, ibx[0, 0, 0][3]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][5] and jp[0, 0, 1] >= layreffr[0, 0, 0][5]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][5]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][5]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][5]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][5]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][5]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][5] * colm2
                    specmult = specwt[0, 0, 0][5] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj5 in range(NG21):
                        sfluxzen[0, 0][ngs[0, 0][5] + jj5] = sfluxref03[0, 0, 0][
                            jj5, js, ibx[0, 0, 0][5]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj5, js + 1, ibx[0, 0, 0][5]]
                            - sfluxref03[0, 0, 0][jj5, js, ibx[0, 0, 0][5]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][6] and jp[0, 0, 1] >= layreffr[0, 0, 0][6]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][6]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][6]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][6]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][6]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][6]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][6] * colm2
                    specmult = specwt[0, 0, 0][6] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj6 in range(NG22):
                        sfluxzen[0, 0][ngs[0, 0][6] + jj6] = sfluxref03[0, 0, 0][
                            jj6, js, ibx[0, 0, 0][6]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj6, js + 1, ibx[0, 0, 0][6]]
                            - sfluxref03[0, 0, 0][jj6, js, ibx[0, 0, 0][6]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][8] and jp[0, 0, 1] >= layreffr[0, 0, 0][8]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][8]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][8]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][8]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][8]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][8]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][8] * colm2
                    specmult = specwt[0, 0, 0][8] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj7 in range(NG24):
                        sfluxzen[0, 0][ngs[0, 0][8] + jj7] = sfluxref03[0, 0, 0][
                            jj7, js, ibx[0, 0, 0][8]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj7, js + 1, ibx[0, 0, 0][8]]
                            - sfluxref03[0, 0, 0][jj7, js, ibx[0, 0, 0][8]]
                        )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict16["rayl"],
        "oneminus": oneminus,
        "NG16": NG16,
        "NS16": NS16,
    },
)
def taumol16(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG16))],
    forref: Field[(DTYPE_FLT, (3, NG16))],
    absa: Field[(DTYPE_FLT, (585, NG16))],
    absb: Field[(DTYPE_FLT, (235, NG16))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG16, NS16

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG16):
            taur[0, 0, 0][NS16 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][0] * colamt[0, 0, 0][4]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs

            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][0] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][0] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10
            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG16):
                taug[0, 0, 0][NS16 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )

        else:
            ind01 = id0[0, 0, 0][0]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][0]
            ind12 = ind11 + 1

            for j3 in range(NG16):
                taug[0, 0, 0][NS16 + j3] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict17["rayl"],
        "oneminus": oneminus,
        "NG17": NG17,
        "NS17": NS17,
    },
)
def taumol17(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG17))],
    forref: Field[(DTYPE_FLT, (3, NG17))],
    absa: Field[(DTYPE_FLT, (585, NG17))],
    absb: Field[(DTYPE_FLT, (235, NG17))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG17, NS17

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG17):
            taur[0, 0, 0][NS17 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][1] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][1] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][1] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG17):
                taug[0, 0, 0][NS17 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][1] * colamt[0, 0, 0][1]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][1] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][1] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG17):
                taug[0, 0, 0][NS17 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                ) + colamt[0, 0, 0][0] * forfac * (
                    forref[0, 0, 0][indf, j3]
                    + forfrac * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict18["rayl"],
        "oneminus": oneminus,
        "NG18": NG18,
        "NS18": NS18,
    },
)
def taumol18(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG18))],
    forref: Field[(DTYPE_FLT, (3, NG18))],
    absa: Field[(DTYPE_FLT, (585, NG18))],
    absb: Field[(DTYPE_FLT, (235, NG18))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG18, NS18

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG18):
            taur[0, 0, 0][NS18 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][2] * colamt[0, 0, 0][4]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][2] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][2] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG18):
                taug[0, 0, 0][NS18 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            ind01 = id0[0, 0, 0][2]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][2]
            ind12 = ind11 + 1

            for j3 in range(NG18):
                taug[0, 0, 0][NS18 + j3] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict19["rayl"],
        "oneminus": oneminus,
        "NG19": NG19,
        "NS19": NS19,
    },
)
def taumol19(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG19))],
    forref: Field[(DTYPE_FLT, (3, NG19))],
    absa: Field[(DTYPE_FLT, (585, NG19))],
    absb: Field[(DTYPE_FLT, (235, NG19))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG19, NS19

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG19):
            taur[0, 0, 0][NS19 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][3] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][3] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][3] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG19):
                taug[0, 0, 0][NS19 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            ind01 = id0[0, 0, 0][3]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][3]
            ind12 = ind11 + 1

            for j3 in range(NG19):
                taug[0, 0, 0][NS19 + j3] = colamt[0, 0, 0][1] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict20["rayl"],
        "NG20": NG20,
        "NS20": NS20,
    },
)
def taumol20(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    selfref: Field[(DTYPE_FLT, (10, NG20))],
    forref: Field[(DTYPE_FLT, (4, NG20))],
    absa: Field[(DTYPE_FLT, (65, NG20))],
    absb: Field[(DTYPE_FLT, (235, NG20))],
    absch4: Field[(DTYPE_FLT, (NG20,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, NG20, NS20

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG20):
            taur[0, 0, 0][NS20 + j] = tauray

        if laytrop:
            ind01 = id0[0, 0, 0][4]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][4]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG20):
                taug[0, 0, 0][NS20 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        (
                            fac00 * absa[0, 0, 0][ind01, j2]
                            + fac10 * absa[0, 0, 0][ind02, j2]
                            + fac01 * absa[0, 0, 0][ind11, j2]
                            + fac11 * absa[0, 0, 0][ind12, j2]
                        )
                        + selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + colamt[0, 0, 0][4] * absch4[0, 0, 0][j2]
                )
        else:
            ind01 = id0[0, 0, 0][4]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][4]
            ind12 = ind11 + 1

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG20):
                taug[0, 0, 0][NS20 + j3] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j3]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                        )
                    )
                    + colamt[0, 0, 0][4] * absch4[0, 0, 0][j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict21["rayl"],
        "oneminus": oneminus,
        "NG21": NG21,
        "NS21": NS21,
    },
)
def taumol21(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG21))],
    forref: Field[(DTYPE_FLT, (4, NG21))],
    absa: Field[(DTYPE_FLT, (65, NG21))],
    absb: Field[(DTYPE_FLT, (235, NG21))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG21, NS21

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG21):
            taur[0, 0, 0][NS21 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][5] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][5] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][5] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG21):
                taug[0, 0, 0][NS21 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][5] * colamt[0, 0, 0][1]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][5] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][5] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG21):
                taug[0, 0, 0][NS21 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                ) + colamt[0, 0, 0][0] * forfac * (
                    forref[0, 0, 0][indf, j3]
                    + forfrac * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "rayl": lookupdict22["rayl"],
        "oneminus": oneminus,
        "NG22": NG22,
        "NS22": NS22,
    },
)
def taumol22(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG22))],
    forref: Field[(DTYPE_FLT, (4, NG22))],
    absa: Field[(DTYPE_FLT, (65, NG22))],
    absb: Field[(DTYPE_FLT, (235, NG22))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import rayl, oneminus, NG22, NS22

    with computation(PARALLEL), interval(1, None):

        #  --- ...  the following factor is the ratio of total o2 band intensity (lines
        #           and mate continuum) to o2 band intensity (line only). it is needed
        #           to adjust the optical depths since the k's include only lines.

        o2adj = 1.6
        o2tem = 4.35e-4 / (350.0 * 2.0)

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG22):
            taur[0, 0, 0][NS22 + j] = tauray

        if laytrop:
            o2cont = o2tem * colamt[0, 0, 0][5]
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][6] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][6] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][6] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG22):
                taug[0, 0, 0][NS22 + j2] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ind01, j2]
                        + fac100 * absa[0, 0, 0][ind02, j2]
                        + fac010 * absa[0, 0, 0][ind03, j2]
                        + fac110 * absa[0, 0, 0][ind04, j2]
                        + fac001 * absa[0, 0, 0][ind11, j2]
                        + fac101 * absa[0, 0, 0][ind12, j2]
                        + fac011 * absa[0, 0, 0][ind13, j2]
                        + fac111 * absa[0, 0, 0][ind14, j2]
                    )
                    + colamt[0, 0, 0][0]
                    * (
                        selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + o2cont
                )
        else:
            o2cont = o2tem * colamt[0, 0, 0][5]

            ind01 = id0[0, 0, 0][6]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][6]
            ind12 = ind11 + 1

            for j3 in range(NG22):
                taug[0, 0, 0][NS22 + j3] = (
                    colamt[0, 0, 0][5]
                    * o2adj
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + o2cont
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "givfac": lookupdict23["givfac"],
        "NG23": NG23,
        "NS23": NS23,
    },
)
def taumol23(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    selfref: Field[(DTYPE_FLT, (10, NG23))],
    forref: Field[(DTYPE_FLT, (3, NG23))],
    absa: Field[(DTYPE_FLT, (65, NG23))],
    rayl: Field[(DTYPE_FLT, (NG23,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import givfac, NG23, NS23

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG23):
            taur[0, 0, 0][NS23 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][7]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][7]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG23):
                taug[0, 0, 0][NS23 + j2] = colamt[0, 0, 0][0] * (
                    givfac
                    * (
                        fac00 * absa[0, 0, 0][ind01, j2]
                        + fac10 * absa[0, 0, 0][ind02, j2]
                        + fac01 * absa[0, 0, 0][ind11, j2]
                        + fac11 * absa[0, 0, 0][ind12, j2]
                    )
                    + selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            for j3 in range(NG23):
                taug[0, 0, 0][NS23 + j3] = 0.0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG24": NG24,
        "NS24": NS24,
    },
)
def taumol24(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG24))],
    forref: Field[(DTYPE_FLT, (3, NG24))],
    absa: Field[(DTYPE_FLT, (585, NG24))],
    absb: Field[(DTYPE_FLT, (235, NG24))],
    rayla: Field[(DTYPE_FLT, (NG24, 9))],
    raylb: Field[(DTYPE_FLT, (NG24,))],
    abso3a: Field[(DTYPE_FLT, (NG24,))],
    abso3b: Field[(DTYPE_FLT, (NG24,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    js: FIELD_INT,
):

    from __externals__ import oneminus, NG24, NS24

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][8] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][8] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][8] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG24):
                taug[0, 0, 0][NS24 + j2] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ind01, j2]
                        + fac100 * absa[0, 0, 0][ind02, j2]
                        + fac010 * absa[0, 0, 0][ind03, j2]
                        + fac110 * absa[0, 0, 0][ind04, j2]
                        + fac001 * absa[0, 0, 0][ind11, j2]
                        + fac101 * absa[0, 0, 0][ind12, j2]
                        + fac011 * absa[0, 0, 0][ind13, j2]
                        + fac111 * absa[0, 0, 0][ind14, j2]
                    )
                    + colamt[0, 0, 0][2] * abso3a[0, 0, 0][j2]
                    + colamt[0, 0, 0][0]
                    * (
                        selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                )

                taur[0, 0, 0][NS24 + j2] = colmol * (
                    rayla[0, 0, 0][j2, js - 1]
                    + fs * (rayla[0, 0, 0][j2, js] - rayla[0, 0, 0][j2, js - 1])
                )
        else:
            ind01 = id0[0, 0, 0][8]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][8]
            ind12 = ind11 + 1

            for j3 in range(NG24):
                taug[0, 0, 0][NS24 + j3] = (
                    colamt[0, 0, 0][5]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + colamt[0, 0, 0][2] * abso3b[0, 0, 0][j3]
                )

                taur[0, 0, 0][NS24 + j3] = colmol * raylb[0, 0, 0][j3]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG25": NG25,
        "NS25": NS25,
    },
)
def taumol25(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    absa: Field[(DTYPE_FLT, (65, NG25))],
    rayl: Field[(DTYPE_FLT, (NG25,))],
    abso3a: Field[(DTYPE_FLT, (NG25,))],
    abso3b: Field[(DTYPE_FLT, (NG25,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
):

    from __externals__ import NG25, NS25

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG25):
            taur[0, 0, 0][NS25 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][9]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][9]
            ind12 = ind11 + 1

            for j2 in range(NG25):
                taug[0, 0, 0][NS25 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ind01, j2]
                        + fac10 * absa[0, 0, 0][ind02, j2]
                        + fac01 * absa[0, 0, 0][ind11, j2]
                        + fac11 * absa[0, 0, 0][ind12, j2]
                    )
                    + colamt[0, 0, 0][2] * abso3a[0, 0, 0][j2]
                )
        else:
            for j3 in range(NG25):
                taug[0, 0, 0][NS25 + j3] = colamt[0, 0, 0][2] * abso3b[0, 0, 0][j3]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG26": NG26,
        "NS26": NS26,
    },
)
def taumol26(
    colmol: FIELD_FLT,
    rayl: Field[(DTYPE_FLT, (NG26,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
):

    from __externals__ import NG26, NS26

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG26):
            taug[0, 0, 0][NS26 + j] = 0.0
            taur[0, 0, 0][NS26 + j] = colmol * rayl[0, 0, 0][j]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG27": NG27,
        "NS27": NS27,
    },
)
def taumol27(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    absa: Field[(DTYPE_FLT, (65, NG27))],
    absb: Field[(DTYPE_FLT, (235, NG27))],
    rayl: Field[(DTYPE_FLT, (NG27,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
):

    from __externals__ import NG27, NS27

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG27):
            taur[0, 0, 0][NS27 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][11]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][11]
            ind12 = ind11 + 1

            for j2 in range(NG27):
                taug[0, 0, 0][NS27 + j2] = colamt[0, 0, 0][2] * (
                    fac00 * absa[0, 0, 0][ind01, j2]
                    + fac10 * absa[0, 0, 0][ind02, j2]
                    + fac01 * absa[0, 0, 0][ind11, j2]
                    + fac11 * absa[0, 0, 0][ind12, j2]
                )
        else:
            ind01 = id0[0, 0, 0][11]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][11]
            ind12 = ind11 + 1

            for j3 in range(NG27):
                taug[0, 0, 0][NS27 + j3] = colamt[0, 0, 0][2] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG28": NG28,
        "NS28": NS28,
        "rayl": lookupdict28["rayl"],
        "oneminus": oneminus,
    },
)
def taumol28(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    strrat: Field[type_nbandssw_flt],
    absa: Field[(DTYPE_FLT, (585, NG28))],
    absb: Field[(DTYPE_FLT, (1175, NG28))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    js: FIELD_INT,
):

    from __externals__ import NG28, NS28, rayl, oneminus

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG28):
            taur[0, 0, 0][NS28 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][2] + strrat[0, 0, 0][12] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][2] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][12] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][12] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            for j2 in range(NG28):
                taug[0, 0, 0][NS28 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                )

        else:
            speccomb = colamt[0, 0, 0][2] + strrat[0, 0, 0][12] * colamt[0, 0, 0][5]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][2] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][12] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][12] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            for j3 in range(NG28):
                taug[0, 0, 0][NS28 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG29": NG29,
        "NS29": NS29,
        "rayl": lookupdict29["rayl"],
    },
)
def taumol29(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forref: Field[(DTYPE_FLT, (4, NG29))],
    absa: Field[(DTYPE_FLT, (65, NG29))],
    absb: Field[(DTYPE_FLT, (235, NG29))],
    selfref: Field[(DTYPE_FLT, (10, NG29))],
    absh2o: Field[(DTYPE_FLT, (NG29,))],
    absco2: Field[(DTYPE_FLT, (NG29,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import NG29, NS29, rayl

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG29):
            taur[0, 0, 0][NS29 + j] = tauray

        if laytrop:
            ind01 = id0[0, 0, 0][13]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][13]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG29):
                taug[0, 0, 0][NS29 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        (
                            fac00 * absa[0, 0, 0][ind01, j2]
                            + fac10 * absa[0, 0, 0][ind02, j2]
                            + fac01 * absa[0, 0, 0][ind11, j2]
                            + fac11 * absa[0, 0, 0][ind12, j2]
                        )
                        + selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + colamt[0, 0, 0][1] * absco2[0, 0, 0][j2]
                )
        else:
            ind01 = id0[0, 0, 0][13]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][13]
            ind12 = ind11 + 1

            for j3 in range(NG29):
                taug[0, 0, 0][NS29 + j3] = (
                    colamt[0, 0, 0][1]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + colamt[0, 0, 0][0] * absh2o[0, 0, 0][j3]
                )


taumolsetup(
    indict_gt4py["colamt"],
    indict_gt4py["jp"],
    indict_gt4py["jt"],
    indict_gt4py["jt1"],
    indict_gt4py["laytrop"],
    indict_gt4py["laytropind"],
    indict_gt4py["idxday"],
    outdict_gt4py["sfluxzen"],
    locdict_gt4py["layind"],
    locdict_gt4py["nspa"],
    locdict_gt4py["nspb"],
    locdict_gt4py["ngs"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["js"],
    locdict_gt4py["jsa"],
    locdict_gt4py["colm1"],
    locdict_gt4py["colm2"],
    lookupdict_ref["sfluxref01"],
    lookupdict_ref["sfluxref02"],
    lookupdict_ref["sfluxref03"],
    lookupdict_ref["layreffr"],
    lookupdict_ref["ix1"],
    lookupdict_ref["ix2"],
    lookupdict_ref["ibx"],
    lookupdict_ref["strrat"],
    lookupdict_ref["specwt"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol16(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict16["selfref"],
    lookupdict16["forref"],
    lookupdict16["absa"],
    lookupdict16["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol17(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict17["selfref"],
    lookupdict17["forref"],
    lookupdict17["absa"],
    lookupdict17["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol18(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict18["selfref"],
    lookupdict18["forref"],
    lookupdict18["absa"],
    lookupdict18["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol19(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict19["selfref"],
    lookupdict19["forref"],
    lookupdict19["absa"],
    lookupdict19["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol20(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict20["selfref"],
    lookupdict20["forref"],
    lookupdict20["absa"],
    lookupdict20["absb"],
    lookupdict20["absch4"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol21(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict21["selfref"],
    lookupdict21["forref"],
    lookupdict21["absa"],
    lookupdict21["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol22(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict22["selfref"],
    lookupdict22["forref"],
    lookupdict22["absa"],
    lookupdict22["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol23(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict23["selfref"],
    lookupdict23["forref"],
    lookupdict23["absa"],
    lookupdict23["rayl"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol24(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict24["selfref"],
    lookupdict24["forref"],
    lookupdict24["absa"],
    lookupdict24["absb"],
    lookupdict24["rayla"],
    lookupdict24["raylb"],
    lookupdict24["abso3a"],
    lookupdict24["abso3b"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["js"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol25(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    lookupdict25["absa"],
    lookupdict25["rayl"],
    lookupdict25["abso3a"],
    lookupdict25["abso3b"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol26(
    indict_gt4py["colmol"],
    lookupdict26["rayl"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol27(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    lookupdict27["absa"],
    lookupdict27["absb"],
    lookupdict27["rayl"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol28(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    lookupdict_ref["strrat"],
    lookupdict28["absa"],
    lookupdict28["absb"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["js"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol29(
    indict_gt4py["colamt"],
    indict_gt4py["colmol"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["laytrop"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    lookupdict29["forref"],
    lookupdict29["absa"],
    lookupdict29["absb"],
    lookupdict29["selfref"],
    lookupdict29["absh2o"],
    lookupdict29["absco2"],
    outdict_gt4py["taug"],
    outdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

outvars = ["taug", "taur", "sfluxzen"]

outdict_np = dict()
valdict_np = dict()

for var in outvars:
    if var == "sfluxzen":
        outdict_np[var] = outdict_gt4py[var].view(np.ndarray).squeeze()
    else:
        outdict_np[var] = outdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()

    valdict_np[var] = serializer.read(
        var, serializer.savepoint["swrad-taumol-output-000000"]
    )

compare_data(outdict_np, valdict_np)
