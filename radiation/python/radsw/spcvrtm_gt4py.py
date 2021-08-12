import sys
import numpy as np
import xarray as xr
from gt4py.gtscript import (
    stencil,
    computation,
    PARALLEL,
    interval,
    log,
    index,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from config import *
from radsw_param import ngptsw, nblow, nbhgh, NGB, idxsfc
from util import create_storage_from_array, create_storage_zeros, compare_data

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radsw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = [
    "ssolar",
    "cosz1",
    "sntz1",
    "albbm",
    "albdf",
    "sfluxzen",
    "cldfmc",
    "zcf1",
    "zcf0",
    "taug",
    "taur",
    "tauae",
    "ssaae",
    "asyae",
    "taucw",
    "ssacw",
    "asycw",
    "nlay",
    "nlp1",
]

outvars = [
    "fxupc",
    "fxdnc",
    "fxup0",
    "fxdn0",
    "ftoauc",
    "ftoau0",
    "ftoadc",
    "fsfcuc",
    "fsfcu0",
    "fsfcdc",
    "fsfcd0",
    "sfbmc",
    "sfdfc",
    "sfbm0",
    "sfdf0",
    "suvbfc",
    "suvbf0",
]

locvars = [
    "ztaus",
    "zssas",
    "zasys",
    "zldbt0",
    "zrefb",
    "zrefd",
    "ztrab",
    "ztrad",
    "ztdbt",
    "zldbt",
    "zfu",
    "zfd",
    "ztau1",
    "zssa1",
    "zasy1",
    "ztau0",
    "zssa0",
    "zasy0",
    "zasy3",
    "zssaw",
    "zasyw",
    "zgam1",
    "zgam2",
    "zgam3",
    "zgam4",
    "za1",
    "za2",
    "zb1",
    "zb2",
    "zrk",
    "zrk2",
    "zrp",
    "zrp1",
    "zrm1",
    "zrpp",
    "zrkg1",
    "zrkg3",
    "zrkg4",
    "zexp1",
    "zexm1",
    "zexp2",
    "zexm2",
    "zden1",
    "zexp3",
    "zexp4",
    "ze1r45",
    "ftind",
    "zsolar",
    "ztdbt0",
    "zr1",
    "zr2",
    "zr3",
    "zr4",
    "zr5",
    "zt1",
    "zt2",
    "zt3",
    "zf1",
    "zf2",
    "zrpp1",
    "jb",
    "ib",
    "ibd",
]

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-spcvrtm-input-000000"])
    if var in [
        "taug",
        "taur",
        "cldfmc",
        "taucw",
        "ssacw",
        "asycw",
        "tauae",
        "ssaae",
        "asyae",
    ]:
        tmp2 = np.insert(tmp, 0, 0, axis=0)
        indict[var] = np.tile(tmp2[None, None, :, :], (npts, 1, 1, 1))
    elif var in ["sfluxzen", "albbm", "albdf"]:
        indict[var] = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
    else:
        indict[var] = np.tile(tmp[None, None], (npts, 1))

indict_gt4py = dict()
for var in invars:
    if var in ["taug", "taur", "cldfmc", "sfluxzen"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_ngptsw
        )
    elif var in ["taucw", "ssacw", "asycw", "tauae", "ssaae", "asyae"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_nbdsw
        )
    elif var in ["albbm", "albdf"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (2,))
        )
    else:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, FIELD_2D
        )

outdict_gt4py = dict()
for var in outvars:
    if var in ["fxupc", "fxdnc", "fxup0", "fxdn0"]:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    elif var in ["sfbmc", "sfdfc", "sfbm0", "sfdf0"]:
        outdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_FLT, (2,))
        )
    elif var in [
        "suvbfc",
        "suvbf0",
        "ftoadc",
        "ftoauc",
        "ftoau0",
        "fsfcuc",
        "fsfcu0",
        "fsfcdc",
        "fsfcd0",
    ]:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)

locdict_gt4py = dict()
for var in locvars:
    if var in ["jb", "ib", "ibd"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

NGB = np.tile(np.array(NGB)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["NGB"] = create_storage_from_array(NGB, backend, shape_nlp1, type_ngptsw)

idxsfc = np.tile(np.array(idxsfc)[None, None, None, :], (npts, 1, nlp1, 14))
locdict_gt4py["idxsfc"] = create_storage_from_array(
    idxsfc, backend, shape_nlp1, (DTYPE_FLT, (14,))
)


@stencil(backend=backend, rebuild=rebuild, externals={"ngptsw": ngptsw, "nblow": nblow})
def spcvrtm(
    ssolar: FIELD_2D,
    cosz: FIELD_2D,
    sntz: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    sfluxzen: Field[type_ngptsw],
    cldfmc: Field[type_ngptsw],
    cf1: FIELD_2D,
    cf0: FIELD_2D,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    jb: FIELD_INT,
    ib: FIELD_INT,
    ibd: FIELD_INT,
    NGB: Field[type_ngptsw],
    idxsfc: Field[(DTYPE_FLT, (14,))],
):
    from __externals__ import ngptsw, nblow

    with computation(PARALLEL), interval(1, None):
        #  --- ...  loop over all g-points in each band
        for jg in range(ngptsw):
            jb = NGB[0, 0, 0][jg] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb]

            zsolar = ssolar * sfluxzen[0, 0, 0][jg]

            #  --- ...  set up toa direct beam and surface values (beam and diff)
