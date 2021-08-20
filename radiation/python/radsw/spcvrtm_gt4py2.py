import sys
import numpy as np
import xarray as xr
from gt4py.gtscript import (
    stencil,
    computation,
    PARALLEL,
    BACKWARD,
    interval,
    log,
    index,
    sqrt,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from config import *
from radsw_param import ngptsw, nblow, nbhgh, NGB, idxsfc, ntbmx
from radphysparam import iswmode
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

idxsfc = np.tile(np.array(idxsfc)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["idxsfc"] = create_storage_from_array(
    idxsfc, backend, shape_nlp1, (DTYPE_FLT, (14,))
)

eps = 1.0e-6
oneminus = 1.0 - eps

bpade = 1.0 / 0.278
ftiny = 1.0e-12
flimit = 1.0e-20

zcrit = 0.9999995  # thresold for conservative scattering
zsr3 = np.sqrt(3.0)
od_lo = 0.06
eps1 = 1.0e-8


@stencil(backend=backend, rebuild=rebuild, externals={"ngptsw": ngptsw})
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
    from __externals__ import ngptsw

    with computation(PARALLEL), interval(1, None):
        #  --- ...  loop over all g-points in each band
        for jg in range(ngptsw):
            jb = NGB[0, 0, 0][jg] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb]

            zsolar = ssolar * sfluxzen[0, 0, 0][jg]

    with computation(PARALLEL), interval(-1, None):
        ztdbt = 1.0
    with computation(PARALLEL), interval(1, 2):
        zldbt = 0.0
        if ibd != 0:
            zrefb = albbm[0, 0, 0][ibd]
            zrefd = albdf[0, 0, 0][ibd]
        else:
            zrefb = 0.5 * (albbm[0, 0, 0][0] + albbm[0, 0, 0][1])
            zrefd = 0.5 * (albdf[0, 0, 0][0] + albdf[0, 0, 0][1])
        ztrab = 0.0
        ztrad = 0.0

    with computation(PARALLEL), interval(1, None):
        # Compute clear-sky optical parameters, layer reflectance and
        #    transmittance.
        #    - Set up toa direct beam and surface values (beam and diff)
        #    - Delta scaling for clear-sky condition
        #    - General two-stream expressions for physparam::iswmode
        #    - Compute homogeneous reflectance and transmittance for both
        #      conservative and non-conservative scattering
        #    - Pre-delta-scaling clear and cloudy direct beam transmittance
        #    - Call swflux() to compute the upward and downward radiation fluxes

        ztdbt0 = 1.0

        for jg3 in range(ngptsw):
            jb = NGB[0, 0, 0][jg3] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb]

            ztau0 = max(
                ftiny, taur[0, 0, 0][jg3] + taug[0, 0, 0][jg3] + tauae[0, 0, 0][ib]
            )
            zssa0 = taur[0, 0, 0][jg3] + tauae[0, 0, 0][ib] * ssaae[0, 0, 0][ib]
            zasy0 = asyae[0, 0, 0][ib] * ssaae[0, 0, 0][ib] * tauae[0, 0, 0][ib]
            zssaw = min(oneminus, zssa0 / ztau0)
            zasyw = zasy0 / max(ftiny, zssa0)

            #  --- ...  saving clear-sky quantities for later total-sky usage
            ztaus = ztau0
            zssas = zssa0
            zasys = zasy0

            #  --- ...  delta scaling for clear-sky condition
            za1 = zasyw * zasyw
            za2 = zssaw * za1

            ztau1 = (1.0 - za2) * ztau0
            zssa1 = (zssaw - za2) / (1.0 - za2)
            zasy1 = zasyw / (1.0 + zasyw)  # to reduce truncation error
            zasy3 = 0.75 * zasy1


spcvrtm(
    indict_gt4py["ssolar"],
    indict_gt4py["cosz1"],
    indict_gt4py["sntz1"],
    indict_gt4py["albbm"],
    indict_gt4py["albdf"],
    indict_gt4py["sfluxzen"],
    indict_gt4py["cldfmc"],
    indict_gt4py["zcf1"],
    indict_gt4py["zcf0"],
    indict_gt4py["taug"],
    indict_gt4py["taur"],
    indict_gt4py["tauae"],
    indict_gt4py["ssaae"],
    indict_gt4py["asyae"],
    indict_gt4py["taucw"],
    indict_gt4py["ssacw"],
    indict_gt4py["asycw"],
    locdict_gt4py["jb"],
    locdict_gt4py["ib"],
    locdict_gt4py["ibd"],
    locdict_gt4py["NGB"],
    locdict_gt4py["idxsfc"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
