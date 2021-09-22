import sys
import numpy as np
import xarray as xr
import time
from gt4py.gtscript import (
    BACKWARD,
    FORWARD,
    stencil,
    function,
    computation,
    PARALLEL,
    interval,
    abs,
    max,
    min,
    sqrt,
)

# sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
sys.path.insert(0, "/work/radiation/python")
from config import *
from radsw.radsw_param import (
    ngptsw,
    nblow,
    NGB,
    idxsfc,
    ntbmx,
    bpade,
    oneminus,
    ftiny,
    flimit,
    eps,
    nuvb,
    bpade,
    flimit,
    zcrit,
    zsr3,
    od_lo,
    eps1,
    eps,
)
from radphysparam import iswmode
from util import create_storage_from_array, create_storage_zeros, compare_data

# SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/work/radiation/fortran/radsw/dump"
# ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radsw/dump"
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
    "exp_tbl",
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
    "zrupb",
    "zrupd",
    "ztdn",
    "zrdnd",
    "jb",
    "ib",
    "ibd",
    "itind",
    "zb11",
    "zb22",
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
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var in ["albbm", "albdf"]:
        indict[var] = np.tile(tmp[:, None, None, :], (1, 1, nlp1, 1))
    elif var == "sfluxzen":
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    elif var in ["zcf1", "zcf0"]:
        indict[var] = np.tile(tmp[:, None, None], (1, 1, nlp1))
    elif var == "exp_tbl":
        indict[var] = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
    else:
        indict[var] = np.tile(tmp[:, None], (1, 1))

indict_gt4py = dict()
for var in invars:
    if var in ["taug", "taur", "cldfmc"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_ngptsw
        )
    elif var == "sfluxzen":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, type_ngptsw
        )
    elif var in ["taucw", "ssacw", "asycw", "tauae", "ssaae", "asyae"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_nbdsw
        )
    elif var in ["albbm", "albdf"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (2,))
        )
    elif var == "exp_tbl":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_ntbmx
        )
    elif var in ["zcf1", "zcf0"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
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
        outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, (DTYPE_FLT, (2,)))
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
    if var in ["jb", "ib", "ibd", "itind"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var in ["zb11", "zb22"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, type_ngptsw)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw)

NGB = np.tile(np.array(NGB)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["NGB"] = create_storage_from_array(NGB, backend, shape_nlp1, type_ngptsw)

idxsfc = np.tile(np.array(idxsfc)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["idxsfc"] = create_storage_from_array(
    idxsfc, backend, shape_nlp1, (DTYPE_FLT, (14,))
)

zcrit = 0.9999995  # thresold for conservative scattering
zsr3 = np.sqrt(3.0)
od_lo = 0.06
eps1 = 1.0e-8

start = time.time()


@stencil(
    backend=backend,
    rebuild=rebuild,
    verbose=True,
    externals={
        "ngptsw": ngptsw,
        "bpade": bpade,
        "oneminus": oneminus,
        "ftiny": ftiny,
        "flimit": flimit,
        "zcrit": zcrit,
        "zsr3": zsr3,
        "od_lo": od_lo,
        "eps1": eps1,
        "eps": eps,
    },
)
def spcvrtm_clearsky(
    ssolar: FIELD_2D,
    cosz: FIELD_2D,
    sntz: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    cldfmc: Field[type_ngptsw],
    cf1: FIELD_FLT,
    cf0: FIELD_FLT,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    exp_tbl: Field[type_ntbmx],
    ztaus: Field[type_ngptsw],
    zssas: Field[type_ngptsw],
    zasys: Field[type_ngptsw],
    zldbt0: Field[type_ngptsw],
    zrefb: Field[type_ngptsw],
    zrefd: Field[type_ngptsw],
    ztrab: Field[type_ngptsw],
    ztrad: Field[type_ngptsw],
    ztdbt: Field[type_ngptsw],
    zldbt: Field[type_ngptsw],
    zfu: Field[type_ngptsw],
    zfd: Field[type_ngptsw],
    ztau1: Field[type_ngptsw],
    zssa1: Field[type_ngptsw],
    zasy1: Field[type_ngptsw],
    ztau0: Field[type_ngptsw],
    zssa0: Field[type_ngptsw],
    zasy0: Field[type_ngptsw],
    zasy3: Field[type_ngptsw],
    zssaw: Field[type_ngptsw],
    zasyw: Field[type_ngptsw],
    zgam1: Field[type_ngptsw],
    zgam2: Field[type_ngptsw],
    zgam3: Field[type_ngptsw],
    zgam4: Field[type_ngptsw],
    za1: Field[type_ngptsw],
    za2: Field[type_ngptsw],
    zb1: Field[type_ngptsw],
    zb2: Field[type_ngptsw],
    zrk: Field[type_ngptsw],
    zrk2: Field[type_ngptsw],
    zrp: Field[type_ngptsw],
    zrp1: Field[type_ngptsw],
    zrm1: Field[type_ngptsw],
    zrpp: Field[type_ngptsw],
    zrkg1: Field[type_ngptsw],
    zrkg3: Field[type_ngptsw],
    zrkg4: Field[type_ngptsw],
    zexp1: Field[type_ngptsw],
    zexm1: Field[type_ngptsw],
    zexp2: Field[type_ngptsw],
    zexm2: Field[type_ngptsw],
    zden1: Field[type_ngptsw],
    zexp3: Field[type_ngptsw],
    zexp4: Field[type_ngptsw],
    ze1r45: Field[type_ngptsw],
    ftind: Field[type_ngptsw],
    zsolar: Field[type_ngptsw],
    ztdbt0: Field[type_ngptsw],
    zr1: Field[type_ngptsw],
    zr2: Field[type_ngptsw],
    zr3: Field[type_ngptsw],
    zr4: Field[type_ngptsw],
    zr5: Field[type_ngptsw],
    zt1: Field[type_ngptsw],
    zt2: Field[type_ngptsw],
    zt3: Field[type_ngptsw],
    zf1: Field[type_ngptsw],
    zf2: Field[type_ngptsw],
    zrpp1: Field[type_ngptsw],
    zrupd: Field[type_ngptsw],
    zrupb: Field[type_ngptsw],
    ztdn: Field[type_ngptsw],
    zrdnd: Field[type_ngptsw],
    zb11: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    zb22: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    jb: FIELD_INT,
    ib: FIELD_INT,
    ibd: FIELD_INT,
    NGB: Field[type_ngptsw],
    idxsfc: Field[(DTYPE_FLT, (14,))],
    itind: FIELD_INT,
    fxupc: Field[type_nbdsw],
    fxdnc: Field[type_nbdsw],
    fxup0: Field[type_nbdsw],
    fxdn0: Field[type_nbdsw],
    ftoauc: FIELD_2D,
    ftoau0: FIELD_2D,
    ftoadc: FIELD_2D,
    fsfcuc: FIELD_2D,
    fsfcu0: FIELD_2D,
    fsfcdc: FIELD_2D,
    fsfcd0: FIELD_2D,
    sfbmc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdfc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfbm0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdf0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    suvbfc: FIELD_2D,
    suvbf0: FIELD_2D,
):
    from __externals__ import (
        ngptsw,
        bpade,
        oneminus,
        ftiny,
        zcrit,
        zsr3,
        od_lo,
        eps1,
        eps,
    )

    with computation(PARALLEL), interval(...):
        #  --- ...  loop over all g-points in each band
        for jg in range(ngptsw):
            jb = NGB[0, 0, 0][jg] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

            zsolar[0, 0, 0][jg] = ssolar * sfluxzen[0, 0][jg]
            ztdbt0[0, 0, 0][jg] = 1.0

    with computation(PARALLEL), interval(-1, None):
        for n in range(ngptsw):
            ztdbt[0, 0, 0][n] = 1.0
    with computation(PARALLEL), interval(0, 1):
        for n2 in range(ngptsw):
            jb = NGB[0, 0, 0][n2] - 1
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            zldbt[0, 0, 0][n2] = 0.0
            if ibd != -1:
                zrefb[0, 0, 0][n2] = albbm[0, 0, 0][ibd]
                zrefd[0, 0, 0][n2] = albdf[0, 0, 0][ibd]
            else:
                zrefb[0, 0, 0][n2] = 0.5 * (albbm[0, 0, 0][0] + albbm[0, 0, 0][1])
                zrefd[0, 0, 0][n2] = 0.5 * (albdf[0, 0, 0][0] + albdf[0, 0, 0][1])
            ztrab[0, 0, 0][n2] = 0.0
            ztrad[0, 0, 0][n2] = 0.0

    with computation(PARALLEL), interval(0, -1):
        # Compute clear-sky optical parameters, layer reflectance and
        #    transmittance.
        #    - Set up toa direct beam and surface values (beam and diff)
        #    - Delta scaling for clear-sky condition
        #    - General two-stream expressions for physparam::iswmode
        #    - Compute homogeneous reflectance and transmittance for both
        #      conservative and non-conservative scattering
        #    - Pre-delta-scaling clear and cloudy direct beam transmittance
        #    - Call swflux() to compute the upward and downward radiation fluxes

        for n3 in range(ngptsw):
            jb = NGB[0, 0, 0][n3] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

            ztau0[0, 0, 0][n3] = max(
                ftiny, taur[0, 0, 1][n3] + taug[0, 0, 1][n3] + tauae[0, 0, 1][ib]
            )
            zssa0[0, 0, 0][n3] = (
                taur[0, 0, 1][n3] + tauae[0, 0, 1][ib] * ssaae[0, 0, 1][ib]
            )
            zasy0[0, 0, 0][n3] = (
                asyae[0, 0, 1][ib] * ssaae[0, 0, 1][ib] * tauae[0, 0, 1][ib]
            )
            zssaw[0, 0, 0][n3] = min(oneminus, zssa0[0, 0, 0][n3] / ztau0[0, 0, 0][n3])
            zasyw[0, 0, 0][n3] = zasy0[0, 0, 0][n3] / max(ftiny, zssa0[0, 0, 0][n3])

            #  --- ...  saving clear-sky quantities for later total-sky usage
            ztaus[0, 0, 0][n3] = ztau0[0, 0, 0][n3]
            zssas[0, 0, 0][n3] = zssa0[0, 0, 0][n3]
            zasys[0, 0, 0][n3] = zasy0[0, 0, 0][n3]

            #  --- ...  delta scaling for clear-sky condition
            za1[0, 0, 0][n3] = zasyw[0, 0, 0][n3] * zasyw[0, 0, 0][n3]
            za2[0, 0, 0][n3] = zssaw[0, 0, 0][n3] * za1[0, 0, 0][n3]

            ztau1[0, 0, 0][n3] = (1.0 - za2[0, 0, 0][n3]) * ztau0[0, 0, 0][n3]
            zssa1[0, 0, 0][n3] = (zssaw[0, 0, 0][n3] - za2[0, 0, 0][n3]) / (
                1.0 - za2[0, 0, 0][n3]
            )
            zasy1[0, 0, 0][n3] = zasyw[0, 0, 0][n3] / (
                1.0 + zasyw[0, 0, 0][n3]
            )  # to reduce truncation error
            zasy3[0, 0, 0][n3] = 0.75 * zasy1[0, 0, 0][n3]

            #  --- ...  general two-stream expressions
            if iswmode == 1:
                zgam1[0, 0, 0][n3] = 1.75 - zssa1[0, 0, 0][n3] * (
                    1.0 + zasy3[0, 0, 0][n3]
                )
                zgam2[0, 0, 0][n3] = -0.25 + zssa1[0, 0, 0][n3] * (
                    1.0 - zasy3[0, 0, 0][n3]
                )
                zgam3[0, 0, 0][n3] = 0.5 - zasy3[0, 0, 0][n3] * cosz
            elif iswmode == 2:  # pifm
                zgam1[0, 0, 0][n3] = 2.0 - zssa1[0, 0, 0][n3] * (
                    1.25 + zasy3[0, 0, 0][n3]
                )
                zgam2[0, 0, 0][n3] = (
                    0.75 * zssa1[0, 0, 0][n3] * (1.0 - zasy1[0, 0, 0][n3])
                )
                zgam3[0, 0, 0][n3] = 0.5 - zasy3[0, 0, 0][n3] * cosz
            elif iswmode == 3:  # discrete ordinates
                zgam1[0, 0, 0][n3] = (
                    zsr3 * (2.0 - zssa1[0, 0, 0][n3] * (1.0 + zasy1[0, 0, 0][n3])) * 0.5
                )
                zgam2[0, 0, 0][n3] = (
                    zsr3 * zssa1[0, 0, 0][n3] * (1.0 - zasy1[0, 0, 0][n3]) * 0.5
                )
                zgam3[0, 0, 0][n3] = (1.0 - zsr3 * zasy1[0, 0, 0][n3] * cosz) * 0.5

            zgam4[0, 0, 0][n3] = 1.0 - zgam3[0, 0, 0][n3]

            #  --- ...  compute homogeneous reflectance and transmittance

            if zssaw[0, 0, 0][n3] >= zcrit:  # for conservative scattering
                za1[0, 0, 0][n3] = zgam1[0, 0, 0][n3] * cosz - zgam3[0, 0, 0][n3]
                za2[0, 0, 0][n3] = zgam1[0, 0, 0][n3] * ztau1[0, 0, 0][n3]

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1[0, 0, 0][n3] = min(ztau1[0, 0, 0][n3] * sntz, 500.0)
                if zb1[0, 0, 0][n3] <= od_lo:
                    zb2[0, 0, 0][n3] = (
                        1.0
                        - zb1[0, 0, 0][n3]
                        + 0.5 * zb1[0, 0, 0][n3] * zb1[0, 0, 0][n3]
                    )
                else:
                    ftind[0, 0, 0][n3] = zb1[0, 0, 0][n3] / (bpade + zb1[0, 0, 0][n3])
                    itind = ftind[0, 0, 0][n3] * ntbmx + 0.5
                    zb2[0, 0, 0][n3] = exp_tbl[0, 0, 0][itind]

    with computation(FORWARD), interval(1, None):
        for nn3 in range(ngptsw):
            jb = NGB[0, 0, 0][nn3] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            if zssaw[0, 0, -1][nn3] >= zcrit:
                # collimated beam
                zrefb[0, 0, 0][nn3] = max(
                    0.0,
                    min(
                        1.0,
                        (
                            za2[0, 0, -1][nn3]
                            - za1[0, 0, -1][nn3] * (1.0 - zb2[0, 0, -1][nn3])
                        )
                        / (1.0 + za2[0, 0, -1][nn3]),
                    ),
                )
                ztrab[0, 0, 0][nn3] = max(0.0, min(1.0, 1.0 - zrefb[0, 0, 0][nn3]))

                #      ...  isotropic incidence
                zrefd[0, 0, 0][nn3] = max(
                    0.0, min(1.0, za2[0, 0, -1][nn3] / (1.0 + za2[0, 0, -1][nn3]))
                )
                ztrad[0, 0, 0][nn3] = max(0.0, min(1.0, 1.0 - zrefd[0, 0, 0][nn3]))

    with computation(BACKWARD), interval(0, -1):
        for n4 in range(ngptsw):
            jb = NGB[0, 0, 0][n4] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            if zssaw[0, 0, 0][n4] < zcrit:  # for non-conservative scattering
                za1[0, 0, 0][n4] = (
                    zgam1[0, 0, 0][n4] * zgam4[0, 0, 0][n4]
                    + zgam2[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                )
                za2[0, 0, 0][n4] = (
                    zgam1[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                    + zgam2[0, 0, 0][n4] * zgam4[0, 0, 0][n4]
                )
                zrk[0, 0, 0][n4] = sqrt(
                    (zgam1[0, 0, 0][n4] - zgam2[0, 0, 0][n4])
                    * (zgam1[0, 0, 0][n4] + zgam2[0, 0, 0][n4])
                )
                zrk2[0, 0, 0][n4] = 2.0 * zrk[0, 0, 0][n4]

                zrp[0, 0, 0][n4] = zrk[0, 0, 0][n4] * cosz
                zrp1[0, 0, 0][n4] = 1.0 + zrp[0, 0, 0][n4]
                zrm1[0, 0, 0][n4] = 1.0 - zrp[0, 0, 0][n4]
                zrpp1[0, 0, 0][n4] = 1.0 - zrp[0, 0, 0][n4] * zrp[0, 0, 0][n4]
                tmp = max(flimit, abs(zrpp1[0, 0, 0][n4]))
                zrpp[0, 0, 0][n4] = (
                    tmp if zrpp1[0, 0, 0][n4] >= 0 else -tmp
                )  # avoid numerical singularity
                zrkg1[0, 0, 0][n4] = zrk[0, 0, 0][n4] + zgam1[0, 0, 0][n4]
                zrkg3[0, 0, 0][n4] = zrk[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                zrkg4[0, 0, 0][n4] = zrk[0, 0, 0][n4] * zgam4[0, 0, 0][n4]

                zr1[0, 0, 0][n4] = zrm1[0, 0, 0][n4] * (
                    za2[0, 0, 0][n4] + zrkg3[0, 0, 0][n4]
                )
                zr2[0, 0, 0][n4] = zrp1[0, 0, 0][n4] * (
                    za2[0, 0, 0][n4] - zrkg3[0, 0, 0][n4]
                )
                zr3[0, 0, 0][n4] = zrk2[0, 0, 0][n4] * (
                    zgam3[0, 0, 0][n4] - za2[0, 0, 0][n4] * cosz
                )
                zr4[0, 0, 0][n4] = zrpp[0, 0, 0][n4] * zrkg1[0, 0, 0][n4]
                zr5[0, 0, 0][n4] = zrpp[0, 0, 0][n4] * (
                    zrk[0, 0, 0][n4] - zgam1[0, 0, 0][n4]
                )

                zt1[0, 0, 0][n4] = zrp1[0, 0, 0][n4] * (
                    za1[0, 0, 0][n4] + zrkg4[0, 0, 0][n4]
                )
                zt2[0, 0, 0][n4] = zrm1[0, 0, 0][n4] * (
                    za1[0, 0, 0][n4] - zrkg4[0, 0, 0][n4]
                )
                zt3[0, 0, 0][n4] = zrk2[0, 0, 0][n4] * (
                    zgam4[0, 0, 0][n4] + za1[0, 0, 0][n4] * cosz
                )

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1[0, 0, 0][n4] = min(zrk[0, 0, 0][n4] * ztau1[0, 0, 0][n4], 500.0)
                if zb1[0, 0, 0][n4] <= od_lo:
                    zexm1[0, 0, 0][n4] = (
                        1.0
                        - zb1[0, 0, 0][n4]
                        + 0.5 * zb1[0, 0, 0][n4] * zb1[0, 0, 0][n4]
                    )
                else:
                    ftind[0, 0, 0][n4] = zb1[0, 0, 0][n4] / (bpade + zb1[0, 0, 0][n4])
                    itind = ftind[0, 0, 0][n4] * ntbmx + 0.5
                    zexm1[0, 0, 0][n4] = exp_tbl[0, 0, 0][itind]

                zexp1[0, 0, 0][n4] = 1.0 / zexm1[0, 0, 0][n4]

                zb2[0, 0, 0][n4] = min(sntz * ztau1[0, 0, 0][n4], 500.0)
                if zb2[0, 0, 0][n4] <= od_lo:
                    zexm2[0, 0, 0][n4] = (
                        1.0
                        - zb2[0, 0, 0][n4]
                        + 0.5 * zb2[0, 0, 0][n4] * zb2[0, 0, 0][n4]
                    )
                else:
                    ftind[0, 0, 0][n4] = zb2[0, 0, 0][n4] / (bpade + zb2[0, 0, 0][n4])
                    itind = ftind[0, 0, 0][n4] * ntbmx + 0.5
                    zexm2[0, 0, 0][n4] = exp_tbl[0, 0, 0][itind]

                zexp2[0, 0, 0][n4] = 1.0 / zexm2[0, 0, 0][n4]
                ze1r45[0, 0, 0][n4] = (
                    zr4[0, 0, 0][n4] * zexp1[0, 0, 0][n4]
                    + zr5[0, 0, 0][n4] * zexm1[0, 0, 0][n4]
                )

    with computation(BACKWARD), interval(1, None):
        for nn4 in range(ngptsw):

            if zssaw[0, 0, -1][nn4] < zcrit:
                #      ...  collimated beam
                if ze1r45[0, 0, -1][nn4] >= -eps1 and ze1r45[0, 0, -1][nn4] <= eps1:
                    zrefb[0, 0, 0][nn4] = eps1
                    ztrab[0, 0, 0][nn4] = zexm2[0, 0, -1][nn4]
                else:
                    zden1[0, 0, 0][nn4] = zssa1[0, 0, -1][nn4] / ze1r45[0, 0, -1][nn4]
                    zrefb[0, 0, 0][nn4] = max(
                        0.0,
                        min(
                            1.0,
                            (
                                zr1[0, 0, -1][nn4] * zexp1[0, 0, -1][nn4]
                                - zr2[0, 0, -1][nn4] * zexm1[0, 0, -1][nn4]
                                - zr3[0, 0, -1][nn4] * zexm2[0, 0, -1][nn4]
                            )
                            * zden1[0, 0, 0][nn4],
                        ),
                    )
                    ztrab[0, 0, 0][nn4] = max(
                        0.0,
                        min(
                            1.0,
                            zexm2[0, 0, -1][nn4]
                            * (
                                1.0
                                - (
                                    zt1[0, 0, -1][nn4] * zexp1[0, 0, -1][nn4]
                                    - zt2[0, 0, -1][nn4] * zexm1[0, 0, -1][nn4]
                                    - zt3[0, 0, -1][nn4] * zexp2[0, 0, -1][nn4]
                                )
                                * zden1[0, 0, 0][nn4]
                            ),
                        ),
                    )

                #      ...  diffuse beam
                zden1[0, 0, 0][nn4] = zr4[0, 0, -1][nn4] / (
                    ze1r45[0, 0, -1][nn4] * zrkg1[0, 0, -1][nn4]
                )
                zrefd[0, 0, 0][nn4] = max(
                    0.0,
                    min(
                        1.0,
                        zgam2[0, 0, -1][nn4]
                        * (zexp1[0, 0, -1][nn4] - zexm1[0, 0, -1][nn4])
                        * zden1[0, 0, 0][nn4],
                    ),
                )
                ztrad[0, 0, 0][nn4] = max(
                    0.0, min(1.0, zrk2[0, 0, -1][nn4] * zden1[0, 0, 0][nn4])
                )

    with computation(BACKWARD), interval(0, -1):

        #  --- ...  direct beam transmittance. use exponential lookup table
        #           for transmittance, or expansion of exponential for low
        #           optical depth

        for n5 in range(ngptsw):
            jb = NGB[0, 0, 0][n5] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            zr1[0, 0, 0][n5] = ztau1[0, 0, 0][n5] * sntz
            if zr1[0, 0, 0][n5] <= od_lo:
                zexp3[0, 0, 0][n5] = (
                    1.0 - zr1[0, 0, 0][n5] + 0.5 * zr1[0, 0, 0][n5] * zr1[0, 0, 0][n5]
                )
            else:
                ftind[0, 0, 0][n5] = zr1[0, 0, 0][n5] / (bpade + zr1[0, 0, 0][n5])
                itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][n5]))
                zexp3[0, 0, 0][n5] = exp_tbl[0, 0, 0][itind]

            ztdbt[0, 0, 0][n5] = zexp3[0, 0, 0][n5] * ztdbt[0, 0, 1][n5]

            #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
            #           (must use 'orig', unscaled cloud optical depth)

            zr1[0, 0, 0][n5] = ztau0[0, 0, 0][n5] * sntz
            if zr1[0, 0, 0][n5] <= od_lo:
                zexp4[0, 0, 0][n5] = (
                    1.0 - zr1[0, 0, 0][n5] + 0.5 * zr1[0, 0, 0][n5] * zr1[0, 0, 0][n5]
                )
            else:
                ftind[0, 0, 0][n5] = zr1[0, 0, 0][n5] / (bpade + zr1[0, 0, 0][n5])
                itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][n5]))
                zexp4[0, 0, 0][n5] = exp_tbl[0, 0, 0][itind]

            zldbt0[0, 0, 0][n5] = zexp4[0, 0, 0][n5]
            ztdbt0[0, 0, 0][n5] = zexp4[0, 0, 0][n5] * ztdbt0[0, 0, 1][n5]

    with computation(PARALLEL), interval(1, None):
        for nn5 in range(ngptsw):
            zldbt[0, 0, 0][nn5] = zexp3[0, 0, -1][nn5]

    # Need to incorporate the vrtqdr subroutine here, since I don't know
    # of a way to specify the interval within a gtscript function. Could be pulled
    # out later if we can call a stencil from a stencil

    with computation(FORWARD):
        with interval(0, 1):
            for nnn5 in range(ngptsw):
                # Link lowest layer with surface.
                zrupb[0, 0, 0][nnn5] = zrefb[0, 0, 0][nnn5]  # direct beam
                zrupd[0, 0, 0][nnn5] = zrefd[0, 0, 0][nnn5]  # diffused

        with interval(1, None):
            for n6 in range(ngptsw):
                zden1[0, 0, 0][n6] = 1.0 / (
                    1.0 - zrupd[0, 0, -1][n6] * zrefd[0, 0, 0][n6]
                )

                zrupb[0, 0, 0][n6] = (
                    zrefb[0, 0, 0][n6]
                    + (
                        ztrad[0, 0, 0][n6]
                        * (
                            (ztrab[0, 0, 0][n6] - zldbt[0, 0, 0][n6])
                            * zrupd[0, 0, -1][n6]
                            + zldbt[0, 0, 0][n6] * zrupb[0, 0, -1][n6]
                        )
                    )
                    * zden1[0, 0, 0][n6]
                )
                zrupd[0, 0, 0][n6] = (
                    zrefd[0, 0, 0][n6]
                    + ztrad[0, 0, 0][n6]
                    * ztrad[0, 0, 0][n6]
                    * zrupd[0, 0, -1][n6]
                    * zden1[0, 0, 0][n6]
                )

    with computation(PARALLEL):
        with interval(-2, -1):
            for n7 in range(ngptsw):
                ztdn[0, 0, 0][n7] = ztrab[0, 0, 1][n7]
                zrdnd[0, 0, 0][n7] = zrefd[0, 0, 1][n7]
        with interval(-1, None):
            for n8 in range(ngptsw):
                ztdn[0, 0, 0][n8] = 1.0
                zrdnd[0, 0, 0][n8] = 0.0

    with computation(BACKWARD), interval(0, -1):
        for n9 in range(ngptsw):
            zden1[0, 0, 0][n9] = 1.0 / (1.0 - zrefd[0, 0, 1][n9] * zrdnd[0, 0, 1][n9])
            ztdn[0, 0, 0][n9] = (
                ztdbt[0, 0, 1][n9] * ztrab[0, 0, 1][n9]
                + (
                    ztrad[0, 0, 1][n9]
                    * (
                        (ztdn[0, 0, 1][n9] - ztdbt[0, 0, 1][n9])
                        + ztdbt[0, 0, 1][n9] * zrefb[0, 0, 1][n9] * zrdnd[0, 0, 1][n9]
                    )
                )
                * zden1[0, 0, 0][n9]
            )
            zrdnd[0, 0, 0][n9] = (
                zrefd[0, 0, 1][n9]
                + ztrad[0, 0, 1][n9]
                * ztrad[0, 0, 1][n9]
                * zrdnd[0, 0, 1][n9]
                * zden1[0, 0, 0][n9]
            )

    # Up and down-welling fluxes at levels.
    with computation(PARALLEL), interval(...):
        for n10 in range(ngptsw):
            jb = NGB[0, 0, 0][n10] - 1
            ib = jb + 1 - nblow

            zden1[0, 0, 0][n10] = 1.0 / (
                1.0 - zrdnd[0, 0, 0][n10] * zrupd[0, 0, 0][n10]
            )
            zfu[0, 0, 0][n10] = (
                ztdbt[0, 0, 0][n10] * zrupb[0, 0, 0][n10]
                + (ztdn[0, 0, 0][n10] - ztdbt[0, 0, 0][n10]) * zrupd[0, 0, 0][n10]
            ) * zden1[0, 0, 0][n10]
            zfd[0, 0, 0][n10] = (
                ztdbt[0, 0, 0][n10]
                + (
                    ztdn[0, 0, 0][n10]
                    - ztdbt[0, 0, 0][n10]
                    + ztdbt[0, 0, 0][n10] * zrupb[0, 0, 0][n10] * zrdnd[0, 0, 0][n10]
                )
                * zden1[0, 0, 0][n10]
            )

            #  --- ...  compute upward and downward fluxes at levels
            fxup0[0, 0, 0][ib] = (
                fxup0[0, 0, 0][ib] + zsolar[0, 0, 0][n10] * zfu[0, 0, 0][n10]
            )
            fxdn0[0, 0, 0][ib] = (
                fxdn0[0, 0, 0][ib] + zsolar[0, 0, 0][n10] * zfd[0, 0, 0][n10]
            )

    with computation(FORWARD), interval(0, 1):
        for n11 in range(ngptsw):
            jb = NGB[0, 0, 0][n11] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            # --- ...  surface downward beam/diffuse flux components
            zb11[0, 0][n11] = zsolar[0, 0, 0][n11] * ztdbt0[0, 0, 0][n11]
            zb22[0, 0][n11] = zsolar[0, 0, 0][n11] * (
                zfd[0, 0, 0][n11] - ztdbt0[0, 0, 0][n11]
            )

            if ibd != -1:
                sfbm0[0, 0][ibd] = sfbm0[0, 0][ibd] + zb11[0, 0][n11]
                sfdf0[0, 0][ibd] = sfdf0[0, 0][ibd] + zb22[0, 0][n11]
            else:
                zf1[0, 0, 0][n11] = 0.5 * zb11[0, 0][n11]
                zf2[0, 0, 0][n11] = 0.5 * zb22[0, 0][n11]
                sfbm0[0, 0][0] = sfbm0[0, 0][0] + zf1[0, 0, 0][n11]
                sfdf0[0, 0][0] = sfdf0[0, 0][0] + zf2[0, 0, 0][n11]
                sfbm0[0, 0][1] = sfbm0[0, 0][1] + zf1[0, 0, 0][n11]
                sfdf0[0, 0][1] = sfdf0[0, 0][1] + zf2[0, 0, 0][n11]

            zldbt[0, 0, 0][n11] = 0.0


ibd0 = nuvb - nblow


@stencil(
    backend=backend,
    rebuild=rebuild,
    verbose=True,
    externals={
        "ngptsw": ngptsw,
        "bpade": bpade,
        "oneminus": oneminus,
        "ftiny": ftiny,
        "flimit": flimit,
        "zcrit": zcrit,
        "zsr3": zsr3,
        "od_lo": od_lo,
        "eps1": eps1,
        "eps": eps,
        "ibd0": ibd0,
    },
)
def spcvrtm_allsky(
    ssolar: FIELD_2D,
    cosz: FIELD_2D,
    sntz: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    cldfmc: Field[type_ngptsw],
    cf1: FIELD_FLT,
    cf0: FIELD_FLT,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    exp_tbl: Field[type_ntbmx],
    ztaus: Field[type_ngptsw],
    zssas: Field[type_ngptsw],
    zasys: Field[type_ngptsw],
    zldbt0: Field[type_ngptsw],
    zrefb: Field[type_ngptsw],
    zrefd: Field[type_ngptsw],
    ztrab: Field[type_ngptsw],
    ztrad: Field[type_ngptsw],
    ztdbt: Field[type_ngptsw],
    zldbt: Field[type_ngptsw],
    zfu: Field[type_ngptsw],
    zfd: Field[type_ngptsw],
    ztau1: Field[type_ngptsw],
    zssa1: Field[type_ngptsw],
    zasy1: Field[type_ngptsw],
    ztau0: Field[type_ngptsw],
    zssa0: Field[type_ngptsw],
    zasy0: Field[type_ngptsw],
    zasy3: Field[type_ngptsw],
    zssaw: Field[type_ngptsw],
    zasyw: Field[type_ngptsw],
    zgam1: Field[type_ngptsw],
    zgam2: Field[type_ngptsw],
    zgam3: Field[type_ngptsw],
    zgam4: Field[type_ngptsw],
    za1: Field[type_ngptsw],
    za2: Field[type_ngptsw],
    zb1: Field[type_ngptsw],
    zb2: Field[type_ngptsw],
    zrk: Field[type_ngptsw],
    zrk2: Field[type_ngptsw],
    zrp: Field[type_ngptsw],
    zrp1: Field[type_ngptsw],
    zrm1: Field[type_ngptsw],
    zrpp: Field[type_ngptsw],
    zrkg1: Field[type_ngptsw],
    zrkg3: Field[type_ngptsw],
    zrkg4: Field[type_ngptsw],
    zexp1: Field[type_ngptsw],
    zexm1: Field[type_ngptsw],
    zexp2: Field[type_ngptsw],
    zexm2: Field[type_ngptsw],
    zden1: Field[type_ngptsw],
    zexp3: Field[type_ngptsw],
    zexp4: Field[type_ngptsw],
    ze1r45: Field[type_ngptsw],
    ftind: Field[type_ngptsw],
    zsolar: Field[type_ngptsw],
    ztdbt0: Field[type_ngptsw],
    zr1: Field[type_ngptsw],
    zr2: Field[type_ngptsw],
    zr3: Field[type_ngptsw],
    zr4: Field[type_ngptsw],
    zr5: Field[type_ngptsw],
    zt1: Field[type_ngptsw],
    zt2: Field[type_ngptsw],
    zt3: Field[type_ngptsw],
    zf1: Field[type_ngptsw],
    zf2: Field[type_ngptsw],
    zrpp1: Field[type_ngptsw],
    zrupd: Field[type_ngptsw],
    zrupb: Field[type_ngptsw],
    ztdn: Field[type_ngptsw],
    zrdnd: Field[type_ngptsw],
    zb11: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    zb22: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    jb: FIELD_INT,
    ib: FIELD_INT,
    ibd: FIELD_INT,
    NGB: Field[type_ngptsw],
    idxsfc: Field[(DTYPE_FLT, (14,))],
    itind: FIELD_INT,
    fxupc: Field[type_nbdsw],
    fxdnc: Field[type_nbdsw],
    fxup0: Field[type_nbdsw],
    fxdn0: Field[type_nbdsw],
    ftoauc: FIELD_2D,
    ftoau0: FIELD_2D,
    ftoadc: FIELD_2D,
    fsfcuc: FIELD_2D,
    fsfcu0: FIELD_2D,
    fsfcdc: FIELD_2D,
    fsfcd0: FIELD_2D,
    sfbmc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdfc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfbm0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdf0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    suvbfc: FIELD_2D,
    suvbf0: FIELD_2D,
):
    from __externals__ import (
        ngptsw,
        bpade,
        oneminus,
        ftiny,
        zcrit,
        zsr3,
        od_lo,
        eps1,
        eps,
    )

    # -# Compute total sky optical parameters, layer reflectance and
    #    transmittance.
    #    - Set up toa direct beam and surface values (beam and diff)
    #    - Delta scaling for total-sky condition
    #    - General two-stream expressions for physparam::iswmode
    #    - Compute homogeneous reflectance and transmittance for
    #      conservative scattering and non-conservative scattering
    #    - Pre-delta-scaling clear and cloudy direct beam transmittance
    #    - Call swflux() to compute the upward and downward radiation fluxes

    with computation(BACKWARD):
        with interval(-1, None):
            if cf1 > eps:
                for m0 in range(ngptsw):
                    ztdbt0[0, 0, 0][m0] = 1.0
        with interval(0, -1):
            if cf1 > eps:
                for m in range(ngptsw):
                    jb = NGB[0, 0, 0][m] - 1
                    ib = jb + 1 - nblow
                    ibd = idxsfc[0, 0, 0][jb - 15] - 1

                    if cldfmc[0, 0, 1][m] > ftiny:
                        ztau0[0, 0, 0][m] = ztaus[0, 0, 0][m] + taucw[0, 0, 1][ib]
                        zssa0[0, 0, 0][m] = zssas[0, 0, 0][m] + ssacw[0, 0, 1][ib]
                        zasy0[0, 0, 0][m] = zasys[0, 0, 0][m] + asycw[0, 0, 1][ib]
                        zssaw[0, 0, 0][m] = min(
                            oneminus, zssa0[0, 0, 0][m] / ztau0[0, 0, 0][m]
                        )
                        zasyw[0, 0, 0][m] = zasy0[0, 0, 0][m] / max(
                            ftiny, zssa0[0, 0, 0][m]
                        )

                        #  --- ...  delta scaling for total-sky condition
                        za1[0, 0, 0][m] = zasyw[0, 0, 0][m] * zasyw[0, 0, 0][m]
                        za2[0, 0, 0][m] = zssaw[0, 0, 0][m] * za1[0, 0, 0][m]

                        ztau1[0, 0, 0][m] = (1.0 - za2[0, 0, 0][m]) * ztau0[0, 0, 0][m]
                        zssa1[0, 0, 0][m] = (zssaw[0, 0, 0][m] - za2[0, 0, 0][m]) / (
                            1.0 - za2[0, 0, 0][m]
                        )
                        zasy1[0, 0, 0][m] = zasyw[0, 0, 0][m] / (
                            1.0 + zasyw[0, 0, 0][m]
                        )
                        zasy3[0, 0, 0][m] = 0.75 * zasy1[0, 0, 0][m]

                        #  --- ...  general two-stream expressions
                        if iswmode == 1:
                            zgam1[0, 0, 0][m] = 1.75 - zssa1[0, 0, 0][m] * (
                                1.0 + zasy3[0, 0, 0][m]
                            )
                            zgam2[0, 0, 0][m] = -0.25 + zssa1[0, 0, 0][m] * (
                                1.0 - zasy3[0, 0, 0][m]
                            )
                            zgam3[0, 0, 0][m] = 0.5 - zasy3[0, 0, 0][m] * cosz
                        elif iswmode == 2:  # pifm
                            zgam1[0, 0, 0][m] = 2.0 - zssa1[0, 0, 0][m] * (
                                1.25 + zasy3[0, 0, 0][m]
                            )
                            zgam2[0, 0, 0][m] = (
                                0.75 * zssa1[0, 0, 0][m] * (1.0 - zasy1[0, 0, 0][m])
                            )
                            zgam3[0, 0, 0][m] = 0.5 - zasy3[0, 0, 0][m] * cosz
                        elif iswmode == 3:  # discrete ordinates
                            zgam1[0, 0, 0][m] = (
                                zsr3
                                * (2.0 - zssa1[0, 0, 0][m] * (1.0 + zasy1[0, 0, 0][m]))
                                * 0.5
                            )
                            zgam2[0, 0, 0][m] = (
                                zsr3
                                * zssa1[0, 0, 0][m]
                                * (1.0 - zasy1[0, 0, 0][m])
                                * 0.5
                            )
                            zgam3[0, 0, 0][m] = (
                                1.0 - zsr3 * zasy1[0, 0, 0][m] * cosz
                            ) * 0.5

                        zgam4[0, 0, 0][m] = 1.0 - zgam3[0, 0, 0][m]

                        #  --- ...  compute homogeneous reflectance and transmittance
                        if zssaw[0, 0, 0][m] >= zcrit:  # for conservative scattering
                            za1[0, 0, 0][m] = (
                                zgam1[0, 0, 0][m] * cosz - zgam3[0, 0, 0][m]
                            )
                            za2[0, 0, 0][m] = zgam1[0, 0, 0][m] * ztau1[0, 0, 0][m]

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1[0, 0, 0][m] = min(ztau1[0, 0, 0][m] * sntz, 500.0)
                            if zb1[0, 0, 0][m] <= od_lo:
                                zb2[0, 0, 0][m] = (
                                    1.0
                                    - zb1[0, 0, 0][m]
                                    + 0.5 * zb1[0, 0, 0][m] * zb1[0, 0, 0][m]
                                )
                            else:
                                ftind[0, 0, 0][m] = zb1[0, 0, 0][m] / (
                                    bpade + zb1[0, 0, 0][m]
                                )
                                itind = ftind[0, 0, 0][m] * ntbmx + 0.5
                                zb2[0, 0, 0][m] = exp_tbl[0, 0, 0][itind]

    with computation(BACKWARD), interval(1, None):
        if cf1 > eps:
            for m2 in range(ngptsw):
                if cldfmc[0, 0, 0][m2] > ftiny:
                    if zssaw[0, 0, -1][m2] >= zcrit:
                        #      ...  collimated beam
                        zrefb[0, 0, 0][m2] = max(
                            0.0,
                            min(
                                1.0,
                                (
                                    za2[0, 0, -1][m2]
                                    - za1[0, 0, -1][m2] * (1.0 - zb2[0, 0, -1][m2])
                                )
                                / (1.0 + za2[0, 0, -1][m2]),
                            ),
                        )
                        ztrab[0, 0, 0][m2] = max(
                            0.0, min(1.0, 1.0 - zrefb[0, 0, 0][m2])
                        )

                        #      ...  isotropic incidence
                        zrefd[0, 0, 0][m2] = max(
                            0.0, min(1.0, za2[0, 0, -1][m2] / (1.0 + za2[0, 0, -1][m2]))
                        )
                        ztrad[0, 0, 0][m2] = max(
                            0.0, min(1.0, 1.0 - zrefd[0, 0, 0][m2])
                        )

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m3 in range(ngptsw):
                if cldfmc[0, 0, 1][m3] > ftiny:
                    if zssaw[0, 0, 0][m3] < zcrit:  # for non-conservative scattering
                        za1[0, 0, 0][m3] = (
                            zgam1[0, 0, 0][m3] * zgam4[0, 0, 0][m3]
                            + zgam2[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                        )
                        za2[0, 0, 0][m3] = (
                            zgam1[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                            + zgam2[0, 0, 0][m3] * zgam4[0, 0, 0][m3]
                        )
                        zrk[0, 0, 0][m3] = sqrt(
                            (zgam1[0, 0, 0][m3] - zgam2[0, 0, 0][m3])
                            * (zgam1[0, 0, 0][m3] + zgam2[0, 0, 0][m3])
                        )
                        zrk2[0, 0, 0][m3] = 2.0 * zrk[0, 0, 0][m3]

                        zrp[0, 0, 0][m3] = zrk[0, 0, 0][m3] * cosz
                        zrp1[0, 0, 0][m3] = 1.0 + zrp[0, 0, 0][m3]
                        zrm1[0, 0, 0][m3] = 1.0 - zrp[0, 0, 0][m3]
                        zrpp1[0, 0, 0][m3] = 1.0 - zrp[0, 0, 0][m3] * zrp[0, 0, 0][m3]
                        tmp = max(flimit, abs(zrpp1[0, 0, 0][m3]))
                        zrpp[0, 0, 0][m3] = (
                            tmp if zrpp1[0, 0, 0][m3] >= 0 else -tmp
                        )  # avoid numerical singularity
                        zrkg1[0, 0, 0][m3] = zrk[0, 0, 0][m3] + zgam1[0, 0, 0][m3]
                        zrkg3[0, 0, 0][m3] = zrk[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                        zrkg4[0, 0, 0][m3] = zrk[0, 0, 0][m3] * zgam4[0, 0, 0][m3]

                        zr1[0, 0, 0][m3] = zrm1[0, 0, 0][m3] * (
                            za2[0, 0, 0][m3] + zrkg3[0, 0, 0][m3]
                        )
                        zr2[0, 0, 0][m3] = zrp1[0, 0, 0][m3] * (
                            za2[0, 0, 0][m3] - zrkg3[0, 0, 0][m3]
                        )
                        zr3[0, 0, 0][m3] = zrk2[0, 0, 0][m3] * (
                            zgam3[0, 0, 0][m3] - za2[0, 0, 0][m3] * cosz
                        )
                        zr4[0, 0, 0][m3] = zrpp[0, 0, 0][m3] * zrkg1[0, 0, 0][m3]
                        zr5[0, 0, 0][m3] = zrpp[0, 0, 0][m3] * (
                            zrk[0, 0, 0][m3] - zgam1[0, 0, 0][m3]
                        )

                        zt1[0, 0, 0][m3] = zrp1[0, 0, 0][m3] * (
                            za1[0, 0, 0][m3] + zrkg4[0, 0, 0][m3]
                        )
                        zt2[0, 0, 0][m3] = zrm1[0, 0, 0][m3] * (
                            za1[0, 0, 0][m3] - zrkg4[0, 0, 0][m3]
                        )
                        zt3[0, 0, 0][m3] = zrk2[0, 0, 0][m3] * (
                            zgam4[0, 0, 0][m3] + za1[0, 0, 0][m3] * cosz
                        )

                        #  --- ...  use exponential lookup table for transmittance, or expansion
                        #           of exponential for low optical depth

                        zb1[0, 0, 0][m3] = min(
                            zrk[0, 0, 0][m3] * ztau1[0, 0, 0][m3], 500.0
                        )
                        if zb1[0, 0, 0][m3] <= od_lo:
                            zexm1[0, 0, 0][m3] = (
                                1.0
                                - zb1[0, 0, 0][m3]
                                + 0.5 * zb1[0, 0, 0][m3] * zb1[0, 0, 0][m3]
                            )
                        else:
                            ftind[0, 0, 0][m3] = zb1[0, 0, 0][m3] / (
                                bpade + zb1[0, 0, 0][m3]
                            )
                            itind = ftind[0, 0, 0][m3] * ntbmx + 0.5
                            zexm1[0, 0, 0][m3] = exp_tbl[0, 0, 0][itind]

                        zexp1[0, 0, 0][m3] = 1.0 / zexm1[0, 0, 0][m3]

                        zb2[0, 0, 0][m3] = min(ztau1[0, 0, 0][m3] * sntz, 500.0)
                        if zb2[0, 0, 0][m3] <= od_lo:
                            zexm2[0, 0, 0][m3] = (
                                1.0
                                - zb2[0, 0, 0][m3]
                                + 0.5 * zb2[0, 0, 0][m3] * zb2[0, 0, 0][m3]
                            )
                        else:
                            ftind[0, 0, 0][m3] = zb2[0, 0, 0][m3] / (
                                bpade + zb2[0, 0, 0][m3]
                            )
                            itind = ftind[0, 0, 0][m3] * ntbmx + 0.5
                            zexm2[0, 0, 0][m3] = exp_tbl[0, 0, 0][itind]

                        zexp2[0, 0, 0][m3] = 1.0 / zexm2[0, 0, 0][m3]
                        ze1r45[0, 0, 0][m3] = (
                            zr4[0, 0, 0][m3] * zexp1[0, 0, 0][m3]
                            + zr5[0, 0, 0][m3] * zexm1[0, 0, 0][m3]
                        )

    with computation(BACKWARD), interval(1, None):
        if cf1 > eps:
            for mm3 in range(ngptsw):
                if cldfmc[0, 0, 0][mm3] > ftiny:
                    if zssaw[0, 0, -1][mm3] < zcrit:
                        #      ...  collimated beam
                        if (
                            ze1r45[0, 0, -1][mm3] >= -eps1
                            and ze1r45[0, 0, -1][mm3] <= eps1
                        ):
                            zrefb[0, 0, 0][mm3] = eps1
                            ztrab[0, 0, 0][mm3] = zexm2[0, 0, -1][mm3]
                        else:
                            zden1[0, 0, 0][mm3] = (
                                zssa1[0, 0, -1][mm3] / ze1r45[0, 0, -1][mm3]
                            )
                            zrefb[0, 0, 0][mm3] = max(
                                0.0,
                                min(
                                    1.0,
                                    (
                                        zr1[0, 0, -1][mm3] * zexp1[0, 0, -1][mm3]
                                        - zr2[0, 0, -1][mm3] * zexm1[0, 0, -1][mm3]
                                        - zr3[0, 0, -1][mm3] * zexm2[0, 0, -1][mm3]
                                    )
                                    * zden1[0, 0, 0][mm3],
                                ),
                            )
                            ztrab[0, 0, 0][mm3] = max(
                                0.0,
                                min(
                                    1.0,
                                    zexm2[0, 0, -1][mm3]
                                    * (
                                        1.0
                                        - (
                                            zt1[0, 0, -1][mm3] * zexp1[0, 0, -1][mm3]
                                            - zt2[0, 0, -1][mm3] * zexm1[0, 0, -1][mm3]
                                            - zt3[0, 0, -1][mm3] * zexp2[0, 0, -1][mm3]
                                        )
                                        * zden1[0, 0, 0][mm3]
                                    ),
                                ),
                            )

                        #      ...  diffuse beam
                        zden1[0, 0, 0][mm3] = zr4[0, 0, -1][mm3] / (
                            ze1r45[0, 0, -1][mm3] * zrkg1[0, 0, -1][mm3]
                        )
                        zrefd[0, 0, 0][mm3] = max(
                            0.0,
                            min(
                                1.0,
                                zgam2[0, 0, -1][mm3]
                                * (zexp1[0, 0, -1][mm3] - zexm1[0, 0, -1][mm3])
                                * zden1[0, 0, 0][mm3],
                            ),
                        )
                        ztrad[0, 0, 0][mm3] = max(
                            0.0, min(1.0, zrk2[0, 0, -1][mm3] * zden1[0, 0, 0][mm3])
                        )

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m4 in range(ngptsw):
                if cldfmc[0, 0, 1][m4] > ftiny:
                    #  --- ...  direct beam transmittance. use exponential lookup table
                    #           for transmittance, or expansion of exponential for low
                    #           optical depth

                    zr1[0, 0, 0][m4] = ztau1[0, 0, 0][m4] * sntz
                    if zr1[0, 0, 0][m4] <= od_lo:
                        zexp3[0, 0, 0][m4] = (
                            1.0
                            - zr1[0, 0, 0][m4]
                            + 0.5 * zr1[0, 0, 0][m4] * zr1[0, 0, 0][m4]
                        )
                    else:
                        ftind[0, 0, 0][m4] = zr1[0, 0, 0][m4] / (
                            bpade + zr1[0, 0, 0][m4]
                        )
                        itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][m4]))
                        zexp3[0, 0, 0][m4] = exp_tbl[0, 0, 0][itind]

                    ztdbt[0, 0, 0][m4] = zexp3[0, 0, 0][m4] * ztdbt[0, 0, 1][m4]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    #           (must use 'orig', unscaled cloud optical depth)

                    zr1[0, 0, 0][m4] = ztau0[0, 0, 0][m4] * sntz
                    if zr1[0, 0, 0][m4] <= od_lo:
                        zexp4[0, 0, 0][m4] = (
                            1.0
                            - zr1[0, 0, 0][m4]
                            + 0.5 * zr1[0, 0, 0][m4] * zr1[0, 0, 0][m4]
                        )
                    else:
                        ftind[0, 0, 0][m4] = zr1[0, 0, 0][m4] / (
                            bpade + zr1[0, 0, 0][m4]
                        )
                        itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][m4]))
                        zexp4[0, 0, 0][m4] = exp_tbl[0, 0, 0][itind]

                    ztdbt0[0, 0, 0][m4] = zexp4[0, 0, 0][m4] * ztdbt0[0, 0, 1][m4]

                else:
                    #  --- ...  direct beam transmittance
                    ztdbt[0, 0, 0][m4] = zldbt[0, 0, 1][m4] * ztdbt[0, 0, 1][m4]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    ztdbt0[0, 0, 0][m4] = zldbt0[0, 0, 0][m4] * ztdbt0[0, 0, 1][m4]

    # Need to incorporate the vrtqdr subroutine here, since I don't know
    # of a way to specify the interval within a gtscript function. Could be pulled
    # out later

    with computation(FORWARD):
        with interval(0, 1):
            if cf1 > eps:
                for m5 in range(ngptsw):
                    # Link lowest layer with surface.
                    zrupb[0, 0, 0][m5] = zrefb[0, 0, 0][m5]  # direct beam
                    zrupd[0, 0, 0][m5] = zrefd[0, 0, 0][m5]  # diffused

        with interval(1, None):
            if cf1 > eps:
                for m6 in range(ngptsw):
                    if cldfmc[0, 0, 0][m6] > ftiny:
                        zldbt[0, 0, 0][m6] = zexp3[0, 0, -1][m6]

                    zden1[0, 0, 0][m6] = 1.0 / (
                        1.0 - zrupd[0, 0, -1][m6] * zrefd[0, 0, 0][m6]
                    )

                    zrupb[0, 0, 0][m6] = (
                        zrefb[0, 0, 0][m6]
                        + (
                            ztrad[0, 0, 0][m6]
                            * (
                                (ztrab[0, 0, 0][m6] - zldbt[0, 0, 0][m6])
                                * zrupd[0, 0, -1][m6]
                                + zldbt[0, 0, 0][m6] * zrupb[0, 0, -1][m6]
                            )
                        )
                        * zden1[0, 0, 0][m6]
                    )
                    zrupd[0, 0, 0][m6] = (
                        zrefd[0, 0, 0][m6]
                        + ztrad[0, 0, 0][m6]
                        * ztrad[0, 0, 0][m6]
                        * zrupd[0, 0, -1][m6]
                        * zden1[0, 0, 0][m6]
                    )

                    ztdn[0, 0, 0][m6] = 0.0
                    zrdnd[0, 0, 0][m6] = 0.0

    with computation(PARALLEL):
        with interval(-2, -1):
            if cf1 > eps:
                for m7 in range(ngptsw):
                    ztdn[0, 0, 0][m7] = ztrab[0, 0, 1][m7]
                    zrdnd[0, 0, 0][m7] = zrefd[0, 0, 1][m7]
        with interval(-1, None):
            if cf1 > eps:
                for m8 in range(ngptsw):
                    ztdn[0, 0, 0][m8] = 1.0
                    zrdnd[0, 0, 0][m8] = 0.0

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m9 in range(ngptsw):
                zden1[0, 0, 0][m9] = 1.0 / (
                    1.0 - zrefd[0, 0, 1][m9] * zrdnd[0, 0, 1][m9]
                )
                ztdn[0, 0, 0][m9] = (
                    ztdbt[0, 0, 1][m9] * ztrab[0, 0, 1][m9]
                    + (
                        ztrad[0, 0, 1][m9]
                        * (
                            (ztdn[0, 0, 1][m9] - ztdbt[0, 0, 1][m9])
                            + ztdbt[0, 0, 1][m9]
                            * zrefb[0, 0, 1][m9]
                            * zrdnd[0, 0, 1][m9]
                        )
                    )
                    * zden1[0, 0, 0][m9]
                )
                zrdnd[0, 0, 0][m9] = (
                    zrefd[0, 0, 1][m9]
                    + ztrad[0, 0, 1][m9]
                    * ztrad[0, 0, 1][m9]
                    * zrdnd[0, 0, 1][m9]
                    * zden1[0, 0, 0][m9]
                )

    # Up and down-welling fluxes at levels.
    with computation(FORWARD), interval(...):
        #  -# Process and save outputs.
        # --- ...  surface downward beam/diffused flux components
        if cf1 > eps:
            for m10 in range(ngptsw):
                jb = NGB[0, 0, 0][m10] - 1
                ib = jb + 1 - nblow

                zden1[0, 0, 0][m10] = 1.0 / (
                    1.0 - zrdnd[0, 0, 0][m10] * zrupd[0, 0, 0][m10]
                )
                zfu[0, 0, 0][m10] = (
                    ztdbt[0, 0, 0][m10] * zrupb[0, 0, 0][m10]
                    + (ztdn[0, 0, 0][m10] - ztdbt[0, 0, 0][m10]) * zrupd[0, 0, 0][m10]
                ) * zden1[0, 0, 0][m10]
                zfd[0, 0, 0][m10] = (
                    ztdbt[0, 0, 0][m10]
                    + (
                        ztdn[0, 0, 0][m10]
                        - ztdbt[0, 0, 0][m10]
                        + ztdbt[0, 0, 0][m10]
                        * zrupb[0, 0, 0][m10]
                        * zrdnd[0, 0, 0][m10]
                    )
                    * zden1[0, 0, 0][m10]
                )

                fxupc[0, 0, 0][ib] = (
                    fxupc[0, 0, 0][ib] + zsolar[0, 0, 0][m10] * zfu[0, 0, 0][m10]
                )
                fxdnc[0, 0, 0][ib] = (
                    fxdnc[0, 0, 0][ib] + zsolar[0, 0, 0][m10] * zfd[0, 0, 0][m10]
                )

    # -# Process and save outputs.
    with computation(FORWARD), interval(0, 1):
        if cf1 > eps:
            for m11 in range(ngptsw):
                jb = NGB[0, 0, 0][m11] - 1
                ib = jb + 1 - nblow
                ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

                # --- ...  surface downward beam/diffused flux components
                zb11[0, 0][m11] = zsolar[0, 0, 0][m11] * ztdbt0[0, 0, 0][m11]
                zb22[0, 0][m11] = zsolar[0, 0, 0][m11] * (
                    zfd[0, 0, 0][m11] - ztdbt0[0, 0, 0][m11]
                )

                if ibd != -1:
                    sfbmc[0, 0][ibd] = sfbmc[0, 0][ibd] + zb11[0, 0][m11]
                    sfdfc[0, 0][ibd] = sfdfc[0, 0][ibd] + zb22[0, 0][m11]
                else:
                    zf1[0, 0, 0][m11] = 0.5 * zb11[0, 0][m11]
                    zf2[0, 0, 0][m11] = 0.5 * zb22[0, 0][m11]
                    sfbmc[0, 0][0] = sfbmc[0, 0][0] + zf1[0, 0, 0][m11]
                    sfdfc[0, 0][0] = sfdfc[0, 0][0] + zf2[0, 0, 0][m11]
                    sfbmc[0, 0][1] = sfbmc[0, 0][1] + zf1[0, 0, 0][m11]
                    sfdfc[0, 0][1] = sfdfc[0, 0][1] + zf2[0, 0, 0][m11]

    with computation(FORWARD):
        with interval(0, 1):
            for b in range(nbdsw):
                fsfcu0 = fsfcu0 + fxup0[0, 0, 0][b]
                fsfcd0 = fsfcd0 + fxdn0[0, 0, 0][b]

            # --- ...  uv-b surface downward flux
            suvbf0 = fxdn0[0, 0, 0][ibd0]
        with interval(-1, None):
            for bb in range(nbdsw):
                ftoadc = ftoadc + fxdn0[0, 0, 0][bb]
                ftoau0 = ftoau0 + fxup0[0, 0, 0][bb]

    with computation(PARALLEL), interval(...):
        if cf1 <= eps:  # clear column
            for b2 in range(nbdsw):
                fxupc[0, 0, 0][b2] = fxup0[0, 0, 0][b2]
                fxdnc[0, 0, 0][b2] = fxdn0[0, 0, 0][b2]

    with computation(FORWARD):
        with interval(0, 1):
            if cf1 <= eps:
                ftoauc = ftoau0
                fsfcuc = fsfcu0
                fsfcdc = fsfcd0

                # --- ...  surface downward beam/diffused flux components
                sfbmc[0, 0][0] = sfbm0[0, 0][0]
                sfdfc[0, 0][0] = sfdf0[0, 0][0]
                sfbmc[0, 0][1] = sfbm0[0, 0][1]
                sfdfc[0, 0][1] = sfdf0[0, 0][1]

                # --- ...  uv-b surface downward flux
                suvbfc = suvbf0
            else:  # cloudy column, compute total-sky fluxes
                for b3 in range(nbdsw):
                    fsfcuc = fsfcuc + fxupc[0, 0, 0][b3]
                    fsfcdc = fsfcdc + fxdnc[0, 0, 0][b3]

                # --- ...  uv-b surface downward flux
                suvbfc = fxdnc[0, 0, 0][ibd0]

        with interval(-1, None):
            if cf1 > eps:  # cloudy column, compute total-sky fluxes
                for b4 in range(nbdsw):
                    ftoauc = ftoauc + fxupc[0, 0, 0][b4]


end = time.time()


spcvrtm_clearsky(
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
    indict_gt4py["exp_tbl"],
    locdict_gt4py["ztaus"],
    locdict_gt4py["zssas"],
    locdict_gt4py["zasys"],
    locdict_gt4py["zldbt0"],
    locdict_gt4py["zrefb"],
    locdict_gt4py["zrefd"],
    locdict_gt4py["ztrab"],
    locdict_gt4py["ztrad"],
    locdict_gt4py["ztdbt"],
    locdict_gt4py["zldbt"],
    locdict_gt4py["zfu"],
    locdict_gt4py["zfd"],
    locdict_gt4py["ztau1"],
    locdict_gt4py["zssa1"],
    locdict_gt4py["zasy1"],
    locdict_gt4py["ztau0"],
    locdict_gt4py["zssa0"],
    locdict_gt4py["zasy0"],
    locdict_gt4py["zasy3"],
    locdict_gt4py["zssaw"],
    locdict_gt4py["zasyw"],
    locdict_gt4py["zgam1"],
    locdict_gt4py["zgam2"],
    locdict_gt4py["zgam3"],
    locdict_gt4py["zgam4"],
    locdict_gt4py["za1"],
    locdict_gt4py["za2"],
    locdict_gt4py["zb1"],
    locdict_gt4py["zb2"],
    locdict_gt4py["zrk"],
    locdict_gt4py["zrk2"],
    locdict_gt4py["zrp"],
    locdict_gt4py["zrp1"],
    locdict_gt4py["zrm1"],
    locdict_gt4py["zrpp"],
    locdict_gt4py["zrkg1"],
    locdict_gt4py["zrkg3"],
    locdict_gt4py["zrkg4"],
    locdict_gt4py["zexp1"],
    locdict_gt4py["zexm1"],
    locdict_gt4py["zexp2"],
    locdict_gt4py["zexm2"],
    locdict_gt4py["zden1"],
    locdict_gt4py["zexp3"],
    locdict_gt4py["zexp4"],
    locdict_gt4py["ze1r45"],
    locdict_gt4py["ftind"],
    locdict_gt4py["zsolar"],
    locdict_gt4py["ztdbt0"],
    locdict_gt4py["zr1"],
    locdict_gt4py["zr2"],
    locdict_gt4py["zr3"],
    locdict_gt4py["zr4"],
    locdict_gt4py["zr5"],
    locdict_gt4py["zt1"],
    locdict_gt4py["zt2"],
    locdict_gt4py["zt3"],
    locdict_gt4py["zf1"],
    locdict_gt4py["zf2"],
    locdict_gt4py["zrpp1"],
    locdict_gt4py["zrupd"],
    locdict_gt4py["zrupb"],
    locdict_gt4py["ztdn"],
    locdict_gt4py["zrdnd"],
    locdict_gt4py["zb11"],
    locdict_gt4py["zb22"],
    locdict_gt4py["jb"],
    locdict_gt4py["ib"],
    locdict_gt4py["ibd"],
    locdict_gt4py["NGB"],
    locdict_gt4py["idxsfc"],
    locdict_gt4py["itind"],
    outdict_gt4py["fxupc"],
    outdict_gt4py["fxdnc"],
    outdict_gt4py["fxup0"],
    outdict_gt4py["fxdn0"],
    outdict_gt4py["ftoauc"],
    outdict_gt4py["ftoau0"],
    outdict_gt4py["ftoadc"],
    outdict_gt4py["fsfcuc"],
    outdict_gt4py["fsfcu0"],
    outdict_gt4py["fsfcdc"],
    outdict_gt4py["fsfcd0"],
    outdict_gt4py["sfbmc"],
    outdict_gt4py["sfdfc"],
    outdict_gt4py["sfbm0"],
    outdict_gt4py["sfdf0"],
    outdict_gt4py["suvbfc"],
    outdict_gt4py["suvbf0"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

spcvrtm_allsky(
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
    indict_gt4py["exp_tbl"],
    locdict_gt4py["ztaus"],
    locdict_gt4py["zssas"],
    locdict_gt4py["zasys"],
    locdict_gt4py["zldbt0"],
    locdict_gt4py["zrefb"],
    locdict_gt4py["zrefd"],
    locdict_gt4py["ztrab"],
    locdict_gt4py["ztrad"],
    locdict_gt4py["ztdbt"],
    locdict_gt4py["zldbt"],
    locdict_gt4py["zfu"],
    locdict_gt4py["zfd"],
    locdict_gt4py["ztau1"],
    locdict_gt4py["zssa1"],
    locdict_gt4py["zasy1"],
    locdict_gt4py["ztau0"],
    locdict_gt4py["zssa0"],
    locdict_gt4py["zasy0"],
    locdict_gt4py["zasy3"],
    locdict_gt4py["zssaw"],
    locdict_gt4py["zasyw"],
    locdict_gt4py["zgam1"],
    locdict_gt4py["zgam2"],
    locdict_gt4py["zgam3"],
    locdict_gt4py["zgam4"],
    locdict_gt4py["za1"],
    locdict_gt4py["za2"],
    locdict_gt4py["zb1"],
    locdict_gt4py["zb2"],
    locdict_gt4py["zrk"],
    locdict_gt4py["zrk2"],
    locdict_gt4py["zrp"],
    locdict_gt4py["zrp1"],
    locdict_gt4py["zrm1"],
    locdict_gt4py["zrpp"],
    locdict_gt4py["zrkg1"],
    locdict_gt4py["zrkg3"],
    locdict_gt4py["zrkg4"],
    locdict_gt4py["zexp1"],
    locdict_gt4py["zexm1"],
    locdict_gt4py["zexp2"],
    locdict_gt4py["zexm2"],
    locdict_gt4py["zden1"],
    locdict_gt4py["zexp3"],
    locdict_gt4py["zexp4"],
    locdict_gt4py["ze1r45"],
    locdict_gt4py["ftind"],
    locdict_gt4py["zsolar"],
    locdict_gt4py["ztdbt0"],
    locdict_gt4py["zr1"],
    locdict_gt4py["zr2"],
    locdict_gt4py["zr3"],
    locdict_gt4py["zr4"],
    locdict_gt4py["zr5"],
    locdict_gt4py["zt1"],
    locdict_gt4py["zt2"],
    locdict_gt4py["zt3"],
    locdict_gt4py["zf1"],
    locdict_gt4py["zf2"],
    locdict_gt4py["zrpp1"],
    locdict_gt4py["zrupd"],
    locdict_gt4py["zrupb"],
    locdict_gt4py["ztdn"],
    locdict_gt4py["zrdnd"],
    locdict_gt4py["zb11"],
    locdict_gt4py["zb22"],
    locdict_gt4py["jb"],
    locdict_gt4py["ib"],
    locdict_gt4py["ibd"],
    locdict_gt4py["NGB"],
    locdict_gt4py["idxsfc"],
    locdict_gt4py["itind"],
    outdict_gt4py["fxupc"],
    outdict_gt4py["fxdnc"],
    outdict_gt4py["fxup0"],
    outdict_gt4py["fxdn0"],
    outdict_gt4py["ftoauc"],
    outdict_gt4py["ftoau0"],
    outdict_gt4py["ftoadc"],
    outdict_gt4py["fsfcuc"],
    outdict_gt4py["fsfcu0"],
    outdict_gt4py["fsfcdc"],
    outdict_gt4py["fsfcd0"],
    outdict_gt4py["sfbmc"],
    outdict_gt4py["sfdfc"],
    outdict_gt4py["sfbm0"],
    outdict_gt4py["sfdf0"],
    outdict_gt4py["suvbfc"],
    outdict_gt4py["suvbf0"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

print(f"Elapsed time = {end-start}")

outdict_np = dict()
valdict = dict()
for var in outvars:
    outdict_np[var] = outdict_gt4py[var][:, ...].view(np.ndarray).squeeze()
    valdict[var] = serializer.read(
        var, serializer.savepoint["swrad-spcvrtm-output-000000"]
    )

compare_data(outdict_np, valdict)
