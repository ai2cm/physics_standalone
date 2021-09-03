import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    mod,
)

sys.path.insert(0, "..")

from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *
from phys_const import con_g, con_avgd, con_amd, con_amw, amdw, amdo3
from radphysparam import iswrgas, iswcliq, iovrsw

from radsw.radsw_param import ftiny, oneminus, s0

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/data/SW"
ddir2 = "../../fortran/radsw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank1")
serializer2 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Serialized_rank1")

invars = [
    "plyr",
    "plvl",
    "tlyr",
    "tlvl",
    "qlyr",
    "olyr",
    "gasvmr",
    "clouds",
    "faersw",
    "sfcalb",
    "dz",
    "delp",
    "de_lgth",
    "coszen",
    "solcon",
    "nday",
    "idxday",
    "im",
    "lmk",
    "lmp",
    "lprnt",
]

locvars = [
    "cosz1",
    "sntz1",
    "ssolar",
    "albbm",
    "albdf",
    "pavel",
    "tavel",
    "h2ovmr",
    "o3vmr",
    "coldry",
    "temcol",
    "colamt",
    "colmol",
    "tauae",
    "ssaae",
    "asyae",
    "cfrac",
    "cliqp",
    "reliq",
    "cicep",
    "reice",
    "cdat1",
    "cdat2",
    "cdat3",
    "cdat4",
    "zcf0",
    "zcf1",
]

temvars = ["tem0", "tem1", "tem2"]

indict = dict()

for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-in-000000"])
    if var in ["plyr", "tlyr", "qlyr", "olyr", "dz", "delp"]:
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
    elif var in ["plvl", "tlvl"]:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    elif var in ["gasvmr", "clouds"]:
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var == "sfcalb":
        indict[var] = np.tile(tmp[:, None, None, :], (1, 1, nlp1, 1))
    elif var == "faersw":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :, :], (1, 1, 1, 1, 1))
    elif var in ["coszen", "de_lgth"]:
        indict[var] = np.tile(tmp[:, None], (1, 1))
    elif var == "idxday":
        tmp2 = np.zeros(npts, dtype=bool)
        for n in range(npts):
            if tmp[n] > 1 and tmp[n] < 25:
                tmp2[tmp[n] - 1] = True

        indict[var] = np.tile(tmp2[:, None], (1, 1))
    else:
        indict[var] = tmp[0]

indict_gt4py = dict()

for var in invars:
    if var in ["plyr", "tlyr", "qlyr", "olyr", "dz", "delp", "plvl", "tlvl"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    elif var == "gasvmr":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_10
        )
    elif var == "clouds":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_9
        )
    elif var == "sfcalb":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (4,))
        )
    elif var == "faersw":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (nbdsw, 3))
        )
    elif var in ["coszen", "de_lgth"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, DTYPE_FLT
        )
    elif var == "idxday":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, DTYPE_BOOL
        )
    elif var == "solcon":
        indict_gt4py[var] = indict["solcon"]

locdict_gt4py = dict()
for var in locvars:
    if var in ["tauae", "ssaae", "asyae", "taucw", "ssacw", "asycw"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    elif var in ["zcf0", "zcf1"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
    elif var in ["albbm", "albdf"]:
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_FLT, (2,))
        )
    elif var == "colamt":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_maxgas)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

for var in temvars:
    locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "s0": s0,
        "con_g": con_g,
        "con_avgd": con_avgd,
        "con_amd": con_amd,
        "con_amw": con_amw,
        "amdw": amdw,
        "amdo3": amdo3,
        "iswrgas": iswrgas,
        "iswcliq": iswcliq,
        "iovrsw": iovrsw,
        "nbdsw": nbdsw,
        "ftiny": ftiny,
        "oneminus": oneminus,
    },
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[type_10],
    clouds: Field[type_9],
    aerosols: Field[(DTYPE_FLT, (nbdsw, 3))],
    sfcalb: Field[(DTYPE_FLT, (4,))],
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_2D,
    cosz: FIELD_2D,
    idxday: FIELD_2DBOOL,
    solcon: float,
    cosz1: FIELD_FLT,
    sntz1: FIELD_FLT,
    ssolar: FIELD_FLT,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    tem1: FIELD_FLT,
    tem2: FIELD_FLT,
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    tem0: FIELD_FLT,
    coldry: FIELD_FLT,
    temcol: FIELD_FLT,
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    zcf0: FIELD_2D,
    zcf1: FIELD_2D,
):
    from __externals__ import (
        s0,
        con_g,
        con_avgd,
        con_amd,
        con_amw,
        amdw,
        amdo3,
        iswrgas,
        iswcliq,
        iovrsw,
        nbdsw,
        ftiny,
        oneminus,
    )

    with computation(FORWARD), interval(0, 1):

        s0fac = solcon / s0

        if idxday:
            cosz1 = cosz
            sntz1 = 1.0 / cosz
            ssolar = s0fac * cosz

            # Prepare surface albedo: bm,df - dir,dif; 1,2 - nir,uvv.
            albbm[0, 0, 0][0] = sfcalb[0, 0, 0][0]
            albdf[0, 0, 0][0] = sfcalb[0, 0, 0][1]
            albbm[0, 0, 0][1] = sfcalb[0, 0, 0][2]
            albdf[0, 0, 0][1] = sfcalb[0, 0, 0][3]

            zcf0 = 1.0
            zcf1 = 1.0

    with computation(FORWARD), interval(1, None):
        if idxday:
            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            pavel = plyr
            tavel = tlyr

            h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
            o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delpin / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
            colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
            colamt[0, 0, 0][2] = max(0.0, coldry * o3vmr)  # o3
            colmol = coldry + colamt[0, 0, 0][0]

            #  --- ...  set up gas column amount, convert from volume mixing ratio
            #           to molec/cm2 based on coldry (scaled to 1.0e-20)

            if iswrgas > 0:
                colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                colamt[0, 0, 0][5] = max(temcol, coldry * gasvmr[0, 0, 0][3])  # o2
            else:
                colamt[0, 0, 0][3] = temcol  # n2o
                colamt[0, 0, 0][4] = temcol  # ch4
                colamt[0, 0, 0][5] = temcol

            #  --- ...  set aerosol optical properties
            for ib in range(nbdsw):
                tauae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 0]
                ssaae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 1]
                asyae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 2]

            if iswcliq > 0:  # use prognostic cloud method
                cfrac = clouds[0, 0, 0][0]  # cloud fraction
                cliqp = clouds[0, 0, 0][1]  # cloud liq path
                reliq = clouds[0, 0, 0][2]  # liq partical effctive radius
                cicep = clouds[0, 0, 0][3]  # cloud ice path
                reice = clouds[0, 0, 0][4]  # ice partical effctive radius
                cdat1 = clouds[0, 0, 0][5]  # cloud rain drop path
                cdat2 = clouds[0, 0, 0][6]  # rain partical effctive radius
                cdat3 = clouds[0, 0, 0][7]  # cloud snow path
                cdat4 = clouds[0, 0, 0][8]  # snow partical effctive radius
            else:  # use diagnostic cloud method
                cfrac = clouds[0, 0, 0][0]  # cloud fraction
                cdat1 = clouds[0, 0, 0][1]  # cloud optical depth
                cdat2 = clouds[0, 0, 0][2]  # cloud single scattering albedo
                cdat3 = clouds[0, 0, 0][3]  # cloud asymmetry factor

            # -# Compute fractions of clear sky view:
            #    - random overlapping
            #    - max/ran overlapping
            #    - maximum overlapping

            if iovrsw == 0:
                zcf0 = zcf0 * (1.0 - cfrac)
            elif iovrsw == 1:
                if cfrac > ftiny:  # cloudy layer
                    zcf1 = min(zcf1, 1.0 - cfrac)
                elif zcf1 < 1.0:  # clear layer
                    zcf0 = zcf0 * zcf1
                    zcf1 = 1.0

            elif iovrsw >= 2:
                zcf0 = min(zcf0, 1.0 - cfrac)  # used only as clear/cloudy indicator

    with computation(FORWARD), interval(0, 1):
        if idxday:
            if iovrsw == 1:
                zcf0 = zcf0 * zcf1

            if zcf0 <= ftiny:
                zcf0 = 0.0
            if zcf0 > oneminus:
                zcf0 = 1.0

            zcf1 = 1.0 - zcf0


firstloop(
    indict_gt4py["plyr"],
    indict_gt4py["plvl"],
    indict_gt4py["tlyr"],
    indict_gt4py["tlvl"],
    indict_gt4py["qlyr"],
    indict_gt4py["olyr"],
    indict_gt4py["gasvmr"],
    indict_gt4py["clouds"],
    indict_gt4py["faersw"],
    indict_gt4py["sfcalb"],
    indict_gt4py["dz"],
    indict_gt4py["delp"],
    indict_gt4py["de_lgth"],
    indict_gt4py["coszen"],
    indict_gt4py["idxday"],
    indict_gt4py["solcon"],
    locdict_gt4py["cosz1"],
    locdict_gt4py["sntz1"],
    locdict_gt4py["ssolar"],
    locdict_gt4py["albbm"],
    locdict_gt4py["albdf"],
    locdict_gt4py["tem1"],
    locdict_gt4py["tem2"],
    locdict_gt4py["pavel"],
    locdict_gt4py["tavel"],
    locdict_gt4py["h2ovmr"],
    locdict_gt4py["o3vmr"],
    locdict_gt4py["tem0"],
    locdict_gt4py["coldry"],
    locdict_gt4py["temcol"],
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["tauae"],
    locdict_gt4py["ssaae"],
    locdict_gt4py["asyae"],
    locdict_gt4py["cfrac"],
    locdict_gt4py["cliqp"],
    locdict_gt4py["reliq"],
    locdict_gt4py["cicep"],
    locdict_gt4py["reice"],
    locdict_gt4py["cdat1"],
    locdict_gt4py["cdat2"],
    locdict_gt4py["cdat3"],
    locdict_gt4py["cdat4"],
    locdict_gt4py["zcf0"],
    locdict_gt4py["zcf1"],
)


valdict = dict()
outdict_np = dict()

for var in locvars:
    valdict[var] = serializer2.read(
        var, serializer2.savepoint["swrad-firstloop-output-000000"]
    )
    if var in ["cosz1", "sntz1", "ssolar", "albbm", "albdf", "tem0", "tem1", "tem2"]:
        outdict_np[var] = locdict_gt4py[var][:, :, 0, ...].view(np.ndarray).squeeze()
    elif var in ["zcf1", "zcf0"]:
        outdict_np[var] = locdict_gt4py[var].view(np.ndarray).squeeze()
    else:
        outdict_np[var] = locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()

compare_data(valdict, outdict_np)
