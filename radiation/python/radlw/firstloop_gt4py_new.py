import gt4py
import os
import sys
import numpy as np
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval, stencil, exp

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from radlw_param import a0, a1, a2, nbands, eps
from util import (
    view_gt4py_storage,
    compare_data,
    create_storage_from_array,
    create_storage_zeros,
    create_storage_ones,
)
from config import *

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3

semiss0_np = np.ones(nbands)

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data/LW"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")

savepoints = serializer.savepoint_list()

invars = [
    "plyr",
    "plvl",
    "tlyr",
    "tlvl",
    "qlyr",
    "olyr",
    "gasvmr",
    "clouds",
    "icsdlw",
    "faerlw",
    "semis",
    "tsfg",
    "dz",
    "delp",
    "de_lgth",
    "im",
    "lmk",
    "lmp",
    "lprnt",
]

nlay_vars = ["plyr", "tlyr", "qlyr", "olyr", "dz", "delp"]
nlp1_vars = ["plvl", "tlvl"]

print("Loading input vars...")
indict = dict()

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-in-000000"])
    if var in ["semis", "icsdlw", "tsfg", "de_lgth"]:
        indict[var] = np.tile(tmp[:, None, None], (1, 1, nlp1))
    elif var == "faerlw":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :, :], (1, 1, 1, 1, 1))
    elif var == "gasvmr" or var == "clouds":
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var in ["plyr", "tlyr", "qlyr", "olyr", "dz", "delp"]:
        tmp2 = np.insert(tmp, 0, 0, axis=1)
        indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
        print(f"{var} = {indict[var].shape}")
    elif var in ["plvl", "tlvl"]:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    else:
        indict[var] = tmp[0]

indict_gt4py = dict()

for var in invars:
    if var == "faerlw":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (nbands, 3))
        )
    elif var == "gasvmr" or var == "clouds":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, (DTYPE_FLT, (indict[var].shape[3],))
        )
    elif var == "icsdlw":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_INT
        )
    elif indict[var].size > 1:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    else:
        indict_gt4py[var] = indict[var]

locvars = [
    "pavel",
    "tavel",
    "delp",
    "colbrd",
    "h2ovmr",
    "o3vmr",
    "coldry",
    "colamt",
    "temcol",
    "tauaer",
    "taucld",
    "semiss0",
    "semiss",
    "tz",
    "dz",
    "wx",
    "cldfrc",
    "clwp",
    "ciwp",
    "relw",
    "reiw",
    "cda1",
    "cda2",
    "cda3",
    "cda4",
    "pwvcm",
    "secdiff",
    "tem00",
    "tem11",
    "tem22",
    "summol",
]

locdict_gt4py = dict()

for var in locvars:
    if var == "colamt":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_maxgas)
    elif var == "wx":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_maxxsec)
    elif var == "pwvcm" or var == "tem00":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
    elif var == "tauaer" or var == "taucld":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var == "semiss0":
        locdict_gt4py[var] = create_storage_ones(backend, shape_nlp1, type_nbands)
    elif var == "semiss" or var == "secdiff":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var == "tz":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    elif var == "cldfrc":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

locdict_gt4py["A0"] = create_storage_from_array(a0, backend, shape_nlp1, type_nbands)
locdict_gt4py["A1"] = create_storage_from_array(a1, backend, shape_nlp1, type_nbands)
locdict_gt4py["A2"] = create_storage_from_array(a2, backend, shape_nlp1, type_nbands)

print("Done")


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={"nbands": nbands, "ilwcliq": ilwcliq, "ilwrgas": ilwrgas},
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[(DTYPE_FLT, (10,))],
    clouds: Field[(DTYPE_FLT, (9,))],
    icseed: FIELD_INT,
    aerosols: Field[(DTYPE_FLT, (nbands, 3))],
    sfemis: FIELD_FLT,
    sfgtmp: FIELD_FLT,
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_FLT,
    cldfrc: FIELD_FLT,
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    delp: FIELD_FLT,
    dz: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    tauaer: Field[type_nbands],
    semiss0: Field[type_nbands],
    semiss: Field[type_nbands],
    tem11: FIELD_FLT,
    tem22: FIELD_FLT,
    tem00: FIELD_2D,
    summol: FIELD_FLT,
    pwvcm: FIELD_2D,
    clwp: FIELD_FLT,
    relw: FIELD_FLT,
    ciwp: FIELD_FLT,
    reiw: FIELD_FLT,
    cda1: FIELD_FLT,
    cda2: FIELD_FLT,
    cda3: FIELD_FLT,
    cda4: FIELD_FLT,
    secdiff: Field[type_nbands],
    a0: Field[type_nbands],
    a1: Field[type_nbands],
    a2: Field[type_nbands],
):
    from __externals__ import nbands, ilwcliq, ilwrgas

    with computation(PARALLEL):
        with interval(1, None):
            if sfemis > eps and sfemis <= 1.0:
                for j in range(nbands):
                    semiss[0, 0, 0][j] = sfemis
            else:
                for j2 in range(nbands):
                    semiss[0, 0, 0][j2] = semiss0[0, 0, 0][j2]

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

    with computation(PARALLEL):
        with interval(1, None):
            pavel = plyr
            delp = delpin
            tavel = tlyr
            dz = dzlyr

            h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
            o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delp / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
            colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
            colamt[0, 0, 0][2] = max(temcol, coldry * o3vmr)  # o3

            if ilwrgas > 0:
                colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                colamt[0, 0, 0][5] = max(0.0, coldry * gasvmr[0, 0, 0][3])  # o2
                colamt[0, 0, 0][6] = max(0.0, coldry * gasvmr[0, 0, 0][4])  # co

                wx[0, 0, 0][0] = max(0.0, coldry * gasvmr[0, 0, 0][8])  # ccl4
                wx[0, 0, 0][1] = max(0.0, coldry * gasvmr[0, 0, 0][5])  # cf11
                wx[0, 0, 0][2] = max(0.0, coldry * gasvmr[0, 0, 0][6])  # cf12
                wx[0, 0, 0][3] = max(0.0, coldry * gasvmr[0, 0, 0][7])  # cf22

            else:
                colamt[0, 0, 0][3] = 0.0  # n2o
                colamt[0, 0, 0][4] = 0.0  # ch4
                colamt[0, 0, 0][5] = 0.0  # o2
                colamt[0, 0, 0][6] = 0.0  # co

                wx[0, 0, 0][0] = 0.0
                wx[0, 0, 0][1] = 0.0
                wx[0, 0, 0][2] = 0.0
                wx[0, 0, 0][3] = 0.0

            for j3 in range(nbands):
                tauaer[0, 0, 0][j3] = aerosols[0, 0, 0][j3, 0] * (
                    1.0 - aerosols[0, 0, 0][j3, 1]
                )

    with computation(PARALLEL):
        with interval(1, None):
            cldfrc = clouds[0, 0, 0][0]

    with computation(PARALLEL):
        with interval(1, None):
            # Workaround for variables first referenced inside if statements
            # Can be removed at next gt4py release
            clwp = clwp
            relw = relw
            ciwp = ciwp
            reiw = reiw
            cda1 = cda1
            cda2 = cda2
            cda3 = cda3
            cda4 = cda4
            clouds = clouds
            if ilwcliq > 0:
                clwp = clouds[0, 0, 0][1]
                relw = clouds[0, 0, 0][2]
                ciwp = clouds[0, 0, 0][3]
                reiw = clouds[0, 0, 0][4]
                cda1 = clouds[0, 0, 0][5]
                cda2 = clouds[0, 0, 0][6]
                cda3 = clouds[0, 0, 0][7]
                cda4 = clouds[0, 0, 0][8]
            else:
                cda1 = clouds[0, 0, 0][1]

    with computation(FORWARD):
        with interval(0, 1):
            cldfrc = 1.0
        with interval(1, 2):
            tem11 = coldry[0, 0, 0] + colamt[0, 0, 0][0]
            tem22 = colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(2, None):
            #  --- ...  compute precipitable water vapor for diffusivity angle adjustments
            tem11 = tem11[0, 0, -1] + coldry + colamt[0, 0, 0][0]
            tem22 = tem22[0, 0, -1] + colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(-1, None):
            tem00 = 10.0 * tem22 / (amdw * tem11 * con_g)
    with computation(FORWARD):
        with interval(0, 1):
            pwvcm[0, 0] = tem00[0, 0] * plvl[0, 0, 0]

    with computation(PARALLEL):
        with interval(1, None):
            for m in range(1, maxgas):
                summol += colamt[0, 0, 0][m]
            colbrd = coldry - summol

            tem1 = 1.80
            tem2 = 1.50
            for j4 in range(nbands):
                if j4 == 0 or j4 == 3 or j4 == 9:
                    secdiff[0, 0, 0][j4] = 1.66
                else:
                    # Workaround for native functions not working inside for loops
                    # Can be refactored at next gt4py release
                    secdiff[0, 0, 0][j4] = min(
                        tem1,
                        max(
                            tem2,
                            a0[0, 0, 0][j4]
                            + a1[0, 0, 0][j4] * exp(a2[0, 0, 0][j4] * pwvcm),
                        ),
                    )


firstloop(
    indict_gt4py["plyr"],
    indict_gt4py["plvl"],
    indict_gt4py["tlyr"],
    indict_gt4py["tlvl"],
    indict_gt4py["qlyr"],
    indict_gt4py["olyr"],
    indict_gt4py["gasvmr"],
    indict_gt4py["clouds"],
    indict_gt4py["icsdlw"],
    indict_gt4py["faerlw"],
    indict_gt4py["semis"],
    indict_gt4py["tsfg"],
    indict_gt4py["dz"],
    indict_gt4py["delp"],
    indict_gt4py["de_lgth"],
    locdict_gt4py["cldfrc"],
    locdict_gt4py["pavel"],
    locdict_gt4py["tavel"],
    locdict_gt4py["delp"],
    locdict_gt4py["dz"],
    locdict_gt4py["h2ovmr"],
    locdict_gt4py["o3vmr"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colbrd"],
    locdict_gt4py["colamt"],
    locdict_gt4py["wx"],
    locdict_gt4py["tauaer"],
    locdict_gt4py["semiss0"],
    locdict_gt4py["semiss"],
    locdict_gt4py["tem11"],
    locdict_gt4py["tem22"],
    locdict_gt4py["tem00"],
    locdict_gt4py["summol"],
    locdict_gt4py["pwvcm"],
    locdict_gt4py["clwp"],
    locdict_gt4py["relw"],
    locdict_gt4py["ciwp"],
    locdict_gt4py["reiw"],
    locdict_gt4py["cda1"],
    locdict_gt4py["cda2"],
    locdict_gt4py["cda3"],
    locdict_gt4py["cda4"],
    locdict_gt4py["secdiff"],
    locdict_gt4py["A0"],
    locdict_gt4py["A1"],
    locdict_gt4py["A2"],
    domain=(npts, 1, nlp1),
    origin=default_origin,
    validate_args=validate,
)

valvars = [
    "pavel",
    "tavel",
    "delp",
    "colbrd",
    "cldfrc",
    "taucld",
    "semiss0",
    "dz",
    "semiss",
    "coldry",
    "colamt",
    "tauaer",
    "h2ovmr",
    "o3vmr",
    "wx",
    "clwp",
    "relw",
    "ciwp",
    "reiw",
    "cda1",
    "cda2",
    "cda3",
    "cda4",
    "pwvcm",
    "secdiff",
]

# Load serialized data to validate against
ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()

outdict = dict()
valdict = dict()
for var in valvars:
    if var == "cldfrc":
        tmp = locdict_gt4py[var].view(np.ndarray).squeeze()
        outdict[var] = np.append(tmp, np.zeros((npts, 1)), axis=1)
    elif var == "pwvcm":
        outdict[var] = locdict_gt4py[var].view(np.ndarray).squeeze()
    elif var == "taucld" or var == "tauaer":
        tmp = locdict_gt4py[var][:, :, 1:, :].view(np.ndarray).squeeze()
        outdict[var] = np.transpose(tmp, [0, 2, 1])
    elif var == "semiss" or var == "secdiff":
        outdict[var] = locdict_gt4py[var][:, :, 1, :].view(np.ndarray).squeeze()
    else:
        outdict[var] = locdict_gt4py[var][:, :, 1:].view(np.ndarray).squeeze()

    valdict[var] = serializer.read(var, serializer.savepoint["lw_firstloop_out_000000"])

compare_data(outdict, valdict)
