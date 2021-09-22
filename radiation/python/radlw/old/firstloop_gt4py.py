import gt4py
import os
import sys
import numpy as np
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval, stencil

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

os.environ["DYLD_LIBRARY_PATH"] = "/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3

semiss0_np = np.ones(nbands)

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data"
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

for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-in-000000"])
    if var in nlay_vars or var in nlp1_vars:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    elif var == "faerlw":
        indict[var] = np.tile(tmp[:, None, :, :, :], (1, 1, 1, 1, 1))
    elif var == "semis":
        indict[var] = np.tile(tmp[:, None, None], (1, 1, 1))
    elif var == "gasvmr" or var == "clouds":
        indict[var] = np.tile(tmp[:, None, :, :], (1, 1, 1, 1))
    else:
        indict[var] = tmp

print("Done")
print(" ")
print("Creating input storages...")

indict_gt4py = dict()

for var in invars:
    if var in nlay_vars:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, DTYPE_FLT
        )
    elif var in nlp1_vars:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    elif var == "faerlw":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, type_nbands3
        )
    elif var == "semis":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape, DTYPE_FLT
        )
    elif var == "gasvmr":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, type_10
        )
    elif var == "clouds":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, type_9
        )
    else:
        indict_gt4py[var] = indict[var]

print("Done")
print(" ")
print("Creating local storages...")

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
]

locdict_gt4py = dict()

for var in locvars:
    if var == "colamt":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_maxgas)
    elif var == "wx":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_maxxsec)
    elif var == "pwvcm":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
    elif var == "tauaer" or var == "taucld":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_nbands)
    elif var == "semiss0":
        locdict_gt4py[var] = create_storage_ones(backend, shape, type_nbands)
    elif var == "semiss" or var == "secdiff":
        locdict_gt4py[var] = create_storage_zeros(backend, shape, type_nbands)
    elif var == "tz":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    elif var == "cldfrc":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp2, DTYPE_FLT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)

print("Done")


@stencil(backend=backend, rebuild=rebuild)
def get_surface_emissivity(
    sfemis: FIELD_FLT,
    semiss0: Field[type_nbands],
    semiss: Field[type_nbands],
    value: int,
):
    with computation(PARALLEL), interval(...):
        if sfemis[0, 0, 0] > eps and sfemis[0, 0, 0] <= 1.0:
            semiss[0, 0, 0][value] = sfemis[0, 0, 0]
        else:
            semiss[0, 0, 0][value] = semiss0[0, 0, 0][value]


@stencil(backend=backend, rebuild=rebuild)
def set_aerosols(aerosols: Field[type_nbands3], tauaer: Field[type_nbands], value: int):
    with computation(PARALLEL), interval(...):
        tauaer[0, 0, 0][value] = aerosols[0, 0, 0][value, 0] * (
            1.0 - aerosols[0, 0, 0][value, 1]
        )


tem1 = 100.0 * con_g
tem2 = 1.0e-20 * 1.0e3 * con_avgd


@stencil(backend=backend, rebuild=rebuild)
def set_absorbers(
    plyr: FIELD_FLT,
    delpin: FIELD_FLT,
    tlyr: FIELD_FLT,
    dzlyr: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[type_10],
    pavel: FIELD_FLT,
    delp: FIELD_FLT,
    tavel: FIELD_FLT,
    dz: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    coldry: FIELD_FLT,
    temcol: FIELD_FLT,
    tem0: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
):
    with computation(PARALLEL), interval(...):

        pavel[0, 0, 0] = plyr[0, 0, 0]
        delp[0, 0, 0] = delpin[0, 0, 0]
        tavel[0, 0, 0] = tlyr[0, 0, 0]
        dz[0, 0, 0] = dzlyr[0, 0, 0]

        h2ovmr[0, 0, 0] = max(0.0, qlyr[0, 0, 0] * amdw / (1.0 - qlyr[0, 0, 0]))
        o3vmr[0, 0, 0] = max(0.0, olyr[0, 0, 0] * amdo3)

        tem0[0, 0, 0] = (1.0 - h2ovmr[0, 0, 0]) * con_amd + h2ovmr[0, 0, 0] * con_amw
        coldry[0, 0, 0] = (
            tem2 * delp[0, 0, 0] / (tem1 * tem0[0, 0, 0] * (1.0 + h2ovmr[0, 0, 0]))
        )
        temcol[0, 0, 0] = 1.0e-12 * coldry[0, 0, 0]

        colamt[0, 0, 0][0] = max(0.0, coldry[0, 0, 0] * h2ovmr[0, 0, 0])
        colamt[0, 0, 0][1] = max(temcol[0, 0, 0], coldry[0, 0, 0] * gasvmr[0, 0, 0][0])
        colamt[0, 0, 0][2] = max(temcol[0, 0, 0], coldry[0, 0, 0] * o3vmr[0, 0, 0])

        if ilwrgas > 0:
            colamt[0, 0, 0][3] = max(
                temcol[0, 0, 0], coldry[0, 0, 0] * gasvmr[0, 0, 0][1]
            )  # n2o
            colamt[0, 0, 0][4] = max(
                temcol[0, 0, 0], coldry[0, 0, 0] * gasvmr[0, 0, 0][2]
            )  # ch4
            colamt[0, 0, 0][5] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][3])  # o2
            colamt[0, 0, 0][6] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][4])  # co

            wx[0, 0, 0][0] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][8])  # ccl4
            wx[0, 0, 0][1] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][5])  # cf11
            wx[0, 0, 0][2] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][6])  # cf12
            wx[0, 0, 0][3] = max(0.0, coldry[0, 0, 0] * gasvmr[0, 0, 0][7])  # cf22


@stencil(backend=backend, rebuild=rebuild)
def set_clouds(
    clouds: Field[type_9],
    cldfrc: FIELD_FLT,
    clwp: FIELD_FLT,
    relw: FIELD_FLT,
    ciwp: FIELD_FLT,
    reiw: FIELD_FLT,
    cda1: FIELD_FLT,
    cda2: FIELD_FLT,
    cda3: FIELD_FLT,
    cda4: FIELD_FLT,
):
    with computation(PARALLEL), interval(1, None):
        if ilwcliq > 0:
            cldfrc = clouds[0, 0, -1][0]
    with computation(PARALLEL), interval(...):
        if ilwcliq > 0:
            clwp[0, 0, 0] = clouds[0, 0, 0][1]
            relw[0, 0, 0] = clouds[0, 0, 0][2]
            ciwp[0, 0, 0] = clouds[0, 0, 0][3]
            reiw[0, 0, 0] = clouds[0, 0, 0][4]
            cda1[0, 0, 0] = clouds[0, 0, 0][5]
            cda2[0, 0, 0] = clouds[0, 0, 0][6]
            cda3[0, 0, 0] = clouds[0, 0, 0][7]
            cda4[0, 0, 0] = clouds[0, 0, 0][8]
        else:
            cda1[0, 0, 0] = clouds[0, 0, 0][1]
    with computation(PARALLEL), interval(0, 1):
        cldfrc[0, 0, 0] = 1.0


@stencil(backend=backend, rebuild=rebuild)
def compute_temps_for_pwv(
    tem00: FIELD_2D,
    tem11: FIELD_FLT,
    tem22: FIELD_FLT,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    plvl: FIELD_FLT,
    pwvcm: FIELD_2D,
):
    with computation(FORWARD), interval(0, 1):
        tem11[0, 0, 0] = coldry[0, 0, 0] + colamt[0, 0, 0][0]
        tem22[0, 0, 0] = colamt[0, 0, 0][0]
    with computation(FORWARD), interval(1, None):
        tem11[0, 0, 0] = tem11[0, 0, -1] + coldry[0, 0, 0] + colamt[0, 0, 0][0]
        tem22[0, 0, 0] = tem22[0, 0, -1] + colamt[0, 0, 0][0]
    with computation(FORWARD), interval(-1, None):
        tem00 = 10.0 * tem22 / (amdw * tem11 * con_g)
    with computation(FORWARD), interval(0, 1):
        pwvcm[0, 0] = tem00[0, 0] * plvl[0, 0, 0]


# Execute code from here

for j in range(nbands):
    get_surface_emissivity(
        indict_gt4py["semis"],
        locdict_gt4py["semiss0"],
        locdict_gt4py["semiss"],
        value=j,
        origin=default_origin,
        domain=domain,
        validate_args=validate,
    )


locdict_gt4py["tz"] = indict_gt4py["tlvl"].copy()

tem0 = gt4py.storage.zeros(
    backend=backend, default_origin=default_origin, shape=shape_nlay, dtype=DTYPE_FLT
)

set_absorbers(
    indict_gt4py["plyr"],
    indict_gt4py["delp"],
    indict_gt4py["tlyr"],
    indict_gt4py["dz"],
    indict_gt4py["qlyr"],
    indict_gt4py["olyr"],
    indict_gt4py["gasvmr"],
    locdict_gt4py["pavel"],
    locdict_gt4py["delp"],
    locdict_gt4py["tavel"],
    locdict_gt4py["dz"],
    locdict_gt4py["h2ovmr"],
    locdict_gt4py["o3vmr"],
    locdict_gt4py["coldry"],
    locdict_gt4py["temcol"],
    tem0,
    locdict_gt4py["colamt"],
    locdict_gt4py["wx"],
    origin=default_origin,
    domain=domain2,
    validate_args=validate,
)

for j in range(nbands):
    set_aerosols(
        indict_gt4py["faerlw"],
        locdict_gt4py["tauaer"],
        value=j,
        origin=default_origin,
        domain=domain2,
        validate_args=validate,
    )

set_clouds(
    indict_gt4py["clouds"],
    locdict_gt4py["cldfrc"],
    locdict_gt4py["clwp"],
    locdict_gt4py["relw"],
    locdict_gt4py["ciwp"],
    locdict_gt4py["reiw"],
    locdict_gt4py["cda1"],
    locdict_gt4py["cda2"],
    locdict_gt4py["cda3"],
    locdict_gt4py["cda4"],
    domain=domain2,
    origin=default_origin,
    validate_args=validate,
)

tem00 = gt4py.storage.zeros(
    backend=backend, default_origin=default_origin, shape=shape_2D, dtype=DTYPE_FLT
)
tem11 = gt4py.storage.zeros(
    backend=backend, default_origin=default_origin, shape=shape_nlay, dtype=DTYPE_FLT
)
tem22 = gt4py.storage.zeros(
    backend=backend, default_origin=default_origin, shape=shape_nlay, dtype=DTYPE_FLT
)

# This stencil below didn't work, but I realized it's just a sum over k.

compute_temps_for_pwv(
    tem00,
    tem11,
    tem22,
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    indict_gt4py["plvl"],
    locdict_gt4py["pwvcm"],
    origin=default_origin,
    domain=domain2,
    validate_args=validate,
)

# Load serialized data to validate against
ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()
print(savepoints)
sp = serializer.savepoint["lw_firstloop_out_000000"]

valdict = dict()
for var in locvars:
    if var != "secdiff":
        valdict[var] = serializer.read(var, sp)

locdict_np = view_gt4py_storage(locdict_gt4py)

compare_data(valdict, locdict_np)

# Start second loop here

# @stencil(backend=backend, rebuild=rebuild)
# def compute_broadening_gases(colamt: Field[type_maxgas],
#                              coldry: FIELD_FLT,
#                              colbrd: FIELD_FLT):

print(locdict_gt4py["secdiff"].shape)

A0 = create_storage_from_array(a0, backend, (1, 1, 1), type_nbands)
A1 = create_storage_from_array(a1, backend, (1, 1, 1), type_nbands)
A2 = create_storage_from_array(np.exp(a2), backend, (1, 1, 1), type_nbands)

tem1 = 1.80
tem2 = 1.50


@stencil(backend=backend, rebuild=rebuild)
def compute_diffusivity_angle_adj(
    secdiff: Field[type_nbands],
    A0: Field[type_nbands],
    A1: Field[type_nbands],
    expval: FIELD_2D,
    value: int,
):
    with computation(PARALLEL), interval(...):
        if j == 1 or j == 4 or j == 10:
            secdiff[0, 0, 0][value] = 1.66
        else:
            secdiff[0, 0, 0][value] = min(
                tem1, max(tem2, A0[0, 0, 0][value] + A1[0, 0, 0][value] * expval)
            )


for j in range(nbands):
    compute_diffusivity_angle_adj(
        locdict_gt4py["secdiff"],
        A0,
        A1,
        np.exp(a2[j] * locdict_gt4py["pwvcm"]),
        value=j,
        domain=domain,
        origin=default_origin,
        validate_args=validate,
    )
