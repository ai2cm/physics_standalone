import gt4py
import os
import sys
from gt4py import type_hints
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    FORWARD,
    PARALLEL,
    Field,
    computation,
    interval,
    stencil,
    floor,
    __externals__,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from phys_const import con_amw, con_amd, con_amo3
from radlw_param import ngptlw, nbands, abssnow0, absrain, ipat, cldmin
from radphysparam import ilwcice, isubclw
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

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()

amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3

semiss0_np = np.ones(nbands)

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = [
    "nlay",
    "nlp1",
    "ipseed",
    "cldfrc",
    "clwp",
    "relw",
    "ciwp",
    "reiw",
    "cda1",
    "cda2",
    "cda3",
    "cda4",
    "dz",
    "delgth",
    "cldfmc",
    "taucld",
]
nlay_vars = [
    "clwp",
    "relw",
    "ciwp",
    "reiw",
    "cda1",
    "cda2",
    "cda3",
    "cda4",
    "dz",
]

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-cldprop-input-000000"])
    if var in nlay_vars:
        tmp2 = np.append(np.insert(tmp, 0, 0), 0)
        indict[var] = np.tile(tmp2[None, None, :], (npts, 1, 1))
    elif var == "cldfrc":
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    elif var == "delgth":
        indict[var] = np.tile(tmp, (npts, 1))
    elif var == "cldfmc" or var == "taucld":
        tmp2 = np.append(
            np.insert(tmp, 0, np.zeros((1, 1)), axis=1),
            np.zeros((tmp.shape[0], 1)),
            axis=1,
        )
        indict[var] = np.tile(tmp2.T[None, None, :, :], (npts, 1, 1, 1))
    else:
        indict[var] = tmp


indict_gt4py = dict()

for var in invars:
    if var in nlay_vars or var == "cldfrc":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp2, type1
        )
    elif var == "delgth":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, type1
        )
    elif var == "cldfmc":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp2, type_ngptlw
        )
    elif var == "taucld":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp2, type_nbands
        )
    else:
        indict_gt4py[var] = indict[var]


locvars = [
    "tauliq",
    "tauice",
    "cldf",
    "dgeice",
    "factor",
    "fint",
    "tauran",
    "tausnw",
    "cldliq",
    "refliq",
    "cldice",
    "refice",
    "index",
]
bandvars = ["tauliq", "tauice"]

locdict_gt4py = dict()

for var in locvars:
    if var in bandvars:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp2, type_nbands)
    elif var == "index":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp2, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp2, DTYPE_FLT)

ds = xr.open_dataset("../lookupdata/radlw_cldprlw_data.nc")

absliq1 = ds["absliq1"]
absice0 = ds["absice0"]
absice1 = ds["absice1"]
absice2 = ds["absice2"]
absice3 = ds["absice3"]

lookup_dict = dict()
lookup_dict["absliq1"] = create_storage_from_array(
    absliq1, backend, (npts, 1, nlp1 + 1), (np.float64, (58, nbands))
)
lookup_dict["absice0"] = create_storage_from_array(
    absice0, backend, (npts, 1, nlp1 + 1), (np.float64, (2,))
)
lookup_dict["absice1"] = create_storage_from_array(
    absice1, backend, (npts, 1, nlp1 + 1), (np.float64, (2, 5))
)
lookup_dict["absice2"] = create_storage_from_array(
    absice2, backend, (npts, 1, nlp1 + 1), (np.float64, (43, nbands))
)
lookup_dict["absice3"] = create_storage_from_array(
    absice3, backend, (npts, 1, nlp1 + 1), (np.float64, (46, nbands))
)
lookup_dict["ipat"] = create_storage_from_array(
    ipat, backend, (npts, 1, nlp1 + 1), (DTYPE_INT, (nbands,))
)


@gtscript.function
def mcica_subcol(cldf, nlay, ipseed, dz, de_lgth):
    lcloudy = True
    return lcloudy


@gtscript.stencil(backend=backend, rebuild=rebuild, externals={"nbands": nbands})
def cldprop(
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    dz: FIELD_FLT,
    de_lgth: FIELD_2D,
    cldfmc: Field[type_ngptlw],
    taucld: Field[type_nbands],
    absliq1: Field[(DTYPE_FLT, (58, nbands))],
    absice0: Field[(DTYPE_FLT, (2,))],
    absice1: Field[(DTYPE_FLT, (2, 5))],
    absice2: Field[(DTYPE_FLT, (43, nbands))],
    absice3: Field[(DTYPE_FLT, (46, nbands))],
    ipat: Field[(DTYPE_INT, (nbands,))],
    tauliq: Field[type_nbands],
    tauice: Field[type_nbands],
    cldf: FIELD_FLT,
    dgeice: FIELD_FLT,
    factor: FIELD_FLT,
    fint: FIELD_FLT,
    tauran: FIELD_FLT,
    tausnw: FIELD_FLT,
    cldliq: FIELD_FLT,
    refliq: FIELD_FLT,
    cldice: FIELD_FLT,
    refice: FIELD_FLT,
    index: FIELD_INT,
):
    from __externals__ import nbands

    with computation(FORWARD), interval(1, -1):
        tauliq = tauliq
        tauice = tauice
        cldf = cldf
        dgeice = dgeice
        factor = factor
        fint = fint
        tauran = tauran
        tausnw = tausnw
        cldliq = cldliq
        refliq = refliq
        cldice = cldice
        refice = refice
        cfrac = cfrac
        cliqp = cliqp
        reliq = reliq
        cicep = cicep
        reice = reice
        cdat1 = cdat1
        cdat2 = cdat2
        cdat3 = cdat3
        cdat4 = cdat4
        dz = dz
        index = index
        absliq1 = absliq1

        if ilwcliq > 0:
            if cfrac > cldmin:
                tauran = absrain * cdat1
                if cdat3 > 0.0 and cdat4 > 10.0:
                    tausnw = abssnow0 * 1.05756 * cdat3 / cdat4
                else:
                    tausnw = 0.0

                cldliq = cliqp
                cldice = cicep
                refliq = reliq
                refice = reice

                if cldliq <= 0:
                    for i in range(nbands):
                        tauliq[0, 0, 0][i] = 0.0
                else:
                    if ilwcliq == 1:
                        factor = refliq - 1.5
                        index = max(1, min(57, factor))
                        fint = factor - index

                        for ib in range(nbands):
                            tauliq[0, 0, 0][ib] = max(
                                0.0,
                                cldliq
                                * (
                                    absliq1[0, 0, 0][index, ib]
                                    + fint
                                    * (
                                        absliq1[0, 0, 0][index + 1, ib]
                                        - absliq1[0, 0, 0][index, ib]
                                    )
                                ),
                            )


cldprop(
    indict_gt4py["cldfrc"],
    indict_gt4py["clwp"],
    indict_gt4py["relw"],
    indict_gt4py["ciwp"],
    indict_gt4py["reiw"],
    indict_gt4py["cda1"],
    indict_gt4py["cda2"],
    indict_gt4py["cda3"],
    indict_gt4py["cda4"],
    indict_gt4py["dz"],
    indict_gt4py["delgth"],
    indict_gt4py["cldfmc"],
    indict_gt4py["taucld"],
    lookup_dict["absliq1"],
    lookup_dict["absice0"],
    lookup_dict["absice1"],
    lookup_dict["absice2"],
    lookup_dict["absice3"],
    lookup_dict["ipat"],
    locdict_gt4py["tauliq"],
    locdict_gt4py["tauice"],
    locdict_gt4py["cldf"],
    locdict_gt4py["dgeice"],
    locdict_gt4py["factor"],
    locdict_gt4py["fint"],
    locdict_gt4py["tauran"],
    locdict_gt4py["tausnw"],
    locdict_gt4py["cldliq"],
    locdict_gt4py["refliq"],
    locdict_gt4py["cldice"],
    locdict_gt4py["refice"],
    locdict_gt4py["index"],
    domain=(npts, 1, nlp1 + 1),
    origin=default_origin,
    validate_args=validate,
)

print(locdict_gt4py["index"])
print(locdict_gt4py["fint"])
