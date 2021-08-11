import gt4py
import os
import sys
import time
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
from radphysparam import ilwcice, ilwcliq
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

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

isubclw = 2

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
        tmp2 = np.append(tmp, 0)
        indict[var] = np.tile(tmp2[None, None, :], (npts, 1, 1))
    elif var == "cldfrc":
        indict[var] = np.tile(tmp[None, None, :-1], (npts, 1, 1))
    elif var == "delgth":
        indict[var] = np.tile(tmp, (npts, 1))
    elif var == "cldfmc" or var == "taucld":
        tmp2 = np.append(
            tmp,
            np.zeros((tmp.shape[0], 1)),
            axis=1,
        )
        indict[var] = np.tile(tmp2.T[None, None, :, :], (npts, 1, 1, 1))
    else:
        indict[var] = tmp

# Read in 2-D array of random numbers used in mcica_subcol, this will change
# in the future once there is a solution for the RNG in python/gt4py
ds = xr.open_dataset("../lookupdata/rand2d.nc")
rand2d = ds["rand2d"][0, :].data
cdfunc = np.reshape(rand2d, (ngptlw, nlay), order="C")
cdfunc = np.append(
    cdfunc,
    np.zeros((cdfunc.shape[0], 1)),
    axis=1,
)
indict["cdfunc"] = np.tile(cdfunc.T[None, None, :, :], (npts, 1, 1, 1))


indict_gt4py = dict()

for var in invars:
    if var in nlay_vars or var == "cldfrc":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    elif var == "delgth":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_2D, DTYPE_FLT
        )
    elif var == "cldfmc":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_ngptlw
        )
    elif var == "taucld":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_nbands
        )
    else:
        indict_gt4py[var] = indict[var]

indict_gt4py["cdfunc"] = create_storage_from_array(
    indict["cdfunc"], backend, shape_nlp1, type_ngptlw
)

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
    "ia",
    "lcloudy",
    "tem1",
    "lcf1",
    "cldsum",
]
bandvars = ["tauliq", "tauice"]

locdict_gt4py = dict()

for var in locvars:
    if var in bandvars:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var == "lcloudy":
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_INT, (ngptlw))
        )
    elif var == "index" or var == "ia":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var == "lcf1":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_BOOL)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)


# Read in lookup table data for cldprop calculations
ds = xr.open_dataset("../lookupdata/radlw_cldprlw_data.nc")

absliq1 = ds["absliq1"].data
absice0 = ds["absice0"].data
absice1 = ds["absice1"].data
absice2 = ds["absice2"].data
absice3 = ds["absice3"].data

lookup_dict = dict()
lookup_dict["absliq1"] = create_storage_from_array(
    absliq1, backend, (npts, 1, nlp1), (np.float64, (58, nbands))
)
lookup_dict["absice0"] = create_storage_from_array(
    absice0, backend, (npts, 1, nlp1), (np.float64, (2,))
)
lookup_dict["absice1"] = create_storage_from_array(
    absice1, backend, (npts, 1, nlp1), (np.float64, (2, 5))
)
lookup_dict["absice2"] = create_storage_from_array(
    absice2, backend, (npts, 1, nlp1), (np.float64, (43, nbands))
)
lookup_dict["absice3"] = create_storage_from_array(
    absice3, backend, (npts, 1, nlp1), (np.float64, (46, nbands))
)
lookup_dict["ipat"] = create_storage_from_array(
    ipat, backend, (npts, 1, nlp1), (DTYPE_INT, (nbands,))
)


@gtscript.stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbands": nbands,
        "ilwcliq": ilwcliq,
        "ngptlw": ngptlw,
        "isubclw": isubclw,
    },
)
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
    cldfmc: Field[type_ngptlw],
    taucld: Field[type_nbands],
    absliq1: Field[(DTYPE_FLT, (58, nbands))],
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
    ia: FIELD_INT,
    lcloudy: Field[(DTYPE_INT, (ngptlw,))],
    cdfunc: Field[type_ngptlw],
    tem1: FIELD_FLT,
    lcf1: FIELD_2DBOOL,
    cldsum: FIELD_FLT,
):
    from __externals__ import nbands, ilwcliq, ngptlw, isubclw

    # Compute flag for whether or not there is cloud in the vertical column
    with computation(FORWARD):
        with interval(0, 1):
            cldsum = cfrac[0, 0, 1]
        with interval(1, -1):
            cldsum = cldsum[0, 0, -1] + cfrac[0, 0, 1]
    with computation(FORWARD), interval(-2, -1):
        lcf1 = cldsum > 0

    with computation(FORWARD), interval(0, -1):
        # Workaround for bug where variables first used inside if statements cause
        # problems. Can be removed after next tag of gt4py is released
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

        if lcf1:
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
                            index = max(1, min(57, factor)) - 1
                            fint = factor - (index + 1)

                            for ib in range(nbands):
                                tmp = cldliq * (
                                    absliq1[0, 0, 0][index, ib]
                                    + fint
                                    * (
                                        absliq1[0, 0, 0][index + 1, ib]
                                        - absliq1[0, 0, 0][index, ib]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauliq[0, 0, 0][ib] = tmp if tmp > 0.0 else 0.0

                    if cldice <= 0.0:
                        for ib2 in range(nbands):
                            tauice[0, 0, 0][ib2] = 0.0
                    else:
                        if ilwcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib3 in range(nbands):
                                ia = ipat[0, 0, 0][ib3] - 1
                                tmp = cldice * (
                                    absice1[0, 0, 0][0, ia]
                                    + absice1[0, 0, 0][1, ia] / refice
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib3] = tmp if tmp > 0.0 else 0.0
                        elif ilwcice == 2:
                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, factor)) - 1
                            fint = factor - (index + 1)

                            for ib4 in range(nbands):
                                tmp = cldice * (
                                    absice2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        absice2[0, 0, 0][index + 1, ib4]
                                        - absice2[0, 0, 0][index, ib4]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib4] = tmp if tmp > 0.0 else 0.0

                        elif ilwcice == 3:
                            dgeice = max(5.0, 1.0315 * refice)  # v4.71 value
                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, factor)) - 1
                            fint = factor - (index + 1)

                            for ib5 in range(nbands):
                                tmp = cldice * (
                                    absice3[0, 0, 0][index, ib5]
                                    + fint
                                    * (
                                        absice3[0, 0, 0][index + 1, ib5]
                                        - absice3[0, 0, 0][index, ib5]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib5] = tmp if tmp > 0.0 else 0.0

                    for ib6 in range(nbands):
                        taucld[0, 0, 0][ib6] = (
                            tauice[0, 0, 0][ib6]
                            + tauliq[0, 0, 0][ib6]
                            + tauran
                            + tausnw
                        )

            else:
                if cfrac[0, 0, 1] > cldmin:
                    for ib7 in range(nbands):
                        taucld[0, 0, 0][ib7] = cdat1

            if isubclw > 0:
                if cfrac[0, 0, 1] < cldmin:
                    cldf = 0.0
                else:
                    cldf = cfrac[0, 0, 1]

    # This section builds mcica_subcol from the fortran into cldprop.
    # Here I've read in the generated random numbers until we figure out
    # what to do with them. This will definitely need to change in future.
    # Only the iovrlw = 1 option is ported from Fortran
    with computation(PARALLEL), interval(1, -1):
        cldf = cldf
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, -1]

            for n in range(ngptlw):
                if cdfunc[0, 0, -1][n] > tem1:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
                else:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(0, -1):
        cldf = cldf
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, 0]

            for n2 in range(ngptlw):
                if cdfunc[0, 0, 0][n2] >= tem1:
                    lcloudy[0, 0, 0][n2] = 1
                else:
                    lcloudy[0, 0, 0][n2] = 0

            for n3 in range(ngptlw):
                if lcloudy[0, 0, 0][n3] == 1:
                    cldfmc[0, 0, 0][n3] = 1.0
                else:
                    cldfmc[0, 0, 0][n3] = 0.0


start = time.time()
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
    indict_gt4py["cldfmc"],
    indict_gt4py["taucld"],
    lookup_dict["absliq1"],
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
    locdict_gt4py["ia"],
    locdict_gt4py["lcloudy"],
    indict_gt4py["cdfunc"],
    locdict_gt4py["tem1"],
    locdict_gt4py["lcf1"],
    locdict_gt4py["cldsum"],
    domain=(npts, 1, nlp1),
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

outdict_gt4py = dict()
outdict_val = dict()

outvars = ["cldfmc", "taucld"]

for var in outvars:
    outdict_gt4py[var] = indict_gt4py[var][0, :, :-1, :].squeeze().T
    outdict_val[var] = serializer.read(
        var, serializer.savepoint["lwrad-cldprop-output-000000"]
    )

compare_data(outdict_gt4py, outdict_val)
