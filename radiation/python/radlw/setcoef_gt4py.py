import sys
import numpy as np
import xarray as xr
from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    FORWARD,
    PARALLEL,
    log,
    index,
    K,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radlw_param import nbands, nrates, delwave, nplnk
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = [
    "pavel",
    "tavel",
    "tz",
    "stemp",
    "h2ovmr",
    "colamt",
    "coldry",
    "colbrd",
    "nlay",
    "nlp1",
]

nlay_vars = [
    "pavel",
    "tavel",
    "h2ovmr",
    "coldry",
    "colbrd",
]

outvars = [
    "laytrop",
    "pklay",
    "pklev",
    "jp",
    "jt",
    "jt1",
    "rfrate",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "indself",
    "forfac",
    "forfrac",
    "indfor",
    "minorfrac",
    "scaleminor",
    "scaleminorn2",
    "indminor",
]

locvars = [
    "tlvlfr",
    "tlyrfr",
    "plog",
    "fp",
    "ft",
    "ft1",
    "tem1",
    "tem2",
    "indlay",
    "indlev",
    "jp1",
    "tzint",
    "stempint",
    "tavelint",
]

indict = dict()

for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-setcoef-input-000000"])

    if var == "colamt":
        tmp2 = np.insert(tmp, 0, np.zeros((1, 1)), axis=0)
        indict[var] = np.tile(tmp2[None, None, :, :], (npts, 1, 1, 1))
    elif var in nlay_vars:
        tmp2 = np.insert(tmp, 0, 0)
        indict[var] = np.tile(tmp2[None, None, :], (npts, 1, 1))
    elif var == "tz":
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    elif var == "stemp":
        indict[var] = np.tile(tmp, (npts, 1, nlp1))
    elif tmp.size == 1:
        indict[var] = tmp[0]
    else:
        indict[var] = tmp

indict_gt4py = dict()

for var in invars:
    if var == "colamt":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_maxgas
        )
    elif var in nlay_vars or var == "tz" or var == "stemp":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    else:
        indict_gt4py[var] = indict[var]

outdict_gt4py = dict()
for var in outvars:
    if var == "rfrate":
        outdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_FLT, (nrates, 2))
        )
    elif var == "pklay" or var == "pklev":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var == "laytrop":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, bool)
    elif var in ["jp", "jt", "jt1", "indself", "indfor", "indminor"]:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    else:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

locdict_gt4py = dict()
for var in locvars:
    if var in ["jp1", "tzint", "stempint", "indlev", "indlay", "tavelint"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

lookupdict_gt4py = dict()
ds = xr.open_dataset("../lookupdata/totplnk.nc")
totplnk = ds["totplnk"].data

totplnk = np.tile(totplnk[None, None, None, :, :], (npts, 1, nlp1, 1, 1))
lookupdict_gt4py["totplnk"] = create_storage_from_array(
    totplnk, backend, shape_nlp1, (DTYPE_FLT, (nplnk, nbands))
)

refvars = ["pref", "preflog", "tref", "chi_mls"]
ds2 = xr.open_dataset("../lookupdata/radlw_ref_data.nc")

for var in refvars:
    tmp = ds2[var].data

    if var == "chi_mls":
        tmp = np.tile(tmp[None, None, None, :, :], (npts, 1, nlp1, 1, 1))
        lookupdict_gt4py[var] = create_storage_from_array(
            tmp, backend, shape_nlp1, (DTYPE_FLT, (7, 59))
        )
    else:
        tmp = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
        lookupdict_gt4py[var] = create_storage_from_array(
            tmp, backend, shape_nlp1, (DTYPE_FLT, (59,))
        )

delwave = np.tile(delwave[None, None, None, :], (npts, 1, nlp1, 1))
delwave = create_storage_from_array(delwave, backend, shape_nlp1, type_nbands)

stpfac = 296.0 / 1013.0


@stencil(
    backend=backend, rebuild=rebuild, externals={"nbands": nbands, "stpfac": stpfac}
)
def setcoef(
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    tz: FIELD_FLT,
    stemp: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    colamt: Field[type_maxgas],
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    totplnk: Field[(DTYPE_FLT, (nplnk, nbands))],
    pref: Field[(DTYPE_FLT, (59,))],
    preflog: Field[(DTYPE_FLT, (59,))],
    tref: Field[(DTYPE_FLT, (59,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    delwave: Field[type_nbands],
    laytrop: Field[bool],
    pklay: Field[type_nbands],
    pklev: Field[type_nbands],
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminor: FIELD_FLT,
    scaleminorn2: FIELD_FLT,
    indminor: FIELD_INT,
    tzint: FIELD_INT,
    stempint: FIELD_INT,
    tavelint: FIELD_INT,
    indlay: FIELD_INT,
    indlev: FIELD_INT,
    tlyrfr: FIELD_FLT,
    tlvlfr: FIELD_FLT,
    jp1: FIELD_INT,
    plog: FIELD_FLT,
):
    from __externals__ import nbands, stpfac

    with computation(PARALLEL):
        #  --- ...  calculate information needed by the radiative transfer routine
        #           that is specific to this atmosphere, especially some of the
        #           coefficients and indices needed to compute the optical depths
        #           by interpolating data from stored reference atmospheres.
        with interval(0, 1):
            indlay = min(180, max(1, stemp - 159.0))
            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            stempint = stemp
            tlyrfr = stemp - stempint
            tlvlfr = tz - tzint

            for i0 in range(nbands):
                tem1 = totplnk[0, 0, 0][indlay, i0] - totplnk[0, 0, 0][indlay - 1, i0]
                tem2 = totplnk[0, 0, 0][indlev, i0] - totplnk[0, 0, 0][indlev - 1, i0]
                pklay[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlay - 1, i0] + tlyrfr * tem1
                )
                pklev[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlev - 1, i0] + tlvlfr * tem2
                )

        #           calculate the integrated Planck functions for each band at the
        #           surface, level, and layer temperatures.
        with interval(1, None):
            indlay = min(180, max(1, tavel - 159.0))
            tavelint = tavel
            tlyrfr = tavel - tavelint

            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            tlvlfr = tz - tzint

            #  --- ...  begin spectral band loop
            for i in range(nbands):
                pklay[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlay - 1, i]
                    + tlyrfr
                    * (totplnk[0, 0, 0][indlay, i] - totplnk[0, 0, 0][indlay - 1, i])
                )
                pklev[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlev - 1, i]
                    + tlvlfr
                    * (totplnk[0, 0, 0][indlev, i] - totplnk[0, 0, 0][indlev - 1, i])
                )

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure. store them in jp and jp1. store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = log(pavel)
            jp = max(1, min(58, 36.0 - 5.0 * (plog + 0.04))) - 1
            jp1 = jp + 1
            #  --- ...  limit pressure extrapolation at the top
            fp = max(0.0, min(1.0, 5.0 * (preflog[0, 0, 0][jp] - plog)))

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #           reference temperature (these are different for each
            #           reference pressure) is nearest the layer temperature but does
            #           not exceed it. store these indices in jt and jt1, resp.
            #           store in ft (resp. ft1) the fraction of the way between jt
            #           (jt1) and the next highest reference temperature that the
            #           layer temperature falls.

            tem1 = (tavel - tref[0, 0, 0][jp]) / 15.0
            tem2 = (tavel - tref[0, 0, 0][jp1]) / 15.0
            jt = max(1, min(4, 3.0 + tem1)) - 1
            jt1 = max(1, min(4, 3.0 + tem2)) - 1
            # --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
            ft = max(-0.5, min(1.5, tem1 - (jt - 2)))
            ft1 = max(-0.5, min(1.5, tem2 - (jt1 - 2)))

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n)

            tem1 = 1.0 - fp
            fac10 = tem1 * ft
            fac00 = tem1 * (1.0 - ft)
            fac11 = fp * ft1
            fac01 = fp * (1.0 - ft1)

            forfac = pavel * stpfac / (tavel * (1.0 + h2ovmr))
            selffac = h2ovmr * forfac

            #  --- ...  set up factors needed to separately include the minor gases
            #           in the calculation of absorption coefficient

            scaleminor = pavel / tavel
            scaleminorn2 = (pavel / tavel) * (colbrd / (coldry + colamt[0, 0, 0][0]))

            tem1 = (tavel - 180.8) / 7.2
            indminor = min(18, max(1, tem1))
            minorfrac = tem1 - indminor

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            indfor = indfor
            forfrac = forfrac
            indself = indself
            selffrac = selffrac
            rfrate = rfrate
            chi_mls = chi_mls
            laytrop = laytrop

            if plog > 4.56:

                # compute troposphere mask, True in troposphere, False otherwise
                laytrop = True

                tem1 = (332.0 - tavel) / 36.0
                indfor = min(2, max(1, tem1))
                forfrac = tem1 - indfor

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem1 = (tavel - 188.0) / 7.2
                indself = min(9, max(1, tem1 - 7))
                selffrac = tem1 - (indself + 7)

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in lower atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][1, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][2, jp]
                )
                rfrate[0, 0, 0][1, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][2, jp + 1]
                )
                rfrate[0, 0, 0][2, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][3, jp]
                )
                rfrate[0, 0, 0][2, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][3, jp + 1]
                )
                rfrate[0, 0, 0][3, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][5, jp]
                )
                rfrate[0, 0, 0][3, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][5, jp + 1]
                )
                rfrate[0, 0, 0][4, 0] = (
                    chi_mls[0, 0, 0][3, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][4, 1] = (
                    chi_mls[0, 0, 0][3, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            else:
                laytrop = False

                tem1 = (tavel - 188.0) / 36.0
                indfor = 3
                forfrac = tem1 - 1.0

                indself = 0
                selffrac = 0.0

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in upper atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][5, 0] = (
                    chi_mls[0, 0, 0][2, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][5, 1] = (
                    chi_mls[0, 0, 0][2, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            #  --- ...  rescale selffac and forfac for use in taumol

            selffac = colamt[0, 0, 0][0] * selffac
            forfac = colamt[0, 0, 0][0] * forfac

            #  --- ...  add one to computed indices for compatibility with later
            #           subroutines

            jp += 1
            jt += 1
            jt1 += 1


setcoef(
    indict_gt4py["pavel"],
    indict_gt4py["tavel"],
    indict_gt4py["tz"],
    indict_gt4py["stemp"],
    indict_gt4py["h2ovmr"],
    indict_gt4py["colamt"],
    indict_gt4py["coldry"],
    indict_gt4py["colbrd"],
    lookupdict_gt4py["totplnk"],
    lookupdict_gt4py["pref"],
    lookupdict_gt4py["preflog"],
    lookupdict_gt4py["tref"],
    lookupdict_gt4py["chi_mls"],
    delwave,
    outdict_gt4py["laytrop"],
    outdict_gt4py["pklay"],
    outdict_gt4py["pklev"],
    outdict_gt4py["jp"],
    outdict_gt4py["jt"],
    outdict_gt4py["jt1"],
    outdict_gt4py["rfrate"],
    outdict_gt4py["fac00"],
    outdict_gt4py["fac01"],
    outdict_gt4py["fac10"],
    outdict_gt4py["fac11"],
    outdict_gt4py["selffac"],
    outdict_gt4py["selffrac"],
    outdict_gt4py["indself"],
    outdict_gt4py["forfac"],
    outdict_gt4py["forfrac"],
    outdict_gt4py["indfor"],
    outdict_gt4py["minorfrac"],
    outdict_gt4py["scaleminor"],
    outdict_gt4py["scaleminorn2"],
    outdict_gt4py["indminor"],
    locdict_gt4py["tzint"],
    locdict_gt4py["stempint"],
    locdict_gt4py["tavelint"],
    locdict_gt4py["indlay"],
    locdict_gt4py["indlev"],
    locdict_gt4py["tlyrfr"],
    locdict_gt4py["tlvlfr"],
    locdict_gt4py["jp1"],
    locdict_gt4py["plog"],
    domain=(npts, 1, nlp1),
    origin=default_origin,
    validate_args=validate,
)

outdict_np = dict()
outdict_val = dict()

for var in outvars:
    outdict_val[var] = serializer.read(
        var, serializer.savepoint["lwrad-setcoef-output-000000"]
    )
    if var != "laytrop":
        outdict_np[var] = outdict_gt4py[var][0, 0, ...].squeeze().view(np.ndarray)
        if var == "pklay" or var == "pklev":
            outdict_np[var] = outdict_np[var].T
        else:
            outdict_np[var] = outdict_np[var][1:, ...]
    else:
        outdict_np[var] = (
            outdict_gt4py[var][0, :, 1:]
            .squeeze()
            .view(np.ndarray)
            .astype(np.int32)
            .sum()
        )

compare_data(outdict_np, outdict_val)
