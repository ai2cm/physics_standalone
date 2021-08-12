import sys
import numpy as np
import xarray as xr
from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    log,
    index,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from config import *
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

invars = ["pavel", "tavel", "h2ovmr", "nlay", "nlp1"]

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-setcoef-input-000000"])
    if var in ["pavel", "tavel", "h2ovmr"]:
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    else:
        indict[var] = tmp[0]

indict_gt4py = dict()
for var in invars:
    if var in ["pavel", "tavel", "h2ovmr"]:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlay, DTYPE_FLT
        )
    else:
        indict_gt4py[var] = indict[var]

outvars_flt = [
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "forfac",
    "forfrac",
]

outvars_int = ["indself", "indfor", "jp", "jt", "jt1"]

outdict_gt4py = dict()

for var in outvars_flt:
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)
for var in outvars_int:
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_INT)

outdict_gt4py["laytrop"] = create_storage_zeros(backend, shape_nlay, DTYPE_BOOL)

locvars = ["plog", "fp", "fp1", "ft", "ft1", "tem1", "tem2", "jp1"]

locdict_gt4py = dict()
for var in locvars:
    if var == "jp1":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)

ds = xr.open_dataset("../lookupdata/radsw_ref_data.nc")
preflog = ds["preflog"].data
preflog = np.tile(preflog[None, None, None, :], (npts, 1, nlay, 1))
tref = ds["tref"].data
tref = np.tile(tref[None, None, None, :], (npts, 1, nlay, 1))
lookupdict_gt4py = dict()

lookupdict_gt4py["preflog"] = create_storage_from_array(
    preflog, backend, shape_nlay, (DTYPE_FLT, (59,))
)
lookupdict_gt4py["tref"] = create_storage_from_array(
    tref, backend, shape_nlay, (DTYPE_FLT, (59,))
)

stpfac = 296.0 / 1013.0


@stencil(backend=backend, rebuild=rebuild, externals={"stpfac": stpfac})
def setcoef(
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    laytrop: FIELD_BOOL,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
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
    plog: FIELD_FLT,
    fp: FIELD_FLT,
    fp1: FIELD_FLT,
    ft: FIELD_FLT,
    ft1: FIELD_FLT,
    tem1: FIELD_FLT,
    tem2: FIELD_FLT,
    jp1: FIELD_INT,
    preflog: Field[(DTYPE_FLT, (59,))],
    tref: Field[(DTYPE_FLT, (59,))],
):
    from __externals__ import stpfac

    with computation(PARALLEL), interval(...):
        forfac = pavel * stpfac / (tavel * (1.0 + h2ovmr))

        #  --- ...  find the two reference pressures on either side of the
        #           layer pressure.  store them in jp and jp1.  store in fp the
        #           fraction of the difference (in ln(pressure)) between these
        #           two values that the layer pressure lies.

        plog = log(pavel)
        jp = max(1, min(58, 36.0 - 5.0 * (plog + 0.04))) - 1
        jp1 = jp + 1
        fp = 5.0 * (preflog[0, 0, 0][jp] - plog)

        #  --- ...  determine, for each reference pressure (jp and jp1), which
        #          reference temperature (these are different for each reference
        #          pressure) is nearest the layer temperature but does not exceed it.
        #          store these indices in jt and jt1, resp. store in ft (resp. ft1)
        #          the fraction of the way between jt (jt1) and the next highest
        #          reference temperature that the layer temperature falls.

        tem1 = (tavel - tref[0, 0, 0][jp]) / 15.0
        tem2 = (tavel - tref[0, 0, 0][jp1]) / 15.0
        jt = max(1, min(4, 3.0 + tem1)) - 1
        jt1 = max(1, min(4, 3.0 + tem2)) - 1
        ft = tem1 - (jt - 2)
        ft1 = tem2 - (jt1 - 2)

        #  --- ...  we have now isolated the layer ln pressure and temperature,
        #           between two reference pressures and two reference temperatures
        #           (for each reference pressure).  we multiply the pressure
        #           fraction fp with the appropriate temperature fractions to get
        #           the factors that will be needed for the interpolation that yields
        #           the optical depths (performed in routines taugbn for band n).

        fp1 = 1.0 - fp
        fac10 = fp1 * ft
        fac00 = fp1 * (1.0 - ft)
        fac11 = fp * ft1
        fac01 = fp * (1.0 - ft1)

        #  --- ...  if the pressure is less than ~100mb, perform a different
        #           set of species interpolations.

        if plog > 4.56:

            laytrop = True

            #  --- ...  set up factors needed to separately include the water vapor
            #           foreign-continuum in the calculation of absorption coefficient.

            tem1 = (332.0 - tavel) / 36.0
            indfor = min(2, max(1, tem1))
            forfrac = tem1 - indfor

            #  --- ...  set up factors needed to separately include the water vapor
            #           self-continuum in the calculation of absorption coefficient.

            tem2 = (tavel - 188.0) / 7.2
            indself = min(9, max(1, tem2 - 7))
            selffrac = tem2 - (indself + 7)
            selffac = h2ovmr * forfac

        else:

            #  --- ...  set up factors needed to separately include the water vapor
            #           foreign-continuum in the calculation of absorption coefficient.

            tem1 = (tavel - 188.0) / 36.0
            indfor = 3
            forfrac = tem1 - 1.0

            indself = 0
            selffrac = 0.0
            selffac = 0.0

        # Add one to indices for consistency with Fortran
        jp += 1
        jt += 1
        jt1 += 1


setcoef(
    indict_gt4py["pavel"],
    indict_gt4py["tavel"],
    indict_gt4py["h2ovmr"],
    outdict_gt4py["laytrop"],
    outdict_gt4py["jp"],
    outdict_gt4py["jt"],
    outdict_gt4py["jt1"],
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
    locdict_gt4py["plog"],
    locdict_gt4py["fp"],
    locdict_gt4py["fp1"],
    locdict_gt4py["ft"],
    locdict_gt4py["ft1"],
    locdict_gt4py["tem1"],
    locdict_gt4py["tem2"],
    locdict_gt4py["jp1"],
    lookupdict_gt4py["preflog"],
    lookupdict_gt4py["tref"],
    domain=shape_nlay,
    origin=default_origin,
    validate_args=validate,
)

outvars = [
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "forfac",
    "forfrac",
    "indself",
    "indfor",
    "jp",
    "jt",
    "jt1",
    "laytrop",
]

outdict_np = dict()
outdict_val = dict()
for var in outvars:
    if var == "laytrop":
        outdict_np[var] = (
            outdict_gt4py[var][0, :, :].view(np.ndarray).astype(int).squeeze().sum()
        )
    else:
        outdict_np[var] = outdict_gt4py[var][0, :, :].view(np.ndarray).squeeze()
    outdict_val[var] = serializer.read(
        var, serializer.savepoint["swrad-setcoef-output-000000"]
    )

compare_data(outdict_np, outdict_val)
