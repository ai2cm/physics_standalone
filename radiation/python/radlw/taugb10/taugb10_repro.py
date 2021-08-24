import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    PARALLEL,
    Field,
    computation,
    interval,
    stencil,
    mod,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
sys.path.insert(
    0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python/radlw"
)
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data
from radlw.radlw_param import (
    nspa,
    nspb,
    ng10,
    ns10,
)

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = True
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = [
    "laytrop",
    "colamt",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "jp",
    "jt",
    "jt1",
    "selffac",
    "selffrac",
    "indself",
    "forfac",
    "forfrac",
    "indfor",
    "fracs",
]

integervars = ["jp", "jt", "jt1", "indself", "indfor"]
fltvars = [
    "colamt",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "forfac",
    "forfrac",
]

ddir = "serdat"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")

savepoints = serializer.savepoint_list()

indict = dict()

for var in invars:
    tmp = serializer.read(var, serializer.savepoint["lwrad-taumol-input-000000"])
    if var == "colamt":
        tmp2 = np.append(tmp, np.zeros((npts, 1, tmp.shape[2])), axis=1)
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var == "fracs":
        tmp2 = np.transpose(
            np.append(tmp, np.zeros((npts, tmp.shape[1], 1)), axis=2), (0, 2, 1)
        )
        indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
    elif var in integervars or var in fltvars:
        tmp2 = np.append(tmp, np.zeros((npts, 1)), axis=1)
        indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
    elif var == "laytrop":
        indict[var] = tmp
    else:
        indict[var] = tmp[0]

laytrop_arr = np.zeros(shape_nlp1, dtype=bool)
for n in range(npts):
    lim = indict["laytrop"][n]
    laytrop_arr[n, :, :lim] = True

indict["laytrop"] = laytrop_arr

indict_gt4py = dict()

for var in invars:
    if var == "colamt":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_maxgas
        )
    elif var in integervars:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_INT
        )
    elif var in fltvars:
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_FLT
        )
    elif var == "fracs":
        indict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptlw)
    elif var == "laytrop":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, DTYPE_BOOL
        )
    else:
        indict_gt4py[var] = indict[var]

locdict_gt4py = dict()

locvars_int = [
    "ind0",
    "ind0p",
    "ind1",
    "ind1p",
    "inds",
    "indsp",
    "indf",
    "indfp",
    "indm",
    "indmp",
]
locvars_flt = [
    "taug",
    "tauself",
    "taufor",
]

for var in locvars_int:
    locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)

for var in locvars_flt:
    if var == "taug":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptlw)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)


def loadlookupdata(name):
    """
    Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables
    """
    ds = xr.open_dataset("lookupdat/radlw_" + name + "_data.nc")

    lookupdict = dict()
    lookupdict_gt4py = dict()

    for var in ds.data_vars.keys():
        # print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :], (npts, 1, nlp1, 1)
            )
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1)
            )
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :, :], (npts, 1, nlp1, 1, 1, 1)
            )

        lookupdict_gt4py[var] = create_storage_from_array(
            lookupdict[var], backend, shape_nlp1, (DTYPE_FLT, ds[var].shape)
        )

    ds2 = xr.open_dataset("lookupdat/radlw_ref_data.nc")
    tmp = np.tile(ds2["chi_mls"].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1))

    lookupdict_gt4py["chi_mls"] = create_storage_from_array(
        tmp, backend, shape_nlp1, (DTYPE_FLT, ds2["chi_mls"].shape)
    )

    return lookupdict_gt4py


lookupdict_gt4py10 = loadlookupdata("kgb10")


@stencil(
    backend=backend,
    rebuild=rebuild,
    verbose=True,
    externals={
        "nspa": nspa[9],
        "nspb": nspb[9],
        "ng10": ng10,
        "ns10": ns10,
    },
)
def taugb10(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng10, 65))],
    absb: Field[(DTYPE_FLT, (ng10, 235))],
    selfref: Field[(DTYPE_FLT, (ng10, 10))],
    forref: Field[(DTYPE_FLT, (ng10, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng10,))],
    fracrefb: Field[(DTYPE_FLT, (ng10,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng10, ns10

    with computation(PARALLEL), interval(0, -1):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            for ig in range(ng10):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns10 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1

            for ig2 in range(ng10):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )

                taug[0, 0, 0][ns10 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig2] = fracrefb[0, 0, 0][ig2]


taugb10(
    indict_gt4py["laytrop"],
    indict_gt4py["colamt"],
    indict_gt4py["fac00"],
    indict_gt4py["fac01"],
    indict_gt4py["fac10"],
    indict_gt4py["fac11"],
    indict_gt4py["jp"],
    indict_gt4py["jt"],
    indict_gt4py["jt1"],
    indict_gt4py["selffac"],
    indict_gt4py["selffrac"],
    indict_gt4py["indself"],
    indict_gt4py["forfac"],
    indict_gt4py["forfrac"],
    indict_gt4py["indfor"],
    indict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py10["absa"],
    lookupdict_gt4py10["absb"],
    lookupdict_gt4py10["selfref"],
    lookupdict_gt4py10["forref"],
    lookupdict_gt4py10["fracrefa"],
    lookupdict_gt4py10["fracrefb"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

print(" ")
print(f"There should be no zeros in the following vector:")
print(f"fracs = {indict_gt4py['fracs'][:, :, 0, 108]}")
