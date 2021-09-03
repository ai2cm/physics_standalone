import numpy as np
import xarray as xr
import sys

sys.path.insert(0, "..")
from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *
from stencils_sw_gt4py import *
from radsw.radsw_param import idxebc

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

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

locvars_firstloop = [
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

locvars_cldprop = [
    "tauliq",
    "tauice",
    "ssaliq",
    "ssaice",
    "ssaran",
    "ssasnw",
    "asyliq",
    "asyice",
    "asyran",
    "asysnw",
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
    "cldran",
    "cldsnw",
    "refsnw",
    "extcoliq",
    "ssacoliq",
    "asycoliq",
    "extcoice",
    "ssacoice",
    "asycoice",
    "dgesnw",
    "lcloudy",
    "index",
    "ia",
    "jb",
    "cldfmc",
    "taucw",
    "ssacw",
    "asycw",
    "cldfrc",
]

locvars_setcoef = [
    "plog",
    "fp",
    "fp1",
    "ft",
    "ft1",
    "jp1",
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
for var in locvars_firstloop:
    if var in ["tauae", "ssaae", "asyae"]:
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

# Read in 2-D array of random numbers used in mcica_subcol, this will change
# in the future once there is a solution for the RNG in python/gt4py
ds = xr.open_dataset("../lookupdata/rand2d_sw.nc")
rand2d = ds["rand2d"].data
cdfunc = np.zeros((npts, nlay, ngptsw))
idxday = serializer.read("idxday", serializer.savepoint["swrad-in-000000"])
for n in range(npts):
    myind = idxday[n]
    if myind > 1 and myind < 25:
        cdfunc[myind - 1, :, :] = np.reshape(rand2d[n, :], (nlay, ngptsw), order="F")
cdfunc = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))
cdfunc = np.insert(cdfunc, 0, 0, axis=2)

locdict_gt4py["cdfunc"] = create_storage_from_array(
    cdfunc, backend, shape_nlp1, type_ngptsw
)

for var in locvars_cldprop:
    if var in [
        "tauliq",
        "tauice",
        "ssaliq",
        "ssaice",
        "ssaran",
        "ssasnw",
        "asyliq",
        "asyice",
        "asyran",
        "asysnw",
    ]:
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, type_nbandssw_flt
        )
    elif var in ["taucw", "ssacw", "asycw"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    elif var == "cldfmc":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw)
    elif var == "lcloudy":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw_bool)
    elif var in ["index", "ia", "jb"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

idxebc = np.tile(np.array(idxebc)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["idxebc"] = create_storage_from_array(
    idxebc, backend, shape_nlp1, type_nbandssw_int
)

for var in locvars_setcoef:
    if var == "jp1":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var in [
        "fac00",
        "fac01",
        "fac10",
        "fac11",
        "selffac",
        "selffrac",
        "forfac",
        "forfrac",
    ]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    elif var in ["indself", "indfor", "jp", "jt", "jt1"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var == "laytrop":
        locdict_gt4py["laytrop"] = create_storage_zeros(backend, shape_nlp1, DTYPE_BOOL)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)


def loadlookupdata(name):
    """
    Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables
    """
    ds = xr.open_dataset("../lookupdata/radsw_" + name + "_data.nc")

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

        if len(ds.data_vars[var].shape) >= 1:
            lookupdict_gt4py[var] = create_storage_from_array(
                lookupdict[var], backend, shape_nlp1, (DTYPE_FLT, ds[var].shape)
            )
        else:
            lookupdict_gt4py[var] = float(ds[var].data)

    return lookupdict_gt4py


lookupdict = loadlookupdata("cldprtb")

# Load lookup data for setcoef
ds = xr.open_dataset("../lookupdata/radsw_ref_data.nc")
preflog = ds["preflog"].data
preflog = np.tile(preflog[None, None, None, :], (npts, 1, nlp1, 1))
tref = ds["tref"].data
tref = np.tile(tref[None, None, None, :], (npts, 1, nlp1, 1))
lookupdict_setcoef = dict()

lookupdict_setcoef["preflog"] = create_storage_from_array(
    preflog, backend, shape_nlp1, (DTYPE_FLT, (59,))
)
lookupdict_setcoef["tref"] = create_storage_from_array(
    tref, backend, shape_nlp1, (DTYPE_FLT, (59,))
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


valdict_firstloop = dict()
outdict_firstloop = dict()

for var in locvars_firstloop:
    valdict_firstloop[var] = serializer2.read(
        var, serializer2.savepoint["swrad-firstloop-output-000000"]
    )
    if var in ["cosz1", "sntz1", "ssolar", "albbm", "albdf", "tem0", "tem1", "tem2"]:
        outdict_firstloop[var] = (
            locdict_gt4py[var][:, :, 0, ...].view(np.ndarray).squeeze()
        )
    elif var in ["zcf1", "zcf0"]:
        outdict_firstloop[var] = locdict_gt4py[var].view(np.ndarray).squeeze()
    else:
        outdict_firstloop[var] = (
            locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()
        )

compare_data(valdict_firstloop, outdict_firstloop)

cldprop(
    locdict_gt4py["cfrac"],
    locdict_gt4py["cliqp"],
    locdict_gt4py["reliq"],
    locdict_gt4py["cicep"],
    locdict_gt4py["reice"],
    locdict_gt4py["cdat1"],
    locdict_gt4py["cdat2"],
    locdict_gt4py["cdat3"],
    locdict_gt4py["cdat4"],
    locdict_gt4py["zcf1"],
    indict_gt4py["dz"],
    indict_gt4py["de_lgth"],
    indict_gt4py["idxday"],
    locdict_gt4py["cldfmc"],
    locdict_gt4py["taucw"],
    locdict_gt4py["ssacw"],
    locdict_gt4py["asycw"],
    locdict_gt4py["cldfrc"],
    locdict_gt4py["tauliq"],
    locdict_gt4py["tauice"],
    locdict_gt4py["ssaliq"],
    locdict_gt4py["ssaice"],
    locdict_gt4py["ssaran"],
    locdict_gt4py["ssasnw"],
    locdict_gt4py["asyliq"],
    locdict_gt4py["asyice"],
    locdict_gt4py["asyran"],
    locdict_gt4py["asysnw"],
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
    locdict_gt4py["cldran"],
    locdict_gt4py["cldsnw"],
    locdict_gt4py["refsnw"],
    locdict_gt4py["extcoliq"],
    locdict_gt4py["ssacoliq"],
    locdict_gt4py["asycoliq"],
    locdict_gt4py["extcoice"],
    locdict_gt4py["ssacoice"],
    locdict_gt4py["asycoice"],
    locdict_gt4py["dgesnw"],
    locdict_gt4py["lcloudy"],
    locdict_gt4py["index"],
    locdict_gt4py["ia"],
    locdict_gt4py["jb"],
    locdict_gt4py["idxebc"],
    locdict_gt4py["cdfunc"],
    lookupdict["extliq1"],
    lookupdict["extliq2"],
    lookupdict["ssaliq1"],
    lookupdict["ssaliq2"],
    lookupdict["asyliq1"],
    lookupdict["asyliq2"],
    lookupdict["extice2"],
    lookupdict["ssaice2"],
    lookupdict["asyice2"],
    lookupdict["extice3"],
    lookupdict["ssaice3"],
    lookupdict["asyice3"],
    lookupdict["fdlice3"],
    lookupdict["abari"],
    lookupdict["bbari"],
    lookupdict["cbari"],
    lookupdict["dbari"],
    lookupdict["ebari"],
    lookupdict["fbari"],
    lookupdict["b0s"],
    lookupdict["b1s"],
    lookupdict["c0s"],
    lookupdict["b0r"],
    lookupdict["c0r"],
    lookupdict["a0r"],
    lookupdict["a1r"],
    lookupdict["a0s"],
    lookupdict["a1s"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

valdict_cldprop = dict()
outdict_cldprop = dict()

outvars_cldprop = ["cldfmc", "taucw", "ssacw", "asycw", "cldfrc"]

for var in outvars_cldprop:
    valdict_cldprop[var] = serializer2.read(
        var, serializer2.savepoint["swrad-cldprop-output-000000"]
    )
    outdict_cldprop[var] = locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()

compare_data(outdict_cldprop, valdict_cldprop)

setcoef(
    locdict_gt4py["pavel"],
    locdict_gt4py["tavel"],
    locdict_gt4py["h2ovmr"],
    indict_gt4py["idxday"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["plog"],
    locdict_gt4py["fp"],
    locdict_gt4py["fp1"],
    locdict_gt4py["ft"],
    locdict_gt4py["ft1"],
    locdict_gt4py["tem1"],
    locdict_gt4py["tem2"],
    locdict_gt4py["jp1"],
    lookupdict_setcoef["preflog"],
    lookupdict_setcoef["tref"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

outvars_setcoef = [
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

outdict_setcoef = dict()
valdict_setcoef = dict()
for var in outvars_setcoef:
    if var == "laytrop":
        outdict_setcoef[var] = (
            locdict_gt4py[var].view(np.ndarray).astype(int).squeeze().sum(axis=1)
        )
    else:
        outdict_setcoef[var] = (
            locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()
        )
    valdict_setcoef[var] = serializer2.read(
        var, serializer2.savepoint["swrad-setcoef-output-000000"]
    )

compare_data(outdict_setcoef, valdict_setcoef)
