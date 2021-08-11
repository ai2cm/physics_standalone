import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radlw_param import a0, a1, a2, nbands, nrates, ipat, delwave
from stencils_gt4py import *
from util import (
    view_gt4py_storage,
    compare_data,
    create_storage_from_array,
    create_storage_zeros,
    create_storage_ones,
    loadlookupdata,
)
from config import *

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")
savepoints = serializer.savepoint_list()

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

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

outvars = [
    "hlwc",
    "cldtau",
    "upfxc_t",
    "upfx0_t",
    "upfxc_s",
    "upfx0_s",
    "dnfxc_s",
    "dnfx0_s",
]

locvars = [
    "cldfrc",
    "totuflux",
    "totdflux",
    "totuclfl",
    "totdclfl",
    "tz",
    "htr",
    "htrcl",
    "pavel",
    "tavel",
    "delp",
    "clwp",
    "ciwp",
    "relw",
    "reiw",
    "cda1",
    "cda2",
    "cda3",
    "cda4",
    "coldry",
    "colbrd",
    "h2ovmr",
    "o3vmr",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "forfac",
    "forfrac",
    "minorfrac",
    "scaleminor",
    "scaleminorn2",
    "temcol",
    "dz",
    "pklev",
    "pklay",
    "htrb" "taucld",
    "tauaer",
    "fracs",
    "tautot",
    "cldfmc",
    "taucld",
    "semiss",
    "semiss0",
    "secdiff",
    "colamt",
    "wx",
    "rfrate",
    "tem0",
    "tem1",
    "tem2",
    "pwvcm",
    "summol",
    "stemp",
    "delgth",
    "ipseed",
    "jp",
    "jt",
    "jt1",
    "indself",
    "indfor",
    "indminor",
    "tem00",
    "tem11",
    "tem22",
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
    "summol",
    "expval",
    "lcf1",
    "cldsum",
    "tlvlfr",
    "tlyrfr",
    "plog",
    "fp",
    "ft",
    "ft1",
    "indlay",
    "indlev",
    "jp1",
    "tzint",
    "stempint",
    "tavelint",
    "laytrop",
]

locvars_int = [
    "ib",
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
    "js",
    "js1",
    "jmn2o",
    "jmn2op",
    "jpl",
    "jplp",
    "id000",
    "id010",
    "id100",
    "id110",
    "id200",
    "id210",
    "id001",
    "id011",
    "id101",
    "id111",
    "id201",
    "id211",
    "jmo3",
    "jmo3p",
    "jmco2",
    "jmco2p",
    "jmco",
    "jmcop",
    "jmn2",
    "jmn2p",
    "NGB",
]
locvars_flt = [
    "taug",
    "pp",
    "corradj",
    "scalen2",
    "tauself",
    "taufor",
    "taun2",
    "fpl",
    "speccomb",
    "speccomb1",
    "fac001",
    "fac101",
    "fac201",
    "fac011",
    "fac111",
    "fac211",
    "fac000",
    "fac100",
    "fac200",
    "fac010",
    "fac110",
    "fac210",
    "specparm",
    "specparm1",
    "specparm_planck",
    "ratn2o",
    "ratco2",
]

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

outdict_gt4py = dict()

for var in outvars:
    outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

locdict_gt4py = dict()

for var in locvars:
    if var == "rfrate":
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_FLT, (nrates, 2))
        )
    elif var == "wx":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_maxxsec)
    elif var == "pwvcm" or var == "tem00":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
    elif var == "lcf1":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_BOOL)
    elif var == "colamt":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_maxgas)
    elif var in ["semiss", "secdiff", "expval", "pklay", "pklev"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var == "semiss0":
        locdict_gt4py[var] = create_storage_ones(backend, shape_nlp1, type_nbands)
    elif var in ["taucld", "tauaer", "tauliq", "tauice", "htrb"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbands)
    elif var in ["fracs", "tautot", "cldfmc"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptlw)
    elif var in ["ipseed", "jp", "jt", "jt1", "indself", "indfor", "indminor"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var in ["jp1", "tzint", "stempint", "indlev", "indlay", "tavelint"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var == "lcloudy":
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, (DTYPE_INT, (ngptlw))
        )
    elif var == "laytrop":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, bool)
    elif var == "index" or var == "ia":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

# Initialize local vars for taumol
for var in locvars_int:
    if var == "ib":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_INT)
    elif var == "NGB":
        locdict_gt4py[var] = create_storage_from_array(
            np.tile(np.array(ngb)[None, None, :], (npts, 1, 1)),
            backend,
            shape_2D,
            (DTYPE_INT, (ngptlw,)),
            default_origin=(0, 0),
        )
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)

for var in locvars_flt:
    if var == "taug":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptlw)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

locdict_gt4py["A0"] = create_storage_from_array(a0, backend, shape_nlp1, type_nbands)
locdict_gt4py["A1"] = create_storage_from_array(a1, backend, shape_nlp1, type_nbands)
locdict_gt4py["A2"] = create_storage_from_array(a2, backend, shape_nlp1, type_nbands)

# Read in lookup table data for cldprop calculations
ds = xr.open_dataset("../lookupdata/radlw_cldprlw_data.nc")

absliq1 = ds["absliq1"].data
absice0 = ds["absice0"].data
absice1 = ds["absice1"].data
absice2 = ds["absice2"].data
absice3 = ds["absice3"].data

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
cdfunc = np.tile(cdfunc.T[None, None, :, :], (npts, 1, 1, 1))
locdict_gt4py["cdfunc"] = create_storage_from_array(
    cdfunc, backend, shape_nlp1, type_ngptlw
)

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

print("Loading lookup table data . . .")
lookupdict_gt4py1 = loadlookupdata("kgb01")
lookupdict_gt4py2 = loadlookupdata("kgb02")
lookupdict_gt4py3 = loadlookupdata("kgb03")
lookupdict_gt4py4 = loadlookupdata("kgb04")
lookupdict_gt4py5 = loadlookupdata("kgb05")
lookupdict_gt4py6 = loadlookupdata("kgb06")
lookupdict_gt4py7 = loadlookupdata("kgb07")
lookupdict_gt4py8 = loadlookupdata("kgb08")
lookupdict_gt4py9 = loadlookupdata("kgb09")
lookupdict_gt4py10 = loadlookupdata("kgb10")
lookupdict_gt4py11 = loadlookupdata("kgb11")
lookupdict_gt4py12 = loadlookupdata("kgb12")
lookupdict_gt4py13 = loadlookupdata("kgb13")
lookupdict_gt4py14 = loadlookupdata("kgb14")
lookupdict_gt4py15 = loadlookupdata("kgb15")
lookupdict_gt4py16 = loadlookupdata("kgb16")
print("Done")
print(" ")

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
    locdict_gt4py["expval"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)


cldprop(
    locdict_gt4py["cldfrc"],
    locdict_gt4py["clwp"],
    locdict_gt4py["relw"],
    locdict_gt4py["ciwp"],
    locdict_gt4py["reiw"],
    locdict_gt4py["cda1"],
    locdict_gt4py["cda2"],
    locdict_gt4py["cda3"],
    locdict_gt4py["cda4"],
    locdict_gt4py["dz"],
    locdict_gt4py["cldfmc"],
    locdict_gt4py["taucld"],
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
    locdict_gt4py["cdfunc"],
    locdict_gt4py["tem1"],
    locdict_gt4py["lcf1"],
    locdict_gt4py["cldsum"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

setcoef(
    locdict_gt4py["pavel"],
    locdict_gt4py["tavel"],
    locdict_gt4py["tz"],
    locdict_gt4py["stemp"],
    locdict_gt4py["h2ovmr"],
    locdict_gt4py["colamt"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colbrd"],
    lookupdict_gt4py["totplnk"],
    lookupdict_gt4py["pref"],
    lookupdict_gt4py["preflog"],
    lookupdict_gt4py["tref"],
    lookupdict_gt4py["chi_mls"],
    delwave,
    locdict_gt4py["laytrop"],
    locdict_gt4py["pklay"],
    locdict_gt4py["pklev"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["rfrate"],
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
    locdict_gt4py["minorfrac"],
    locdict_gt4py["scaleminor"],
    locdict_gt4py["scaleminorn2"],
    locdict_gt4py["indminor"],
    locdict_gt4py["tzint"],
    locdict_gt4py["stempint"],
    locdict_gt4py["tavelint"],
    locdict_gt4py["indlay"],
    locdict_gt4py["indlev"],
    locdict_gt4py["tlyrfr"],
    locdict_gt4py["tlvlfr"],
    locdict_gt4py["jp1"],
    locdict_gt4py["plog"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

print(locdict_gt4py["laytrop"])

start0 = time.time()
start = time.time()
taugb01(
    locdict_gt4py["laytrop"],
    locdict_gt4py["pavel"],
    locdict_gt4py["colamt"],
    locdict_gt4py["colbrd"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["scaleminorn2"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py1["absa"],
    lookupdict_gt4py1["absb"],
    lookupdict_gt4py1["selfref"],
    lookupdict_gt4py1["forref"],
    lookupdict_gt4py1["fracrefa"],
    lookupdict_gt4py1["fracrefb"],
    lookupdict_gt4py1["ka_mn2"],
    lookupdict_gt4py1["kb_mn2"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["pp"],
    locdict_gt4py["corradj"],
    locdict_gt4py["scalen2"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["taun2"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

print("Running taugb02")
start = time.time()
taugb02(
    locdict_gt4py["laytrop"],
    locdict_gt4py["pavel"],
    locdict_gt4py["colamt"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py2["absa"],
    lookupdict_gt4py2["absb"],
    lookupdict_gt4py2["selfref"],
    lookupdict_gt4py2["forref"],
    lookupdict_gt4py2["fracrefa"],
    lookupdict_gt4py2["fracrefb"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["corradj"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

print("Running taugb03")
start = time.time()
taugb03(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py3["absa"],
    lookupdict_gt4py3["absb"],
    lookupdict_gt4py3["selfref"],
    lookupdict_gt4py3["forref"],
    lookupdict_gt4py3["fracrefa"],
    lookupdict_gt4py3["fracrefb"],
    lookupdict_gt4py3["ka_mn2o"],
    lookupdict_gt4py3["kb_mn2o"],
    lookupdict_gt4py3["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jmn2o"],
    locdict_gt4py["jmn2op"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    locdict_gt4py["ratn2o"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb04(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py4["absa"],
    lookupdict_gt4py4["absb"],
    lookupdict_gt4py4["selfref"],
    lookupdict_gt4py4["forref"],
    lookupdict_gt4py4["fracrefa"],
    lookupdict_gt4py4["fracrefb"],
    lookupdict_gt4py4["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb05(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["wx"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py5["absa"],
    lookupdict_gt4py5["absb"],
    lookupdict_gt4py5["selfref"],
    lookupdict_gt4py5["forref"],
    lookupdict_gt4py5["fracrefa"],
    lookupdict_gt4py5["fracrefb"],
    lookupdict_gt4py5["ka_mo3"],
    lookupdict_gt4py5["ccl4"],
    lookupdict_gt4py5["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["jmo3"],
    locdict_gt4py["jmo3p"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb06(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["wx"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py6["absa"],
    lookupdict_gt4py6["selfref"],
    lookupdict_gt4py6["forref"],
    lookupdict_gt4py6["fracrefa"],
    lookupdict_gt4py6["ka_mco2"],
    lookupdict_gt4py6["cfc11adj"],
    lookupdict_gt4py6["cfc12"],
    lookupdict_gt4py6["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["ratco2"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb07(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py7["absa"],
    lookupdict_gt4py7["absb"],
    lookupdict_gt4py7["selfref"],
    lookupdict_gt4py7["forref"],
    lookupdict_gt4py7["fracrefa"],
    lookupdict_gt4py7["fracrefb"],
    lookupdict_gt4py7["ka_mco2"],
    lookupdict_gt4py7["kb_mco2"],
    lookupdict_gt4py7["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jmco2"],
    locdict_gt4py["jmco2p"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    locdict_gt4py["ratco2"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb08(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["wx"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py8["absa"],
    lookupdict_gt4py8["absb"],
    lookupdict_gt4py8["selfref"],
    lookupdict_gt4py8["forref"],
    lookupdict_gt4py8["fracrefa"],
    lookupdict_gt4py8["fracrefb"],
    lookupdict_gt4py8["ka_mo3"],
    lookupdict_gt4py8["ka_mco2"],
    lookupdict_gt4py8["kb_mco2"],
    lookupdict_gt4py8["cfc12"],
    lookupdict_gt4py8["ka_mn2o"],
    lookupdict_gt4py8["kb_mn2o"],
    lookupdict_gt4py8["cfc22adj"],
    lookupdict_gt4py8["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["ratco2"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb09(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py9["absa"],
    lookupdict_gt4py9["absb"],
    lookupdict_gt4py9["selfref"],
    lookupdict_gt4py9["forref"],
    lookupdict_gt4py9["fracrefa"],
    lookupdict_gt4py9["fracrefb"],
    lookupdict_gt4py9["ka_mn2o"],
    lookupdict_gt4py9["kb_mn2o"],
    lookupdict_gt4py9["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jmco2"],
    locdict_gt4py["jmco2p"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    locdict_gt4py["ratn2o"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb10(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
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
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb11(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["indminor"],
    locdict_gt4py["scaleminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py11["absa"],
    lookupdict_gt4py11["absb"],
    lookupdict_gt4py11["selfref"],
    lookupdict_gt4py11["forref"],
    lookupdict_gt4py11["fracrefa"],
    lookupdict_gt4py11["fracrefb"],
    lookupdict_gt4py11["ka_mo2"],
    lookupdict_gt4py11["kb_mo2"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind0p"],
    locdict_gt4py["ind1"],
    locdict_gt4py["ind1p"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb12(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py12["absa"],
    lookupdict_gt4py12["selfref"],
    lookupdict_gt4py12["forref"],
    lookupdict_gt4py12["fracrefa"],
    lookupdict_gt4py12["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    locdict_gt4py["specparm_planck"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb13(
    locdict_gt4py["laytrop"],
    locdict_gt4py["coldry"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["indminor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py13["absa"],
    lookupdict_gt4py13["selfref"],
    lookupdict_gt4py13["forref"],
    lookupdict_gt4py13["fracrefa"],
    lookupdict_gt4py13["fracrefb"],
    lookupdict_gt4py13["ka_mco"],
    lookupdict_gt4py13["ka_mco2"],
    lookupdict_gt4py13["kb_mo3"],
    lookupdict_gt4py13["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["jmco"],
    locdict_gt4py["jmcop"],
    locdict_gt4py["jmco2"],
    locdict_gt4py["jmco2p"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    locdict_gt4py["ratco2"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb14(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py14["absa"],
    lookupdict_gt4py14["absb"],
    lookupdict_gt4py14["selfref"],
    lookupdict_gt4py14["forref"],
    lookupdict_gt4py14["fracrefa"],
    lookupdict_gt4py14["fracrefb"],
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
end = time.time()
print(f"Elapsed time = {end-start}")


start = time.time()
taugb15(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["colbrd"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["indminor"],
    locdict_gt4py["minorfrac"],
    locdict_gt4py["scaleminor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py15["absa"],
    lookupdict_gt4py15["selfref"],
    lookupdict_gt4py15["forref"],
    lookupdict_gt4py15["fracrefa"],
    lookupdict_gt4py15["ka_mn2"],
    lookupdict_gt4py15["chi_mls"],
    locdict_gt4py["ind0"],
    locdict_gt4py["ind1"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["indm"],
    locdict_gt4py["indmp"],
    locdict_gt4py["tauself"],
    locdict_gt4py["taufor"],
    locdict_gt4py["taun2"],
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["jmn2"],
    locdict_gt4py["jmn2p"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["fpl"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

start = time.time()
taugb16(
    locdict_gt4py["laytrop"],
    locdict_gt4py["colamt"],
    locdict_gt4py["rfrate"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["fracs"],
    locdict_gt4py["taug"],
    lookupdict_gt4py16["absa"],
    lookupdict_gt4py16["absb"],
    lookupdict_gt4py16["selfref"],
    lookupdict_gt4py16["forref"],
    lookupdict_gt4py16["fracrefa"],
    lookupdict_gt4py16["fracrefb"],
    lookupdict_gt4py16["chi_mls"],
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
    locdict_gt4py["js"],
    locdict_gt4py["js1"],
    locdict_gt4py["jpl"],
    locdict_gt4py["jplp"],
    locdict_gt4py["id000"],
    locdict_gt4py["id010"],
    locdict_gt4py["id100"],
    locdict_gt4py["id110"],
    locdict_gt4py["id200"],
    locdict_gt4py["id210"],
    locdict_gt4py["id001"],
    locdict_gt4py["id011"],
    locdict_gt4py["id101"],
    locdict_gt4py["id111"],
    locdict_gt4py["id201"],
    locdict_gt4py["id211"],
    locdict_gt4py["fpl"],
    locdict_gt4py["speccomb"],
    locdict_gt4py["speccomb1"],
    locdict_gt4py["fac000"],
    locdict_gt4py["fac100"],
    locdict_gt4py["fac200"],
    locdict_gt4py["fac010"],
    locdict_gt4py["fac110"],
    locdict_gt4py["fac210"],
    locdict_gt4py["fac001"],
    locdict_gt4py["fac101"],
    locdict_gt4py["fac201"],
    locdict_gt4py["fac011"],
    locdict_gt4py["fac111"],
    locdict_gt4py["fac211"],
    locdict_gt4py["specparm"],
    locdict_gt4py["specparm1"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
end = time.time()
print(f"Elapsed time = {end-start}")

combine_optical_depth(
    locdict_gt4py["NGB"],
    locdict_gt4py["ib"],
    locdict_gt4py["taug"],
    locdict_gt4py["tauaer"],
    locdict_gt4py["tautot"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)
print(" ")
end0 = time.time()
print(f"Total time taken = {end0 - start0}")
