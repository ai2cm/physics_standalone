import numpy as np
from numpy.lib.shape_base import _column_stack_dispatcher
import xarray as xr
import sys
import time

sys.path.insert(0, "..")
from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *
from stencils_sw_gt4py import *
from radsw.radsw_param import idxebc, nspa, nspb, ngs, ng, NGB, idxsfc
from radphysparam import iswrate
from phys_const import con_g, con_cp

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = False
validate = True

# Flag to say whether or not to do intermediate tests
do_subtests = False

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
    "exp_tbl",
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

locvars_taumol = [
    "id0",
    "id1",
    "ind01",
    "ind02",
    "ind03",
    "ind04",
    "ind11",
    "ind12",
    "ind13",
    "ind14",
    "inds",
    "indsp",
    "indf",
    "indfp",
    "fs",
    "js",
    "jsa",
    "colm1",
    "colm2",
    "sfluxzen",
    "taug",
    "taur",
]

locvars_spcvrtm = [
    "ztaus",
    "zssas",
    "zasys",
    "zldbt0",
    "zrefb",
    "zrefd",
    "ztrab",
    "ztrad",
    "ztdbt",
    "zldbt",
    "zfu",
    "zfd",
    "ztau1",
    "zssa1",
    "zasy1",
    "ztau0",
    "zssa0",
    "zasy0",
    "zasy3",
    "zssaw",
    "zasyw",
    "zgam1",
    "zgam2",
    "zgam3",
    "zgam4",
    "za1",
    "za2",
    "zb1",
    "zb2",
    "zrk",
    "zrk2",
    "zrp",
    "zrp1",
    "zrm1",
    "zrpp",
    "zrkg1",
    "zrkg3",
    "zrkg4",
    "zexp1",
    "zexm1",
    "zexp2",
    "zexm2",
    "zden1",
    "zexp3",
    "zexp4",
    "ze1r45",
    "ftind",
    "zsolar",
    "ztdbt0",
    "zr1",
    "zr2",
    "zr3",
    "zr4",
    "zr5",
    "zt1",
    "zt2",
    "zt3",
    "zf1",
    "zf2",
    "zrpp1",
    "zrupb",
    "zrupd",
    "ztdn",
    "zrdnd",
    "jb",
    "ib",
    "ibd",
    "itind",
    "zb11",
    "zb22",
    "fxupc",
    "fxdnc",
    "fxup0",
    "fxdn0",
    "ftoauc",
    "ftoau0",
    "ftoadc",
    "fsfcuc",
    "fsfcu0",
    "fsfcdc",
    "fsfcd0",
    "sfbmc",
    "sfdfc",
    "sfbm0",
    "sfdf0",
    "suvbfc",
    "suvbf0",
]

locvars_finalloop = [
    "flxuc",
    "flxdc",
    "flxu0",
    "flxd0",
    "fnet",
    "fnetc",
    "fnetb",
    "heatfac",
    "rfdelp",
]

temvars = ["tem0", "tem1", "tem2"]

outvars = [
    "upfxc_t",
    "dnfxc_t",
    "upfx0_t",
    "upfxc_s",
    "dnfxc_s",
    "upfx0_s",
    "dnfx0_s",
    "htswc",
    "htsw0",
    "htswb",
    "uvbf0",
    "uvbfc",
    "nirbm",
    "nirdf",
    "visbm",
    "visdf",
    "cldtausw",
]

outdict_gt4py = dict()
for var in outvars:
    if var in ["htswc", "cldtausw", "htsw0"]:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    elif var == "htswb":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    else:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)

indict = dict()

for var in invars:
    if var == "exp_tbl":
        tmp = serializer2.read(var, serializer2.savepoint["swrad-spcvrtm-input-000000"])
    else:
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
    elif var == "exp_tbl":
        indict[var] = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
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
    elif var == "exp_tbl":
        indict_gt4py[var] = create_storage_from_array(
            indict[var], backend, shape_nlp1, type_ntbmx
        )


locdict_gt4py = dict()
for var in locvars_firstloop:
    if var in ["tauae", "ssaae", "asyae"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    elif var in ["zcf0", "zcf1", "cosz1", "sntz1", "ssolar"]:
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

for var in locvars_taumol:
    if var in ["id0", "id1"]:
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlp1, type_nbandssw_int
        )
    elif var in ["colm1", "colm2", "fs"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)
    elif var == "sfluxzen":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, type_ngptsw)
    elif var in ["taug", "taur"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)

layind = np.arange(nlay, dtype=np.int32)
layind = np.insert(layind, 0, 0)
layind = np.tile(layind[None, None, :], (npts, 1, 1))
nspa = np.tile(np.array(nspa)[None, None, :], (npts, 1, 1))
nspb = np.tile(np.array(nspb)[None, None, :], (npts, 1, 1))
ngtmp = np.tile(np.array(ng)[None, None, :], (npts, 1, 1))
ngs = np.tile(np.array(ngs)[None, None, :], (npts, 1, 1))
locdict_gt4py["layind"] = create_storage_from_array(
    layind, backend, shape_nlp1, DTYPE_INT
)
locdict_gt4py["nspa"] = create_storage_from_array(
    nspa, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["nspb"] = create_storage_from_array(
    nspb, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ng"] = create_storage_from_array(
    ngtmp, backend, shape_2D, type_nbandssw_int
)
locdict_gt4py["ngs"] = create_storage_from_array(
    ngs, backend, shape_2D, type_nbandssw_int
)

for var in locvars_spcvrtm:
    if var in ["jb", "ib", "ibd", "itind"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_INT)
    elif var in ["zb11", "zb22"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, type_ngptsw)
    elif var in ["fxupc", "fxdnc", "fxup0", "fxdn0"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_nbdsw)
    elif var in ["sfbmc", "sfdfc", "sfbm0", "sfdf0"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, (DTYPE_FLT, (2,)))
    elif var in [
        "suvbfc",
        "suvbf0",
        "ftoadc",
        "ftoauc",
        "ftoau0",
        "fsfcuc",
        "fsfcu0",
        "fsfcdc",
        "fsfcd0",
    ]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, type_ngptsw)

NGB = np.tile(np.array(NGB)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["NGB"] = create_storage_from_array(NGB, backend, shape_nlp1, type_ngptsw)

idxsfc = np.tile(np.array(idxsfc)[None, None, None, :], (npts, 1, nlp1, 1))
locdict_gt4py["idxsfc"] = create_storage_from_array(
    idxsfc, backend, shape_nlp1, (DTYPE_FLT, (14,))
)


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
                lookupdict[var], backend, shape_nlp1, (ds[var].dtype, ds[var].shape)
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

for var in locvars_finalloop:
    if var == "heatfac":
        if iswrate == 1:
            locdict_gt4py[var] = con_g * 864.0 / con_cp
        else:
            locdict_gt4py[var] = con_g * 1.0e-2 / con_cp
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, DTYPE_FLT)

lookupdict_ref = loadlookupdata("sflux")
lookupdict16 = loadlookupdata("kgb16")
lookupdict17 = loadlookupdata("kgb17")
lookupdict18 = loadlookupdata("kgb18")
lookupdict19 = loadlookupdata("kgb19")
lookupdict20 = loadlookupdata("kgb20")
lookupdict21 = loadlookupdata("kgb21")
lookupdict22 = loadlookupdata("kgb22")
lookupdict23 = loadlookupdata("kgb23")
lookupdict24 = loadlookupdata("kgb24")
lookupdict25 = loadlookupdata("kgb25")
lookupdict26 = loadlookupdata("kgb26")
lookupdict27 = loadlookupdata("kgb27")
lookupdict28 = loadlookupdata("kgb28")
lookupdict29 = loadlookupdata("kgb29")

# Subtract one from indexing variables for Fortran -> Python conversion
lookupdict_ref["ix1"] = lookupdict_ref["ix1"] - 1
lookupdict_ref["ix2"] = lookupdict_ref["ix2"] - 1
lookupdict_ref["ibx"] = lookupdict_ref["ibx"] - 1

start = time.time()
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
    locdict_gt4py["pavel"],
    locdict_gt4py["tavel"],
    locdict_gt4py["h2ovmr"],
    locdict_gt4py["o3vmr"],
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
    outdict_gt4py["cldtausw"],
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

# Compute integer indices of troposphere height
laytropind = locdict_gt4py["laytrop"].view(np.ndarray).astype(int).squeeze().sum(axis=1)
locdict_gt4py["laytropind"] = create_storage_from_array(
    laytropind[:, None] - 1, backend, shape_2D, DTYPE_INT
)

taumolsetup(
    locdict_gt4py["colamt"],
    locdict_gt4py["jp"],
    locdict_gt4py["jt"],
    locdict_gt4py["jt1"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["laytropind"],
    indict_gt4py["idxday"],
    locdict_gt4py["sfluxzen"],
    locdict_gt4py["layind"],
    locdict_gt4py["nspa"],
    locdict_gt4py["nspb"],
    locdict_gt4py["ngs"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["fs"],
    locdict_gt4py["js"],
    locdict_gt4py["jsa"],
    locdict_gt4py["colm1"],
    locdict_gt4py["colm2"],
    lookupdict_ref["sfluxref01"],
    lookupdict_ref["sfluxref02"],
    lookupdict_ref["sfluxref03"],
    lookupdict_ref["layreffr"],
    lookupdict_ref["ix1"],
    lookupdict_ref["ix2"],
    lookupdict_ref["ibx"],
    lookupdict_ref["strrat"],
    lookupdict_ref["specwt"],
    lookupdict_ref["scalekur"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol16(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict16["selfref"],
    lookupdict16["forref"],
    lookupdict16["absa"],
    lookupdict16["absb"],
    lookupdict16["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol17(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict17["selfref"],
    lookupdict17["forref"],
    lookupdict17["absa"],
    lookupdict17["absb"],
    lookupdict17["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol18(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict18["selfref"],
    lookupdict18["forref"],
    lookupdict18["absa"],
    lookupdict18["absb"],
    lookupdict18["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol19(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict19["selfref"],
    lookupdict19["forref"],
    lookupdict19["absa"],
    lookupdict19["absb"],
    lookupdict19["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol20(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict20["selfref"],
    lookupdict20["forref"],
    lookupdict20["absa"],
    lookupdict20["absb"],
    lookupdict20["absch4"],
    lookupdict20["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol21(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict21["selfref"],
    lookupdict21["forref"],
    lookupdict21["absa"],
    lookupdict21["absb"],
    lookupdict21["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol22(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict22["selfref"],
    lookupdict22["forref"],
    lookupdict22["absa"],
    lookupdict22["absb"],
    lookupdict22["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol23(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict23["selfref"],
    lookupdict23["forref"],
    lookupdict23["absa"],
    lookupdict23["rayl"],
    lookupdict23["givfac"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol24(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict_ref["strrat"],
    lookupdict24["selfref"],
    lookupdict24["forref"],
    lookupdict24["absa"],
    lookupdict24["absb"],
    lookupdict24["rayla"],
    lookupdict24["raylb"],
    lookupdict24["abso3a"],
    lookupdict24["abso3b"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    locdict_gt4py["js"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol25(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    lookupdict25["absa"],
    lookupdict25["rayl"],
    lookupdict25["abso3a"],
    lookupdict25["abso3b"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol26(
    locdict_gt4py["colmol"],
    lookupdict26["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol27(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    lookupdict27["absa"],
    lookupdict27["absb"],
    lookupdict27["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol28(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    lookupdict_ref["strrat"],
    lookupdict28["absa"],
    lookupdict28["absb"],
    lookupdict28["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind03"],
    locdict_gt4py["ind04"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["ind13"],
    locdict_gt4py["ind14"],
    locdict_gt4py["js"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

taumol29(
    locdict_gt4py["colamt"],
    locdict_gt4py["colmol"],
    locdict_gt4py["fac00"],
    locdict_gt4py["fac01"],
    locdict_gt4py["fac10"],
    locdict_gt4py["fac11"],
    locdict_gt4py["laytrop"],
    locdict_gt4py["forfac"],
    locdict_gt4py["forfrac"],
    locdict_gt4py["indfor"],
    locdict_gt4py["selffac"],
    locdict_gt4py["selffrac"],
    locdict_gt4py["indself"],
    lookupdict29["forref"],
    lookupdict29["absa"],
    lookupdict29["absb"],
    lookupdict29["selfref"],
    lookupdict29["absh2o"],
    lookupdict29["absco2"],
    lookupdict29["rayl"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["id0"],
    locdict_gt4py["id1"],
    locdict_gt4py["ind01"],
    locdict_gt4py["ind02"],
    locdict_gt4py["ind11"],
    locdict_gt4py["ind12"],
    locdict_gt4py["inds"],
    locdict_gt4py["indsp"],
    locdict_gt4py["indf"],
    locdict_gt4py["indfp"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)


spcvrtm_clearsky(
    locdict_gt4py["ssolar"],
    locdict_gt4py["cosz1"],
    locdict_gt4py["sntz1"],
    locdict_gt4py["albbm"],
    locdict_gt4py["albdf"],
    locdict_gt4py["sfluxzen"],
    locdict_gt4py["cldfmc"],
    locdict_gt4py["zcf1"],
    locdict_gt4py["zcf0"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["tauae"],
    locdict_gt4py["ssaae"],
    locdict_gt4py["asyae"],
    locdict_gt4py["taucw"],
    locdict_gt4py["ssacw"],
    locdict_gt4py["asycw"],
    indict_gt4py["exp_tbl"],
    locdict_gt4py["ztaus"],
    locdict_gt4py["zssas"],
    locdict_gt4py["zasys"],
    locdict_gt4py["zldbt0"],
    locdict_gt4py["zrefb"],
    locdict_gt4py["zrefd"],
    locdict_gt4py["ztrab"],
    locdict_gt4py["ztrad"],
    locdict_gt4py["ztdbt"],
    locdict_gt4py["zldbt"],
    locdict_gt4py["zfu"],
    locdict_gt4py["zfd"],
    locdict_gt4py["ztau1"],
    locdict_gt4py["zssa1"],
    locdict_gt4py["zasy1"],
    locdict_gt4py["ztau0"],
    locdict_gt4py["zssa0"],
    locdict_gt4py["zasy0"],
    locdict_gt4py["zasy3"],
    locdict_gt4py["zssaw"],
    locdict_gt4py["zasyw"],
    locdict_gt4py["zgam1"],
    locdict_gt4py["zgam2"],
    locdict_gt4py["zgam3"],
    locdict_gt4py["zgam4"],
    locdict_gt4py["za1"],
    locdict_gt4py["za2"],
    locdict_gt4py["zb1"],
    locdict_gt4py["zb2"],
    locdict_gt4py["zrk"],
    locdict_gt4py["zrk2"],
    locdict_gt4py["zrp"],
    locdict_gt4py["zrp1"],
    locdict_gt4py["zrm1"],
    locdict_gt4py["zrpp"],
    locdict_gt4py["zrkg1"],
    locdict_gt4py["zrkg3"],
    locdict_gt4py["zrkg4"],
    locdict_gt4py["zexp1"],
    locdict_gt4py["zexm1"],
    locdict_gt4py["zexp2"],
    locdict_gt4py["zexm2"],
    locdict_gt4py["zden1"],
    locdict_gt4py["zexp3"],
    locdict_gt4py["zexp4"],
    locdict_gt4py["ze1r45"],
    locdict_gt4py["ftind"],
    locdict_gt4py["zsolar"],
    locdict_gt4py["ztdbt0"],
    locdict_gt4py["zr1"],
    locdict_gt4py["zr2"],
    locdict_gt4py["zr3"],
    locdict_gt4py["zr4"],
    locdict_gt4py["zr5"],
    locdict_gt4py["zt1"],
    locdict_gt4py["zt2"],
    locdict_gt4py["zt3"],
    locdict_gt4py["zf1"],
    locdict_gt4py["zf2"],
    locdict_gt4py["zrpp1"],
    locdict_gt4py["zrupd"],
    locdict_gt4py["zrupb"],
    locdict_gt4py["ztdn"],
    locdict_gt4py["zrdnd"],
    locdict_gt4py["zb11"],
    locdict_gt4py["zb22"],
    locdict_gt4py["jb"],
    locdict_gt4py["ib"],
    locdict_gt4py["ibd"],
    locdict_gt4py["NGB"],
    locdict_gt4py["idxsfc"],
    locdict_gt4py["itind"],
    locdict_gt4py["fxupc"],
    locdict_gt4py["fxdnc"],
    locdict_gt4py["fxup0"],
    locdict_gt4py["fxdn0"],
    locdict_gt4py["ftoauc"],
    locdict_gt4py["ftoau0"],
    locdict_gt4py["ftoadc"],
    locdict_gt4py["fsfcuc"],
    locdict_gt4py["fsfcu0"],
    locdict_gt4py["fsfcdc"],
    locdict_gt4py["fsfcd0"],
    locdict_gt4py["sfbmc"],
    locdict_gt4py["sfdfc"],
    locdict_gt4py["sfbm0"],
    locdict_gt4py["sfdf0"],
    locdict_gt4py["suvbfc"],
    locdict_gt4py["suvbf0"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

spcvrtm_allsky(
    locdict_gt4py["ssolar"],
    locdict_gt4py["cosz1"],
    locdict_gt4py["sntz1"],
    locdict_gt4py["albbm"],
    locdict_gt4py["albdf"],
    locdict_gt4py["sfluxzen"],
    locdict_gt4py["cldfmc"],
    locdict_gt4py["zcf1"],
    locdict_gt4py["zcf0"],
    locdict_gt4py["taug"],
    locdict_gt4py["taur"],
    locdict_gt4py["tauae"],
    locdict_gt4py["ssaae"],
    locdict_gt4py["asyae"],
    locdict_gt4py["taucw"],
    locdict_gt4py["ssacw"],
    locdict_gt4py["asycw"],
    indict_gt4py["exp_tbl"],
    locdict_gt4py["ztaus"],
    locdict_gt4py["zssas"],
    locdict_gt4py["zasys"],
    locdict_gt4py["zldbt0"],
    locdict_gt4py["zrefb"],
    locdict_gt4py["zrefd"],
    locdict_gt4py["ztrab"],
    locdict_gt4py["ztrad"],
    locdict_gt4py["ztdbt"],
    locdict_gt4py["zldbt"],
    locdict_gt4py["zfu"],
    locdict_gt4py["zfd"],
    locdict_gt4py["ztau1"],
    locdict_gt4py["zssa1"],
    locdict_gt4py["zasy1"],
    locdict_gt4py["ztau0"],
    locdict_gt4py["zssa0"],
    locdict_gt4py["zasy0"],
    locdict_gt4py["zasy3"],
    locdict_gt4py["zssaw"],
    locdict_gt4py["zasyw"],
    locdict_gt4py["zgam1"],
    locdict_gt4py["zgam2"],
    locdict_gt4py["zgam3"],
    locdict_gt4py["zgam4"],
    locdict_gt4py["za1"],
    locdict_gt4py["za2"],
    locdict_gt4py["zb1"],
    locdict_gt4py["zb2"],
    locdict_gt4py["zrk"],
    locdict_gt4py["zrk2"],
    locdict_gt4py["zrp"],
    locdict_gt4py["zrp1"],
    locdict_gt4py["zrm1"],
    locdict_gt4py["zrpp"],
    locdict_gt4py["zrkg1"],
    locdict_gt4py["zrkg3"],
    locdict_gt4py["zrkg4"],
    locdict_gt4py["zexp1"],
    locdict_gt4py["zexm1"],
    locdict_gt4py["zexp2"],
    locdict_gt4py["zexm2"],
    locdict_gt4py["zden1"],
    locdict_gt4py["zexp3"],
    locdict_gt4py["zexp4"],
    locdict_gt4py["ze1r45"],
    locdict_gt4py["ftind"],
    locdict_gt4py["zsolar"],
    locdict_gt4py["ztdbt0"],
    locdict_gt4py["zr1"],
    locdict_gt4py["zr2"],
    locdict_gt4py["zr3"],
    locdict_gt4py["zr4"],
    locdict_gt4py["zr5"],
    locdict_gt4py["zt1"],
    locdict_gt4py["zt2"],
    locdict_gt4py["zt3"],
    locdict_gt4py["zf1"],
    locdict_gt4py["zf2"],
    locdict_gt4py["zrpp1"],
    locdict_gt4py["zrupd"],
    locdict_gt4py["zrupb"],
    locdict_gt4py["ztdn"],
    locdict_gt4py["zrdnd"],
    locdict_gt4py["zb11"],
    locdict_gt4py["zb22"],
    locdict_gt4py["jb"],
    locdict_gt4py["ib"],
    locdict_gt4py["ibd"],
    locdict_gt4py["NGB"],
    locdict_gt4py["idxsfc"],
    locdict_gt4py["itind"],
    locdict_gt4py["fxupc"],
    locdict_gt4py["fxdnc"],
    locdict_gt4py["fxup0"],
    locdict_gt4py["fxdn0"],
    locdict_gt4py["ftoauc"],
    locdict_gt4py["ftoau0"],
    locdict_gt4py["ftoadc"],
    locdict_gt4py["fsfcuc"],
    locdict_gt4py["fsfcu0"],
    locdict_gt4py["fsfcdc"],
    locdict_gt4py["fsfcd0"],
    locdict_gt4py["sfbmc"],
    locdict_gt4py["sfdfc"],
    locdict_gt4py["sfbm0"],
    locdict_gt4py["sfdf0"],
    locdict_gt4py["suvbfc"],
    locdict_gt4py["suvbf0"],
    domain=shape_nlp1,
    origin=default_origin,
    validate_args=validate,
)

finalloop(
    indict_gt4py["idxday"],
    indict_gt4py["delp"],
    locdict_gt4py["fxupc"],
    locdict_gt4py["fxdnc"],
    locdict_gt4py["fxup0"],
    locdict_gt4py["fxdn0"],
    locdict_gt4py["suvbf0"],
    locdict_gt4py["suvbfc"],
    locdict_gt4py["sfbmc"],
    locdict_gt4py["sfdfc"],
    locdict_gt4py["ftoauc"],
    locdict_gt4py["ftoadc"],
    locdict_gt4py["ftoau0"],
    locdict_gt4py["fsfcuc"],
    locdict_gt4py["fsfcdc"],
    locdict_gt4py["fsfcu0"],
    locdict_gt4py["fsfcd0"],
    outdict_gt4py["upfxc_t"],
    outdict_gt4py["dnfxc_t"],
    outdict_gt4py["upfx0_t"],
    outdict_gt4py["upfxc_s"],
    outdict_gt4py["dnfxc_s"],
    outdict_gt4py["upfx0_s"],
    outdict_gt4py["dnfx0_s"],
    outdict_gt4py["htswc"],
    outdict_gt4py["htsw0"],
    outdict_gt4py["htswb"],
    outdict_gt4py["uvbf0"],
    outdict_gt4py["uvbfc"],
    outdict_gt4py["nirbm"],
    outdict_gt4py["nirdf"],
    outdict_gt4py["visbm"],
    outdict_gt4py["visdf"],
    locdict_gt4py["rfdelp"],
    locdict_gt4py["fnet"],
    locdict_gt4py["fnetc"],
    locdict_gt4py["fnetb"],
    locdict_gt4py["flxuc"],
    locdict_gt4py["flxdc"],
    locdict_gt4py["flxu0"],
    locdict_gt4py["flxd0"],
    locdict_gt4py["heatfac"],
)

end = time.time()
print(f"Elapsed time = {end-start}")

outdict_final = dict()
valdict_final = dict()

for var in outvars:
    if var in ["htswc", "cldtausw", "htsw0"]:
        outdict_final[var] = outdict_gt4py[var][:, :, 1:].view(np.ndarray).squeeze()
    elif var == "htswb":
        if lhswb:
            outdict_final[var] = (
                outdict_gt4py[var][:, :, 1:, :].view(np.ndarray).squeeze()
            )
    else:
        outdict_final[var] = outdict_gt4py[var].view(np.ndarray).squeeze()

    if lhswb:
        valdict_final[var] = serializer.read(
            var, serializer.savepoint["swrad-out-000000"]
        )
    else:
        if var != "htswb":
            valdict_final[var] = serializer.read(
                var, serializer.savepoint["swrad-out-000000"]
            )

compare_data(outdict_final, valdict_final)

if do_subtests:
    # Run tests for output of first loop
    valdict_firstloop = dict()
    outdict_firstloop = dict()

    for var in locvars_firstloop:
        valdict_firstloop[var] = serializer2.read(
            var, serializer2.savepoint["swrad-firstloop-output-000000"]
        )
        if var in ["albbm", "albdf", "tem0", "tem1", "tem2"]:
            outdict_firstloop[var] = (
                locdict_gt4py[var][:, :, 0, ...].view(np.ndarray).squeeze()
            )
        elif var in ["zcf1", "zcf0", "cosz1", "sntz1", "ssolar"]:
            outdict_firstloop[var] = locdict_gt4py[var].view(np.ndarray).squeeze()
        else:
            outdict_firstloop[var] = (
                locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()
            )

    compare_data(valdict_firstloop, outdict_firstloop)

    # Run test for cldprop output
    valdict_cldprop = dict()
    outdict_cldprop = dict()

    outvars_cldprop = ["cldfmc", "taucw", "ssacw", "asycw", "cldfrc"]

    for var in outvars_cldprop:
        valdict_cldprop[var] = serializer2.read(
            var, serializer2.savepoint["swrad-cldprop-output-000000"]
        )
        outdict_cldprop[var] = (
            locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()
        )

    compare_data(outdict_cldprop, valdict_cldprop)

    # Run tests for output of setcoef
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

    # Test output from taumol
    outvars_taumol = ["taug", "taur", "sfluxzen"]

    outdict_taumol = dict()
    valdict_taumol = dict()

    for var in outvars_taumol:
        if var == "sfluxzen":
            outdict_taumol[var] = locdict_gt4py[var].view(np.ndarray).squeeze()
        else:
            outdict_taumol[var] = (
                locdict_gt4py[var][:, :, 1:, ...].view(np.ndarray).squeeze()
            )

        valdict_taumol[var] = serializer2.read(
            var, serializer2.savepoint["swrad-taumol-output-000000"]
        )

    compare_data(outdict_taumol, valdict_taumol)

    # Test output for spcvrtm
    outvars_spcvrtm = [
        "fxupc",
        "fxdnc",
        "fxup0",
        "fxdn0",
        "ftoauc",
        "ftoau0",
        "ftoadc",
        "fsfcuc",
        "fsfcu0",
        "fsfcdc",
        "fsfcd0",
        "sfbmc",
        "sfdfc",
        "sfbm0",
        "sfdf0",
        "suvbfc",
        "suvbf0",
    ]

    outdict_spcvrtm = dict()
    valdict_spcvrtm = dict()
    for var in outvars_spcvrtm:
        outdict_spcvrtm[var] = locdict_gt4py[var][:, ...].view(np.ndarray).squeeze()
        valdict_spcvrtm[var] = serializer2.read(
            var, serializer2.savepoint["swrad-spcvrtm-output-000000"]
        )

    compare_data(outdict_spcvrtm, valdict_spcvrtm)
