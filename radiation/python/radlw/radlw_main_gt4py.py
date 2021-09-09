import numpy as np
import xarray as xr
import sys
import time

sys.path.insert(0, "..")
from radphysparam import (
    ilwrgas as ilwrgas,
    icldflg as icldflg,
    ilwcliq as ilwcliq,
    ilwrate as ilwrate,
    ilwcice as ilwcice,
)
from radlw.radlw_param import *
from phys_const import con_g, con_cp, con_amd, con_amw, con_amo3
from util import (
    create_storage_from_array,
    create_storage_zeros,
    create_storage_ones,
    loadlookupdata,
    compare_data,
    view_gt4py_storage,
)
from config import *
from stencils_gt4py import *

import serialbox as ser


class RadLWClass:
    VTAGLW = "NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82"
    expeps = 1.0e-20

    bpade = 1.0 / 0.278
    eps = 1.0e-6
    oneminus = 1.0 - eps
    cldmin = 1.0e-80
    stpfac = 296.0 / 1013.0
    wtdiff = 0.5
    tblint = ntbl

    ipsdlw0 = ngptlw

    amdw = con_amd / con_amw
    amdo3 = con_amd / con_amo3

    nspa = [1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9]
    nspb = [1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

    delwave = np.array(
        [
            340.0,
            150.0,
            130.0,
            70.0,
            120.0,
            160.0,
            100.0,
            100.0,
            210.0,
            90.0,
            320.0,
            280.0,
            170.0,
            130.0,
            220.0,
            650.0,
        ]
    )

    a0 = [
        1.66,
        1.55,
        1.58,
        1.66,
        1.54,
        1.454,
        1.89,
        1.33,
        1.668,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
    ]
    a1 = [
        0.00,
        0.25,
        0.22,
        0.00,
        0.13,
        0.446,
        -0.10,
        0.40,
        -0.006,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
    a2 = [
        0.00,
        -12.0,
        -11.7,
        0.00,
        -0.72,
        -0.243,
        0.19,
        -0.062,
        0.414,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]

    A0 = create_storage_from_array(a0, backend, shape_nlp1, type_nbands)
    A1 = create_storage_from_array(a1, backend, shape_nlp1, type_nbands)
    A2 = create_storage_from_array(a2, backend, shape_nlp1, type_nbands)

    NGB = create_storage_from_array(
        np.tile(np.array(ngb)[None, None, :], (npts, 1, 1)),
        backend,
        shape_2D,
        (DTYPE_INT, (ngptlw,)),
        default_origin=(0, 0),
    )

    def __init__(self, me, iovrlw, isubclw):
        self.lhlwb = False
        self.lhlw0 = False
        self.lflxprf = False

        self.semiss0 = np.ones(nbands)

        self.iovrlw = iovrlw
        self.isubclw = isubclw

        self.exp_tbl = np.zeros(ntbl + 1)
        self.tau_tbl = np.zeros(ntbl + 1)
        self.tfn_tbl = np.zeros(ntbl + 1)

        expeps = 1e-20

        if self.iovrlw < 0 or self.iovrlw > 3:
            print(
                f"  *** Error in specification of cloud overlap flag",
                f" IOVRLW={self.iovrlw}, in RLWINIT !!",
            )
        elif iovrlw >= 2 and isubclw == 0:
            if me == 0:
                print(
                    f"  *** IOVRLW={self.iovrlw} is not available for",
                    " ISUBCLW=0 setting!!",
                )
                print("      The program uses maximum/random overlap instead.")

        if me == 0:
            print(f"- Using AER Longwave Radiation, Version: {self.VTAGLW}")

            if ilwrgas > 0:
                print(
                    "   --- Include rare gases N2O, CH4, O2, CFCs ", "absorptions in LW"
                )
            else:
                print("   --- Rare gases effect is NOT included in LW")

            if self.isubclw == 0:
                print(
                    "   --- Using standard grid average clouds, no ",
                    "   sub-column clouds approximation applied",
                )
            elif self.isubclw == 1:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with a prescribed sequence of permutaion seeds",
                )
            elif self.isubclw == 2:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with provided input array of permutation seeds",
                )
            else:
                print(
                    f"  *** Error in specification of sub-column cloud ",
                    f" control flag isubclw = {self.isubclw}!!",
                )

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
            print(
                "*** Model cloud scheme inconsistent with LW",
                "radiation cloud radiative property setup !!",
            )

        #  --- ...  setup constant factors for flux and heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        pival = 2.0 * np.arcsin(1.0)
        self.fluxfac = pival * 2.0e4

        if ilwrate == 1:
            self.heatfac = con_g * 864.0 / con_cp  #   (in k/day)
        else:
            self.heatfac = con_g * 1.0e-2 / con_cp  #   (in k/second)

        #  --- ...  compute lookup tables for transmittance, tau transition
        #           function, and clear sky tau (for the cloudy sky radiative
        #           transfer).  tau is computed as a function of the tau
        #           transition function, transmittance is calculated as a
        #           function of tau, and the tau transition function is
        #           calculated using the linear in tau formulation at values of
        #           tau above 0.01.  tf is approximated as tau/6 for tau < 0.01.
        #           all tables are computed at intervals of 0.001.  the inverse
        #           of the constant used in the pade approximation to the tau
        #           transition function is set to b.

        self.tau_tbl[0] = 0.0
        self.exp_tbl[0] = 1.0
        self.tfn_tbl[0] = 0.0

        self.tau_tbl[ntbl] = 1.0e10
        self.exp_tbl[ntbl] = expeps
        self.tfn_tbl[ntbl] = 1.0

        explimit = int(np.floor(-np.log(np.finfo(float).tiny)))

        for i in range(1, ntbl):
            tfn = (i) / (ntbl - i)
            self.tau_tbl[i] = self.bpade * tfn
            if self.tau_tbl[i] >= explimit:
                self.exp_tbl[i] = expeps
            else:
                self.exp_tbl[i] = np.exp(-self.tau_tbl[i])

            if self.tau_tbl[i] < 0.06:
                self.tfn_tbl[i] = self.tau_tbl[i] / 6.0
            else:
                self.tfn_tbl[i] = 1.0 - 2.0 * (
                    (1.0 / self.tau_tbl[i])
                    - (self.exp_tbl[i] / (1.0 - self.exp_tbl[i]))
                )

        self.exp_tbl = np.tile(self.exp_tbl[None, None, None, :], (npts, 1, nlp1, 1))
        self.tau_tbl = np.tile(self.tau_tbl[None, None, None, :], (npts, 1, nlp1, 1))
        self.tfn_tbl = np.tile(self.tfn_tbl[None, None, None, :], (npts, 1, nlp1, 1))

        self.exp_tbl = create_storage_from_array(
            self.exp_tbl, backend, shape_nlp1, type_ntbmx
        )
        self.tau_tbl = create_storage_from_array(
            self.tau_tbl, backend, shape_nlp1, type_ntbmx
        )
        self.tfn_tbl = create_storage_from_array(
            self.tfn_tbl, backend, shape_nlp1, type_ntbmx
        )

        self._load_lookup_table_data()

    def return_initdata(self):
        """
        Return output of init routine for validation against Fortran
        """

        outdict = {
            "semiss0": self.semiss0,
            "fluxfac": self.fluxfac,
            "heatfac": self.heatfac,
            "exp_tbl": self.exp_tbl,
            "tau_tbl": self.tau_tbl,
            "tfn_tbl": self.tfn_tbl,
        }
        return outdict

    def create_input_data(self, tile):
        """
        Load input data from serialized Fortran model output and transform into
        gt4py storages. Also creates the necessary local variables as gt4py storages
        """

        ddir = "../../fortran/data/LW"
        self.serializer = ser.Serializer(
            ser.OpenModeKind.Read, ddir, "Generator_rank" + str(tile)
        )

        ddir2 = "../../fortran/radlw/dump"
        self.serializer2 = ser.Serializer(
            ser.OpenModeKind.Read, ddir2, "Serialized_rank" + str(tile)
        )

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
            "htlwc",
            "htlw0",
            "cldtaulw",
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
            "htrb",
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
            "htrb",
            "taucld",
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

        locvars_rtrnmc = [
            "clrurad",
            "clrdrad",
            "toturad",
            "totdrad",
            "gassrcu",
            "totsrcu",
            "trngas",
            "efclrfr",
            "rfdelp",
            "fnet",
            "fnetc",
            "totsrcd",
            "gassrcd",
            "tblind",
            "odepth",
            "odtot",
            "odcld",
            "atrtot",
            "atrgas",
            "reflct",
            "totfac",
            "gasfac",
            "flxfac",
            "plfrac",
            "blay",
            "bbdgas",
            "bbdtot",
            "bbugas",
            "bbutot",
            "dplnku",
            "dplnkd",
            "radtotu",
            "radclru",
            "radtotd",
            "radclrd",
            "rad0",
            "clfm",
            "trng",
            "gasu",
            "itgas",
            "ittot",
            "ib",
        ]

        indict = dict()
        for var in invars:
            tmp = self.serializer.read(
                var, self.serializer.savepoint["lwrad-in-000000"]
            )

            if var in ["semis", "icsdlw", "tsfg", "de_lgth"]:
                # These fields are shape npts, tile to be 3D fields
                indict[var] = np.tile(tmp[:, None, None], (1, 1, nlp1))
            elif var == "faerlw":
                # This is shape(npts, nlay, nbands, 3).
                # Pad k axis with 0 and tile to give them the extra
                # horizontal dimension
                tmp2 = np.insert(tmp, 0, 0, axis=1)
                indict[var] = np.tile(tmp2[:, None, :, :, :], (1, 1, 1, 1, 1))
            elif var == "gasvmr" or var == "clouds":
                # These fields are size (npts, nlay, 9).
                # Pad k axis with 0 and tile to give them the extra
                # horizontal dimension
                tmp2 = np.insert(tmp, 0, 0, axis=1)
                indict[var] = np.tile(tmp2[:, None, :, :], (1, 1, 1, 1))
            elif var in ["plyr", "tlyr", "qlyr", "olyr", "dz", "delp"]:
                # These fields are size (npts, nlay).
                # Pad k axis with 0 and tile to give them the extra
                # horizontal dimension
                tmp2 = np.insert(tmp, 0, 0, axis=1)
                indict[var] = np.tile(tmp2[:, None, :], (1, 1, 1))
            elif var in ["plvl", "tlvl"]:
                # These fields are size (npts, nlp1).
                # Tile to give them the extra
                # horizontal dimension
                indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
            else:
                # Otherwise input is a scalar, grab from array.
                indict[var] = tmp[0]

        indict_gt4py = dict()

        for var in invars:
            if var == "faerlw":
                indict_gt4py[var] = create_storage_from_array(
                    indict[var], backend, shape_nlp1, (DTYPE_FLT, (nbands, 3))
                )
            elif var == "gasvmr" or var == "clouds":
                indict_gt4py[var] = create_storage_from_array(
                    indict[var],
                    backend,
                    shape_nlp1,
                    (DTYPE_FLT, (indict[var].shape[3],)),
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
            if var in ["htlwc", "htlw0", "cldtaulw"]:
                outdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_FLT
                )
            else:
                outdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)

        locdict_gt4py = dict()

        for var in locvars:
            if var == "rfrate":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, (DTYPE_FLT, (nrates, 2))
                )
            elif var == "wx":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_maxxsec
                )
            elif var == "pwvcm" or var == "tem00":
                locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_FLT)
            elif var == "lcf1":
                locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_BOOL)
            elif var == "colamt":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_maxgas
                )
            elif var in [
                "semiss",
                "secdiff",
                "pklay",
                "pklev",
                "htrb",
            ]:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_nbands
                )
            elif var == "semiss0":
                locdict_gt4py[var] = create_storage_ones(
                    backend, shape_nlp1, type_nbands
                )
            elif var in ["taucld", "tauaer", "tauliq", "tauice", "htrb"]:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_nbands
                )
            elif var in ["fracs", "tautot", "cldfmc"]:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_ngptlw
                )
            elif var in ["ipseed", "jp", "jt", "jt1", "indself", "indfor", "indminor"]:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_INT
                )
            elif var in ["jp1", "tzint", "stempint", "indlev", "indlay", "tavelint"]:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_INT
                )
            elif var == "lcloudy":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, (DTYPE_INT, (ngptlw))
                )
            elif var == "laytrop":
                locdict_gt4py[var] = create_storage_zeros(backend, shape_nlp1, bool)
            elif var == "index" or var == "ia":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_INT
                )
            else:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_FLT
                )

        # Initialize local vars for taumol
        for var in locvars_int:
            if var == "ib":
                locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, DTYPE_INT)
            else:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_INT
                )

        for var in locvars_flt:
            if var == "taug":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_ngptlw
                )
            else:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_FLT
                )

        # Initialize local vars for rtrnmc
        for var in locvars_rtrnmc:
            if var[-3:] == "rad":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_nbands
                )
            elif var == "fnet" or var == "fnetc" or var == "rfdelp":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, DTYPE_FLT
                )
            elif var == "itgas" or var == "ittot":
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, (np.int32, (ngptlw,))
                )
            elif var == "ib":
                locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, np.int32)
            else:
                locdict_gt4py[var] = create_storage_zeros(
                    backend, shape_nlp1, type_ngptlw
                )

        self.indict_gt4py = indict_gt4py
        self.locdict_gt4py = locdict_gt4py
        self.outdict_gt4py = outdict_gt4py
        self.outvars = outvars

    def _load_lookup_table_data(self):
        """
        Read in lookup table data from netcdf data that has been serialized out from
        radlw_datatb.F
        """

        ds = xr.open_dataset("../lookupdata/radlw_cldprlw_data.nc")

        cldprop_types = {
            "absliq1": {"ctype": (DTYPE_FLT, (58, nbands)), "data": ds["absliq1"].data},
            "absice0": {"ctype": (DTYPE_FLT, (2,)), "data": ds["absice0"].data},
            "absice1": {"ctype": (DTYPE_FLT, (2, 5)), "data": ds["absice1"].data},
            "absice2": {"ctype": (DTYPE_FLT, (43, nbands)), "data": ds["absice2"].data},
            "absice3": {"ctype": (DTYPE_FLT, (46, nbands)), "data": ds["absice3"].data},
            "ipat": {"ctype": (DTYPE_INT, (nbands,)), "data": ipat},
        }

        lookupdict_gt4py = dict()

        for name, info in cldprop_types.items():
            lookupdict_gt4py[name] = create_storage_from_array(
                info["data"], backend, shape_nlp1, info["ctype"]
            )

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

        delwave = np.tile(self.delwave[None, None, None, :], (npts, 1, nlp1, 1))
        delwave = create_storage_from_array(delwave, backend, shape_nlp1, type_nbands)
        lookupdict_gt4py["delwave"] = delwave

        print("Loading lookup table data . . .")
        self.lookupdict_gt4py1 = loadlookupdata("kgb01")
        self.lookupdict_gt4py2 = loadlookupdata("kgb02")
        self.lookupdict_gt4py3 = loadlookupdata("kgb03")
        self.lookupdict_gt4py4 = loadlookupdata("kgb04")
        self.lookupdict_gt4py5 = loadlookupdata("kgb05")
        self.lookupdict_gt4py6 = loadlookupdata("kgb06")
        self.lookupdict_gt4py7 = loadlookupdata("kgb07")
        self.lookupdict_gt4py8 = loadlookupdata("kgb08")
        self.lookupdict_gt4py9 = loadlookupdata("kgb09")
        self.lookupdict_gt4py10 = loadlookupdata("kgb10")
        self.lookupdict_gt4py11 = loadlookupdata("kgb11")
        self.lookupdict_gt4py12 = loadlookupdata("kgb12")
        self.lookupdict_gt4py13 = loadlookupdata("kgb13")
        self.lookupdict_gt4py14 = loadlookupdata("kgb14")
        self.lookupdict_gt4py15 = loadlookupdata("kgb15")
        self.lookupdict_gt4py16 = loadlookupdata("kgb16")
        print("Done")
        print(" ")

        self.lookupdict_gt4py = lookupdict_gt4py

    def _load_random_numbers(self, tile):
        """
        Read in 2-D array of random numbers used in mcica_subcol, this will change
        in the future once there is a solution for the RNG in python/gt4py

        This serialized set of random numbers will be used for testing, and the python
        RNG for running the model.

        rand2d is shape (npts, ngptlw*nlay), and I will reshape it to (npts, 1, nlp1, ngptlw)
        - First reshape to (npts, ngptlw, nlay)
        - Second pad k axis with one zero
        - Third switch order of k and data axes
        """
        ds = xr.open_dataset("../lookupdata/rand2d_tile" + str(tile) + "_lw.nc")
        rand2d = ds["rand2d"][:, :].data
        cdfunc = np.zeros((npts, ngptlw, nlay))
        for n in range(npts):
            cdfunc[n, :, :] = np.reshape(rand2d[n, :], (ngptlw, nlay), order="C")
        cdfunc = np.insert(cdfunc, 0, 0, axis=2)
        cdfunc = np.transpose(cdfunc, (0, 2, 1))

        cdfunc = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))
        self.lookupdict_gt4py["cdfunc"] = create_storage_from_array(
            cdfunc, backend, shape_nlp1, type_ngptlw
        )

    def lwrad(self, tile, do_subtest=False):
        """
        Run the main longwave radiation scheme

        Requires create_input_data to have been run before calling
        Currently uses serialized random number arrays in cldprop

        Inputs:
        - tile: integer denoting current tile
        - do_subtest: flag to test individual stencil outputs
        """

        start0 = time.time()
        firstloop(
            self.indict_gt4py["plyr"],
            self.indict_gt4py["plvl"],
            self.indict_gt4py["tlyr"],
            self.indict_gt4py["tlvl"],
            self.indict_gt4py["qlyr"],
            self.indict_gt4py["olyr"],
            self.indict_gt4py["gasvmr"],
            self.indict_gt4py["clouds"],
            self.indict_gt4py["icsdlw"],
            self.indict_gt4py["faerlw"],
            self.indict_gt4py["semis"],
            self.indict_gt4py["tsfg"],
            self.indict_gt4py["dz"],
            self.indict_gt4py["delp"],
            self.indict_gt4py["de_lgth"],
            self.locdict_gt4py["cldfrc"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.locdict_gt4py["delp"],
            self.locdict_gt4py["dz"],
            self.locdict_gt4py["h2ovmr"],
            self.locdict_gt4py["o3vmr"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["tauaer"],
            self.locdict_gt4py["semiss0"],
            self.locdict_gt4py["semiss"],
            self.locdict_gt4py["tem11"],
            self.locdict_gt4py["tem22"],
            self.locdict_gt4py["tem00"],
            self.locdict_gt4py["summol"],
            self.locdict_gt4py["pwvcm"],
            self.locdict_gt4py["clwp"],
            self.locdict_gt4py["relw"],
            self.locdict_gt4py["ciwp"],
            self.locdict_gt4py["reiw"],
            self.locdict_gt4py["cda1"],
            self.locdict_gt4py["cda2"],
            self.locdict_gt4py["cda3"],
            self.locdict_gt4py["cda4"],
            self.locdict_gt4py["secdiff"],
            self.A0,
            self.A1,
            self.A2,
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:

            outvars_firstloop = [
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

            outdict_firstloop = dict()
            valdict_firstloop = dict()
            for var in outvars_firstloop:
                if var == "cldfrc":
                    tmp = self.locdict_gt4py[var].view(np.ndarray).squeeze()
                    outdict_firstloop[var] = np.append(tmp, np.zeros((npts, 1)), axis=1)
                elif var == "pwvcm":
                    outdict_firstloop[var] = (
                        self.locdict_gt4py[var].view(np.ndarray).squeeze()
                    )
                elif var == "taucld" or var == "tauaer":
                    tmp = (
                        self.locdict_gt4py[var][:, :, 1:, :].view(np.ndarray).squeeze()
                    )
                    outdict_firstloop[var] = np.transpose(tmp, [0, 2, 1])
                elif var == "semiss" or var == "secdiff":
                    outdict_firstloop[var] = (
                        self.locdict_gt4py[var][:, :, 1, :].view(np.ndarray).squeeze()
                    )
                else:
                    outdict_firstloop[var] = (
                        self.locdict_gt4py[var][:, :, 1:].view(np.ndarray).squeeze()
                    )

                valdict_firstloop[var] = self.serializer2.read(
                    var, self.serializer2.savepoint["lw_firstloop_out_000000"]
                )

            print("Testing firstloop...")
            print(" ")
            compare_data(outdict_firstloop, valdict_firstloop)
            print(" ")
            print("Firstloop validates!")
            print(" ")

        self._load_random_numbers(tile)

        cldprop(
            self.locdict_gt4py["cldfrc"],
            self.locdict_gt4py["clwp"],
            self.locdict_gt4py["relw"],
            self.locdict_gt4py["ciwp"],
            self.locdict_gt4py["reiw"],
            self.locdict_gt4py["cda1"],
            self.locdict_gt4py["cda2"],
            self.locdict_gt4py["cda3"],
            self.locdict_gt4py["cda4"],
            self.locdict_gt4py["dz"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["taucld"],
            self.outdict_gt4py["cldtaulw"],
            self.lookupdict_gt4py["absliq1"],
            self.lookupdict_gt4py["absice1"],
            self.lookupdict_gt4py["absice2"],
            self.lookupdict_gt4py["absice3"],
            self.lookupdict_gt4py["ipat"],
            self.locdict_gt4py["tauliq"],
            self.locdict_gt4py["tauice"],
            self.locdict_gt4py["cldf"],
            self.locdict_gt4py["dgeice"],
            self.locdict_gt4py["factor"],
            self.locdict_gt4py["fint"],
            self.locdict_gt4py["tauran"],
            self.locdict_gt4py["tausnw"],
            self.locdict_gt4py["cldliq"],
            self.locdict_gt4py["refliq"],
            self.locdict_gt4py["cldice"],
            self.locdict_gt4py["refice"],
            self.locdict_gt4py["index"],
            self.locdict_gt4py["ia"],
            self.locdict_gt4py["lcloudy"],
            self.lookupdict_gt4py["cdfunc"],
            self.locdict_gt4py["tem1"],
            self.locdict_gt4py["lcf1"],
            self.locdict_gt4py["cldsum"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outdict_cldprop = dict()
            valdict_cldprop = dict()

            outvars_cldprop = ["cldfmc", "taucld"]

            for var in outvars_cldprop:
                outdict_cldprop[var] = (
                    self.locdict_gt4py[var][:, :, 1:, :].squeeze().transpose((0, 2, 1))
                )
                valdict_cldprop[var] = self.serializer2.read(
                    var, self.serializer2.savepoint["lw-cldprop-output-000000"]
                )

            print("Testing cldprop...")
            print(" ")
            compare_data(outdict_cldprop, valdict_cldprop)
            print(" ")
            print("cldprop validates!")
            print(" ")

        setcoef(
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.indict_gt4py["tlvl"],
            self.indict_gt4py["tsfg"],
            self.locdict_gt4py["h2ovmr"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colbrd"],
            self.lookupdict_gt4py["totplnk"],
            self.lookupdict_gt4py["pref"],
            self.lookupdict_gt4py["preflog"],
            self.lookupdict_gt4py["tref"],
            self.lookupdict_gt4py["chi_mls"],
            self.lookupdict_gt4py["delwave"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pklay"],
            self.locdict_gt4py["pklev"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["scaleminorn2"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["tzint"],
            self.locdict_gt4py["stempint"],
            self.locdict_gt4py["tavelint"],
            self.locdict_gt4py["indlay"],
            self.locdict_gt4py["indlev"],
            self.locdict_gt4py["tlyrfr"],
            self.locdict_gt4py["tlvlfr"],
            self.locdict_gt4py["jp1"],
            self.locdict_gt4py["plog"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_setcoef = [
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

            outdict_setcoef = dict()
            valdict_setcoef = dict()

            for var in outvars_setcoef:
                valdict_setcoef[var] = self.serializer2.read(
                    var, self.serializer2.savepoint["lwrad-setcoef-output-000000"]
                )
                if var != "laytrop":
                    outdict_setcoef[var] = (
                        self.locdict_gt4py[var][:, 0, ...].squeeze().view(np.ndarray)
                    )
                    if var == "pklay" or var == "pklev":
                        outdict_setcoef[var] = np.transpose(
                            outdict_setcoef[var], (0, 2, 1)
                        )
                    else:
                        outdict_setcoef[var] = outdict_setcoef[var][:, 1:, ...]
                else:
                    outdict_setcoef[var] = (
                        self.locdict_gt4py[var][0, :, 1:]
                        .squeeze()
                        .view(np.ndarray)
                        .astype(np.int32)
                        .sum()
                    )

            print("Testing setcoef...")
            print(" ")
            compare_data(outdict_setcoef, valdict_setcoef)
            print(" ")
            print("setcoef validates!")
            print(" ")

        taugb01(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminorn2"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py1["absa"],
            self.lookupdict_gt4py1["absb"],
            self.lookupdict_gt4py1["selfref"],
            self.lookupdict_gt4py1["forref"],
            self.lookupdict_gt4py1["fracrefa"],
            self.lookupdict_gt4py1["fracrefb"],
            self.lookupdict_gt4py1["ka_mn2"],
            self.lookupdict_gt4py1["kb_mn2"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["pp"],
            self.locdict_gt4py["corradj"],
            self.locdict_gt4py["scalen2"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["taun2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb02(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py2["absa"],
            self.lookupdict_gt4py2["absb"],
            self.lookupdict_gt4py2["selfref"],
            self.lookupdict_gt4py2["forref"],
            self.lookupdict_gt4py2["fracrefa"],
            self.lookupdict_gt4py2["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["corradj"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb03(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py3["absa"],
            self.lookupdict_gt4py3["absb"],
            self.lookupdict_gt4py3["selfref"],
            self.lookupdict_gt4py3["forref"],
            self.lookupdict_gt4py3["fracrefa"],
            self.lookupdict_gt4py3["fracrefb"],
            self.lookupdict_gt4py3["ka_mn2o"],
            self.lookupdict_gt4py3["kb_mn2o"],
            self.lookupdict_gt4py3["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmn2o"],
            self.locdict_gt4py["jmn2op"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratn2o"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb04(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py4["absa"],
            self.lookupdict_gt4py4["absb"],
            self.lookupdict_gt4py4["selfref"],
            self.lookupdict_gt4py4["forref"],
            self.lookupdict_gt4py4["fracrefa"],
            self.lookupdict_gt4py4["fracrefb"],
            self.lookupdict_gt4py4["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb05(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py5["absa"],
            self.lookupdict_gt4py5["absb"],
            self.lookupdict_gt4py5["selfref"],
            self.lookupdict_gt4py5["forref"],
            self.lookupdict_gt4py5["fracrefa"],
            self.lookupdict_gt4py5["fracrefb"],
            self.lookupdict_gt4py5["ka_mo3"],
            self.lookupdict_gt4py5["ccl4"],
            self.lookupdict_gt4py5["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmo3"],
            self.locdict_gt4py["jmo3p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb06(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py6["absa"],
            self.lookupdict_gt4py6["selfref"],
            self.lookupdict_gt4py6["forref"],
            self.lookupdict_gt4py6["fracrefa"],
            self.lookupdict_gt4py6["ka_mco2"],
            self.lookupdict_gt4py6["cfc11adj"],
            self.lookupdict_gt4py6["cfc12"],
            self.lookupdict_gt4py6["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb07(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py7["absa"],
            self.lookupdict_gt4py7["absb"],
            self.lookupdict_gt4py7["selfref"],
            self.lookupdict_gt4py7["forref"],
            self.lookupdict_gt4py7["fracrefa"],
            self.lookupdict_gt4py7["fracrefb"],
            self.lookupdict_gt4py7["ka_mco2"],
            self.lookupdict_gt4py7["kb_mco2"],
            self.lookupdict_gt4py7["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb08(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py8["absa"],
            self.lookupdict_gt4py8["absb"],
            self.lookupdict_gt4py8["selfref"],
            self.lookupdict_gt4py8["forref"],
            self.lookupdict_gt4py8["fracrefa"],
            self.lookupdict_gt4py8["fracrefb"],
            self.lookupdict_gt4py8["ka_mo3"],
            self.lookupdict_gt4py8["ka_mco2"],
            self.lookupdict_gt4py8["kb_mco2"],
            self.lookupdict_gt4py8["cfc12"],
            self.lookupdict_gt4py8["ka_mn2o"],
            self.lookupdict_gt4py8["kb_mn2o"],
            self.lookupdict_gt4py8["cfc22adj"],
            self.lookupdict_gt4py8["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb09(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py9["absa"],
            self.lookupdict_gt4py9["absb"],
            self.lookupdict_gt4py9["selfref"],
            self.lookupdict_gt4py9["forref"],
            self.lookupdict_gt4py9["fracrefa"],
            self.lookupdict_gt4py9["fracrefb"],
            self.lookupdict_gt4py9["ka_mn2o"],
            self.lookupdict_gt4py9["kb_mn2o"],
            self.lookupdict_gt4py9["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratn2o"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb10(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py10["absa"],
            self.lookupdict_gt4py10["absb"],
            self.lookupdict_gt4py10["selfref"],
            self.lookupdict_gt4py10["forref"],
            self.lookupdict_gt4py10["fracrefa"],
            self.lookupdict_gt4py10["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb11(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py11["absa"],
            self.lookupdict_gt4py11["absb"],
            self.lookupdict_gt4py11["selfref"],
            self.lookupdict_gt4py11["forref"],
            self.lookupdict_gt4py11["fracrefa"],
            self.lookupdict_gt4py11["fracrefb"],
            self.lookupdict_gt4py11["ka_mo2"],
            self.lookupdict_gt4py11["kb_mo2"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb12(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py12["absa"],
            self.lookupdict_gt4py12["selfref"],
            self.lookupdict_gt4py12["forref"],
            self.lookupdict_gt4py12["fracrefa"],
            self.lookupdict_gt4py12["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["specparm_planck"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb13(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py13["absa"],
            self.lookupdict_gt4py13["selfref"],
            self.lookupdict_gt4py13["forref"],
            self.lookupdict_gt4py13["fracrefa"],
            self.lookupdict_gt4py13["fracrefb"],
            self.lookupdict_gt4py13["ka_mco"],
            self.lookupdict_gt4py13["ka_mco2"],
            self.lookupdict_gt4py13["kb_mo3"],
            self.lookupdict_gt4py13["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmco"],
            self.locdict_gt4py["jmcop"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb14(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py14["absa"],
            self.lookupdict_gt4py14["absb"],
            self.lookupdict_gt4py14["selfref"],
            self.lookupdict_gt4py14["forref"],
            self.lookupdict_gt4py14["fracrefa"],
            self.lookupdict_gt4py14["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb15(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py15["absa"],
            self.lookupdict_gt4py15["selfref"],
            self.lookupdict_gt4py15["forref"],
            self.lookupdict_gt4py15["fracrefa"],
            self.lookupdict_gt4py15["ka_mn2"],
            self.lookupdict_gt4py15["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["taun2"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmn2"],
            self.locdict_gt4py["jmn2p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["fpl"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb16(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py16["absa"],
            self.lookupdict_gt4py16["absb"],
            self.lookupdict_gt4py16["selfref"],
            self.lookupdict_gt4py16["forref"],
            self.lookupdict_gt4py16["fracrefa"],
            self.lookupdict_gt4py16["fracrefb"],
            self.lookupdict_gt4py16["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["fpl"],
            self.locdict_gt4py["speccomb"],
            self.locdict_gt4py["speccomb1"],
            self.locdict_gt4py["fac000"],
            self.locdict_gt4py["fac100"],
            self.locdict_gt4py["fac200"],
            self.locdict_gt4py["fac010"],
            self.locdict_gt4py["fac110"],
            self.locdict_gt4py["fac210"],
            self.locdict_gt4py["fac001"],
            self.locdict_gt4py["fac101"],
            self.locdict_gt4py["fac201"],
            self.locdict_gt4py["fac011"],
            self.locdict_gt4py["fac111"],
            self.locdict_gt4py["fac211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        combine_optical_depth(
            self.NGB,
            self.locdict_gt4py["ib"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["tauaer"],
            self.locdict_gt4py["tautot"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outdict_taumol = {
                "fracs": np.transpose(
                    self.locdict_gt4py["fracs"][:, :, 1:, :].squeeze(), (0, 2, 1)
                ),
                "tautot": np.transpose(
                    self.locdict_gt4py["tautot"][:, :, 1:, :].squeeze(), (0, 2, 1)
                ),
            }

            outvars_t = ["fracs", "tautot"]

            valdict_taumol = dict()
            for var in outvars_t:
                valdict_taumol[var] = self.serializer2.read(
                    var, self.serializer2.savepoint["lwrad-taumol-output-000000"]
                )

            print("Testing taumol...")
            print(" ")
            compare_data(outdict_taumol, valdict_taumol)
            print(" ")
            print("taumol validates!")
            print(" ")

        rtrnmc(
            self.locdict_gt4py["semiss"],
            self.locdict_gt4py["secdiff"],
            self.locdict_gt4py["delp"],
            self.locdict_gt4py["taucld"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["tautot"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["pklay"],
            self.locdict_gt4py["pklev"],
            self.exp_tbl,
            self.tau_tbl,
            self.tfn_tbl,
            self.NGB,
            self.locdict_gt4py["totuflux"],
            self.locdict_gt4py["totdflux"],
            self.locdict_gt4py["totuclfl"],
            self.locdict_gt4py["totdclfl"],
            self.outdict_gt4py["upfxc_t"],
            self.outdict_gt4py["upfx0_t"],
            self.outdict_gt4py["upfxc_s"],
            self.outdict_gt4py["upfx0_s"],
            self.outdict_gt4py["dnfxc_s"],
            self.outdict_gt4py["dnfx0_s"],
            self.outdict_gt4py["htlwc"],
            self.outdict_gt4py["htlw0"],
            self.locdict_gt4py["clrurad"],
            self.locdict_gt4py["clrdrad"],
            self.locdict_gt4py["toturad"],
            self.locdict_gt4py["totdrad"],
            self.locdict_gt4py["gassrcu"],
            self.locdict_gt4py["totsrcu"],
            self.locdict_gt4py["trngas"],
            self.locdict_gt4py["efclrfr"],
            self.locdict_gt4py["rfdelp"],
            self.locdict_gt4py["fnet"],
            self.locdict_gt4py["fnetc"],
            self.locdict_gt4py["totsrcd"],
            self.locdict_gt4py["gassrcd"],
            self.locdict_gt4py["tblind"],
            self.locdict_gt4py["odepth"],
            self.locdict_gt4py["odtot"],
            self.locdict_gt4py["odcld"],
            self.locdict_gt4py["atrtot"],
            self.locdict_gt4py["atrgas"],
            self.locdict_gt4py["reflct"],
            self.locdict_gt4py["totfac"],
            self.locdict_gt4py["gasfac"],
            self.locdict_gt4py["plfrac"],
            self.locdict_gt4py["blay"],
            self.locdict_gt4py["bbdgas"],
            self.locdict_gt4py["bbdtot"],
            self.locdict_gt4py["bbugas"],
            self.locdict_gt4py["bbutot"],
            self.locdict_gt4py["dplnku"],
            self.locdict_gt4py["dplnkd"],
            self.locdict_gt4py["radtotu"],
            self.locdict_gt4py["radclru"],
            self.locdict_gt4py["radtotd"],
            self.locdict_gt4py["radclrd"],
            self.locdict_gt4py["rad0"],
            self.locdict_gt4py["clfm"],
            self.locdict_gt4py["trng"],
            self.locdict_gt4py["gasu"],
            self.locdict_gt4py["itgas"],
            self.locdict_gt4py["ittot"],
            self.locdict_gt4py["ib"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_rtrnmc = [
                "totuflux",
                "totdflux",
                "totuclfl",
                "totdclfl",
            ]
            outdict_rtrnmc = dict()

            for var in outvars_rtrnmc:
                outdict_rtrnmc[var] = self.locdict_gt4py[var]
            outdict_rtrnmc = view_gt4py_storage(outdict_rtrnmc)

            for var in outdict_rtrnmc.keys():
                outdict_rtrnmc[var] = outdict_rtrnmc[var][:, :]

            valdict_rtrnmc = dict()
            for var in outvars_rtrnmc:
                valdict_rtrnmc[var] = self.serializer2.read(
                    var, self.serializer2.savepoint["lwrad-rtrnmc-output-000000"]
                )

            print("Testing rtrnmc...")
            print(" ")
            compare_data(outdict_rtrnmc, valdict_rtrnmc)
            print(" ")
            print("rtrnmc validates!")
            print(" ")

        end0 = time.time()
        print(f"Total time taken = {end0 - start0}")

        valdict = dict()
        outdict_np = dict()

        for var in self.outvars:
            valdict[var] = self.serializer.read(
                var, self.serializer.savepoint["lwrad-out-000000"]
            )
            if var == "htlwc" or var == "htlw0" or var == "cldtaulw":
                outdict_np[var] = (
                    self.outdict_gt4py[var][:, :, 1:].view(np.ndarray).squeeze()
                )
            else:
                outdict_np[var] = self.outdict_gt4py[var].view(np.ndarray).squeeze()

        print("Testing final output...")
        print(" ")
        compare_data(valdict, outdict_np)
        print(" ")
        print("lwrad validates!")
        print(" ")
