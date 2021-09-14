import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import ntbmx, NGB, nbandssw
from radphysparam import iswmode, iswrgas, iswrate, iswcice, iswcliq
from phys_const import con_amd, con_amw, con_amo3, con_g, con_cp, con_avgd
from util import *
from config import *


class RadSWClass:
    VTAGSW = "NCEP SW v5.1  Nov 2012 -RRTMG-SW v3.8"

    # constant values
    eps = 1.0e-6
    oneminus = 1.0 - eps
    # pade approx constant
    bpade = 1.0 / 0.278
    stpfac = 296.0 / 1013.0
    ftiny = 1.0e-12
    flimit = 1.0e-20
    # internal solar constant
    s0 = 1368.22
    f_zero = 0.0
    f_one = 1.0

    # atomic weights for conversion from mass to volume mixing ratios
    amdw = con_amd / con_amw
    amdo3 = con_amd / con_amo3

    # band indices
    nspa = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
    nspb = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]
    # band index for sfc flux
    idxsfc = [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1]
    # band index for cld prop
    idxebc = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5]

    # uv-b band index
    nuvb = 27

    # initial permutation seed used for sub-column cloud scheme
    ipsdsw0 = 1

    def __init__(self, me, iovrsw, isubcsw, icldflg):

        self.iovrsw = iovrsw
        self.isubcsw = isubcsw
        self.icldflg = icldflg

        expeps = 1.0e-20

        #
        # ===> ... begin here
        #
        if self.iovrsw < 0 or self.iovrsw > 3:
            print(
                "*** Error in specification of cloud overlap flag",
                f" IOVRSW={self.iovrsw} in RSWINIT !!",
            )

        if me == 0:
            print(f"- Using AER Shortwave Radiation, Version: {self.VTAGSW}")

            if iswmode == 1:
                print("   --- Delta-eddington 2-stream transfer scheme")
            elif iswmode == 2:
                print("   --- PIFM 2-stream transfer scheme")
            elif iswmode == 3:
                print("   --- Discrete ordinates 2-stream transfer scheme")

            if iswrgas <= 0:
                print("   --- Rare gases absorption is NOT included in SW")
            else:
                print("   --- Include rare gases N2O, CH4, O2, absorptions in SW")

            if self.isubcsw == 0:
                print(
                    "   --- Using standard grid average clouds, no ",
                    "   sub-column clouds approximation applied",
                )
            elif self.isubcsw == 1:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with a prescribed sequence of permutation seeds",
                )
            elif self.isubcsw == 2:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with provided input array of permutation seeds",
                )
            else:
                print(
                    "  *** Error in specification of sub-column cloud ",
                    f" control flag isubcsw = {self.isubcsw} !!",
                )

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and iswcliq != 0) or (icldflg == 1 and iswcliq == 0):
            print(
                "*** Model cloud scheme inconsistent with SW",
                " radiation cloud radiative property setup !!",
            )

        if self.isubcsw == 0 and self.iovrsw > 2:
            if me == 0:
                print(
                    f"*** IOVRSW={self.iovrsw} is not available for",
                    " ISUBCSW=0 setting!!",
                )
                print("The program will use maximum/random overlap", " instead.")
            self.iovrsw = 1

        #  --- ...  setup constant factors for heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        if iswrate == 1:
            self.heatfac = con_g * 864.0 / con_cp  #   (in k/day)
        else:
            self.heatfac = con_g * 1.0e-2 / con_cp  #   (in k/second)

        #  --- ...  define exponential lookup tables for transmittance. tau is
        #           computed as a function of the tau transition function, and
        #           transmittance is calculated as a function of tau.  all tables
        #           are computed at intervals of 0.0001.  the inverse of the
        #           constant used in the Pade approximation to the tau transition
        #           function is set to bpade.

        self.exp_tbl = np.zeros(ntbmx + 1)
        self.exp_tbl[0] = 1.0
        self.exp_tbl[ntbmx] = expeps

        for i in range(ntbmx - 1):
            tfn = i / (ntbmx - i)
            tau = self.bpade * tfn
            self.exp_tbl[i] = np.exp(-tau)

        self.exp_tbl = np.tile(self.exp_tbl[None, None, None, :], (npts, 1, nlp1, 1))

        self.exp_tbl = create_storage_from_array(
            self.exp_tbl, backend, shape_nlp1, type_ntbmx
        )

        self.NGB = np.tile(np.array(NGB)[None, None, None, :], (npts, 1, nlp1, 1))
        self.NGB = create_storage_from_array(NGB, backend, shape_nlp1, type_ngptsw)

        self.idxsfc - np.tile(
            np.array(self.idxsfc)[None, None, None, :], (npts, 1, nlp1, 1)
        )
        self.idxebc - np.tile(
            np.array(self.idxebc)[None, None, None, :], (npts, 1, nlp1, 1)
        )

        self.idxsfc = create_storage_from_array(
            self.idxsfc, backend, shape_nlp1, (DTYPE_FLT, (nbandssw,))
        )

        self.idxebc = create_storage_from_array(
            self.idxebc, backend, shape_nlp1, (DTYPE_FLT, (nbandssw,))
        )

        self._load_lookup_table_data()

    def return_initdata(self):
        outdict = {"heatfac": self.heatfac, "exp_tbl": self.exp_tbl}
        return outdict

    def create_input_data(self, rank):

        self.serializer2 = ser.Serializer(
            ser.OpenModeKind.Read, SW_SERIALIZED_DIR, "Serialized_rank" + str(rank)
        )

        invars = {
            "plyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "plvl": {"shape": (npts, nlp1), "type": DTYPE_FLT},
            "tlyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "tlvl": {"shape": (npts, nlp1), "type": DTYPE_FLT},
            "qlyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "olyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "gasvmr": {"shape": (npts, nlay, 10), "type": type_10},
            "clouds": {"shape": (npts, nlay, 9), "type": type_9},
            "faersw": {"shape": (npts, nlay, nbdsw, 3), "type": type_nbands3},
            "sfcalb": {"shape": (npts, 4), "type": (DTYPE_FLT, (4,))},
            "dz": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "delp": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "de_lgth": {"shape": (npts,), "type": DTYPE_FLT},
            "coszen": {"shape": (npts,), "type": DTYPE_FLT},
            "solcon": {"shape": (), "type": DTYPE_FLT},
            "nday": {"shape": (), "type": DTYPE_INT},
            "idxday": {"shape": (npts,), "type": DTYPE_INT},
            "im": {"shape": (), "type": DTYPE_INT},
            "lmk": {"shape": (), "type": DTYPE_INT},
            "lmp": {"shape": (), "type": DTYPE_INT},
            "lprnt": {"shape": (), "type": DTYPE_BOOL},
        }

        self._indict = read_data(
            os.path.join(FORTRANDATA_DIR, "SW"), "swrad", rank, 0, True, invars
        )
        indict_gt4py = numpy_dict_to_gt4py_dict(self._indict, invars)

        outvars = {
            "upfxc_t": {"shape": (npts,), "type": DTYPE_FLT},
            "dnfxc_t": {"shape": (npts,), "type": DTYPE_FLT},
            "upfx0_t": {"shape": (npts,), "type": DTYPE_FLT},
            "upfxc_s": {"shape": (npts,), "type": DTYPE_FLT},
            "dnfxc_s": {"shape": (npts,), "type": DTYPE_FLT},
            "upfx0_s": {"shape": (npts,), "type": DTYPE_FLT},
            "dnfx0_s": {"shape": (npts,), "type": DTYPE_FLT},
            "htswc": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "htsw0": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "htswb": {"shape": (npts, nlay, nbdsw), "type": DTYPE_FLT},
            "uvbf0": {"shape": (npts,), "type": DTYPE_FLT},
            "uvbfc": {"shape": (npts,), "type": DTYPE_FLT},
            "nirbm": {"shape": (npts,), "type": DTYPE_FLT},
            "nirdf": {"shape": (npts,), "type": DTYPE_FLT},
            "visbm": {"shape": (npts,), "type": DTYPE_FLT},
            "visdf": {"shape": (npts,), "type": DTYPE_FLT},
            "cldtausw": {"shape": (npts, nlay), "type": DTYPE_FLT},
        }

        outdict_gt4py = create_gt4py_dict_zeros(outvars)

        locvars = {
            "cosz1": {"shape": shape_2D, "type": DTYPE_FLT},
            "sntz1": {"shape": shape_2D, "type": DTYPE_FLT},
            "ssolar": {"shape": shape_2D, "type": DTYPE_FLT},
            "albbm": {"shape": shape_nlp1, "type": (DTYPE_FLT, (2,))},
            "albdf": {"shape": shape_nlp1, "type": (DTYPE_FLT, (2,))},
            "pavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "h2ovmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "o3vmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "coldry": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "temcol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "colamt": {"shape": shape_nlp1, "type": type_maxgas},
            "colmol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauae": {"shape": shape_nlp1, "type": type_nbdsw},
            "ssaae": {"shape": shape_nlp1, "type": type_nbdsw},
            "asyae": {"shape": shape_nlp1, "type": type_nbdsw},
            "cfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cliqp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "reliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cicep": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "reice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat3": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat4": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "zcf0": {"shape": shape_2D, "type": DTYPE_FLT},
            "zcf1": {"shape": shape_2D, "type": DTYPE_FLT},
            "tauliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "tauice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaran": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssasnw": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyran": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asysnw": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "cldf": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "dgeice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "factor": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fint": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauran": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tausnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldran": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldsnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refsnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "extcoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ssacoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "asycoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "extcoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ssacoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "asycoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "dgesnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "lcloudy": {"shape": shape_nlp1, "type": type_ngptsw_bool},
            "index": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ia": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jb": {"shape": shape_nlp1, "type": DTYPE_INT},
            "cldfmc": {"shape": shape_nlp1, "type": type_ngptsw},
            "taucw": {"shape": shape_nlp1, "type": type_nbdsw},
            "ssacw": {"shape": shape_nlp1, "type": type_nbdsw},
            "asycw": {"shape": shape_nlp1, "type": type_nbdsw},
            "cldfrc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "plog": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fp1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ft": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ft1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "jp1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "fac00": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac01": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac10": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac11": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "indself": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfor": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "lcloudy": {"shape": shape_nlp1, "type": DTYPE_BOOL},
            "id0": {"shape": shape_nlp1, "type": type_nbandssw_int},
            "id1": {"shape": shape_nlp1, "type": type_nbandssw_int},
            "ind01": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind02": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind03": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind04": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind11": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind12": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind13": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind14": {"shape": shape_nlp1, "type": DTYPE_INT},
            "inds": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indsp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indf": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "fs": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "js": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jsa": {"shape": shape_nlp1, "type": DTYPE_INT},
            "colm1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "colm2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "sfluxzen": {"shape": shape_2D, "type": type_ngptsw},
            "taug": {"shape": shape_nlp1, "type": type_ngptsw},
            "taur": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztaus": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssas": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasys": {"shape": shape_nlp1, "type": type_ngptsw},
            "zldbt0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrefb": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrefd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztrab": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztrad": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdbt": {"shape": shape_nlp1, "type": type_ngptsw},
            "zldbt": {"shape": shape_nlp1, "type": type_ngptsw},
            "zfu": {"shape": shape_nlp1, "type": type_ngptsw},
            "zfd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztau1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssa1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy1": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztau0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssa0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssaw": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasyw": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam4": {"shape": shape_nlp1, "type": type_ngptsw},
            "za1": {"shape": shape_nlp1, "type": type_ngptsw},
            "za2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zb1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zb2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrk": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrk2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrp": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrm1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrpp": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg4": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexm1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexm2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zden1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp4": {"shape": shape_nlp1, "type": type_ngptsw},
            "ze1r45": {"shape": shape_nlp1, "type": type_ngptsw},
            "ftind": {"shape": shape_nlp1, "type": type_ngptsw},
            "zsolar": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdbt0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr4": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr5": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zf1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zf2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrpp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrupb": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrupd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdn": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrdnd": {"shape": shape_nlp1, "type": type_ngptsw},
            "jb": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ib": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ibd": {"shape": shape_nlp1, "type": DTYPE_INT},
            "itind": {"shape": shape_nlp1, "type": DTYPE_INT},
            "zb11": {"shape": shape_2D, "type": type_ngptsw},
            "zb22": {"shape": shape_2D, "type": type_ngptsw},
            "fxupc": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxdnc": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxup0": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxdn0": {"shape": shape_nlp1, "type": type_nbdsw},
            "ftoauc": {"shape": shape_2D, "type": DTYPE_FLT},
            "ftoau0": {"shape": shape_2D, "type": DTYPE_FLT},
            "ftoadc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcuc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcu0": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcdc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcd0": {"shape": shape_2D, "type": DTYPE_FLT},
            "sfbmc": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfdfc": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfbm0": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfdf0": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "suvbfc": {"shape": shape_2D, "type": DTYPE_FLT},
            "suvbf0": {"shape": shape_2D, "type": DTYPE_FLT},
            "flxuc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxdc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxu0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxd0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnet": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnetc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnetb": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "rfdelp": {"shape": shape_nlp1, "type": DTYPE_FLT},
        }

        locdict_gt4py = create_gt4py_dict_zeros(locvars)

        self.indict_gt4py = indict_gt4py
        self.locdict_gt4py = locdict_gt4py
        self.outdict_gt4py = outdict_gt4py
        self.outvars = outvars

    def _load_lookup_table_data(self):
        """
        Read in lookup table data from netcdf data that has been serialized out from
        radsw_datatb.F
        """

        # Load lookup data for setcoef
        ds = xr.open_dataset("../lookupdata/radsw_ref_data.nc")
        preflog = ds["preflog"].data
        preflog = np.tile(preflog[None, None, None, :], (npts, 1, nlp1, 1))
        tref = ds["tref"].data
        tref = np.tile(tref[None, None, None, :], (npts, 1, nlp1, 1))
        lookupdict_gt4py = dict()

        lookupdict_gt4py["preflog"] = create_storage_from_array(
            preflog, backend, shape_nlp1, (DTYPE_FLT, (59,))
        )
        lookupdict_gt4py["tref"] = create_storage_from_array(
            tref, backend, shape_nlp1, (DTYPE_FLT, (59,))
        )

        self.lookupdict_ref = loadlookupdata("sflux", "radsw")
        self.lookupdict16 = loadlookupdata("kgb16", "radsw")
        self.lookupdict17 = loadlookupdata("kgb17", "radsw")
        self.lookupdict18 = loadlookupdata("kgb18", "radsw")
        self.lookupdict19 = loadlookupdata("kgb19", "radsw")
        self.lookupdict20 = loadlookupdata("kgb20", "radsw")
        self.lookupdict21 = loadlookupdata("kgb21", "radsw")
        self.lookupdict22 = loadlookupdata("kgb22", "radsw")
        self.lookupdict23 = loadlookupdata("kgb23", "radsw")
        self.lookupdict24 = loadlookupdata("kgb24", "radsw")
        self.lookupdict25 = loadlookupdata("kgb25", "radsw")
        self.lookupdict26 = loadlookupdata("kgb26", "radsw")
        self.lookupdict27 = loadlookupdata("kgb27", "radsw")
        self.lookupdict28 = loadlookupdata("kgb28", "radsw")
        self.lookupdict29 = loadlookupdata("kgb29", "radsw")

        # Subtract one from indexing variables for Fortran -> Python conversion
        self.lookupdict_ref["ix1"] = self.lookupdict_ref["ix1"] - 1
        self.lookupdict_ref["ix2"] = self.lookupdict_ref["ix2"] - 1
        self.lookupdict_ref["ibx"] = self.lookupdict_ref["ibx"] - 1

        def _load_random_numbers(self, rank):
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
            ds = xr.open_dataset(
                os.path.join(LOOKUP_DIR, "rand2d_tile" + str(rank) + "_sw.nc")
            )
            rand2d = ds["rand2d"].data
            cdfunc = np.zeros((npts, nlay, ngptsw))
            idxday = self._indict["idxday"]
            for n in range(npts):
                myind = idxday[n]
                if myind > 1 and myind < 25:
                    cdfunc[myind - 1, :, :] = np.reshape(
                        rand2d[n, :], (nlay, ngptsw), order="F"
                    )
            cdfunc = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))
            cdfunc = np.insert(cdfunc, 0, 0, axis=2)

            self.lookupdict_gt4py["cdfunc"] = create_storage_from_array(
                cdfunc, backend, shape_nlp1, type_ngptsw
            )
