import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import ntbmx
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

    def return_initdata(self):
        outdict = {"heatfac": self.heatfac, "exp_tbl": self.exp_tbl}
        return outdict

    def create_input_data(self, rank):
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

        indict = read_data(
            os.path.join(FORTRANDATA_DIR, "SW"), "swrad", rank, 0, True, invars
        )
        indict_gt4py = numpy_dict_to_gt4py_dict(indict, invars)
