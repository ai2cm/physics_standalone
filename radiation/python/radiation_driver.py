import gt4py
import numpy as np
import warnings

from config import *
from radphysparam import *
from phys_const import con_eps, con_epsm1, con_rocp, con_fvirt, con_rog, con_epsq
from funcphys import fpvs

from radiation_astronomy import AstronomyClass
from radiation_aerosols import AerosolClass
from radiation_clouds import CloudClass
from radiation_gases import GasClass
from radiation_sfc import SurfaceClass
from radlw.radlw_main_gt4py import RadLWClass
from radsw.radsw_main_gt4py import RadSWClass


class RadiationDriver:

    VTAGRAD = "NCEP-Radiation_driver    v5.2  Jan 2013"

    QMIN = 1.0e-10
    QME5 = 1.0e-7
    QME6 = 1.0e-7
    EPSQ = 1.0e-12

    # lower limit of toa pressure value in mb
    prsmin = 1.0e-6

    # optional extra top layer on top of low ceiling models
    # LTP=0: no extra top layer
    LTP = 0  # no extra top layer

    # control flag for extra top layer
    lextop = LTP > 0

    def __init__(self):
        # control flag for LW surface temperature at air/ground interface
        # (default=0, the value will be set in subroutine radinit)
        self.itsfc = 0

        # new data input control variables (set/reset in subroutines radinit/radupdate):
        self.month0 = 0
        self.iyear0 = 0
        self.monthd = 0

        # control flag for the first time of reading climatological ozone data
        # (set/reset in subroutines radinit/radupdate, it is used only if the
        # control parameter ioznflg=0)
        self.loz1st = True

    def radinit(
        self,
        si,
        NLAY,
        imp_physics,
        me,
        iemsflg,
        ioznflg,
        ictmflg,
        isolar,
        ico2flg,
        iaerflg,
        ialbflg,
        icldflg,
        ivflip,
        iovrsw,
        iovrlw,
        isubcsw,
        isubclw,
        lcrick,
        lcnorm,
        lnoprec,
        iswcliq,
        do_test=False,
    ):
        self.itsfc = iemsflg / 10  # sfc air/ground temp control
        self.loz1st = ioznflg == 0  # first-time clim ozone data read flag
        self.month0 = 0
        self.iyear0 = 0
        self.monthd = 0
        self.isolar = isolar

        if me == 0:
            print("NEW RADIATION PROGRAM STRUCTURES BECAME OPER. May 01 2007")
            print(self.VTAGRAD)  # print out version tag
            print(" ")
            print(f"- Selected Control Flag settings: ICTMflg={ictmflg}")
            print(f"  ISOLar ={isolar}, ICO2flg={ico2flg}, IAERflg={iaerflg}")
            print(f"  IALBflg={ialbflg}, IEMSflg={iemsflg}, ICLDflg={icldflg}")
            print(f"  IMP_PHYSICS={imp_physics}, IOZNflg={ioznflg}")
            print(f"  IVFLIP={ivflip}, IOVRSW={iovrsw}, IOVRLW={iovrlw}")
            print(f"  ISUBCSW={isubcsw}, ISUBCLW={isubclw}")
            print(f"  LCRICK={lcrick}, LCNORM={lcnorm}, LNOPREC={lnoprec}")
            print(f"  LTP ={self.LTP}, add extra top layer ={self.lextop}")
            print(" ")

            if ictmflg == 0 or ictmflg == -2:
                print("Data usage is limited by initial condition!")
                print("No volcanic aerosols")

            if isubclw == 0:
                print(
                    f"- ISUBCLW={isubclw}, No McICA, use grid ",
                    f"averaged cloud in LW radiation",
                )
            elif isubclw == 1:
                print(
                    "- ISUBCLW={isubclw}, Use McICA with fixed ",
                    "permutation seeds for LW random number generator",
                )
            elif isubclw == 2:
                print(
                    f"- ISUBCLW={isubclw}, Use McICA with random ",
                    f"permutation seeds for LW random number generator",
                )
            else:
                raise ValueError(f"- ERROR!!! ISUBCLW={isubclw}, is not a valid option")

            if isubcsw == 0:
                print(
                    "- ISUBCSW={isubcsw}, No McICA, use grid ",
                    "averaged cloud in SW radiation",
                )
            elif isubcsw == 1:
                print(
                    f"- ISUBCSW={isubcsw}, Use McICA with fixed ",
                    "permutation seeds for SW random number generator",
                )
            elif isubcsw == 2:
                print(
                    f"- ISUBCSW={isubcsw}, Use McICA with random ",
                    "permutation seeds for SW random number generator",
                )
            else:
                raise ValueError(f"- ERROR!!! ISUBCSW={isubcsw}, is not a valid option")

            if isubcsw != isubclw:
                warnings.warn(
                    "- *** Notice *** ISUBCSW /= ISUBCLW !!!", f"{isubcsw}, {isubclw}"
                )

        # -# Initialization
        #  --- ...  astronomy initialization routine
        self.sol = AstronomyClass(me, isolar)
        #  --- ...  aerosols initialization routine
        self.aer = AerosolClass(NLAY, me, iaerflg, ivflip)
        #  --- ...  co2 and other gases initialization routine
        self.gas = GasClass(me, ioznflg, ico2flg, ictmflg)
        #  --- ...  surface initialization routine
        self.sfc = SurfaceClass(me, ialbflg, iemsflg)
        #  --- ...  cloud initialization routine
        self.cld = CloudClass(
            si, NLAY, imp_physics, me, ivflip, icldflg, iovrsw, iovrlw
        )
        #  --- ...  lw radiation initialization routine
        self.rlw = RadLWClass(me, iovrlw, isubclw)
        #  --- ...  sw radiation initialization routine
        self.rsw = RadSWClass(me, iovrsw, isubcsw, iswcliq)

        if do_test:
            sol_dict = self.sol.return_initdata()
            aer_dict = self.aer.return_initdata()
            gas_dict = self.gas.return_initdata()
            sfc_dict = self.sfc.return_initdata()
            cld_dict = self.cld.return_initdata()
            rlw_dict = self.rlw.return_initdata()
            rsw_dict = self.rsw.return_initdata()

            return aer_dict, sol_dict, gas_dict, sfc_dict, cld_dict, rlw_dict, rsw_dict

    def radupdate(self, idate, jdate, deltsw, deltim, lsswr, do_test=False):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:   radupdate   calls many update subroutines to check and  !
        #   update radiation required but time varying data sets and module     !
        #   variables.                                                          !
        #                                                                       !
        # usage:        call radupdate                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:  fortran 90                                               !
        #   machine:   ibm sp                                                   !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input parameters:                                                     !
        #   idate(8)       : ncep absolute date and time of initial condition   !
        #                    (yr, mon, day, t-zone, hr, min, sec, mil-sec)      !
        #   jdate(8)       : ncep absolute date and time at fcst time           !
        #                    (yr, mon, day, t-zone, hr, min, sec, mil-sec)      !
        #   deltsw         : sw radiation calling frequency in seconds          !
        #   deltim         : model timestep in seconds                          !
        #   lsswr          : logical flags for sw radiation calculations        !
        #   me             : print control flag                                 !
        #                                                                       !
        #  outputs:                                                             !
        #   slag           : equation of time in radians                        !
        #   sdec, cdec     : sin and cos of the solar declination angle         !
        #   solcon         : sun-earth distance adjusted solar constant (w/m2)  !
        #                                                                       !
        #  external module variables:                                           !
        #   isolar   : solar constant cntrl  (in module physparam)               !
        #              = 0: use the old fixed solar constant in "physcon"       !
        #              =10: use the new fixed solar constant in "physcon"       !
        #              = 1: use noaa ann-mean tsi tbl abs-scale with cycle apprx!
        #              = 2: use noaa ann-mean tsi tbl tim-scale with cycle apprx!
        #              = 3: use cmip5 ann-mean tsi tbl tim-scale with cycl apprx!
        #              = 4: use cmip5 mon-mean tsi tbl tim-scale with cycl apprx!
        #   ictmflg  : =yyyy#, external data ic time/date control flag          !
        #              =   -2: same as 0, but superimpose seasonal cycle        !
        #                      from climatology data set.                       !
        #              =   -1: use user provided external data for the          !
        #                      forecast time, no extrapolation.                 !
        #              =    0: use data at initial cond time, if not            !
        #                      available, use latest, no extrapolation.         !
        #              =    1: use data at the forecast time, if not            !
        #                      available, use latest and extrapolation.         !
        #              =yyyy0: use yyyy data for the forecast time,             !
        #                      no further data extrapolation.                   !
        #              =yyyy1: use yyyy data for the fcst. if needed, do        !
        #                      extrapolation to match the fcst time.            !
        #                                                                       !
        #  module variables:                                                    !
        #   loz1st   : first-time clim ozone data read flag                     !
        #                                                                       !
        #  subroutines called: sol_update, aer_update, gas_update               !
        #                                                                       !
        #  ===================================================================  !
        #

        # -# Set up time stamp at fcst time and that for green house gases
        # (currently co2 only)
        # --- ...  time stamp at fcst time

        iyear = jdate[0]
        imon = jdate[1]
        iday = jdate[2]
        ihour = jdate[4]

        #  --- ...  set up time stamp used for green house gases (** currently co2 only)

        if ictmflg == 0 or ictmflg == -2:  # get external data at initial condition time
            kyear = idate[0]
            kmon = idate[1]
            kday = idate[2]
            khour = idate[4]
        else:  # get external data at fcst or specified time
            kyear = iyear
            kmon = imon
            kday = iday
            khour = ihour

        if self.month0 != imon:
            lmon_chg = True
            self.month0 = imon
        else:
            lmon_chg = False

        # -# Call module_radiation_astronomy::sol_update(), yearly update, no
        # time interpolation.
        if lsswr:
            if self.isolar == 0 or self.isolar == 10:
                lsol_chg = False
            elif self.iyear0 != iyear:
                lsol_chg = True
            else:
                lsol_chg = isolar == 4 and lmon_chg

            self.iyear0 = iyear

            slag, sdec, cdec, solcon = self.sol.sol_update(
                jdate, kyear, deltsw, deltim, lsol_chg, 0
            )

        # Call module_radiation_aerosols::aer_update(), monthly update, no
        # time interpolation
        if lmon_chg:
            self.aer.aer_update(iyear, imon, 0)

        # -# Call co2 and other gases update routine:
        # module_radiation_gases::gas_update()
        if self.monthd != kmon:
            self.monthd = kmon
            lco2_chg = True
        else:
            lco2_chg = False

        self.gas.gas_update(kyear, kmon, kday, khour, self.loz1st, lco2_chg, 0)

        if self.loz1st:
            self.loz1st = False

        if do_test:
            soldict = {"slag": slag, "sdec": sdec, "cdec": cdec, "solcon": solcon}
            aerdict = self.aer.return_updatedata()
            gasdict = self.gas.return_updatedata()

            return soldict, aerdict, gasdict

        else:
            return slag, sdec, cdec, solcon

    def GFS_radiation_driver(
        self,
        Model,
        Statein,
        Sfcprop,
        Coupling,
        Grid,
        Tbd,
        Radtend,
        Diag,
        Rank
    ):

        if not (Model["lsswr"] or Model["lslwr"]):
            return

        # --- set commonly used integers
        me = Model["me"]
        LM = Model["levr"]
        LEVS = Model["levs"]
        IM = Grid["xlon"].shape[0]
        NFXR = Model["nfxr"]
        NTRAC = Model[
            "ntrac"
        ]  # tracers in grrad strip off sphum - start tracer1(2:NTRAC)
        ntcw = Model["ntcw"]
        ntiw = Model["ntiw"]
        ncld = Model["ncld"]
        ntrw = Model["ntrw"]
        ntsw = Model["ntsw"]
        ntgl = Model["ntgl"]
        ncndl = min(Model["ncnd"], 4)

        LP1 = LM + 1  # num of in/out levels

        # tskn = np.zeros(IM)
        tskn = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1),
                                   dtype=DTYPE_FLT)
        # tsfa = np.zeros(IM)
        tsfa = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1),
                                   dtype=DTYPE_FLT)
        # tsfg = np.zeros(IM)
        tsfg = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1),
                                   dtype=DTYPE_FLT)

        # tem1d = np.zeros(IM)
        tem1d = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1),
                                   dtype=DTYPE_FLT)
        # alb1d = np.zeros(IM)
        alb1d = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1),
                                   dtype=DTYPE_FLT)
        idxday = np.zeros(IM, dtype=DTYPE_INT)
        # idxday = gt4py.storage.zeros(backend=backend, 
        #                            default_origin=default_origin,
        #                            shape=(IM, 1),
        #                            dtype=DTYPE_BOOL)

        # plvl = np.zeros((IM, Model["levr"] + self.LTP + 1))
        plvl = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # tlvl = np.zeros((IM, Model["levr"] + self.LTP + 1))
        tlvl = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)

        # tem2db = np.zeros((IM, Model["levr"] + self.LTP + 1))
        tem2db = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)

        # plyr = np.zeros((IM, Model["levr"] + self.LTP))
        plyr = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # tlyr = np.zeros((IM, Model["levr"] + self.LTP))
        tlyr = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # olyr = np.zeros((IM, Model["levr"] + self.LTP))
        olyr = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # qlyr = np.zeros((IM, Model["levr"] + self.LTP))
        qlyr = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # rhly = np.zeros((IM, Model["levr"] + self.LTP))
        rhly = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # tvly = np.zeros((IM, Model["levr"] + self.LTP))
        tvly = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # delp = np.zeros((IM, Model["levr"] + self.LTP))
        delp = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # qstl = np.zeros((IM, Model["levr"] + self.LTP))
        qstl = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # cldcov = np.zeros((IM, Model["levr"] + self.LTP))
        cldcov = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # deltaq = np.zeros((IM, Model["levr"] + self.LTP))
        deltaq = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # cnvc = np.zeros((IM, Model["levr"] + self.LTP))
        cnvc = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # cnvw = np.zeros((IM, Model["levr"] + self.LTP))
        cnvw = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # effrl = np.zeros((IM, Model["levr"] + self.LTP))
        effrl = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # effri = np.zeros((IM, Model["levr"] + self.LTP))
        effri = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # effrr = np.zeros((IM, Model["levr"] + self.LTP))
        effrr = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # effrs = np.zeros((IM, Model["levr"] + self.LTP))
        effrs = gt4py.storage.zeros(backend=backend, 
                                   default_origin=default_origin,
                                   shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                   dtype=DTYPE_FLT)
        # dz = np.zeros((IM, Model["levr"] + self.LTP))
        dz = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)
        # prslk1 = np.zeros((IM, Model["levr"] + self.LTP))
        prslk1 = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)
        # tem2da = np.zeros((IM, Model["levr"] + self.LTP))
        tem2da = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        tracer1 = np.zeros((IM, Model["levr"] + self.LTP, NTRAC))
        # ccnd = np.zeros((IM, Model["levr"] + self.LTP, min(4, Model["ncnd"])))

        ccnd = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=(DTYPE_FLT, (min(4, Model["ncnd"]),)))

        # cldtausw = np.zeros((IM, Model["levr"] + self.LTP))
        cldtausw = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        cldtaulw = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        htswc = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        htsw0 = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        htlwc = gt4py.storage.zeros(backend=backend, 
                                 default_origin=default_origin,
                                 shape=(IM, 1, Model["levr"] + self.LTP + 1),
                                 dtype=DTYPE_FLT)

        scmpsw = dict()

        Diag["topfsw"]["upfxc"] = np.zeros(IM)
        Diag["topfsw"]["dnfxc"] = np.zeros(IM)
        Diag["topfsw"]["upfx0"] = np.zeros(IM)
        Radtend["sfcfsw"]["upfxc"] = np.zeros(IM)
        Radtend["sfcfsw"]["dnfxc"] = np.zeros(IM)
        Radtend["sfcfsw"]["upfx0"] = np.zeros(IM)
        Radtend["sfcfsw"]["dnfx0"] = np.zeros(IM)

        Radtend["htrlw"] = np.zeros((IM, Model["levs"]))

        lhlwb = False
        lhlw0 = True
        lflxprf = False

        # File names for serialized random numbers in mcica_subcol
        sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_sw.nc")
        lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_lw.nc")

        #  --- ...  set local /level/layer indexes corresponding to in/out variables

        LMK = LM + self.LTP  # num of local layers
        LMP = LMK + 1  # num of local levels

        if self.lextop:
            if ivflip == 1:  # vertical from sfc upward
                kd = 0  # index diff between in/out and local
                kt = 1  # index diff between lyr and upper bound
                kb = 0  # index diff between lyr and lower bound
                lla = LMK  # local index at the 2nd level from top
                llb = LMP  # local index at toa level
                lya = LM  # local index for the 2nd layer from top
                lyb = LP1  # local index for the top layer
            else:  # vertical from toa downward
                kd = 1  # index diff between in/out and local
                kt = 0  # index diff between lyr and upper bound
                kb = 1  # index diff between lyr and lower bound
                lla = 2  # local index at the 2nd level from top
                llb = 1  # local index at toa level
                lya = 2  # local index for the 2nd layer from top
                lyb = 1  # local index for the top layer
        else:
            kd = 0
            if ivflip == 1:  # vertical from sfc upward
                kt = 1  # index diff between lyr and upper bound
                kb = 0  # index diff between lyr and lower bound
            else:  # vertical from toa downward
                kt = 0  # index diff between lyr and upper bound
                kb = 1  # index diff between lyr and lower bound

        raddt = min(Model["fhswr"], Model["fhlwr"])

        # -# Setup surface ground temperature and ground/air skin temperature
        # if required.

        if self.itsfc == 0:  # use same sfc skin-air/ground temp
            for i in range(IM):
                tskn[i,0] = Sfcprop["tsfc"][i]
                tsfg[i,0] = Sfcprop["tsfc"][i]
        else:  # use diff sfc skin-air/ground temp
            for i in range(IM):
                tskn[i,0] = Sfcprop["tsfc"][i]
                tsfg[i,0] = Sfcprop["tsfc"][i]

        # Prepare atmospheric profiles for radiation input.
        #
        lsk = 0
        if ivflip == 0 and LM < LEVS:
            lsk = LEVS - LM

        #           convert pressure unit from pa to mb
        for k in range(LM):
            k1 = k + kd
            k2 = k + lsk
            for i in range(IM):
                plvl[i, 0, k1 + kb] = Statein["prsi"][i, 0, k2 + kb] * 0.01  # pa to mb (hpa)
                plyr[i, 0, k1+1] = Statein["prsl"][i, 0, k2+1] * 0.01  # pa to mb (hpa)
                tlyr[i, 0, k1+1] = Statein["tgrs"][i, 0, k2+1]
                prslk1[i, 0, k1+1] = Statein["prslk"][i, 0, k2+1]

                #  - Compute relative humidity.
                es = min(
                    Statein["prsl"][i, 0, k2+1], fpvs(Statein["tgrs"][i, 0, k2+1])
                )  # fpvs and prsl in pa
                qs = max(
                    self.QMIN, con_eps * es / (Statein["prsl"][i, 0, k2+1] + con_epsm1 * es)
                )
                rhly[i, 0, k1+1] = max(
                    0.0, min(1.0, max(self.QMIN, Statein["qgrs"][i, 0, k2+1, 0]) / qs)
                )
                qstl[i, 0, k1+1] = qs

        # --- recast remaining all tracers (except sphum) forcing them all to be positive
        for j in range(1, NTRAC):
            for k in range(LM):
                k1 = k + kd
                k2 = k + lsk
                tracer1[:, k1, j] = np.maximum(0.0, Statein["qgrs"][:, 0, k2+1, j])

        if ivflip == 0:  # input data from toa to sfc
            for i in range(IM):
                plvl[i, 0, 1 + kd] = 0.01 * Statein["prsi"][i, 0, 0]  # pa to mb (hpa)

            if lsk != 0:
                for i in range(IM):
                    plvl[i, 0, 1 + kd] = 0.5 * (plvl[i, 0, 2 + kd] + plvl[i, 0, 1 + kd])
        else:  # input data from sfc to top
            for i in range(IM):
                plvl[i, 0, LP1 + kd - 1] = (
                    0.01 * Statein["prsi"][i, 0, LP1 + lsk - 1]
                )  # pa to mb (hpa)

            if lsk != 0:
                for i in range(IM):
                    plvl[i, 0, LM + kd - 1] = 0.5 * (
                        plvl[i, 0, LP1 + k - 1] + plvl[i, 0, LM + kd - 1]
                    )

        if self.lextop:  # values for extra top layer
            for i in range(IM):
                plvl[i, 0, llb - 1] = self.prsmin
                if plvl[i, 0, lla - 1] <= self.prsmin:
                    plvl[i, 0, lla - 1] = 2.0 * self.prsmin

                plyr[i, 0, lyb - 1 + 1] = 0.5 * plvl[i, 0, lla - 1]
                tlyr[i, 0, lyb - 1 + 1] = tlyr[i, 0, lya - 1 + 1]
                prslk1[i, 0, lyb - 1+1] = (
                    plyr[i, 0, lyb - 1 + 1] * 0.00001
                ) ** con_rocp  # plyr in Pa
                rhly[i, 0, lyb - 1+1] = rhly[i, 0, lya - 1+1]
                qstl[i, 0, lyb - 1+1] = qstl[i, 0, lya - 1+1]

            #  ---  note: may need to take care the top layer amount
            tracer1[:, lyb - 1, :] = tracer1[:, lya - 1, :]

        #  - Get layer ozone mass mixing ratio (if use ozone climatology data,
        #    call getozn()).

        if Model["ntoz"] > 0:  # interactive ozone generation
            for k in range(LMK):
                for i in range(IM):
                    olyr[i, 0, k+1] = max(self.QMIN, tracer1[i, k, Model["ntoz"] - 1])
        else:  # climatological ozone
            print("Climatological ozone not implemented")

        #  - Call coszmn(), to compute cosine of zenith angle (only when SW is called)

        if Model["lsswr"]:
            Radtend["coszen"], Radtend["coszdg"] = self.sol.coszmn(
                Grid["xlon"], Grid["sinlat"], Grid["coslat"], Model["solhr"], IM, me
            )

        #  - Call getgases(), to set up non-prognostic gas volume mixing
        #    ratioes (gasvmr).
        #  - gasvmr(:,:,1)  -  co2 volume mixing ratio
        #  - gasvmr(:,:,2)  -  n2o volume mixing ratio
        #  - gasvmr(:,:,3)  -  ch4 volume mixing ratio
        #  - gasvmr(:,:,4)  -  o2  volume mixing ratio
        #  - gasvmr(:,:,5)  -  co  volume mixing ratio
        #  - gasvmr(:,:,6)  -  cf11 volume mixing ratio
        #  - gasvmr(:,:,7)  -  cf12 volume mixing ratio
        #  - gasvmr(:,:,8)  -  cf22 volume mixing ratio
        #  - gasvmr(:,:,9)  -  ccl4 volume mixing ratio

        #  --- ...  set up non-prognostic gas volume mixing ratioes

        gasvmr = self.gas.getgases(
            plvl,
            Grid["xlon"],
            Grid["xlat"],
            IM,
            LMK,
        )

        #  - Get temperature at layer interface, and layer moisture.
        for k in range(1, LMK):
            for i in range(IM):
                tem2da[i, 0, k+1] = np.log(plyr[i, 0, k+1])
                tem2db[i, 0, k] = np.log(plvl[i, 0, k])

        if ivflip == 0:  # input data from toa to sfc
            for i in range(IM):
                tem1d[i, 0] = self.QME6
                tem2da[i, 0, 0+1] = np.log(plyr[i, 0, 0+1])
                tem2db[i, 0, 0] = np.log(max(self.prsmin, plvl[i, 0, 0]))
                tem2db[i, 0, LMP - 1] = np.log(plvl[i, 0, LMP - 1])
                tsfa[i, 0] = tlyr[i, 0, LMK - 1 + 1]  # sfc layer air temp
                tlvl[i, 0, 0] = tlyr[i, 0, 0+1]
                tlvl[i, 0, LMP - 1] = tskn[i,0]

            for k in range(LM):
                k1 = k + kd
                for i in range(IM):
                    qlyr[i, 0, k1+1] = max(tem1d[i, 0], Statein["qgrs"][i, 0, k+1, 0])
                    tem1d[i, 0] = min(self.QME5, qlyr[i, k1])
                    tvly[i, 0, k1+1] = Statein["tgrs"][i, 0, k+1] * (
                        1.0 + con_fvirt * qlyr[i, 0, k1+1]
                    )  # virtual T (K)
                    delp[i, 0, k1+1] = plvl[i, 0, k1 + 1] - plvl[i, 0, k1]

            if self.lextop:
                for i in range(IM):
                    qlyr[i, 0, lyb - 1+1] = qlyr[i, 0, lya - 1+1]
                    tvly[i, 0, lyb - 1+1] = tvly[i, 0, lya - 1+1]
                    delp[i, 0, lyb - 1+1] = plvl[i, 0, lla - 1] - plvl[i, 0, llb - 1]

            for k in range(1, LMK):
                for i in range(IM):
                    tlvl[i, 0, k] = tlyr[i, 0, k+1] + (tlyr[i, 0, k - 1+1] - tlyr[i, 0, k+1]) * (
                        tem2db[i, 0, k] - tem2da[i, 0, k+1]
                    ) / (tem2da[i, 0, k - 1+1] - tem2da[i, 0, k+1])

            #  ---  ...  level height and layer thickness (km)

            tem0d = 0.001 * con_rog
            for i in range(IM):
                for k in range(LMK):
                    dz[i, 0, k+1] = tem0d * (tem2db[i, 0, k + 1] - tem2db[i, 0, k]) * tvly[i, 0, k+1]
        else:

            for i in range(IM):
                tem1d[i, 0] = self.QME6
                tem2da[i, 0, 0+1] = np.log(plyr[i, 0, 0+1])
                tem2db[i, 0, 0] = np.log(plvl[i, 0, 0])
                tem2db[i, 0, LMP - 1] = np.log(max(self.prsmin, plvl[i, 0, LMP - 1]))
                tsfa[i, 0] = tlyr[i, 0, 0+1]  # sfc layer air temp
                tlvl[i, 0, 0] = tskn[i,0]
                tlvl[i, 0, LMP - 1] = tlyr[i, 0, LMK - 1+1]

            for k in range(LM - 1, -1, -1):
                for i in range(IM):
                    qlyr[i, 0, k+1] = max(tem1d[i, 0], Statein["qgrs"][i, 0, k+1, 0])
                    tem1d[i, 0] = min(self.QME5, qlyr[i, 0, k+1])
                    tvly[i, 0, k+1] = Statein["tgrs"][i, 0, k+1] * (
                        1.0 + con_fvirt * qlyr[i, 0, k+1]
                    )  # virtual T (K)
                    delp[i, 0, k+1] = plvl[i, 0, k] - plvl[i, 0, k + 1]

            if self.lextop:
                for i in range(IM):
                    qlyr[i, 0, lyb - 1+1] = qlyr[i, 0, lya - 1+1]
                    tvly[i, 0, lyb - 1+1] = tvly[i, 0, lya - 1+1]
                    delp[i, 0, lyb - 1+1] = plvl[i, 0, lla - 1] - plvl[i, 0, llb - 1]

            for k in range(LMK - 1):
                for i in range(IM):
                    tlvl[i, 0, k + 1] = tlyr[i, 0, k+1] + (tlyr[i, 0, k + 1+1] - tlyr[i, 0, k+1]) * (
                        tem2db[i, 0, k + 1] - tem2da[i, 0, k+1]
                    ) / (tem2da[i, 0, k + 1+1] - tem2da[i, 0, k+1])

            #  ---  ...  level height and layer thickness (km)

            tem0d = 0.001 * con_rog
            for i in range(IM):
                for k in range(LMK - 1, -1, -1):
                    dz[i, 0, k+1] = tem0d * (tem2db[i, 0, k] - tem2db[i, 0, k + 1]) * tvly[i, 0, k+1]

        #  - Check for daytime points for SW radiation.

        nday = 0
        for i in range(IM):
            if Radtend["coszen"][i] >= 0.0001:
                nday += 1
                idxday[nday - 1] = i + 1

        #  - Call module_radiation_aerosols::setaer(),to setup aerosols
        # property profile for radiation.

        faersw, faerlw, aerodp = self.aer.setaer(
            plvl,
            plyr,
            prslk1,
            tvly,
            rhly,
            Sfcprop["slmsk"],
            tracer1,
            Grid["xlon"],
            Grid["xlat"],
            IM,
            LMK,
            LMP,
            Model["lsswr"],
            Model["lslwr"],
        )

        #  - Obtain cloud information for radiation calculations
        #    (clouds,cldsa,mtopa,mbota)
        #     for  prognostic cloud:
        #    - For Zhao/Moorthi's prognostic cloud scheme,
        #      call module_radiation_clouds::progcld1()
        #    - For Zhao/Moorthi's prognostic cloud+pdfcld,
        #      call module_radiation_clouds::progcld3()
        #      call module_radiation_clouds::progclduni() for unified cloud and ncld=2

        #  --- ...  obtain cloud information for radiation calculations

        if Model["ncnd"] == 1:  # Zhao_Carr_Sundqvist
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, 0, k+1, 0] = tracer1[i, k, ntcw - 1]  # liquid water/ice
        elif Model["ncnd"] == 2:  # MG
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, 0, k+1, 0] = tracer1[i, k, ntcw - 1]  # liquid water
                    ccnd[i, 0, k+1, 1] = tracer1[i, k, ntiw - 1]  # ice water
        elif Model["ncnd"] == 4:  # MG2
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, 0, k+1, 0] = tracer1[i, k, ntcw - 1]  # liquid water
                    ccnd[i, 0, k+1, 1] = tracer1[i, k, ntiw - 1]  # ice water
                    ccnd[i, 0, k+1, 2] = tracer1[i, k, ntrw - 1]  # rain water
                    ccnd[i, 0, k+1, 3] = tracer1[i, k, ntsw - 1]  # snow water
        elif Model["ncnd"] == 5:  # GFDL MP, Thompson, MG3
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, 0, k+1, 0] = tracer1[i, k, ntcw - 1]  # liquid water
                    ccnd[i, 0, k+1, 1] = tracer1[i, k, ntiw - 1]  # ice water
                    ccnd[i, 0, k+1, 2] = tracer1[i, k, ntrw - 1]  # rain water
                    ccnd[i, 0, k+1, 3] = (
                        tracer1[i, k, ntsw - 1] + tracer1[i, k, ntgl - 1]
                    )  # snow + grapuel

        for n in range(ncndl):
            for k in range(LMK):
                for i in range(IM):
                    if ccnd[i, 0, k+1, n] < con_epsq:
                        ccnd[i, 0, k+1, n] = 0.0

        if Model["imp_physics"] == 11:
            if not Model["lgfdlmprad"]:

                # rsun the  summation methods and order make the difference in calculation
                ccnd[:, 0, 1:, 0] = tracer1[:, :LMK, ntcw - 1]
                ccnd[:, 0, 1:, 0] = ccnd[:, 0, 1:, 0] + tracer1[:, :LMK, ntrw - 1]
                ccnd[:, 0, 1:, 0] = ccnd[:, 0, 1:, 0] + tracer1[:, :LMK, ntiw - 1]
                ccnd[:, 0, 1:, 0] = ccnd[:, 0, 1:, 0] + tracer1[:, :LMK, ntsw - 1]
                ccnd[:, 0, 1:, 0] = ccnd[:, 0, 1:, 0] + tracer1[:, :LMK, ntgl - 1]

            for k in range(LMK):
                for i in range(IM):
                    if ccnd[i, 0, k+1, 0] < self.EPSQ:
                        ccnd[i, 0, k+1, 0] = 0.0

        if Model["uni_cld"]:
            if Model["effr_in"]:
                for k in range(LM):
                    k1 = k + kd
                    for i in range(IM):
                        cldcov[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, Model["indcld"] - 1]
                        effrl[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 1]
                        effri[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 2]
                        effrr[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 3]
                        effrs[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 4]
            else:
                for k in range(LM):
                    k1 = k + kd
                    for i in range(IM):
                        cldcov[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, Model["indcld"] - 1]
        elif Model["imp_physics"] == 11:  # GFDL MP
            cldcov[:IM, 0, kd+1 : LM + kd+1] = tracer1[:IM, :LM, Model["ntclamt"] - 1]
            if Model["effr_in"]:
                for k in range(LM):
                    k1 = k + kd
                    for i in range(IM):
                        effrl[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 0]
                        effri[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 1]
                        effrr[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 2]
                        effrs[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 3]
        else:  # neither of the other two cases
            cldcov = 0.0

        #  --- add suspended convective cloud water to grid-scale cloud water
        #      only for cloud fraction & radiation computation
        #      it is to enhance cloudiness due to suspended convec cloud water
        #      for zhao/moorthi's (imp_phys=99) &
        #          ferrier's (imp_phys=5) microphysics schemes

        if (
            Model["num_p3d"] == 4 and Model["npdf3d"] == 3
        ):  # same as Model%imp_physics = 99
            for k in range(LM):
                k1 = k + kd
                for i in range(IM):
                    deltaq[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 4]
                    cnvw[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 5]
                    cnvc[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, 6]
        elif (
            Model["npdf3d"] == 0 and Model["ncnvcld3d"] == 1
        ):  # same as MOdel%imp_physics=98
            for k in range(LM):
                k1 = k + kd
                for i in range(IM):
                    deltaq[i, 0, k1+1] = 0.0
                    cnvw[i, 0, k1+1] = Tbd["phy_f3d"][i, 0, k+1, Model["num_p3d"]]
                    cnvc[i, 0, k1+1] = 0.0
        else:  # all the rest
            for k in range(LMK):
                for i in range(IM):
                    deltaq[i, 0, k+1] = 0.0
                    cnvw[i, 0, k+1] = 0.0
                    cnvc[i, 0, k+1] = 0.0

        if self.lextop:
            for i in range(IM):
                cldcov[i, 0, lyb - 1+1] = cldcov[i, 0, lya - 1+1]
                deltaq[i, 0, lyb - 1+1] = deltaq[i, 0, lya - 1+1]
                cnvw[i, 0, lyb - 1+1] = cnvw[i, 0, lya - 1+1]
                cnvc[i, 0, lyb - 1+1] = cnvc[i, 0, lya - 1+1]

            if Model["effr_in"]:
                for i in range(IM):
                    effrl[i, 0, lyb - 1+1] = effrl[i, 0, lya - 1+1]
                    effri[i, 0, lyb - 1+1] = effri[i, 0, lya - 1+1]
                    effrr[i, 0, lyb - 1+1] = effrr[i, 0, lya - 1+1]
                    effrs[i, 0, lyb - 1+1] = effrs[i, 0, lya - 1+1]

        if Model["imp_physics"] == 99:
            ccnd[:IM, 0, 1:LMK+1, 0] = ccnd[:IM, 0, 1:LMK+1, 0] + cnvw[:IM, 0, :LMK]

        clouds, cldsa, mtopa, mbota, de_lgth = self.cld.progcld4(
            plyr,
            plvl,
            tlyr,
            tvly,
            qlyr,
            qstl,
            rhly,
            ccnd[:IM, 0, :LMK+1, 0],
            cnvw,
            cnvc,
            Grid["xlat"],
            Grid["xlon"],
            Sfcprop["slmsk"],
            cldcov,
            dz,
            delp,
            IM,
            LMK,
            LMP,
        )

        #  --- ...  start radiation calculations
        #           remember to set heating rate unit to k/sec!

        # mg, sfc-perts
        #  ---  scale random patterns for surface perturbations with
        #  perturbation size
        #  ---  turn vegetation fraction pattern into percentile pattern

        if Model["do_sfcperts"]:
            print("Surface perturbation not implemented!")

        # mg, sfc-perts

        if Model["do_only_clearsky_rad"]:
            clouds[:, 0, :, 0] = 0.0  # layer total cloud fraction
            clouds[:, 0, :, 1] = 0.0  # layer cloud liq water path
            clouds[:, 0, :, 3] = 0.0  # layer cloud ice water path
            clouds[:, 0, :, 5] = 0.0  # layer rain water path
            clouds[:, 0, :, 7] = 0.0  # layer snow water path
            cldsa[:, :] = 0.0  # fraction of clouds for low, mid, hi, tot, bl

        # Start SW radiation calculations
        if Model["lsswr"]:

            #  - Call module_radiation_surface::setalb() to setup surface albedo.
            #  for SW radiation.

            sfcalb = self.sfc.setalb(
                Sfcprop["slmsk"],
                Sfcprop["snowd"],
                Sfcprop["sncovr"],
                Sfcprop["snoalb"],
                Sfcprop["zorl"],
                Radtend["coszen"],
                tsfg,
                tsfa,
                Sfcprop["hprime"],
                Sfcprop["alvsf"],
                Sfcprop["alnsf"],
                Sfcprop["alvwf"],
                Sfcprop["alnwf"],
                Sfcprop["facsf"],
                Sfcprop["facwf"],
                Sfcprop["fice"],
                Sfcprop["tisfc"],
                IM,
                alb1d,
                Model["pertalb"],
            )

            # Approximate mean surface albedo from vis- and nir-  diffuse values.
            Radtend["sfalb"][:] = np.maximum(0.01, 0.5 * (sfcalb[:, 1] + sfcalb[:, 3]))

            lhswb = False
            lhsw0 = True
            lflxprf = False
            lfdncmp = True

            if nday > 0:

                #  - Call module_radsw_main::swrad(), to compute SW heating rates and
                #   fluxes.

                self.rsw.create_input_data_rad_driver(
                    plyr,
                    plvl,
                    tlyr,
                    tlvl,
                    qlyr,
                    olyr,
                    gasvmr,
                    clouds,
                    faersw,
                    sfcalb,
                    dz,
                    delp,
                    de_lgth,
                    Radtend["coszen"],
                    Model["solcon"],
                    nday,
                    idxday,
                    IM,
                    LMK,
                    LMP,
                    Model["lprnt"]
                )

                self.rsw.set_locdict_zero()

                if Model["swhtr"]:
                    self.rsw.swrad(rank=Rank)
                else:
                    self.rsw.swrad(rank=Rank)

                # htswc = np.zeros((self.rsw.outdict_gt4py["htswc"].shape[0],
                #                   self.rsw.outdict_gt4py["htswc"].shape[2]-1)) 
                htswc[:,0, 1:] = self.rsw.outdict_gt4py["htswc"][:,0,1:]
                # htsw0 = np.zeros((self.rsw.outdict_gt4py["htsw0"].shape[0],
                #                   self.rsw.outdict_gt4py["htsw0"].shape[2]-1)) 
                htsw0[:,0, 1:] = self.rsw.outdict_gt4py["htsw0"][:,0,1:]
                Diag["topfsw"]["upfxc"] = self.rsw.outdict_gt4py["upfxc_t"][:,0]
                Diag["topfsw"]["dnfxc"] = self.rsw.outdict_gt4py["dnfxc_t"][:,0]
                Diag["topfsw"]["upfx0"] = self.rsw.outdict_gt4py["upfx0_t"][:,0]
                Radtend["sfcfsw"]["upfxc"] = self.rsw.outdict_gt4py["upfxc_s"][:,0]
                Radtend["sfcfsw"]["dnfxc"] = self.rsw.outdict_gt4py["dnfxc_s"][:,0]
                Radtend["sfcfsw"]["upfx0"] = self.rsw.outdict_gt4py["upfx0_s"][:,0]
                Radtend["sfcfsw"]["dnfx0"] = self.rsw.outdict_gt4py["dnfx0_s"][:,0]
                scmpsw["uvbf0"] = self.rsw.outdict_gt4py["uvbf0"][:,0]
                scmpsw["uvbfc"] = self.rsw.outdict_gt4py["uvbfc"][:,0]
                scmpsw["nirbm"] = self.rsw.outdict_gt4py["nirbm"][:,0]
                scmpsw["nirdf"] = self.rsw.outdict_gt4py["nirdf"][:,0]
                scmpsw["visbm"] = self.rsw.outdict_gt4py["visbm"][:,0]
                scmpsw["visdf"] = self.rsw.outdict_gt4py["visdf"][:,0]
                cldtausw[:,0,1:] = self.rsw.outdict_gt4py["cldtausw"][:,0,1:]

                for k in range(LM):
                    k1 = k + kd
                    Radtend["htrsw"][:IM, k] = htswc[:IM, 0, k1+1]

                #     We are assuming that radiative tendencies are from bottom to top
                # --- repopulate the points above levr i.e. LM
                if LM < LEVS:
                    for k in range(LM, LEVS):
                        Radtend["htrsw"][:IM, k] = Radtend["htrsw"][:IM, LM - 1]

                if Model["swhtr"]:
                    for k in range(LM):
                        k1 = k + kd
                        Radtend["swhc"][:IM, k] = htsw0[:IM, 0, k1+1]

                    # --- repopulate the points above levr i.e. LM
                    if LM < LEVS:
                        for k in range(LM, LEVS):
                            Radtend["swhc"][:IM, k] = Radtend["swhc"][:IM, LM - 1]

                #  --- surface down and up spectral component fluxes
                #  - Save two spectral bands' surface downward and upward fluxes for
                #    output.

                for i in range(IM):
                    Coupling["nirbmdi"][i,0] = scmpsw["nirbm"][i]
                    Coupling["nirdfdi"][i,0] = scmpsw["nirdf"][i]
                    Coupling["visbmdi"][i,0] = scmpsw["visbm"][i]
                    Coupling["visdfdi"][i,0] = scmpsw["visdf"][i]

                    Coupling["nirbmui"][i,0] = scmpsw["nirbm"][i] * sfcalb[i, 0]
                    Coupling["nirdfui"][i,0] = scmpsw["nirdf"][i] * sfcalb[i, 1]
                    Coupling["visbmui"][i,0] = scmpsw["visbm"][i] * sfcalb[i, 2]
                    Coupling["visdfui"][i,0] = scmpsw["visdf"][i] * sfcalb[i, 3]

            else:

                Radtend["htrsw"][:, :] = 0.0

                for i in range(IM):
                    Coupling["nirbmdi"][i,0] = 0.0
                    Coupling["nirdfdi"][i,0] = 0.0
                    Coupling["visbmdi"][i,0] = 0.0
                    Coupling["visdfdi"][i,0] = 0.0
                    Coupling["nirbmui"][i,0] = 0.0
                    Coupling["nirdfui"][i,0] = 0.0
                    Coupling["visbmui"][i,0] = 0.0
                    Coupling["visdfui"][i,0] = 0.0

                if Model["swhtr"]:
                    Radtend["swhc"][:, :] = 0
                    cldtausw[:, 0, 1:] = 0.0

            # --- radiation fluxes for other physics processes
            for i in range(IM):
                Coupling["sfcnsw"][i,0] = (
                    Radtend["sfcfsw"]["dnfxc"][i] - Radtend["sfcfsw"]["upfxc"][i]
                )
                Coupling["sfcdsw"][i,0] = Radtend["sfcfsw"]["dnfxc"][i]

        # Start LW radiation calculations
        if Model["lslwr"]:

            #  - Call module_radiation_surface::setemis(),to setup surface
            # emissivity for LW radiation.

            Radtend["semis"] = self.sfc.setemis(
                Grid["xlon"],
                Grid["xlat"],
                Sfcprop["slmsk"],
                Sfcprop["snowd"],
                Sfcprop["sncovr"],
                Sfcprop["zorl"],
                tsfg,
                tsfa,
                Sfcprop["hprime"],
                IM,
            )

            # Take current radiation_driver variables that're inputs to the LW scheme
            # and convert them into storages
            self.rlw.create_input_data_rad_driver(
                plyr,
                plvl,
                tlyr,
                tlvl,
                qlyr,
                olyr,
                gasvmr,
                clouds,
                Tbd["icsdlw"],
                faerlw,
                Radtend["semis"],
                tsfg,
                dz,
                delp,
                de_lgth,
                IM,
                LMK,
                LMP,
                Model["lprnt"],
            )

            self.rlw.set_locdict_zero()

            #  - Call module_radlw_main::lwrad(), to compute LW heating rates and
            #    fluxes.

            if Model["lwhtr"]:
                self.rlw.lwrad(rank=Rank)
            else:
                self.rlw.lwrad(rank=Rank)

            # Write the outputs from the rlw.lwrad execution back into the radiation scheme
            # htlwc = np.zeros((self.rlw.outdict_gt4py["htlwc"].shape[0],
            #                   self.rlw.outdict_gt4py["htlwc"].shape[2]-1)) 
            htlwc[:,0, 1:] = self.rlw.outdict_gt4py["htlwc"][:,0,1:]
            htlw0 = np.zeros((self.rlw.outdict_gt4py["htlw0"].shape[0],
                              self.rlw.outdict_gt4py["htlw0"].shape[2]-1)) 
            htlw0[:,:] = self.rlw.outdict_gt4py["htlw0"][:,0,1:]
            Diag["topflw"]["upfxc"] = self.rlw.outdict_gt4py["upfxc_t"][:,0]
            Diag["topflw"]["upfx0"] = self.rlw.outdict_gt4py["upfx0_t"][:,0]
            Radtend["sfcflw"]["upfxc"] = self.rlw.outdict_gt4py["upfxc_s"][:,0]
            Radtend["sfcflw"]["upfx0"] = self.rlw.outdict_gt4py["upfx0_s"][:,0]
            Radtend["sfcflw"]["dnfxc"] = self.rlw.outdict_gt4py["dnfxc_s"][:,0]
            Radtend["sfcflw"]["dnfx0"] = self.rlw.outdict_gt4py["dnfx0_s"][:,0]
            # cldtaulw = np.zeros((self.rlw.outdict_gt4py["cldtaulw"].shape[0],
            #                      self.rlw.outdict_gt4py["cldtaulw"].shape[2]-1))
            cldtaulw[:, 0, 1:] = self.rlw.outdict_gt4py["cldtaulw"][:,0,1:]

            # Save calculation results
            #  - Save surface air temp for diurnal adjustment at model t-steps
            Radtend["tsflw"][:] = tsfa[:, 0]

            for k in range(LM):
                k1 = k + kd
                Radtend["htrlw"][:IM, k] = htlwc[:IM, 0, k1+1]

            # --- repopulate the points above levr
            if LM < LEVS:
                for k in range(LM, LEVS):
                    Radtend["htrlw"][IM, k] = Radtend["htrlw"][:IM, LM - 1]

            if Model["lwhtr"]:
                for k in range(LM):
                    k1 = k + kd
                    Radtend["lwhc"][:IM, k] = htlw0[:IM, k1]

                # --- repopulate the points above levr
                if LM < LEVS:
                    for k in range(LM, LEVS):
                        Radtend["lwhc"][:IM, k] = Radtend["lwhc"][:IM, LM - 1]

            # --- radiation fluxes for other physics processes
            Coupling["sfcdlw"][:,0] = Radtend["sfcflw"]["dnfxc"]

        #  - For time averaged output quantities (including total-sky and
        #    clear-sky SW and LW fluxes at TOA and surface; conventional
        #    3-domain cloud amount, cloud top and base pressure, and cloud top
        #    temperature; aerosols AOD, etc.), store computed results in
        #    corresponding slots of array fluxr with appropriate time weights.

        #  --- ...  collect the fluxr data for wrtsfc

        if Model["lssav"]:
            if Model["lsswr"]:
                for i in range(IM):
                    Diag["fluxr"][i, 33] = (
                        Diag["fluxr"][i, 33] + Model["fhswr"] * aerodp[i, 0]
                    )  # total aod at 550nm
                    Diag["fluxr"][i, 34] = (
                        Diag["fluxr"][i, 34] + Model["fhswr"] * aerodp[i, 1]
                    )  # DU aod at 550nm
                    Diag["fluxr"][i, 35] = (
                        Diag["fluxr"][i, 35] + Model["fhswr"] * aerodp[i, 2]
                    )  # BC aod at 550nm
                    Diag["fluxr"][i, 36] = (
                        Diag["fluxr"][i, 36] + Model["fhswr"] * aerodp[i, 3]
                    )  # OC aod at 550nm
                    Diag["fluxr"][i, 37] = (
                        Diag["fluxr"][i, 37] + Model["fhswr"] * aerodp[i, 4]
                    )  # SU aod at 550nm
                    Diag["fluxr"][i, 38] = (
                        Diag["fluxr"][i, 38] + Model["fhswr"] * aerodp[i, 5]
                    )  # SS aod at 550nm

            #  ---  save lw toa and sfc fluxes
            if Model["lslwr"]:
                #  ---  lw total-sky fluxes
                for i in range(IM):
                    Diag["fluxr"][i, 0] = (
                        Diag["fluxr"][i, 0]
                        + Model["fhlwr"] * Diag["topflw"]["upfxc"][i]
                    )  # total sky top lw up
                    Diag["fluxr"][i, 18] = (
                        Diag["fluxr"][i, 18]
                        + Model["fhlwr"] * Radtend["sfcflw"]["dnfxc"][i]
                    )  # total sky sfc lw dn
                    Diag["fluxr"][i, 19] = (
                        Diag["fluxr"][i, 19]
                        + Model["fhlwr"] * Radtend["sfcflw"]["upfxc"][i]
                    )  # total sky sfc lw up
                    #  ---  lw clear-sky fluxes
                    Diag["fluxr"][i, 27] = (
                        Diag["fluxr"][i, 27]
                        + Model["fhlwr"] * Diag["topflw"]["upfx0"][i]
                    )  # clear sky top lw up
                    Diag["fluxr"][i, 29] = (
                        Diag["fluxr"][i, 29]
                        + Model["fhlwr"] * Radtend["sfcflw"]["dnfx0"][i]
                    )  # clear sky sfc lw dn
                    Diag["fluxr"][i, 32] = (
                        Diag["fluxr"][i, 32]
                        + Model["fhlwr"] * Radtend["sfcflw"]["upfx0"][i]
                    )  # clear sky sfc lw up

            #  ---  save sw toa and sfc fluxes with proper diurnal sw wgt. coszen=mean cosz over daylight
            #       part of sw calling interval, while coszdg= mean cosz over entire interval
            if Model["lsswr"]:
                for i in range(IM):
                    if Radtend["coszen"][i] > 0.0:
                        #  --- sw total-sky fluxes
                        #      -------------------
                        tem0d = (
                            Model["fhswr"] * Radtend["coszdg"][i] / Radtend["coszen"][i]
                        )
                        Diag["fluxr"][i, 1] = (
                            Diag["fluxr"][i, 1] + Diag["topfsw"]["upfxc"][i] * tem0d
                        )  # total sky top sw up
                        Diag["fluxr"][i, 2] = (
                            Diag["fluxr"][i, 2] + Radtend["sfcfsw"]["upfxc"][i] * tem0d
                        )  # total sky sfc sw up
                        Diag["fluxr"][i, 3] = (
                            Diag["fluxr"][i, 4] + Radtend["sfcfsw"]["dnfxc"][i] * tem0d
                        )  # total sky sfc sw dn
                        #  --- sw uv-b fluxes
                        #      --------------
                        Diag["fluxr"][i, 20] = (
                            Diag["fluxr"][i, 20] + scmpsw["uvbfc"][i] * tem0d
                        )  # total sky uv-b sw dn
                        Diag["fluxr"][i, 21] = (
                            Diag["fluxr"][i, 21] + scmpsw["uvbf0"][i] * tem0d
                        )  # clear sky uv-b sw dn
                        #  --- sw toa incoming fluxes
                        #      ----------------------
                        Diag["fluxr"][i, 22] = (
                            Diag["fluxr"][i, 22] + Diag["topfsw"]["dnfxc"][i] * tem0d
                        )  # top sw dn
                        #  --- sw sfc flux components
                        #      ----------------------
                        Diag["fluxr"][i, 23] = (
                            Diag["fluxr"][i, 23] + scmpsw["visbm"][i] * tem0d
                        )  # uv/vis beam sw dn
                        Diag["fluxr"][i, 24] = (
                            Diag["fluxr"][i, 24] + scmpsw["visdf"][i] * tem0d
                        )  # uv/vis diff sw dn
                        Diag["fluxr"][i, 25] = (
                            Diag["fluxr"][i, 25] + scmpsw["nirbm"][i] * tem0d
                        )  # nir beam sw dn
                        Diag["fluxr"][i, 26] = (
                            Diag["fluxr"][i, 26] + scmpsw["nirdf"][i] * tem0d
                        )  # nir diff sw dn
                        #  --- sw clear-sky fluxes
                        #      -------------------
                        Diag["fluxr"][i, 28] = (
                            Diag["fluxr"][i, 28] + Diag["topfsw"]["upfx0"][i] * tem0d
                        )  # clear sky top sw up
                        Diag["fluxr"][i, 30] = (
                            Diag["fluxr"][i, 30] + Radtend["sfcfsw"]["upfx0"][i] * tem0d
                        )  # clear sky sfc sw up
                        Diag["fluxr"][i, 31] = (
                            Diag["fluxr"][i, 31] + Radtend["sfcfsw"]["dnfx0"][i] * tem0d
                        )  # clear sky sfc sw dn

            #  ---  save total and boundary layer clouds

            if Model["lsswr"] or Model["lslwr"]:
                for i in range(IM):
                    Diag["fluxr"][i, 16] = Diag["fluxr"][i, 16] + raddt * cldsa[i, 3]
                    Diag["fluxr"][i, 17] = Diag["fluxr"][i, 17] + raddt * cldsa[i, 4]

                #  ---  save cld frac,toplyr,botlyr and top temp, note that the order
                #       of h,m,l cloud is reversed for the fluxr output.
                #  ---  save interface pressure (pa) of top/bot

                for j in range(3):
                    for i in range(IM):
                        tem0d = raddt * cldsa[i, j]
                        itop = int(mtopa[i, j] - kd)
                        ibtc = int(mbota[i, j] - kd)
                        Diag["fluxr"][i, 6 - j] = Diag["fluxr"][i, 6 - j] + tem0d
                        Diag["fluxr"][i, 9 - j] = (
                            Diag["fluxr"][i, 9 - j]
                            + tem0d * Statein["prsi"][i, 0, itop + kt - 1]
                        )
                        Diag["fluxr"][i, 12 - j] = (
                            Diag["fluxr"][i, 12 - j]
                            + tem0d * Statein["prsi"][i, 0, ibtc + kb - 1]
                        )
                        Diag["fluxr"][i, 15 - j] = (
                            Diag["fluxr"][i, 15 - j]
                            + tem0d * Statein["tgrs"][i, 0, itop - 1+1]
                        )

                        # Anning adds optical depth and emissivity output
                        tem1 = 0.0
                        tem2 = 0.0
                        for k in range(ibtc - 1, itop):
                            tem1 = tem1 + cldtausw[i, 0, k+1]  # approx .55 mu channel
                            tem2 = tem2 + cldtaulw[i, 0, k+1]  # approx 10. mu channel

                        Diag["fluxr"][i, 41 - j] = (
                            Diag["fluxr"][i, 41 - j] + tem0d * tem1
                        )
                        Diag["fluxr"][i, 44 - j] = Diag["fluxr"][i, 44 - j] + tem0d * (
                            1.0 - np.exp(-tem2)
                        )

        return Radtend, Diag
