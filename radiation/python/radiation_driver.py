import numpy as np

from config import *
from radphysparam import *
from phys_const import con_eps, con_epsm1, con_rocp, con_fvirt, con_rog, con_epsq
from funcphys import fpvs

from radlw.radiation_astronomy import AstronomyClass
from radlw.radiation_aerosols import AerosolClass
from radlw.radiation_clouds import CloudClass
from radlw.radiation_gases import GasClass
from radlw.radiation_sfc import SurfaceClass
from radlw.radlw_main import RadLWClass
from radlw.radsw_main import RadSWClass
from radlw.coszmn import coszmn


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

    def radinit(self, si, NLAY, imp_physics, me):
        self.itsfc = iemsflg / 10  # sfc air/ground temp control
        self.loz1st = ioznflg == 0  # first-time clim ozone data read flag
        self.month0 = 0
        self.iyear0 = 0
        self.monthd = 0

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
                print(f"- ERROR!!! ISUBCLW={isubclw}, is not a valid option")

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
                print(f"- ERROR!!! ISUBCSW={isubcsw}, is not a valid option")

            if isubcsw != isubclw:
                print(
                    "- *** Notice *** ISUBCSW /= ISUBCLW !!!", f"{isubcsw}, {isubclw}"
                )

        # -# Initialization
        #  --- ...  astronomy initialization routine
        self.sol = AstronomyClass(me, isolar)
        #  --- ...  aerosols initialization routine
        self.aer = AerosolClass(NLAY, me, iaerflg)
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

    def radupdate(
        self, idate, jdate, deltsw, deltim, lsswr, me, slag, sdec, cdec, solcon
    ):
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
            if isolar == 0 or isolar == 10:
                lsol_chg = False
            elif self.iyear0 != iyear:
                lsol_chg = True
            else:
                lsol_chg = isolar == 4 and lmon_chg

            self.iyear0 = iyear

            print(f"lsol_chg = {lsol_chg}")

            slag, sdec, cdec, solcon = self.sol.sol_update(
                jdate, kyear, deltsw, deltim, lsol_chg, me
            )

        # Call module_radiation_aerosols::aer_update(), monthly update, no
        # time interpolation
        if lmon_chg:

            NLAY = 63

            self.aer.aer_update(iyear, imon, me)

        # -# Call co2 and other gases update routine:
        # module_radiation_gases::gas_update()
        if self.monthd != kmon:
            self.monthd = kmon
            lco2_chg = True
        else:
            lco2_chg = False

        self.gas.gas_update(kyear, kmon, kday, khour, self.loz1st, lco2_chg, me)

        if self.loz1st:
            self.loz1st = False

    def GFS_radiation_driver(
        self,
        Model,
        Statein,
        Stateout,
        Sfcprop,
        Coupling,
        Grid,
        Tbd,
        Cldprop,
        Radtend,
        Diag,
    ):

        if not (Model.lsswr or Model.lslwr):
            return

        # --- set commonly used integers
        me = Model.me
        LM = Model.levr
        LEVS = Model.levs
        IM = Grid.xlon.shape[0]
        NFXR = Model.nfxr
        NTRAC = Model.ntrac  # tracers in grrad strip off sphum - start tracer1(2:NTRAC)
        ntcw = Model.ntcw
        ntiw = Model.ntiw
        ncld = Model.ncld
        ntrw = Model.ntrw
        ntsw = Model.ntsw
        ntgl = Model.ntgl
        ncndl = min(Model.ncnd, 4)

        LP1 = LM + 1  # num of in/out levels

        tskn = np.zeros(IM)
        tsfa = np.zeros(IM)
        tsfg = np.zeros(IM)
        tem1d = np.zeros(IM)
        idxday = np.zeros(IM, dtype=DTYPE_INT)

        plvl = np.zeros((IM, Model.levr + self.LTP + 1))
        tlvl = np.zeros((IM, Model.levr + self.LTP + 1))
        tem2db = np.zeros((IM, Model.levr + self.LTP + 1))

        plyr = np.zeros((IM, Model.levr + self.LTP))
        tlyr = np.zeros((IM, Model.levr + self.LTP))
        olyr = np.zeros((IM, Model.levr + self.LTP))
        qlyr = np.zeros((IM, Model.levr + self.LTP))
        rhly = np.zeros((IM, Model.levr + self.LTP))
        tvly = np.zeros((IM, Model.levr + self.LTP))
        delp = np.zeros((IM, Model.levr + self.LTP))
        qstl = np.zeros((IM, Model.levr + self.LTP))
        dz = np.zeros((IM, Model.levr + self.LTP))
        prslk1 = np.zeros((IM, Model.levr + self.LTP))
        tem2da = np.zeros((IM, Model.levr + self.LTP))

        tracer1 = np.zeros((IM, Model.levr + self.LTP, NTRAC - 1))

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

        raddt = min(Model.fhswr, Model.fhlwr)

        # -# Setup surface ground temperature and ground/air skin temperature
        # if required.

        if self.itsfc == 0:  # use same sfc skin-air/ground temp
            for i in range(IM):
                tskn[i] = Sfcprop.tsfc[i]
                tsfg[i] = Sfcprop.tsfc[i]
        else:  # use diff sfc skin-air/ground temp
            for i in range(IM):
                tskn[i] = Sfcprop.tsfc[i]
                tsfg[i] = Sfcprop.tsfc[i]

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
                plvl[i, k1 + kb] = Statein.prsi[i, k2 + kb] * 0.01  # pa to mb (hpa)
                plyr[i, k1] = Statein.prsl[i, k2] * 0.01  # pa to mb (hpa)
                tlyr[i, k1] = Statein.tgrs[i, k2]
                prslk1[i, k1] = Statein.prslk[i, k2]

                #  - Compute relative humidity.
                es = min(
                    Statein.prsl[i, k2], fpvs(Statein.tgrs[i, k2])
                )  # fpvs and prsl in pa
                qs = max(
                    self.QMIN, con_eps * es / (Statein.prsl[i, k2] + con_epsm1 * es)
                )
                rhly[i, k1] = max(
                    0.0, min(1.0, max(self.QMIN, Statein.qgrs(i, k2, 1)) / qs)
                )
                qstl[i, k1] = qs

        # --- recast remaining all tracers (except sphum) forcing them all to be positive
        for j in range(1, NTRAC):
            for k in range(LM):
                k1 = k + kd
                k2 = k + lsk
                tracer1[:, k1, j] = max(0.0, Statein.qgrs[:, k2, j])

        if ivflip == 0:  # input data from toa to sfc
            for i in range(IM):
                plvl[i, 1 + kd] = 0.01 * Statein.prsi[i, 0]  # pa to mb (hpa)

            if lsk != 0:
                for i in range(IM):
                    plvl[i, 1 + kd] = 0.5 * (plvl[i, 2 + kd] + plvl[i, 1 + kd])
        else:  # input data from sfc to top
            for i in range(IM):
                plvl[i, LP1 + kd] = 0.01 * Statein.prsi[i, LP1 + lsk]  # pa to mb (hpa)

            if lsk != 0:
                for i in range(IM):
                    plvl[i, LM + kd] = 0.5 * (plvl[i, LP1 + kd] + plvl[i, LM + kd])

        if self.lextop:  # values for extra top layer
            for i in range(IM):
                plvl[i, llb] = self.prsmin
                if plvl[i, lla] <= self.prsmin:
                    plvl[i, lla] = 2.0 * self.prsmin

                plyr[i, lyb] = 0.5 * plvl[i, lla]
                tlyr[i, lyb] = tlyr[i, lya]
                prslk1[i, lyb] = (plyr[i, lyb] * 0.00001) ** con_rocp  # plyr in Pa
                rhly[i, lyb] = rhly[i, lya]
                qstl[i, lyb] = qstl[i, lya]

            #  ---  note: may need to take care the top layer amount
            tracer1[:, lyb, :] = tracer1[:, lya, :]

        #  - Get layer ozone mass mixing ratio (if use ozone climatology data,
        #    call getozn()).

        if Model.ntoz > 0:  # interactive ozone generation
            for k in range(LMK):
                for i in range(IM):
                    olyr[i, k] = max(self.QMIN, tracer1[i, k, Model.ntoz])
        else:  # climatological ozone
            olyr = getozn(prslk1, Grid.xlat, IM, LMK)

        #  - Call coszmn(), to compute cosine of zenith angle (only when SW is called)

        if Model.lsswr:
            coszen, coszdg = coszmn(
                Grid.xlon, Grid.sinlat, Grid.coslat, Model.solhr, IM, me
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

        gasvmr = getgases(
            plvl,
            Grid.xlon,
            Grid.xlat,
            IM,
            LMK,
        )

        #  - Get temperature at layer interface, and layer moisture.
        for k in range(1, LMK):
            for i in range(IM):
                tem2da[i, k] = np.log(plyr[i, k])
                tem2db[i, k] = np.log(plvl[i, k])

        if ivflip == 0:  # input data from toa to sfc
            for i in range(IM):
                tem1d[i] = self.QME6
                tem2da[i, 0] = np.log(plyr[i, 0])
                tem2db[i, 0] = np.log(max(self.prsmin, plvl[i, 0]))
                tem2db[i, LMP] = np.log(plvl[i, LMP])
                tsfa[i] = tlyr[i, LMK]  # sfc layer air temp
                tlvl[i, 0] = tlyr[i, 0]
                tlvl[i, LMP] = tskn[i]

            for k in range(LM):
                k1 = k + kd
                for i in range(IM):
                    qlyr[i, k1] = max(tem1d[i], Statein.qgrs[i, k, 0])
                    tem1d[i] = min(self.QME5, qlyr[i, k1])
                    tvly[i, k1] = Statein.tgrs[i, k] * (
                        1.0 + con_fvirt * qlyr[i, k1]
                    )  # virtual T (K)
                    delp[i, k1] = plvl[i, k1 + 1] - plvl[i, k1]

            if self.lextop:
                for i in range(IM):
                    qlyr[i, lyb] = qlyr[i, lya]
                    tvly[i, lyb] = tvly[i, lya]
                    delp[i, lyb] = plvl[i, lla] - plvl[i, llb]

            for k in range(1, LMK):
                for i in range(IM):
                    tlvl[i, k] = tlyr[i, k] + (tlyr[i, k - 1] - tlyr[i, k]) * (
                        tem2db[i, k] - tem2da[i, k]
                    ) / (tem2da[i, k - 1] - tem2da[i, k])

            #  ---  ...  level height and layer thickness (km)

            tem0d = 0.001 * con_rog
            for i in range(IM):
                for k in range(LMK):
                    dz[i, k] = tem0d * (tem2db[i, k + 1] - tem2db[i, k]) * tvly[i, k]
        else:

            for i in range(IM):
                tem1d[i] = self.QME6
                tem2da[i, 0] = np.log(plyr[i, 0])
                tem2db[i, 0] = np.log(plvl[i, 0])
                tem2db[i, LMP] = np.log(max(self.prsmin, plvl[i, LMP]))
                tsfa[i] = tlyr[i, 0]  # sfc layer air temp
                tlvl[i, 0] = tskn[i]
                tlvl[i, LMP] = tlyr[i, LMK]

            for k in range(LM - 1, -1, -1):
                for i in range(IM):
                    qlyr[i, k] = max(tem1d[i], Statein.qgrs[i, k, 0])
                    tem1d[i] = min(self.QME5, qlyr[i, k])
                    tvly[i, k] = Statein.tgrs[i, k] * (
                        1.0 + con_fvirt * qlyr[i, k]
                    )  # virtual T (K)
                    delp[i, k] = plvl[i, k] - plvl[i, k + 1]

            if self.lextop:
                for i in range(IM):
                    qlyr[i, lyb] = qlyr[i, lya]
                    tvly[i, lyb] = tvly[i, lya]
                    delp[i, lyb] = plvl[i, lla] - plvl[i, llb]

            for k in range(LMK - 1):
                for i in range(IM):
                    tlvl[i, k + 1] = tlyr[i, k] + (tlyr[i, k + 1] - tlyr[i, k]) * (
                        tem2db[i, k + 1] - tem2da[i, k]
                    ) / (tem2da[i, k + 1] - tem2da[i, k])

            #  ---  ...  level height and layer thickness (km)

            tem0d = 0.001 * con_rog
            for i in range(IM):
                for k in range(LMK - 1, -1, -1):
                    dz[i, k] = tem0d * (tem2db[i, k] - tem2db[i, k + 1]) * tvly[i, k]

        #  - Check for daytime points for SW radiation.

        nday = 0
        for i in range(IM):
            if Radtend.coszen[i] >= 0.0001:
                nday += 1
                idxday[nday - 1] = i

        #  - Call module_radiation_aerosols::setaer(),to setup aerosols
        # property profile for radiation.

        faersw, faerlw, aerodp = setaer(
            plvl,
            plyr,
            prslk1,
            tvly,
            rhly,
            Sfcprop.slmsk,
            tracer1,
            Grid.xlon,
            Grid.xlat,
            IM,
            LMK,
            LMP,
            Model.lsswr,
            Model.lslwr,
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

        ccnd = 0.0
        if Model.ncnd == 1:  # Zhao_Carr_Sundqvist
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, k, 0] = tracer1[i, k, ntcw]  # liquid water/ice
        elif Model.ncnd == 2:  # MG
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, k, 0] = tracer1[i, k, ntcw]  # liquid water
                    ccnd[i, k, 1] = tracer1[i, k, ntiw]  # ice water
        elif Model.ncnd == 4:  # MG2
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, k, 0] = tracer1[i, k, ntcw]  # liquid water
                    ccnd[i, k, 1] = tracer1[i, k, ntiw]  # ice water
                    ccnd[i, k, 2] = tracer1[i, k, ntrw]  # rain water
                    ccnd[i, k, 3] = tracer1[i, k, ntsw]  # snow water
        elif Model.ncnd == 5:  # GFDL MP, Thompson, MG3
            for k in range(LMK):
                for i in range(IM):
                    ccnd[i, k, 0] = tracer1[i, k, ntcw]  # liquid water
                    ccnd[i, k, 1] = tracer1[i, k, ntiw]  # ice water
                    ccnd[i, k, 2] = tracer1[i, k, ntrw]  # rain water
                    ccnd[i, k, 3] = (
                        tracer1[i, k, ntsw] + tracer1[i, k, ntgl]
                    )  # snow + grapuel

        for n in range(ncndl):
            for k in range(LMK):
                for i in range(IM):
                    if ccnd[i, k, n] < con_epsq:
                        ccnd[i, k, n] = 0.0

        if Model.imp_physics == 11:
            if not Model.lgfdlmprad:

                # rsun the  summation methods and order make the difference in calculation
                ccnd[:, :, 0] = tracer1[:, :LMK, ntcw]
                ccnd[:, :, 0] = ccnd[:, :, 0] + tracer1[:, :LMK, ntrw]
                ccnd[:, :, 0] = ccnd[:, :, 0] + tracer1[:, :LMK, ntiw]
                ccnd[:, :, 0] = ccnd[:, :, 0] + tracer1[:, :LMK, ntsw]
                ccnd[:, :, 0] = ccnd[:, :, 0] + tracer1[:, :LMK, ntgl]

            for k in range(LMK):
                for i in range(IM):
                    if ccnd[i, k, 0] < self.EPSQ:
                        ccnd[i, k, 0] = 0.0
