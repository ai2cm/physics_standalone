import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import NBDLW, wvnlw1, wvnlw2
from radsw_param import NBDSW, wvnum1, wvnum2, NSWSTR
from phys_const import con_pi, con_plnk, con_c, con_boltz, con_t0c
from radphysparam import aeros_file, iaermdl, lalw1bd

class AerosolClass():
    VTAGAER = 'NCEP-Radiation_aerosols  v5.2  Jan 2013 '

    NF_AESW = 3
    NF_AELW = 3
    NLWSTR  = 1
    NSPC    = 5
    NSPC1   = NSPC + 1

    NWVSOL = 151
    NWVTOT = 57600
    NWVTIR = 4000

    nwvns0 = [100,  11,  14,  18,  24,  33,  50,  83,  12,  12, 
               13,  15,  15,  17,  18,  20,  21,  24,  26,  30,  32,  37,  42,
               47,  55,  64,  76,  91, 111, 139, 179, 238, 333,  41,  42,  45,
               46,  48,  51,  53,  55,  58,  61,  64,  68,  71,  75,  79,  84,
               89,  95, 101, 107, 115, 123, 133, 142, 154, 167, 181, 197, 217,
              238, 263, 293, 326, 368, 417, 476, 549, 641, 758, 909, 101, 103,
              105, 108, 109, 112, 115, 117, 119, 122, 125, 128, 130, 134, 137,
              140, 143, 147, 151, 154, 158, 163, 166, 171, 175, 181, 185, 190,
              196, 201, 207, 213, 219, 227, 233, 240, 248, 256, 264, 274, 282,
              292, 303, 313, 325, 337, 349, 363, 377, 392, 408, 425, 444, 462,
              483, 505, 529, 554, 580, 610, 641, 675, 711, 751, 793, 841, 891,
              947,1008,1075,1150,1231,1323,1425,1538,1667,1633,14300]

    s0intv = [1.60000E-6, 2.88000E-5, 3.60000E-5, 4.59200E-5, 6.13200E-5,
        8.55000E-5, 1.28600E-4, 2.16000E-4, 2.90580E-4, 3.10184E-4,
        3.34152E-4, 3.58722E-4, 3.88050E-4, 4.20000E-4, 4.57056E-4,
        4.96892E-4, 5.45160E-4, 6.00600E-4, 6.53600E-4, 7.25040E-4,
        7.98660E-4, 9.11200E-4, 1.03680E-3, 1.18440E-3, 1.36682E-3,
        1.57560E-3, 1.87440E-3, 2.25500E-3, 2.74500E-3, 3.39840E-3,
        4.34000E-3, 5.75400E-3, 7.74000E-3, 9.53050E-3, 9.90192E-3,
        1.02874E-2, 1.06803E-2, 1.11366E-2, 1.15830E-2, 1.21088E-2,
        1.26420E-2, 1.32250E-2, 1.38088E-2, 1.44612E-2, 1.51164E-2,
        1.58878E-2, 1.66500E-2, 1.75140E-2, 1.84450E-2, 1.94106E-2,
        2.04864E-2, 2.17248E-2, 2.30640E-2, 2.44470E-2, 2.59840E-2,
        2.75940E-2, 2.94138E-2, 3.13950E-2, 3.34800E-2, 3.57696E-2,
        3.84054E-2, 4.13490E-2, 4.46880E-2, 4.82220E-2, 5.22918E-2,
        5.70078E-2, 6.19888E-2, 6.54720E-2, 6.69060E-2, 6.81226E-2,
        6.97788E-2, 7.12668E-2, 7.27100E-2, 7.31610E-2, 7.33471E-2,
        7.34814E-2, 7.34717E-2, 7.35072E-2, 7.34939E-2, 7.35202E-2,
        7.33249E-2, 7.31713E-2, 7.35462E-2, 7.36920E-2, 7.23677E-2,
        7.25023E-2, 7.24258E-2, 7.20766E-2, 7.18284E-2, 7.32757E-2,
        7.31645E-2, 7.33277E-2, 7.36128E-2, 7.33752E-2, 7.28965E-2,
        7.24924E-2, 7.23307E-2, 7.21050E-2, 7.12620E-2, 7.10903E-2,
        7.12714E-2,
        7.08012E-2, 7.03752E-2, 7.00350E-2, 6.98639E-2, 6.90690E-2,
        6.87621E-2, 6.52080E-2, 6.65184E-2, 6.60038E-2, 6.47615E-2,
        6.44831E-2, 6.37206E-2, 6.24102E-2, 6.18698E-2, 6.06320E-2,
        5.83498E-2, 5.67028E-2, 5.51232E-2, 5.48645E-2, 5.12340E-2,
        4.85581E-2, 4.85010E-2, 4.79220E-2, 4.44058E-2, 4.48718E-2,
        4.29373E-2, 4.15242E-2, 3.81744E-2, 3.16342E-2, 2.99615E-2,
        2.92740E-2, 2.67484E-2, 1.76904E-2, 1.40049E-2, 1.46224E-2,
        1.39993E-2, 1.19574E-2, 1.06386E-2, 1.00980E-2, 8.63808E-3,
        6.52736E-3, 4.99410E-3, 4.39350E-3, 2.21676E-3, 1.33812E-3,
        1.12320E-3, 5.59000E-4, 3.60000E-4, 2.98080E-4, 7.46294E-5]

    MINVYR = 1850
    MAXVYR = 1999
    NXC = 5
    NAE = 7
    NDM = 5
    IMXAE = 72
    JMXAE = 37
    NAERBND = 61
    NRHLEV = 8
    NCM1 = 6
    NCM2 = 4
    NCM = NCM1 + NCM2

    rhlev = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    KAERBND = 61
    KRHLEV = 36

    wvn550 = 1.0e4/0.55

    def __init__(self, NLAY, me, iaerflg):
        self.NSWBND = NBDSW
        self.NLWBND = NBDLW
        self.NSWLWBD = NBDSW*NBDLW
        self.lalwflg = True
        self.laswflg = True
        self.lavoflg = True
        self.lmap_new = True
        self.NLAY = NLAY

        self.kyrstr = 1
        self.kyrend = 1
        self.kyrsav = 1
        self.kmonsav = 1

        self.haer = np.zeros((self.NDM, self.NAE))
        self.prsref = np.zeros((self.NDM, self.NAE))
        self.sigref = np.zeros((self.NDM, self.NAE))

        self.cmixg = np.zeros((self.NXC, self.IMXAE, self.JMXAE))
        self.denng = np.zeros((2, self.IMXAE, self.JMXAE))
        self.idxcg = np.zeros((self.NXC, self.IMXAE, self.JMXAE))
        self.kprfg = np.zeros((self.IMXAE, self.JMXAE))

        self.iaerflg = iaerflg

        self.laswflg = self.iaerflg % 10 > 0     # control flag for sw tropospheric aerosol
        self.lalwflg = self.iaerflg/10 % 10 > 0  # control flag for lw tropospheric aerosol
        self.lavoflg = self.iaerflg >= 100       # control flag for stratospheric volcanic aeros

        # -# Call wrt_aerlog() to write aerosol parameter configuration to output logs.

        if me == 0:
            self.wrt_aerlog()      # write aerosol param info to log file

        if self.iaerflg == 0:
            return  # return without any aerosol calculations

            #  --- ...  in sw, aerosols optical properties are computed for each radiation
            #           spectral band; while in lw, optical properties can be calculated
            #           for either only one broad band or for each of the lw radiation bands

        if self.laswflg:
            self.NSWBND = NBDSW
        else:
            self.NSWBND = 0

        if self.lalwflg:
            if lalw1bd:
                self.NLWBND = 1
            else:
                self.NLWBND = NBDLW
        else:
            self.NLWBND = 0

        self.NSWLWBD = self.NSWBND + self.NLWBND

        self.wvn_sw1 = wvnum1
        self.wvn_sw2 = wvnum2
        self.wvn_lw1 = wvnlw1
        self.wvn_lw2 = wvnlw2

        # note: for result consistency, the defalt opac-clim aeros setting still use
        #       old spectral band mapping. use iaermdl=5 to use new mapping method

        if iaermdl == 0:                    # opac-climatology scheme
            self.lmap_new = False

            self.wvn_sw1[1:NBDSW-1] = self.wvn_sw1[1:NBDSW-1] + 1
            self.wvn_lw1[1:NBDLW] = self.wvn_lw1[1:NBDLW] + 1
        else:
           self.lmap_new = True

        if self.iaerflg != 100:

            # -# Call set_spectrum() to set up spectral one wavenumber solar/IR
            # fluxes. 

            self.set_spectrum()

            # -# Call clim_aerinit() to invoke tropospheric aerosol initialization.

            if iaermdl == 0 or iaermdl == 5:      # opac-climatology scheme

                self.clim_aerinit()

            else:
                if me == 0:
                    print('!!! ERROR in aerosol model scheme selection',
                          f' iaermdl = {iaermdl}')

        # -# Call set_volcaer() to invoke stratospheric volcanic aerosol
        # initialization.

        if self.lavoflg:
            self.ivolae = np.zeros((12,4,10))

    def return_initdata(self):
        outdict = {'extrhi': self.extrhi,
                   'scarhi': self.scarhi,
                   'ssarhi': self.ssarhi,
                   'asyrhi': self.asyrhi,
                   'extrhd': self.extrhd,
                   'scarhd': self.scarhd,
                   'ssarhd': self.ssarhd,
                   'asyrhd': self.asyrhd,
                   'extstra': self.extstra,
                   'prsref': self.prsref,
                   'haer': self.haer,
                   'eirfwv': self.eirfwv,
                   'solfwv': self.solfwv}
        return outdict

    def return_updatedata(self):
        outdict = {'kprfg': self.kprfg,
                   'idxcg': self.idxcg,
                   'cmixg': self.cmixg,
                   'denng': self.denng,
                   'ivolae': self.ivolae}
        return outdict

    def wrt_aerlog(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : wrt_aerlog                                             !
        #                                                                      !
        #    write aerosol parameter configuration to run log file.            !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  external module variables:  (in physparam)                          !
        #   iaermdl  - aerosol scheme flag: 0:opac-clm; 1:gocart-clim;         !
        #              2:gocart-prog; 5:opac-clim+new mapping                  !
        #   iaerflg  - aerosol effect control flag: 3-digits (volc,lw,sw)      !
        #   lalwflg  - toposphere lw aerosol effect: =f:no; =t:yes             !
        #   laswflg  - toposphere sw aerosol effect: =f:no; =t:yes             !
        #   lavoflg  - stratospherer volcanic aeros effect: =f:no; =t:yes      !
        #                                                                      !
        #  outputs: ( none )                                                   !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call wrt_aerlog                                           !
        #                                                                      !
        #  ==================================================================  !

        print(self.VTAGAER)    # print out version tag

        if iaermdl == 0 or iaermdl == 5:
            print('- Using OPAC-seasonal climatology for tropospheric',
                  ' aerosol effect')
        elif iaermdl == 1:
            print('- Using GOCART-climatology for tropospheric',
                   ' aerosol effect')
        elif iaermdl == 2:
            print(' - Using GOCART-prognostic aerosols for tropospheric',
                  ' aerosol effect')
        else:
            print('!!! ERROR in selection of aerosol model scheme',
                  f' IAER_MDL = {iaermdl}')

        print(f'IAER={self.iaerflg},  LW-trop-aer={self.lalwflg}',
              f'SW-trop-aer={self.laswflg}, Volc-aer={self.lavoflg}')

        if self.iaerflg <= 0:        # turn off all aerosol effects
            print('- No tropospheric/volcanic aerosol effect included')
            print('Input values of aerosol optical properties to',
                  ' both SW and LW radiations are set to zeros')
        else:
            if self.iaerflg >= 100:    # incl stratospheric volcanic aerosols
                print('- Include stratospheric volcanic aerosol effect')
            else:                       # no stratospheric volcanic aerosols
                print('- No stratospheric volcanic aerosol effect')

            if self.laswflg:          # chcek for sw effect
                print('- Compute multi-band aerosol optical',
                      ' properties for SW input parameters')
            else:
                print('- No SW radiation aerosol effect, values of',
                      ' aerosol properties to SW input are set to zeros')

            if self.lalwflg:          # check for lw effect
                if lalw1bd:
                    print('- Compute 1 broad-band aerosol optical',
                          ' properties for LW input parameters')
                else:
                    print('- Compute multi-band aerosol optical',
                          ' properties for LW input parameters')
            else:
                print('- No LW radiation aerosol effect, values of',
                      ' aerosol properties to LW input are set to zeros')

    # This subroutine defines the one wavenumber solar fluxes based on toa
    # solar spectral distribution, and define the one wavenumber IR fluxes
    # based on black-body emission distribution at a predefined temperature.
    #\section gel_set_spec General Algorithm

    def set_spectrum(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : set_spectrum                                           !
        #                                                                      !
        #    define the one wavenumber solar fluxes based on toa solar spectral!
        #    distrobution, and define the one wavenumber ir fluxes based on    !
        #    black-body emission distribution at a predefined temperature.     !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #> -  inputs:  (module constants)                                 
        #!  -   NWVTOT:  total num of wave numbers used in sw spectrum   
        #!  -   NWVTIR:  total num of wave numbers used in the ir region 
        #!                                                                      
        #> -  outputs: (in-scope variables)                                       
        #!  -   solfwv(NWVTOT):   solar flux for each individual wavenumber
        #!                        (\f$W/m^2\f$)
        #!  -   eirfwv(NWVTIR):   ir flux(273k) for each individual wavenumber
        #!                        (\f$W/m^2\f$)
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call set_spectrum                                         !
        #                                                                      !
        #  ==================================================================  !

        self.solfwv = np.zeros(self.NWVTOT)

        for nb in range(self.NWVSOL):
            if nb == 0:
                nw1 = 1
            else:
                nw1 = nw1 + self.nwvns0[nb-1]

            nw2 = nw1 + self.nwvns0[nb] - 1

            for nw in range(nw1-1, nw2):
                self.solfwv[nw] = self.s0intv[nb]

        #  --- ...  define the one wavenumber ir fluxes based on black-body
        #           emission distribution at a predefined temperature

        tmp1 = (con_pi + con_pi) * con_plnk * con_c* con_c
        tmp2 = con_plnk * con_c / (con_boltz * con_t0c)

        self.eirfwv = np.zeros(self.NWVTIR)

        for nw in range(self.NWVTIR):
            tmp3 = 100.0 * (nw+1)
            self.eirfwv[nw] = (tmp1 * tmp3**3) / (np.exp(tmp2*tmp3) - 1.0)


    def clim_aerinit(self):
        #  ==================================================================  !
        #                                                                      !
        #  clim_aerinit is the opac-climatology aerosol initialization program !
        #  to set up necessary parameters and working arrays.                  !
        #                                                                      !
        #  inputs:                                                             !
        #   solfwv(NWVTOT)   - solar flux for each individual wavenumber (w/m2)!
        #   eirfwv(NWVTIR)   - ir flux(273k) for each individual wavenum (w/m2)!
        #   me               - print message control flag                      !
        #                                                                      !
        #  outputs: (to module variables)                                      !
        #                                                                      !
        #  external module variables: (in physparam)                           !
        #     iaerflg - abc 3-digit integer aerosol flag (abc:volc,lw,sw)      !
        #               a: =0 use background stratospheric aerosol             !
        #                  =1 incl stratospheric vocanic aeros (MINVYR-MAXVYR) !
        #               b: =0 no topospheric aerosol in lw radiation           !
        #                  =1 include tropspheric aerosols for lw radiation    !
        #               c: =0 no topospheric aerosol in sw radiation           !
        #                  =1 include tropspheric aerosols for sw radiation    !
        #     lalwflg - logical lw aerosols effect control flag                !
        #               =t compute lw aerosol optical prop                     !
        #     laswflg - logical sw aerosols effect control flag                !
        #               =t compute sw aerosol optical prop                     !
        #     lalw1bd = logical lw aeros propty 1 band vs multi-band cntl flag !
        #               =t use 1 broad band optical property                   !
        #               =f use multi bands optical property                    !
        #                                                                      !
        #  module constants:                                                   !
        #     NWVSOL  - num of wvnum regions where solar flux is constant      !
        #     NWVTOT  - total num of wave numbers used in sw spectrum          !
        #     NWVTIR  - total num of wave numbers used in the ir region        !
        #     NSWBND  - total number of sw spectral bands                      !
        #     NLWBND  - total number of lw spectral bands                      !
        #     NAERBND - number of bands for climatology aerosol data           !
        #     NCM1    - number of rh independent aeros species                 !
        #     NCM2    - number of rh dependent aeros species                   !
        #                                                                      !
        #  usage:    call clim_aerinit                                         !
        #                                                                      !
        #  subprograms called:  set_aercoef, optavg                            !
        #                                                                      !
        #  ==================================================================  !#

        #  --- ...  invoke tropospheric aerosol initialization

        # - call set_aercoef() to invoke tropospheric aerosol initialization.
        self.set_aercoef()

        # The initialization program for climatological aerosols. The program
        # reads and maps the pre-tabulated aerosol optical spectral data onto
        # corresponding SW radiation spectral bands.
        #\section det_set_aercoef General Algorithm
        # @{

    def set_aercoef(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : set_aercoef                                            !
        #                                                                      !
        #    this is the initialization progrmam for climatological aerosols   !
        #                                                                      !
        #    the program reads and maps the pre-tabulated aerosol optical      !
        #    spectral data onto corresponding sw radiation spectral bands.     !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   solfwv(:)    - real, solar flux for individual wavenumber (w/m2)   !
        #   eirfwv(:)    - real, lw flux(273k) for individual wavenum (w/m2)   !
        #   me           - integer, select cpu number as print control flag    !
        #                                                                      !
        #  outputs: (to the module variables)                                  !
        #                                                                      !
        #  external module variables:  (in physparam)                          !
        #   lalwflg   - module control flag for lw trop-aer: =f:no; =t:yes     !
        #   laswflg   - module control flag for sw trop-aer: =f:no; =t:yes     !
        #   aeros_file- external aerosol data file name                        !
        #                                                                      !
        #  internal module variables:                                          !
        #     IMXAE   - number of longitude points in global aeros data set    !
        #     JMXAE   - number of latitude points in global aeros data set     !
        #     wvnsw1,wvnsw2 (NSWSTR:NSWEND)                                    !
        #             - start/end wavenumbers for each of sw bands             !
        #     wvnlw1,wvnlw2 (     1:NBDLW)                                     !
        #             - start/end wavenumbers for each of lw bands             !
        #     NSWLWBD - total num of bands (sw+lw) for aeros optical properties!
        #     NSWBND  - number of sw spectral bands actually invloved          !
        #     NLWBND  - number of lw spectral bands actually invloved          !
        #     NIAERCM - unit number for reading input data set                 !
        #     extrhi  - extinction coef for rh-indep aeros         NCM1*NSWLWBD!
        #     scarhi  - scattering coef for rh-indep aeros         NCM1*NSWLWBD!
        #     ssarhi  - single-scat-alb for rh-indep aeros         NCM1*NSWLWBD!
        #     asyrhi  - asymmetry factor for rh-indep aeros        NCM1*NSWLWBD!
        #     extrhd  - extinction coef for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     scarhd  - scattering coef for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     ssarhd  - single-scat-alb for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     asyrhd  - asymmetry factor for rh-dep aeros   NRHLEV*NCM2*NSWLWBD!
        #                                                                      !
        #  major local variables:                                              !
        #   for handling spectral band structures                              !
        #     iendwv   - ending wvnum (cm**-1) for each band  NAERBND          !
        #   for handling optical properties of rh independent species (NCM1)   !
        #         1. insoluble        (inso); 2. soot             (soot);      !
        #         3. mineral nuc mode (minm); 4. mineral acc mode (miam);      !
        #         5. mineral coa mode (micm); 6. mineral transport(mitr).      !
        #     rhidext0 - extinction coefficient             NAERBND*NCM1       !
        #     rhidsca0 - scattering coefficient             NAERBND*NCM1       !
        #     rhidssa0 - single scattering albedo           NAERBND*NCM1       !
        #     rhidasy0 - asymmetry parameter                NAERBND*NCM1       !
        #   for handling optical properties of rh ndependent species (NCM2)    !
        #         1. water soluble    (waso); 2. sea salt acc mode(ssam);      !
        #         3. sea salt coa mode(sscm); 4. sulfate droplets (suso).      !
        #         rh level (NRHLEV): 00%, 50%, 70%, 80%, 90%, 95%, 98%, 99%    !
        #     rhdpext0 - extinction coefficient             NAERBND,NRHLEV,NCM2!
        #     rhdpsca0 - scattering coefficient             NAERBND,NRHLEV,NCM2!
        #     rhdpssa0 - single scattering albedo           NAERBND,NRHLEV,NCM2!
        #     rhdpasy0 - asymmetry parameter                NAERBND,NRHLEV,NCM2!
        #   for handling optical properties of stratospheric bkgrnd aerosols   !
        #     straext0 - extingction coefficients             NAERBND          !
        #                                                                      !
        #  usage:    call set_aercoef                                          !
        #                                                                      !
        #  subprograms called:  optavg                                         !
        #                                                                      !
        #  ==================================================================  !

        file_exist = os.path.isfile(aeros_file)

        if file_exist:
            print(f'Using file {aeros_file}')
        else:
            print(f'Requested aerosol data file "{aeros_file}" not found!')
            print('*** Stopped in subroutine aero_init !!')

        extrhi = np.zeros((self.NCM1, self.NSWLWBD))
        scarhi = np.zeros((self.NCM1, self.NSWLWBD))
        ssarhi = np.zeros((self.NCM1, self.NSWLWBD))
        asyrhi = np.zeros((self.NCM1, self.NSWLWBD))

        extrhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        scarhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        ssarhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        asyrhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))

        extstra = np.zeros((self.NSWLWBD))

        #  --- ...  aloocate and input aerosol optical data
        ds = xr.open_dataset(aeros_file)

        iendwv = ds['iendwv'].data
        haer = ds['haer'].data
        prsref = ds['prsref'].data
        rhidext0 = ds['rhidext0'].data
        rhidsca0 = ds['rhidsca0'].data
        rhidssa0 = ds['rhidssa0'].data
        rhidasy0 = ds['rhidasy0'].data
        rhdpext0 = ds['rhdpext0'].data
        rhdpsca0 = ds['rhdpsca0'].data
        rhdpssa0 = ds['rhdpssa0'].data
        rhdpasy0 = ds['rhdpasy0'].data
        straext0 = ds['straext0'].data

        # -# Convert pressure reference level (in mb) to sigma reference level
        #    assume an 1000mb reference surface pressure.

        sigref = 0.001 * prsref

        # -# Compute solar flux weights and interval indices for mapping
        #    spectral bands between SW radiation and aerosol data.

        nv1 = np.zeros(self.NSWBND, dtype=np.int32)
        nv2 = np.zeros(self.NSWBND, dtype=np.int32)

        if self.laswflg:
            solbnd = np.zeros(self.NSWBND)
            solwaer = np.zeros((self.NSWBND, self.NAERBND))

            ibs = 1
            ibe = 1
            wvs = self.wvn_sw1[0]
            wve = self.wvn_sw1[0]
            nv_aod = 1
            for ib in range(1, self.NSWBND):
                mb = ib + NSWSTR - 1
                if self.wvn_sw2[mb] >= self.wvn550 and self.wvn550 >= self.wvn_sw1[mb]:
                    nv_aod = ib                  # sw band number covering 550nm wavelenth

                if self.wvn_sw1[mb] < wvs:
                    wvs = self.wvn_sw1[mb]
                    ibs = ib
                if self.wvn_sw1[mb] > wve:
                    wve = self.wvn_sw1[mb]
                    ibe = ib

#!$o    mp parallel do private(ib,mb,ii,iw1,iw2,iw,sumsol,fac,tmp,ibs,ibe)
            for ib in range(self.NSWBND):
                mb = ib + NSWSTR - 1
                ii = 0
                iw1 = round(self.wvn_sw1[mb])
                iw2 = round(self.wvn_sw2[mb])

                while iw1 > iendwv[ii]:
                    if ii == self.NAERBND-1:
                        break
                    ii += 1

                if self.lmap_new:
                    if ib == ibs:
                        sumsol = 0.0
                    else:
                        sumsol = -0.5 * self.solfwv[iw1-1]

                    if ib == ibe:
                        fac = 0.0
                    else:
                        fac = -0.5
                    solbnd[ib] = sumsol
                else:
                    sumsol = 0.0

                nv1[ib] = ii

                for iw in range(iw1-1, iw2):
                    solbnd[ib] = solbnd[ib] + self.solfwv[iw]
                    sumsol     = sumsol     + self.solfwv[iw]

                    if iw == iendwv[ii]-1:
                        solwaer[ib, ii] = sumsol

                        if ii < self.NAERBND-1:
                            sumsol = 0.0
                            ii += 1

                if iw2 != iendwv[ii]-1:
                    solwaer[ib, ii] = sumsol

                if self.lmap_new:
                    tmp = fac * self.solfwv[iw2-1]
                    solwaer[ib, ii] = solwaer[ib, ii] + tmp
                    solbnd[ib] = solbnd[ib] + tmp

                nv2[ib] = ii

        # -# Compute LW flux weights and interval indices for mapping
        #    spectral bands between lw radiation and aerosol data.

        nr1 = np.zeros(self.NLWBND, dtype=np.int32)
        nr2 = np.zeros(self.NLWBND, dtype=np.int32)
        NLWSTR = 1

        if self.lalwflg:
            eirbnd = np.zeros(self.NLWBND)
            eirwaer = np.zeros((self.NLWBND, self.NAERBND))

            ibs = 1
            ibe = 1
            if self.NLWBND > 1:
                wvs = self.wvn_lw1[0]
                wve = self.wvn_lw1[0]
                for ib in range(1, self.NLWBND):
                    mb = ib + NLWSTR - 1
                    if self.wvn_lw1[mb] < wvs:
                        wvs = self.wvn_lw1[mb]
                        ibs = ib
                    if self.wvn_lw1[mb] > wve:
                        wve = self.wvn_lw1[mb]
                        ibe = ib

            for ib in range(self.NLWBND):
                ii = 0
                if self.NLWBND == 1:
                    iw1 = 400                   # corresponding 25 mu
                    iw2 = 2500                  # corresponding 4  mu
                else:
                    mb = ib + NLWSTR - 1
                    iw1 = round(self.wvn_lw1[mb])
                    iw2 = round(self.wvn_lw2[mb])

                while iw1 > iendwv[ii]:
                    if ii == self.NAERBND-1:
                        break
                    ii += 1

                if self.lmap_new:
                    if ib == ibs:
                        sumir = 0.0
                    else:
                        sumir = -0.5 * self.eirfwv[iw1-1]

                    if ib == ibe:
                        fac = 0.0
                    else:
                        fac = -0.5

                    eirbnd[ib] = sumir
                else:
                    sumir = 0.0

                nr1[ib] = ii

                for iw in range(iw1-1, iw2):
                    eirbnd[ib] = eirbnd[ib] + self.eirfwv[iw]
                    sumir  = sumir  + self.eirfwv[iw]

                    if iw == iendwv[ii]-1:
                        eirwaer[ib, ii] = sumir

                        if ii < self.NAERBND-1:
                            sumir = 0.0
                            ii += 1


                if iw2 != iendwv[ii]-1:
                    eirwaer[ib, ii] = sumir

                if self.lmap_new:
                    tmp = fac * self.eirfwv[iw2-1]
                    eirwaer[ib, ii] = eirwaer[ib, ii] + tmp
                    eirbnd[ib] = eirbnd[ib] + tmp

                nr2[ib] = ii


        # -# Call optavg() to compute spectral band mean properties for each
        # species.

        self.prsref = prsref
        self.haer = haer
        self.solbnd = solbnd
        self.solwaer = solwaer
        self.nv1 = nv1
        self.nv2 = nv2
        self.nr1 = nr1
        self.nr2 = nr2
        self.rhidext0 = rhidext0
        self.rhidsca0 = rhidsca0
        self.rhidssa0 = rhidssa0
        self.rhidasy0 = rhidasy0
        self.rhdpext0 = rhdpext0
        self.rhdpsca0 = rhdpsca0
        self.rhdpssa0 = rhdpssa0
        self.rhdpasy0 = rhdpasy0
        self.straext0 = straext0
        self.extrhi = extrhi
        self.scarhi = scarhi
        self.ssarhi = ssarhi
        self.asyrhi = asyrhi
        self.extrhd = extrhd
        self.scarhd = scarhd
        self.ssarhd = ssarhd
        self.asyrhd = asyrhd
        self.extstra = extstra
        self.eirbnd = eirbnd
        self.eirwaer = eirwaer

        self.optavg()
    

    # This subroutine computes mean aerosols optical properties over each
    # SW radiation spectral band for each of the species components. This
    # program follows GFDL's approach for thick cloud optical property in
    # SW radiation scheme (2000).
    def optavg(self):
        # ==================================================================== !
        #                                                                      !
        # subprogram: optavg                                                   !
        #                                                                      !
        #   compute mean aerosols optical properties over each sw radiation    !
        #   spectral band for each of the species components.  This program    !
        #   follows gfdl's approach for thick cloud opertical property in      !
        #   sw radiation scheme (2000).                                        !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        # major input variables:                                               !
        #   nv1,nv2 (NSWBND) - start/end spectral band indices of aerosol data !
        #                      for each sw radiation spectral band             !
        #   nr1,nr2 (NLWBND) - start/end spectral band indices of aerosol data !
        #                      for each ir radiation spectral band             !
        #   solwaer (NSWBND,NAERBND)                                           !
        #                    - solar flux weight over each sw radiation band   !
        #                      vs each aerosol data spectral band              !
        #   eirwaer (NLWBND,NAERBND)                                           !
        #                    - ir flux weight over each lw radiation band      !
        #                      vs each aerosol data spectral band              !
        #   solbnd  (NSWBND) - solar flux weight over each sw radiation band   !
        #   eirbnd  (NLWBND) - ir flux weight over each lw radiation band      !
        #   NSWBND           - total number of sw spectral bands               !
        #   NLWBND           - total number of lw spectral bands               !
        #                                                                      !
        # external module variables:  (in physparam)                           !
        #   laswflg          - control flag for sw spectral region             !
        #   lalwflg          - control flag for lw spectral region             !
        #                                                                      !
        # output variables: (to module variables)                              !
        #                                                                      !
        #  ==================================================================  !    #

        #  --- ...  loop for each sw radiation spectral band

        print(f'laswflg = {self.laswflg}, lalwflg = {self.lalwflg}')

        if self.laswflg:

            for nb in range(self.NSWBND):
                rsolbd = 1.0 / self.solbnd[nb]

                #  ---  for rh independent aerosol species

                for nc in range(self.NCM1):        #  ---  for rh independent aerosol species
                    sumk    = 0.0
                    sums    = 0.0
                    sumok   = 0.0
                    sumokg  = 0.0
                    sumreft = 0.0

                    for ni in range(self.nv1[nb], self.nv2[nb]+1):
                        sp = np.sqrt((1.0 - self.rhidssa0[ni, nc]) / \
                            (1.0 - self.rhidssa0[ni, nc]*self.rhidasy0[ni, nc]))
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft*self.solwaer[nb, ni]

                        sumk  = sumk  + self.rhidext0[ni, nc]*self.solwaer[nb, ni]
                        sums  = sums  + self.rhidsca0[ni, nc]*self.solwaer[nb, ni]
                        sumok = sumok + self.rhidssa0[ni, nc]*self.solwaer[nb, ni] * self.rhidext0[ni, nc]
                        sumokg = sumokg  + self.rhidssa0[ni, nc]*self.solwaer[nb, ni] * \
                            self.rhidext0[ni, nc]*self.rhidasy0[ni, nc]

                    refb = sumreft * rsolbd

                    self.extrhi[nc, nb] = sumk   * rsolbd
                    self.scarhi[nc, nb] = sums   * rsolbd
                    self.asyrhi[nc, nb] = sumokg / (sumok + 1.0e-10)
                    self.ssarhi[nc, nb] = 4.0*refb / \
                        ((1.0+refb)**2 - self.asyrhi[nc, nb]*(1.0-refb)**2)


                for nc in range(self.NCM2):        #  ---  for rh dependent aerosols species
                    for nh in range(self.NRHLEV):
                        sumk    = 0.0
                        sums    = 0.0
                        sumok   = 0.0
                        sumokg  = 0.0
                        sumreft = 0.0

                        for ni in range(self.nv1[nb], self.nv2[nb]+1):
                            sp = np.sqrt((1.0 - self.rhdpssa0[ni, nh, nc]) / \
                                (1.0 - self.rhdpssa0[ni, nh, nc]*self.rhdpasy0[ni, nh, nc]))
                            reft = (1.0 - sp) / (1.0 + sp)
                            sumreft = sumreft + reft*self.solwaer[nb, ni]

                            sumk  = sumk  + self.rhdpext0[ni, nh, nc]*self.solwaer[nb, ni]
                            sums  = sums  + self.rhdpsca0[ni, nh, nc]*self.solwaer[nb, ni]
                            sumok = sumok + self.rhdpssa0[ni, nh, nc]*self.solwaer[nb, ni] * \
                                self.rhdpext0[ni, nh, nc]
                            sumokg = sumokg + self.rhdpssa0[ni, nh, nc]*self.solwaer[nb, ni] * \
                                self.rhdpext0[ni, nh, nc]*self.rhdpasy0[ni, nh, nc]

                        refb = sumreft * rsolbd

                        self.extrhd[nh, nc, nb] = sumk   * rsolbd
                        self.scarhd[nh, nc, nb] = sums   * rsolbd
                        self.asyrhd[nh, nc, nb] = sumokg / (sumok + 1.0e-10)
                        self.ssarhd[nh, nc, nb] = 4.0*refb / \
                            ((1.0+refb)**2 - self.asyrhd[nh, nc, nb]*(1.0-refb)**2)

                #  ---  for stratospheric background aerosols

                sumk = 0.0
                for ni in range(self.nv1[nb], self.nv2[nb]+1):
                    sumk += self.straext0[ni]*self.solwaer[nb, ni]

                self.extstra[nb] = sumk * rsolbd

        #  --- ...  loop for each lw radiation spectral band

        if self.lalwflg:
            for nb in range(self.NLWBND):
                ib = self.NSWBND + nb
                rirbd = 1.0 / self.eirbnd[nb]

                for nc in range(self.NCM1):        #  ---  for rh independent aerosol species
                    sumk    = 0.0
                    sums    = 0.0
                    sumok   = 0.0
                    sumokg  = 0.0
                    sumreft = 0.0

                    for ni in range(self.nr1[nb], self.nr2[nb]+1):
                        sp = np.sqrt((1.0 - self.rhidssa0[ni, nc]) / \
                            (1.0 - self.rhidssa0[ni, nc]*self.rhidasy0[ni, nc]))
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft*self.eirwaer[nb, ni]

                        sumk  = sumk  + self.rhidext0[ni, nc]*self.eirwaer[nb, ni]
                        sums  = sums  + self.rhidsca0[ni, nc]*self.eirwaer[nb, ni]
                        sumok = sumok + self.rhidssa0[ni, nc]*self.eirwaer[nb, ni] * \
                            self.rhidext0[ni, nc]
                        sumokg += self.rhidssa0[ni, nc]*self.eirwaer[nb, ni] * \
                            self.rhidext0[ni, nc]*self.rhidasy0[ni, nc]

                    refb = sumreft * rirbd

                    self.extrhi[nc, ib] = sumk   * rirbd
                    self.scarhi[nc, ib] = sums   * rirbd
                    self.asyrhi[nc, ib] = sumokg / (sumok + 1.0e-10)
                    self.ssarhi[nc, ib] = 4.0*refb / \
                        ((1.0+refb)**2 - self.asyrhi[nc, ib]*(1.0-refb)**2)

                for nc in range(self.NCM2):  #  ---  for rh dependent aerosols species
                    for nh in range(self.NRHLEV):
                        sumk    = 0.0
                        sums    = 0.0
                        sumok   = 0.0
                        sumokg  = 0.0
                        sumreft = 0.0

                        for ni in range(self.nr1[nb], self.nr2[nb]+1):
                            sp = np.sqrt((1.0 - self.rhdpssa0[ni, nh, nc]) / \
                                (1.0 - self.rhdpssa0[ni, nh, nc]*self.rhdpasy0[ni, nh, nc]))
                            reft = (1.0 - sp) / (1.0 + sp)
                            sumreft = sumreft + reft*self.eirwaer[nb, ni]

                            sumk  = sumk  + self.rhdpext0[ni, nh, nc]*self.eirwaer[nb, ni]
                            sums  = sums  + self.rhdpsca0[ni, nh, nc]*self.eirwaer[nb, ni]
                            sumok = sumok + self.rhdpssa0[ni, nh, nc]*self.eirwaer[nb, ni] * \
                                self.rhdpext0[ni, nh, nc]
                            sumokg += self.rhdpssa0[ni, nh, nc]*self.eirwaer[nb, ni] * \
                                self.rhdpext0[ni, nh, nc]*self.rhdpasy0[ni, nh, nc]

                        refb = sumreft * rirbd

                        self.extrhd[nh, nc, ib] = sumk   * rirbd
                        self.scarhd[nh, nc, ib] = sums   * rirbd
                        self.asyrhd[nh, nc, ib] = sumokg / (sumok + 1.0e-10)
                        self.ssarhd[nh, nc, ib] = 4.0*refb / \
                            ((1.0+refb)**2 - self.asyrhd[nh, nc, ib]*(1.0-refb)**2)

                #  ---  for stratospheric background aerosols

                sumk = 0.0
                for ni in range(self.nr1[nb], self.nr2[nb]+1):
                    sumk += self.straext0[ni]*self.eirwaer[nb, ni]

                self.extstra[ib] = sumk * rirbd

    def aer_update(self, iyear, imon, me):
        #  ================================================================== 
        #                                                                     
        #  aer_update checks and update time varying climatology aerosol      
        #    data sets.                                                       
        #                                                                     
        #  inputs:                                          size              
        #     iyear   - 4-digit calender year                 1               
        #     imon    - month of the year                     1               
        #     me      - print message control flag            1               
        #                                                                     
        #  outputs: ( none )                                                  
        #                                                                     
        #  external module variables: (in physparam)                          
        #     lalwflg     - control flag for tropospheric lw aerosol          
        #     laswflg     - control flag for tropospheric sw aerosol          
        #     lavoflg     - control flag for stratospheric volcanic aerosol   
        #                                                                     
        #  usage:    call aero_update                                         
        #                                                                     
        #  subprograms called:  trop_update, volc_update                      
        #                                                                     
        #  ================================================================== 
        #
        #===> ...  begin here
        #

        self.iyear = iyear
        self.imon = imon
        self.me = me

        if self.imon < 1 or self.imon > 12:
            print('***** ERROR in specifying requested month !!! ',
                f'imon = {imon}')
            print('***** STOPPED in subroutinte aer_update !!!')
    
        # -# Call trop_update() to update monthly tropospheric aerosol data.
        if self.lalwflg or self.laswflg:
            self.trop_update()
    
        # -# Call volc_update() to update yearly stratospheric volcanic aerosol data.
        if self.lavoflg:
            self.volc_update()
    
    
    def trop_update(self):
        # This subroutine updates the monthly global distribution of aerosol
        # profiles in five degree horizontal resolution.
    
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : trop_update                                            !
        #                                                                      !
        #    updates the  monthly global distribution of aerosol profiles in   !
        #    five degree horizontal resolution.                                !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   imon     - integer, month of the year                              !
        #   me       - integer, print message control flag                     !
        #                                                                      !
        #  outputs: (module variables)                                         !
        #                                                                      !
        #  external module variables: (in physparam)                           !
        #    aeros_file   - external aerosol data file name                    !
        #                                                                      !
        #  internal module variables:                                          !
        #    kprfg (    IMXAE*JMXAE)   - aeros profile index                   !
        #    idxcg (NXC*IMXAE*JMXAE)   - aeros component index                 !
        #    cmixg (NXC*IMXAE*JMXAE)   - aeros component mixing ratio          !
        #    denng ( 2 *IMXAE*JMXAE)   - aerosols number density               !
        #                                                                      !
        #    NIAERCM      - unit number for input data set                     !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call trop_update                                          !
        #                                                                      !
        #  ==================================================================  !
        #
        #
        #===>  ...  begin here
        #
        #  --- ...  reading climatological aerosols data

        file_exist = os.path.isfile(aeros_file)
    
        if file_exist:
            if self.me == 0:
                print(f'Opened aerosol data file: {aeros_file}')
        else:
            print(f'Requested aerosol data file "{aeros_file}" not found!')
            print('*** Stopped in subroutine trop_update !!')

        ds = xr.open_dataset(aeros_file)
        self.kprfg = ds['kprfg'].data
        self.idxcg = ds['idxcg'].data
        self.cmixg = ds['cmixg'].data
        self.denng = ds['denng'].data
        cline = ds['cline'].data

        if self.me == 0:
            print(f'  --- Reading {cline[self.imon-1]}') 
    
    
    def volc_update(self):
        # This subroutine searches historical volcanic data sets to find and
        # read in monthly 45-degree lat-zone band of optical depth.
    
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : volc_update                                            !
        #                                                                      !
        #    searches historical volcanic data sets to find and read in        !
        #    monthly 45-degree lat-zone band data of optical depth.            !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   iyear    - integer, 4-digit calender year                 1        !
        #   imon     - integer, month of the year                     1        !
        #   me       - integer, print message control flag            1        !
        #   NIAERCM  - integer, unit number for input data set        1        !
        #                                                                      !
        #  outputs: (module variables)                                         !
        #   ivolae   - integer, monthly, 45-deg lat-zone volc odp      12*4*10 !
        #   kyrstr   - integer, starting year of data in the input file        !
        #   kyrend   - integer, ending   year of data in the input file        !
        #   kyrsav   - integer, the year of data in use in the input file      !
        #   kmonsav  - integer, the month of data in use in the input file     !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call volc_aerinit                                         !
        #                                                                      !
        #  ==================================================================  !
    
        #  ---  locals:
        volcano_file = 'volcanic_aerosols_1850-1859.txt'
        #
        #===>  ...  begin here
        #
        self.kmonsav = self.imon
    
        if self.kyrstr <= self.iyear and self.iyear <= self.kyrend:   # use previously input data
            self.kyrsav = self.iyear
            return
        else:                                            # need to input new data
            self.kyrsav = self.iyear
            self.kyrstr = self.iyear - self.iyear % 10
            self.kyrend = self.kyrstr + 9

            if self.iyear < self.MINVYR or self.iyear > self.MAXVYR:
                self.ivolae = np.ones((12, 4, 10))            # set as lowest value
                if self.me == 0:
                    print('Request volcanic date out of range,',
                        ' optical depth set to lowest value')
            else:
                file_exist = os.path.isfile(volcano_file)
                if file_exist:
                    ds = xr.open_dataset(volcano_file)
                    cline = ds['cline']
                    #  ---  check print
                    if self.me == 0:
                        print(f'Opened volcanic data file: {volcano_file}')
                        print(cline)
    
                    self.ivolae = ds['ivolae']
                else:
                    print(f'Requested volcanic data file "{volcano_file}" not found!')
                    print('*** Stopped in subroutine VOLC_AERINIT !!')

        #  ---  check print
        if self.me == 0:
            k = (self.kyrsav % 10) + 1
            print(f'CHECK: Sample Volcanic data used for month, year: {self.imon}, {self.iyear}')           
            print(self.ivolae[self.kmonsav, :, k])