import numpy as np
import xarray as xr
import sys
import os
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_pi, con_plnk, con_c, con_boltz, con_t0c
from radphysparam import lalw1bd, iaermdl, aeros_file
from radsw_param import NBDSW, wvnum1, wvnum2, NSWSTR
from radlw_param import NBDLW, wvnlw1, wvnlw2

def aer_init(NLAY, me, iaerflg):
    #  ==================================================================  !
    #                                                                      !
    #  aer_init is the initialization program to set up necessary          !
    #    parameters and working arrays.                                    !
    #                                                                      !
    #  inputs:                                                             !
    #     NLAY    - number of model vertical layers  (not used)            !
    #     me      - print message control flag                             !
    #                                                                      !
    #  outputs: (to module variables)                                      !
    #                                                                      !
    #  external module variables: (in physparam)                           !
    #     iaermdl - tropospheric aerosol model scheme flag                 !
    #               =0 opac-clim; =1 gocart-clim, =2 gocart-prognostic     !
    #               =5 opac-clim new spectral mapping                      !
    #     lalwflg - logical lw aerosols effect control flag                !
    #               =t compute lw aerosol optical prop                     !
    #     laswflg - logical sw aerosols effect control flag                !
    #               =t compute sw aerosol optical prop                     !
    #     lavoflg - logical stratosphere volcanic aerosol control flag     !
    #               =t include volcanic aerosol effect                     !
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
    #                                                                      !
    #  usage:    call aer_init                                             !
    #                                                                      !
    #  subprograms called:  clim_aerinit, gcrt_aerinit,                    !
    #                       wrt_aerlog, set_volcaer, set_spectrum,         !
    #                                                                      !
    #  ==================================================================  !

    # global kyrstr  = 1
    # global kyrend  = 1
    # global kyrsav  = 1
    # global kmonsav = 1

    laswflg= iaerflg % 10 > 0     # control flag for sw tropospheric aerosol
    lalwflg= iaerflg/10 % 10 > 0  # control flag for lw tropospheric aerosol
    lavoflg= iaerflg >= 100       # control flag for stratospheric volcanic aeros

    # -# Call wrt_aerlog() to write aerosol parameter configuration to output logs.

    if me == 0:
        wrt_aerlog(iaerflg)      # write aerosol param info to log file

    if iaerflg == 0:
        return  # return without any aerosol calculations

        #  --- ...  in sw, aerosols optical properties are computed for each radiation
        #           spectral band; while in lw, optical properties can be calculated
        #           for either only one broad band or for each of the lw radiation bands

    if laswflg:
        NSWBND = NBDSW
    else:
        NSWBND = 0

    if lalwflg:
        if lalw1bd:
            NLWBND = 1
        else:
            NLWBND = NBDLW
    else:
        NLWBND = 0

    NSWLWBD = NSWBND + NLWBND

    wvn_sw1 = wvnum1
    wvn_sw2 = wvnum2
    wvn_lw1 = wvnlw1
    wvn_lw2 = wvnlw2

    # note: for result consistency, the defalt opac-clim aeros setting still use
    #       old spectral band mapping. use iaermdl=5 to use new mapping method

    if iaermdl == 0:                    # opac-climatology scheme
        lmap_new = False

        wvn_sw1[1:NBDSW-1] = wvn_sw1[1:NBDSW-1] + 1
        wvn_lw1[1:NBDLW] = wvn_lw1[1:NBDLW] + 1
    else:
       lmap_new = True

    if iaerflg != 100:

        # -# Call set_spectrum() to set up spectral one wavenumber solar/IR
        # fluxes. 

        eirfwv, solfwv = set_spectrum()

        # -# Call clim_aerinit() to invoke tropospheric aerosol initialization.

        if iaermdl == 0 or iaermdl == 5:      # opac-climatology scheme

            out_dict = clim_aerinit(solfwv, eirfwv, me,
                                    wvn_sw1, wvn_sw2,
                                    wvn_lw1, wvn_lw2,
                                    lmap_new, laswflg, lalwflg)

        else:
            if me == 0:
                print('!!! ERROR in aerosol model scheme selection',
                      f' iaermdl = {iaermdl}')

    # -# Call set_volcaer() to invoke stratospheric volcanic aerosol
    # initialization.

    if lavoflg:
        ivolae = np.zeros((12,4,10))

    out_dict['eirfwv'] = eirfwv
    out_dict['solfwv'] = solfwv
    # out_dict['ivolae'] = ivolae

    return out_dict

def wrt_aerlog(iaerflg):
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
    
    lalwflg = True
    laswflg = True
    lavoflg = True
    VTAGAER='NCEP-Radiation_aerosols  v5.2  Jan 2013 '
    print(VTAGAER)    # print out version tag

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

    print(f'IAER={iaerflg},  LW-trop-aer={lalwflg}',
          f'SW-trop-aer={laswflg}, Volc-aer={lavoflg}')

    if iaerflg <= 0:        # turn off all aerosol effects
        print('- No tropospheric/volcanic aerosol effect included')
        print('Input values of aerosol optical properties to',
              ' both SW and LW radiations are set to zeros')
    else:
        if iaerflg >= 100:    # incl stratospheric volcanic aerosols
            print('- Include stratospheric volcanic aerosol effect')
        else:                       # no stratospheric volcanic aerosols
            print('- No stratospheric volcanic aerosol effect')

        if laswflg:          # chcek for sw effect
            print('- Compute multi-band aerosol optical',
                  ' properties for SW input parameters')
        else:
            print('- No SW radiation aerosol effect, values of',
                  ' aerosol properties to SW input are set to zeros')

        if lalwflg:          # check for lw effect
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

def set_spectrum():
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

    NWVTIR = 4000
    NWVSOL = 151
    NWVTOT = 57600

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

    solfwv = np.zeros(NWVTOT)

    for nb in range(NWVSOL):
        if nb == 0:
            nw1 = 1
        else:
            nw1 = nw1 + nwvns0[nb-1]

        nw2 = nw1 + nwvns0[nb] - 1

        for nw in range(nw1-1, nw2):
            solfwv[nw] = s0intv[nb]

    #  --- ...  define the one wavenumber ir fluxes based on black-body
    #           emission distribution at a predefined temperature

    tmp1 = (con_pi + con_pi) * con_plnk * con_c* con_c
    tmp2 = con_plnk * con_c / (con_boltz * con_t0c)

    eirfwv = np.zeros(NWVTIR)

    for nw in range(NWVTIR):
        tmp3 = 100.0 * (nw+1)
        eirfwv[nw] = (tmp1 * tmp3**3) / (np.exp(tmp2*tmp3) - 1.0)

    return eirfwv, solfwv


def clim_aerinit(solfwv, eirfwv, me,
                 wvn_sw1, wvn_sw2,
                 wvn_lw1, wvn_lw2,
                 lmap_new, laswflg, lalwflg):
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
    out_dict = set_aercoef(solfwv,
                           eirfwv,
                           aeros_file,
                           laswflg,
                           lalwflg,
                           wvn_sw1,
                           wvn_sw2,
                           wvn_lw1,
                           wvn_lw2,
                           lmap_new)
    
    return out_dict

    # The initialization program for climatological aerosols. The program
    # reads and maps the pre-tabulated aerosol optical spectral data onto
    # corresponding SW radiation spectral bands.
    #\section det_set_aercoef General Algorithm
    # @{

def set_aercoef(solfwv,
                eirfwv,
                aeros_file,
                laswflg,
                lalwflg,
                wvn_sw1,
                wvn_sw2,
                wvn_lw1,
                wvn_lw2,
                lmap_new):
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

    #  --- ...  skip monthly global distribution

#    for m in range(12):
#        read (NIAERCM,12) cline
#  12    format(a80/)
#
#        do j = 1, JMXAE
#          do i = 1, IMXAE
#            read(NIAERCM,*) id
#          enddo
#        enddo
#      enddo   ! end do_m_block

    NCM1 = 6
    NSWLWBD = 30
    NRHLEV = 8
    NCM2 = 4
    NAERBND = 61
    wvn550 = 1.0e4/0.55

    extrhi = np.zeros((NCM1, NSWLWBD))
    scarhi = np.zeros((NCM1, NSWLWBD))
    ssarhi = np.zeros((NCM1, NSWLWBD))
    asyrhi = np.zeros((NCM1, NSWLWBD))

    extrhd = np.zeros((NRHLEV, NCM2, NSWLWBD))
    scarhd = np.zeros((NRHLEV, NCM2, NSWLWBD))
    ssarhd = np.zeros((NRHLEV, NCM2, NSWLWBD))
    asyrhd = np.zeros((NRHLEV, NCM2, NSWLWBD))

    extstra = np.zeros((NSWLWBD))

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

    NSWBND = NBDSW
    nv1 = np.zeros(NSWBND, dtype=np.int32)
    nv2 = np.zeros(NSWBND, dtype=np.int32)

    if laswflg:
        solbnd = np.zeros(NSWBND)
        solwaer = np.zeros((NSWBND, NAERBND))

        ibs = 1
        ibe = 1
        wvs = wvn_sw1[0]
        wve = wvn_sw1[0]
        nv_aod = 1
        for ib in range(1, NSWBND):
            mb = ib + NSWSTR - 1
            if wvn_sw2[mb] >= wvn550 and wvn550 >= wvn_sw1[mb]:
                nv_aod = ib                  # sw band number covering 550nm wavelenth

            if wvn_sw1[mb] < wvs:
                wvs = wvn_sw1[mb]
                ibs = ib
            if wvn_sw1[mb] > wve:
                wve = wvn_sw1[mb]
                ibe = ib

#!$omp parallel do private(ib,mb,ii,iw1,iw2,iw,sumsol,fac,tmp,ibs,ibe)
        for ib in range(NSWBND):
            mb = ib + NSWSTR - 1
            ii = 0
            iw1 = round(wvn_sw1[mb])
            iw2 = round(wvn_sw2[mb])

            while iw1 > iendwv[ii]:
                if ii == NAERBND-1:
                    break
                ii += 1

            if lmap_new:
                if ib == ibs:
                    sumsol = 0.0
                else:
                    sumsol = -0.5 * solfwv[iw1-1]
                
                if ib == ibe:
                    fac = 0.0
                else:
                    fac = -0.5
                solbnd[ib] = sumsol
            else:
                sumsol = 0.0

            nv1[ib] = ii

            for iw in range(iw1-1, iw2):
                solbnd[ib] = solbnd[ib] + solfwv[iw]
                sumsol     = sumsol     + solfwv[iw]

                if iw == iendwv[ii]-1:
                    solwaer[ib, ii] = sumsol

                    if ii < NAERBND-1:
                        sumsol = 0.0
                        ii += 1

            if iw2 != iendwv[ii]-1:
                solwaer[ib, ii] = sumsol

            if lmap_new:
                tmp = fac * solfwv[iw2-1]
                solwaer[ib, ii] = solwaer[ib, ii] + tmp
                solbnd[ib] = solbnd[ib] + tmp

            nv2[ib] = ii

    # -# Compute LW flux weights and interval indices for mapping
    #    spectral bands between lw radiation and aerosol data.

    NLWBND = NBDLW
    nr1 = np.zeros(NLWBND, dtype=np.int32)
    nr2 = np.zeros(NLWBND, dtype=np.int32)
    NLWSTR = 1

    if lalwflg:
        eirbnd = np.zeros(NLWBND)
        eirwaer = np.zeros((NLWBND, NAERBND))

        ibs = 1
        ibe = 1
        if NLWBND > 1:
            wvs = wvn_lw1[0]
            wve = wvn_lw1[0]
            for ib in range(1, NLWBND):
                mb = ib + NLWSTR - 1
                if wvn_lw1[mb] < wvs:
                    wvs = wvn_lw1[mb]
                    ibs = ib
                if wvn_lw1[mb] > wve:
                    wve = wvn_lw1[mb]
                    ibe = ib

        for ib in range(NLWBND):
            ii = 0
            if NLWBND == 1:
                iw1 = 400                   # corresponding 25 mu
                iw2 = 2500                  # corresponding 4  mu
            else:
                mb = ib + NLWSTR - 1
                iw1 = round(wvn_lw1[mb])
                iw2 = round(wvn_lw2[mb])

            while iw1 > iendwv[ii]:
                if ii == NAERBND-1:
                    break
                ii += 1

            if lmap_new:
                if ib == ibs:
                    sumir = 0.0
                else:
                    sumir = -0.5 * eirfwv[iw1-1]

                if ib == ibe:
                    fac = 0.0
                else:
                    fac = -0.5

                eirbnd[ib] = sumir
            else:
                sumir = 0.0

            nr1[ib] = ii

            for iw in range(iw1-1, iw2):
                eirbnd[ib] = eirbnd[ib] + eirfwv[iw]
                sumir  = sumir  + eirfwv[iw]

                if iw == iendwv[ii]-1:
                    eirwaer[ib, ii] = sumir

                    if ii < NAERBND-1:
                        sumir = 0.0
                        ii += 1


            if iw2 != iendwv[ii]-1:
                eirwaer[ib, ii] = sumir

            if lmap_new:
                tmp = fac * eirfwv[iw2-1]
                eirwaer[ib, ii] = eirwaer[ib, ii] + tmp
                eirbnd[ib] = eirbnd[ib] + tmp

            nr2[ib] = ii


    # -# Call optavg() to compute spectral band mean properties for each
    # species.

    out_dict = optavg(laswflg, lalwflg, solbnd, solwaer, nv1, nv2, nr1, nr2,
                      rhidext0, rhidsca0, rhidssa0, rhidasy0,
                      rhdpext0, rhdpsca0, rhdpssa0, rhdpasy0, straext0,
                      extrhi, scarhi, ssarhi, asyrhi,
                      extrhd, scarhd, ssarhd, asyrhd, extstra,
                      eirbnd, eirwaer)
    
    out_dict['prsref'] = prsref
    out_dict['haer'] = haer
    out_dict['eirfwv'] = eirfwv
    out_dict['solfwv'] = solfwv

    return out_dict
    

# This subroutine computes mean aerosols optical properties over each
# SW radiation spectral band for each of the species components. This
# program follows GFDL's approach for thick cloud optical property in
# SW radiation scheme (2000).
def optavg(laswflg, lalwflg, solbnd, solwaer, nv1, nv2, nr1, nr2,
           rhidext0, rhidsca0, rhidssa0, rhidasy0,
           rhdpext0, rhdpsca0, rhdpssa0, rhdpasy0, straext0,
           extrhi, scarhi, ssarhi, asyrhi,
           extrhd, scarhd, ssarhd, asyrhd, extstra,
           eirbnd, eirwaer):
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

    NSWBND = NBDSW
    NLWBND = NBDLW
    NCM1 = 6
    NRHLEV = 8
    NCM2 = 4

    print(f'laswflg = {laswflg}, lalwflg = {lalwflg}')

    if laswflg:

        for nb in range(NSWBND):
            rsolbd = 1.0 / solbnd[nb]

            #  ---  for rh independent aerosol species

            for nc in range(NCM1):        #  ---  for rh independent aerosol species
                sumk    = 0.0
                sums    = 0.0
                sumok   = 0.0
                sumokg  = 0.0
                sumreft = 0.0

                for ni in range(nv1[nb], nv2[nb]+1):
                    sp = np.sqrt((1.0 - rhidssa0[ni, nc]) / \
                        (1.0 - rhidssa0[ni, nc]*rhidasy0[ni, nc]))
                    reft = (1.0 - sp) / (1.0 + sp)
                    sumreft = sumreft + reft*solwaer[nb, ni]

                    sumk  = sumk  + rhidext0[ni, nc]*solwaer[nb, ni]
                    sums  = sums  + rhidsca0[ni, nc]*solwaer[nb, ni]
                    sumok = sumok + rhidssa0[ni, nc]*solwaer[nb, ni] * rhidext0[ni, nc]
                    sumokg = sumokg  + rhidssa0[ni, nc]*solwaer[nb, ni] * \
                        rhidext0[ni, nc]*rhidasy0[ni, nc]

                refb = sumreft * rsolbd

                extrhi[nc, nb] = sumk   * rsolbd
                scarhi[nc, nb] = sums   * rsolbd
                asyrhi[nc, nb] = sumokg / (sumok + 1.0e-10)
                ssarhi[nc, nb] = 4.0*refb / \
                    ((1.0+refb)**2 - asyrhi[nc, nb]*(1.0-refb)**2)


            for nc in range(NCM2):        #  ---  for rh dependent aerosols species
                for nh in range(NRHLEV):
                    sumk    = 0.0
                    sums    = 0.0
                    sumok   = 0.0
                    sumokg  = 0.0
                    sumreft = 0.0

                    for ni in range(nv1[nb], nv2[nb]+1):
                        sp = np.sqrt((1.0 - rhdpssa0[ni, nh, nc]) / \
                            (1.0 - rhdpssa0[ni, nh, nc]*rhdpasy0[ni, nh, nc]))
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft*solwaer[nb, ni]

                        sumk  = sumk  + rhdpext0[ni, nh, nc]*solwaer[nb, ni]
                        sums  = sums  + rhdpsca0[ni, nh, nc]*solwaer[nb, ni]
                        sumok = sumok + rhdpssa0[ni, nh, nc]*solwaer[nb, ni] * \
                            rhdpext0[ni, nh, nc]
                        sumokg = sumokg + rhdpssa0[ni, nh, nc]*solwaer[nb, ni] * \
                            rhdpext0[ni, nh, nc]*rhdpasy0[ni, nh, nc]

                    refb = sumreft * rsolbd

                    extrhd[nh, nc, nb] = sumk   * rsolbd
                    scarhd[nh, nc, nb] = sums   * rsolbd
                    asyrhd[nh, nc, nb] = sumokg / (sumok + 1.0e-10)
                    ssarhd[nh, nc, nb] = 4.0*refb / \
                        ((1.0+refb)**2 - asyrhd[nh, nc, nb]*(1.0-refb)**2)

            #  ---  for stratospheric background aerosols

            sumk = 0.0
            for ni in range(nv1[nb], nv2[nb]+1):
                sumk += straext0[ni]*solwaer[nb, ni]

            extstra[nb] = sumk * rsolbd

    #  --- ...  loop for each lw radiation spectral band

    if lalwflg:
        for nb in range(NLWBND):
            ib = NSWBND + nb
            rirbd = 1.0 / eirbnd[nb]

            for nc in range(NCM1):        #  ---  for rh independent aerosol species
                sumk    = 0.0
                sums    = 0.0
                sumok   = 0.0
                sumokg  = 0.0
                sumreft = 0.0

                for ni in range(nr1[nb], nr2[nb]+1):
                    sp = np.sqrt((1.0 - rhidssa0[ni, nc]) / \
                        (1.0 - rhidssa0[ni, nc]*rhidasy0[ni, nc]))
                    reft = (1.0 - sp) / (1.0 + sp)
                    sumreft = sumreft + reft*eirwaer[nb, ni]

                    sumk  = sumk  + rhidext0[ni, nc]*eirwaer[nb, ni]
                    sums  = sums  + rhidsca0[ni, nc]*eirwaer[nb, ni]
                    sumok = sumok + rhidssa0[ni, nc]*eirwaer[nb, ni] * \
                        rhidext0[ni, nc]
                    sumokg += rhidssa0[ni, nc]*eirwaer[nb, ni] * \
                        rhidext0[ni, nc]*rhidasy0[ni, nc]

                refb = sumreft * rirbd

                extrhi[nc, ib] = sumk   * rirbd
                scarhi[nc, ib] = sums   * rirbd
                asyrhi[nc, ib] = sumokg / (sumok + 1.0e-10)
                ssarhi[nc, ib] = 4.0*refb / \
                    ((1.0+refb)**2 - asyrhi[nc, ib]*(1.0-refb)**2)

            for nc in range(NCM2):  #  ---  for rh dependent aerosols species
                for nh in range(NRHLEV):
                    sumk    = 0.0
                    sums    = 0.0
                    sumok   = 0.0
                    sumokg  = 0.0
                    sumreft = 0.0

                    for ni in range(nr1[nb], nr2[nb]+1):
                        sp = np.sqrt((1.0 - rhdpssa0[ni, nh, nc]) / \
                            (1.0 - rhdpssa0[ni, nh, nc]*rhdpasy0[ni, nh, nc]))
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft*eirwaer[nb, ni]

                        sumk  = sumk  + rhdpext0[ni, nh, nc]*eirwaer[nb, ni]
                        sums  = sums  + rhdpsca0[ni, nh, nc]*eirwaer[nb, ni]
                        sumok = sumok + rhdpssa0[ni, nh, nc]*eirwaer[nb, ni] * \
                            rhdpext0[ni, nh, nc]
                        sumokg += rhdpssa0[ni, nh, nc]*eirwaer[nb, ni] * \
                            rhdpext0[ni, nh, nc]*rhdpasy0[ni, nh, nc]

                    refb = sumreft * rirbd

                    extrhd[nh, nc, ib] = sumk   * rirbd
                    scarhd[nh, nc, ib] = sums   * rirbd
                    asyrhd[nh, nc, ib] = sumokg / (sumok + 1.0e-10)
                    ssarhd[nh, nc, ib] = 4.0*refb / \
                        ((1.0+refb)**2 - asyrhd[nh, nc, ib]*(1.0-refb)**2)

            #  ---  for stratospheric background aerosols

            sumk = 0.0
            for ni in range(nr1[nb], nr2[nb]+1):
                sumk += straext0[ni]*eirwaer[nb, ni]

            extstra[ib] = sumk * rirbd

    out_dict = dict()
    out_dict['extrhi'] = extrhi
    out_dict['scarhi'] = scarhi
    out_dict['ssarhi'] = ssarhi
    out_dict['asyrhi'] = asyrhi

    out_dict['extstra'] = extstra

    out_dict['extrhd'] = extrhd
    out_dict['scarhd'] = scarhd
    out_dict['ssarhd'] = ssarhd
    out_dict['asyrhd'] = asyrhd

    return out_dict