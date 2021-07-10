import numpy as np
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_pi, con_plnk, con_c, con_boltz, con_t0c

def aer_init(NLAY, me):
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

    kyrstr  = 1
    kyrend  = 1
    kyrsav  = 1
    kmonsav = 1

    laswflg= iaerflg % 10 > 0     # control flag for sw tropospheric aerosol
    lalwflg= iaerflg/10 % 10 > 0  # control flag for lw tropospheric aerosol
    lavoflg= iaerflg >= 100       # control flag for stratospheric volcanic aeros

    # -# Call wrt_aerlog() to write aerosol parameter configuration to output logs.

    if me == 0:
        wrt_aerlog()      # write aerosol param info to log file

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

    wvn_sw1 = wvnsw1
    wvn_sw2 = wvnsw2
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

        set_spectrum()

        # -# Call clim_aerinit() to invoke tropospheric aerosol initialization.

        if iaermdl == 0 or iaermdl == 5:      # opac-climatology scheme

            clim_aerinit(solfwv, eirfwv, me)

        else:
            if me == 0:
                print('!!! ERROR in aerosol model scheme selection',
                      f' iaermdl = {iaermdl}')

    # -# Call set_volcaer() to invoke stratospheric volcanic aerosol
    # initialization.

    if lavoflg:
        ivolae = np.zeros((12,4,10))


def wrt_aerlog():
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

    for nb in range(NWVSOL):
        if nb == 0:
            nw1 = 0
        else:
            nw1 = nw1 + nwvns0[nb-1] -1

        nw2 = nw1 + nwvns0[nb] - 1

        for nw in range(nw1, nw2):
            solfwv[nw] = s0intv[nb]

    #  --- ...  define the one wavenumber ir fluxes based on black-body
    #           emission distribution at a predefined temperature

    tmp1 = (con_pi + con_pi) * con_plnk * con_c* con_c
    tmp2 = con_plnk * con_c / (con_boltz * con_t0c)

#omp parallel do private(nw,tmp3)
    for nw in range(NWVTIR):
        tmp3 = 100.0 * (nw+1)
        eirfwv[nw] = (tmp1 * tmp3**3) / (np.exp(tmp2*tmp3) - 1.0)