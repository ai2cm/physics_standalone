import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
import numpy as np
from radphysparam import iswmode, iswrgas, icldflg, iswrate
from phys_const import con_g, con_cp

def rswinit(me, iovrsw, isubcsw, iswcliq, exp_tbl):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    # purpose:  initialize non-varying module variables, conversion factors,!
    # and look-up tables.                                                   !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                              !
    #    me       - print control for parallel process                      !
    #                                                                       !
    #  outputs: (none)                                                      !
    #                                                                       !
    #  external module variables:  (in physparam)                           !
    #   iswrate - heating rate unit selections                              !
    #           =1: output in k/day                                         !
    #           =2: output in k/second                                      !
    #   iswrgas - control flag for rare gases (ch4,n2o,o2, etc.)            !
    #           =0: do not include rare gases                               !
    #           >0: include all rare gases                                  !
    #   iswcliq - liquid cloud optical properties contrl flag               !
    #           =0: input cloud opt depth from diagnostic scheme            !
    #           >0: input cwp,rew, and other cloud content parameters       !
    #   isubcsw - sub-column cloud approximation control flag               !
    #           =0: no sub-col cld treatment, use grid-mean cld quantities  !
    #           =1: mcica sub-col, prescribed seeds to get random numbers   !
    #           =2: mcica sub-col, providing array icseed for random numbers!
    #   icldflg - cloud scheme control flag                                 !
    #           =0: diagnostic scheme gives cloud tau, omiga, and g.        !
    #           =1: prognostic scheme gives cloud liq/ice path, etc.        !
    #   iovrsw  - clouds vertical overlapping control flag                  !
    #           =0: random overlapping clouds                               !
    #           =1: maximum/random overlapping clouds                       !
    #           =2: maximum overlap cloud                                   !
    #           =3: decorrelation-length overlap clouds                     !
    #   iswmode - control flag for 2-stream transfer scheme                 !
    #           =1; delta-eddington    (joseph et al., 1976)                !
    #           =2: pifm               (zdunkowski et al., 1980)            !
    #           =3: discrete ordinates (liou, 1973)                         !
    #                                                                       !
    #  *******************************************************************  !
    #                                                                       !
    # definitions:                                                          !
    #     arrays for 10000-point look-up tables:                            !
    #     tau_tbl  clear-sky optical depth                                  !
    #     exp_tbl  exponential lookup table for transmittance               !
    #                                                                       !
    #  *******************************************************************  !
    #                                                                       !
    #  ======================  end of description block  =================  !

    expeps = 1.e-20
    VTAGSW = 'NCEP SW v5.1  Nov 2012 -RRTMG-SW v3.8'
    NTBMX = 10000
    bpade = 1.0/0.278

    #
    #===> ... begin here
    #
    if iovrsw < 0 or iovrsw > 3:
        print('*** Error in specification of cloud overlap flag',
              f' IOVRSW={iovrsw} in RSWINIT !!')

    if me == 0:
        print(f'- Using AER Shortwave Radiation, Version: {VTAGSW}')

        if iswmode == 1:
            print('   --- Delta-eddington 2-stream transfer scheme')
        elif iswmode == 2:
            print('   --- PIFM 2-stream transfer scheme')
        elif iswmode == 3:
            print('   --- Discrete ordinates 2-stream transfer scheme')

        if iswrgas <= 0:
            print('   --- Rare gases absorption is NOT included in SW')
        else:
            print('   --- Include rare gases N2O, CH4, O2, absorptions in SW')

        if isubcsw == 0:
            print('   --- Using standard grid average clouds, no ',
                  '   sub-column clouds approximation applied')
        elif isubcsw == 1:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  '   with a prescribed sequence of permutation seeds')
        elif isubcsw == 2:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  '   with provided input array of permutation seeds')
        else:
            print('  *** Error in specification of sub-column cloud ',
                  f' control flag isubcsw = {isubcsw} !!')

    #  --- ...  check cloud flags for consistency

    if ((icldflg == 0 and iswcliq != 0) or (icldflg == 1 and iswcliq == 0)):
        print('*** Model cloud scheme inconsistent with SW',
              ' radiation cloud radiative property setup !!')

    if isubcsw == 0 and iovrsw > 2:
        if me == 0:
            print(f'*** IOVRSW={iovrsw} is not available for',
                  ' ISUBCSW=0 setting!!')
            print('The program will use maximum/random overlap',
                  ' instead.')
        iovrsw = 1

    #  --- ...  setup constant factors for heating rate
    #           the 1.0e-2 is to convert pressure from mb to N/m**2

    if iswrate == 1:
        heatfac = con_g * 864.0 / con_cp            #   (in k/day)
    else:
        heatfac = con_g * 1.0e-2 / con_cp           #   (in k/second)

    #  --- ...  define exponential lookup tables for transmittance. tau is
    #           computed as a function of the tau transition function, and
    #           transmittance is calculated as a function of tau.  all tables
    #           are computed at intervals of 0.0001.  the inverse of the
    #           constant used in the Pade approximation to the tau transition
    #           function is set to bpade.

    exp_tbl[0] = 1.0
    exp_tbl[NTBMX] = expeps

    for i in range(NTBMX-1):
        tfn = i / (NTBMX-i)
        tau = bpade * tfn
        exp_tbl[i] = np.exp(-tau)