import numpy as np

from physparam import *
from phys_const import *

def rlwinit(me):
    """
    Initialize non-varying module variables, conversion factors,!
    and look-up tables.

    inputs:
        me       - print control for parallel process

    outputs: (none)

    external module variables:  (in physparam)
        ilwrate - heating rate unit selections                              
                =1: output in k/day                                         
                =2: output in k/second                                      
        ilwrgas - control flag for rare gases (ch4,n2o,o2,cfcs, etc.)       
                =0: do not include rare gases                               
                >0: include all rare gases                                  
        ilwcliq - liquid cloud optical properties contrl flag               
                =0: input cloud opt depth from diagnostic scheme            
                >0: input cwp,rew, and other cloud content parameters       
        isubclw - sub-column cloud approximation control flag               
                =0: no sub-col cld treatment, use grid-mean cld quantities  
                =1: mcica sub-col, prescribed seeds to get random numbers   
                =2: mcica sub-col, providing array icseed for random numbers
        icldflg - cloud scheme control flag                                 
                =0: diagnostic scheme gives cloud tau, omiga, and g.        
                =1: prognostic scheme gives cloud liq/ice path, etc.        
        iovrlw  - clouds vertical overlapping control flag                  
                =0: random overlapping clouds                               
                =1: maximum/random overlapping clouds                       
                =2: maximum overlap cloud (isubcol>0 only)                  
                =3: decorrelation-length overlap (for isubclw>0 only)       

    definitions:                                                          
        arrays for 10000-point look-up tables:                              
        tau_tbl - clear-sky optical depth (used in cloudy radiative transfer
        exp_tbl - exponential lookup table for tansmittance                 
        tfn_tbl - tau transition function; i.e. the transition of the Planck
                  function from that for the mean layer temperature to that 
                  for the layer boundary temperature as a function of optical
                  depth. the "linear in tau" method is used to make the table
"""



    # locals:
    expeps = 1.e-20

    tfn = 0
    pival = 0
    explimit = 0

    #===> ... begin here

    if iovrlw < 0 or iovrlw > 3:
        print(f'  *** Error in specification of cloud overlap flag',
              ' IOVRLW={iovrlw}, in RLWINIT !!')
    elif iovrlw >= 2 and isubclw == 0:
        if me == 0:
            print(f'  *** IOVRLW={iovrlw} is not available for',
                  ' ISUBCLW=0 setting!!')
            print('      The program uses maximum/random overlap instead.')

        iovrlw = 1

    if me == 0:
        print(f' - Using AER Longwave Radiation, Version: {VTAGLW}')

        if ilwrgas > 0:
            print('   --- Include rare gases N2O, CH4, O2, CFCs ',
                  'absorptions in LW')
        else:
            print('   --- Rare gases effect is NOT included in LW')

        if isubclw == 0:
            print('   --- Using standard grid average clouds, no ',
                  'sub-column clouds approximation applied')
        elif isubclw == 1:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  'with a prescribed sequence of permutaion seeds')
        elif isubclw == 2:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  'with provided input array of permutation seeds')
        else:
            print(f'  *** Error in specification of sub-column cloud ',
                  ' control flag isubclw = {isubclw}!!')

    #  --- ...  check cloud flags for consistency

    if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
        print('  *** Model cloud scheme inconsistent with LW',
              ' radiation cloud radiative property setup !!')

    #  --- ...  setup default surface emissivity for each band here

    semiss0 = f_one

    #  --- ...  setup constant factors for flux and heating rate
    #           the 1.0e-2 is to convert pressure from mb to N/m**2

    pival = 2.0 * np.arcsin(f_one)
    fluxfac = pival * 2.0

    if ilwrate == 1:
        heatfac = con_g * 864.0 / con_cp            #   (in k/day)
    else:
        heatfac = con_g * 1.0e-2 / con_cp           #   (in k/second)

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

    tau_tbl[0] = f_zero
    exp_tbl[0] = f_one
    tfn_tbl[0] = f_zero

    tau_tbl[ntbl] = 1.e10
    exp_tbl[ntbl] = expeps
    tfn_tbl[ntbl] = f_one

    explimit = np.log(tiny(exp_tbl(0)))
    explimit = np.sign(explimit) * np.floor(np.abs(explimit))

    for i in range(ntbl-1):
        tfn = (i+1) / (ntbl-i+1)
        tau_tbl[i] = bpade * tfn
        if tau_tbl[i] >= explimit:
            exp_tbl[i] = expeps
        else:
            exp_tbl[i] = exp(-tau_tbl[i])

        if tau_tbl[i] < 0.06:
            tfn_tbl[i] = tau_tbl[i] / 6.0
        else:
            tfn_tbl[i] = f_one - 2.0*((f_one / tau_tbl[i]) -(exp_tbl[i]/(f_one - exp_tbl[i])))
