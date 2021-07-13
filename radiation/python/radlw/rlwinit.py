import numpy as np
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import (isubclw as isubclw,
                          ilwrgas as ilwrgas,
                          icldflg as icldflg,
                          ilwcliq as ilwcliq,
                          ilwrate as ilwrate,
                          iovrlw as iovrlw)
from radlw_param import ntbl
from phys_const import (con_g as g,
                        con_cp as cp)

def rlwinit(me, tau_tbl, exp_tbl, tfn_tbl):
    # locals:
    VTAGLW='NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82'
    expeps = 1.e-20

    tfn = 0
    pival = 0
    explimit = 0
    f_one = 1.0
    f_zero = 0.0
    bpade   = 1.0/0.278

    #===> ... begin here

    if iovrlw < 0 or iovrlw > 3:
        print(f'  *** Error in specification of cloud overlap flag',
              f' IOVRLW={iovrlw}, in RLWINIT !!')
    elif iovrlw >= 2 and isubclw == 0:
        if me == 0:
            print(f'  *** IOVRLW={iovrlw} is not available for',
                  ' ISUBCLW=0 setting!!')
            print('      The program uses maximum/random overlap instead.')

    if me == 0:
        print(f'- Using AER Longwave Radiation, Version: {VTAGLW}')

        if ilwrgas > 0:
            print('   --- Include rare gases N2O, CH4, O2, CFCs ',
                  'absorptions in LW')
        else:
            print('   --- Rare gases effect is NOT included in LW')

        if isubclw == 0:
            print('   --- Using standard grid average clouds, no ',
                  '   sub-column clouds approximation applied')
        elif isubclw == 1:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  '   with a prescribed sequence of permutaion seeds')
        elif isubclw == 2:
            print('   --- Using MCICA sub-colum clouds approximation ',
                  '   with provided input array of permutation seeds')
        else:
            print(f'  *** Error in specification of sub-column cloud ',
                  f' control flag isubclw = {isubclw}!!')

    #  --- ...  check cloud flags for consistency

    if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
        print('*** Model cloud scheme inconsistent with LW',
              'radiation cloud radiative property setup !!')

    #  --- ...  setup default surface emissivity for each band here

    semiss0 = f_one

    #  --- ...  setup constant factors for flux and heating rate
    #           the 1.0e-2 is to convert pressure from mb to N/m**2

    pival = 2.0 * np.arcsin(f_one)
    fluxfac = pival * 2.0

    if ilwrate == 1:
        heatfac = g * 864.0 / cp            #   (in k/day)
    else:
        heatfac = g * 1.0e-2 / cp           #   (in k/second)

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

    explimit = np.finfo(float).tiny
    explimit = np.sign(explimit) * np.floor(np.abs(explimit))

    for i in range(ntbl-1):
        tfn = (i+1) / (ntbl-i+1)
        tau_tbl[i] = bpade * tfn
        if tau_tbl[i] >= explimit:
            exp_tbl[i] = expeps
        else:
            exp_tbl[i] = np.exp(-tau_tbl[i])

        if tau_tbl[i] < 0.06:
            tfn_tbl[i] = tau_tbl[i] / 6.0
        else:
            tfn_tbl[i] = f_one - 2.0*((f_one / tau_tbl[i]) -exp_tbl[i]/(f_one - exp_tbl[i]))
