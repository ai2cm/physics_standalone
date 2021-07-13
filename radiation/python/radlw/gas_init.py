import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import co2cyc_file, co2usr_file

def gas_init(me, ioznflg, ico2flg, ictmflg):
    #  ===================================================================  !
    #                                                                       !
    #  gas_init sets up ozone, co2, etc. parameters.  if climatology ozone  !
    #  then read in monthly ozone data.                                     !
    #                                                                       !
    #  inputs:                                               dimemsion      !
    #     me      - print message control flag                  1           !
    #                                                                       !
    #  outputs: (to the module variables)                                   !
    #    ( none )                                                           !
    #                                                                       !
    #  external module variables:  (in physparam)                           !
    #     ico2flg    - co2 data source control flag                         !
    #                   =0: use prescribed co2 global mean value            !
    #                   =1: use input global mean co2 value (co2_glb)       !
    #                   =2: use input 2-d monthly co2 value (co2vmr_sav)    !
    #     ictmflg    - =yyyy#, data ic time/date control flag               !
    #                  =   -2: same as 0, but superimpose seasonal cycle    !
    #                          from climatology data set.                   !
    #                  =   -1: use user provided external data for the fcst !
    #                          time, no extrapolation.                      !
    #                  =    0: use data at initial cond time, if not existed!
    #                          then use latest, without extrapolation.      !
    #                  =    1: use data at the forecast time, if not existed!
    #                          then use latest and extrapolate to fcst time.!
    #                  =yyyy0: use yyyy data for the forecast time, no      !
    #                          further data extrapolation.                  !
    #                  =yyyy1: use yyyy data for the fcst. if needed, do    !
    #                          extrapolation to match the fcst time.        !
    #     ioznflg    - ozone data control flag                              !
    #                   =0: use climatological ozone profile                !
    #                   >0: use interactive ozone profile                   !
    #     ivflip     - vertical profile indexing flag                       !
    #     co2usr_file- external co2 user defined data table                 !
    #     co2cyc_file- external co2 climotology monthly cycle data table    !
    #                                                                       !
    #  internal module variables:                                           !
    #     pkstr, o3r - arrays for climatology ozone data                    !
    #                                                                       !
    #  usage:    call gas_init                                              !
    #                                                                       !
    #  subprograms called:  none                                            !
    #                                                                       !
    #  ===================================================================  !
    #

    VTAGGAS = 'NCEP-Radiation_gases     v5.1  Nov 2012'
    IMXCO2 = 24
    JMXCO2 = 12
    co2vmr_def = 350.0e-6

    if me == 0:
        print(VTAGGAS)    # print out version tag

    kyrsav  = 0
    kmonsav = 1

    #  --- ...  climatology ozone data section

    if ioznflg > 0:
        if me == 0:
            print(' - Using interactive ozone distribution')
    else:
        print('Climatological ozone data not implemented')

    #  --- ...  co2 data section

    co2_glb = co2vmr_def

    if ico2flg == 0:
        if me == 0:
            print(f'- Using prescribed co2 global mean value={co2vmr_def}')

    else:
        if ictmflg == -1:      # input user provided data
            print('ictmflg = -1 is not implemented')

        else:                     # input from observed data
            if ico2flg == 1:
                print('Using observed co2 global annual mean value')
           
            elif ico2flg == 2:
                if me == 0:
                    print('Using observed co2 monthly 2-d data')
           
            else:
                print(f' ICO2={ico2flg}, is not a valid selection',
                      ' - Stoped in subroutine gas_init!!!')


        if ictmflg == -2:
            file_exist = os.path.isfile(co2cyc_file)
            if not file_exist:
                if me == 0:
                  print('Can not find seasonal cycle CO2 data: ',
                        f'{co2cyc_file} - Stopped in subroutine gas_init !!')
            else:
                co2cyc_sav = np.zeros((IMXCO2, JMXCO2, 12))
                ds = xr.open_dataset(co2cyc_file)
                #  --- ...  read in co2 2-d seasonal cycle data
                cline = ds['cline'].data
                co2g1 = ds['co2g1'].data
                co2g2 = ds['co2g2'].data
                co2dat = ds['co2dat']

                if me == 0:
                    print(' - Superimpose seasonal cycle to mean CO2 data')
                    print('Opened CO2 climatology seasonal cycle data',
                          f' file: {co2cyc_file}')

                gco2cyc = ds['gco2cyc']
                gco2cyc = gco2cyc * 1.0e-6
               
                co2cyc_sav = co2dat