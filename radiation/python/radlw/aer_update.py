import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import aeros_file

def aer_update(iyear, imon, me, lalwflg, laswflg, lavoflg, kyrstr,
               kyrend):
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
    if imon < 1 or imon > 12:
        print('***** ERROR in specifying requested month !!! ',
              f'imon = {imon}')
        print('***** STOPPED in subroutinte aer_update !!!')
   
    # -# Call trop_update() to update monthly tropospheric aerosol data.
    if lalwflg or laswflg:
        kprfg, idxcg, cmixg, denng = trop_update(me, imon)
   
    # -# Call volc_update() to update yearly stratospheric volcanic aerosol data.
    if lavoflg:
        ivolae = volc_update(imon, iyear, kyrstr, kyrend, me)

    return kprfg, idxcg, cmixg, denng, ivolae
   
   
def trop_update(me, imon):
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
   
    NXC = 5
    IMXAE = 72
    JMXAE = 37

    file_exist = os.path.isfile(aeros_file)
   
    if file_exist:
        if me == 0:
            print(f'Opened aerosol data file: {aeros_file}')
    else:
        print(f'Requested aerosol data file "{aeros_file}" not found!')
        print('*** Stopped in subroutine trop_update !!')

    kprf = np.zeros((IMXAE, JMXAE))
    idxcg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.int32)
    cmixg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.float64)
    denng = np.zeros((2, IMXAE, JMXAE), dtype=np.float64)

    ds = xr.open_dataset(aeros_file)
    kprfg = ds['kprfg'].data
    idxcg = ds['idxcg'].data
    cmixg = ds['cmixg'].data
    denng = ds['denng'].data
    cline = ds['cline'].data

    if me == 0:
        print(f'  --- Reading {cline[imon-1]}')

    return kprfg, idxcg, cmixg, denng  
   
   
def volc_update(imon, iyear, kyrstr, kyrend, me):
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
    MINVYR = 1850
    MAXVYR = 1999
    #
    #===>  ...  begin here
    #
    kmonsav = imon
   
    if kyrstr <= iyear and iyear <= kyrend:   # use previously input data
        kyrsav = iyear
        return
    else:                                            # need to input new data
        kyrsav = iyear
        kyrstr = iyear - iyear % 10
        kyrend = kyrstr + 9

        if iyear < MINVYR or iyear > MAXVYR:
            ivolae = np.ones((12, 4, 10))            # set as lowest value
            if me == 0:
                print('Request volcanic date out of range,',
                      ' optical depth set to lowest value')
        else:
            file_exist = os.path.isfile(volcano_file)
            if file_exist:
                ds = xr.open_dataset(volcano_file)
                cline = ds['cline']
                #  ---  check print
                if me == 0:
                    print(f'Opened volcanic data file: {volcano_file}')
                    print(cline)
   
                ivolae = ds['ivolae']
            else:
                print(f'Requested volcanic data file "{volcano_file}" not found!')
                print('*** Stopped in subroutine VOLC_AERINIT !!')

    #  ---  check print
    if me == 0:
        k = (kyrsav % 10) + 1
        print(f'CHECK: Sample Volcanic data used for month, year: {imon}, {iyear}')           
        print(ivolae[kmonsav, :, k])

    return ivolae