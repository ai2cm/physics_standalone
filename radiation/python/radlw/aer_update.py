import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import aeros_file

def aer_update( iyear, imon, me ):
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
        trop_update()
   
    # -# Call volc_update() to update yearly stratospheric volcanic aerosol data.
    if lavoflg:
        volc_update()
   
   
def trop_update():
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

    idxcg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.int32)
    cmixg = np.zeros((NXC, IMXAE, JMXAE), dtype=np.float64)
    denng = np.zeros((2, IMXAE, JMXAE), dtype=np.float64)
   
    #  --- ...  loop over 12 month global distribution
   
    for m in range(12):
   
           read(NIAERCM,12) cline
     12    format(a80/)
   
           if ( m /= imon ) then
   !         if ( me == 0 ) print *,'  *** Skipped ',cline
   
             do j = 1, JMXAE
               do i = 1, IMXAE
                 read(NIAERCM,*) id
               enddo
             enddo
           else
             if ( me == 0 ) print *,'  --- Reading ',cline
   
             do j = 1, JMXAE
               do i = 1, IMXAE
                 read(NIAERCM,14) (idxc(k),cmix(k),k=1,NXC),kprf,denn,nc,  &
        &                         ctyp
     14          format(5(i2,e11.4),i2,f8.2,i3,1x,a3)
   
                 kprfg(i,j)     = kprf
                 denng(1,i,j)   = denn       ! num density of 1st layer
                 if ( kprf >= 6 ) then
                   denng(2,i,j) = cmix(NXC)  ! num density of 2dn layer
                 else
                   denng(2,i,j) = f_zero
                 endif
   
                 tem = f_one
                 do k = 1, NXC-1
                   idxcg(k,i,j) = idxc(k)    ! component index
                   cmixg(k,i,j) = cmix(k)    ! component mixing ratio
                   tem          = tem - cmix(k)
                 enddo
                 idxcg(NXC,i,j) = idxc(NXC)
                 cmixg(NXC,i,j) = tem        ! to make sure all add to 1.
               enddo
             enddo
   
             close (NIAERCM)
             exit  Lab_do_12mon
           endif     ! end if_m_block
   
         enddo  Lab_do_12mon
   
   
   
def volc_update():
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
    kmonsav = imon
   
    if kyrstr <= iyear and iyear <= kyrend:   # use previously input data
        kyrsav = iyear
        return
    else                                            # need to input new data
        kyrsav = iyear
        kyrstr = iyear - iyear % 10
        kyrend = kyrstr + 9

        if iyear < MINVYR or iyear > MAXVYR:
            ivolae += 1            # set as lowest value
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