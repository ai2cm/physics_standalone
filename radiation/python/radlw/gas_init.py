import numpy as np

def gas_init(me):
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

    IMXCO2 = 24
    JMXCO2 = 12

    co2dat = np.zeros((IMXCO2,JMXCO2))
    pstr = np.zeros(LOZ)
    o3clim4 = np.zeros((JMR, LOZ, 12))
    pstr4 = np.zeros(LOZ)

    imond = np.zeros(12)
    ilat = np.zeros((JMR, 12))
    #data  cform  / '(24f7.2)' /       !! data format in IMXCO2*f7.2

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
                print('ico2flg = 1 is not implemented')
           
            elif ico2flg == 2:
                print('ico2flg = 2 is not implemented')
           
            else:
                print(f' ICO2={ico2flg}, is not a valid selection',
                      ' - Stoped in subroutine gas_init!!!')


        if ictmflg == -2:
           inquire (file=co2cyc_file, exist=file_exist)
           if ( .not. file_exist ) then
             if ( me == 0 ) then
               print *,'   Can not find seasonal cycle CO2 data: ',    &
    &               co2cyc_file,' - Stopped in subroutine gas_init !!'
             endif
             stop
           else
             allocate( co2cyc_sav(IMXCO2,JMXCO2,12) )

!  --- ...  read in co2 2-d seasonal cycle data
             close (NICO2CN)
             open (NICO2CN,file=co2cyc_file,form='formatted',          &
    &              status='old')
             rewind NICO2CN
             read (NICO2CN, 35) cline, co2g1, co2g2
 35          format(a98,f7.2,16x,f5.2)
             read (NICO2CN,cform) co2dat        ! skip annual mean part

             if ( me == 0 ) then
               print *,' - Superimpose seasonal cycle to mean CO2 data'
               print *,'   Opened CO2 climatology seasonal cycle data',&
    &                  ' file: ',co2cyc_file
!check          print *, cline(1:98), co2g1, co2g2
             endif

             do imo = 1, 12
               read (NICO2CN,45) cline, gco2cyc(imo)
 45            format(a58,f7.2)
!check          print *, cline(1:58),gco2cyc(imo)
               gco2cyc(imo) = gco2cyc(imo) * 1.0e-6

               read (NICO2CN,cform) co2dat
!check          print cform, co2dat
               do j = 1, JMXCO2
                 do i = 1, IMXCO2
                   co2cyc_sav(i,j,imo) = co2dat(i,j) * 1.0e-6
                 enddo
               enddo
             enddo

             close (NICO2CN)
           endif   ! endif_file_exist_block
         endif

       endif   lab_ictm
     endif   lab_ico2

     return
!
!...................................
     end subroutine gas_init
!-----------------------------------