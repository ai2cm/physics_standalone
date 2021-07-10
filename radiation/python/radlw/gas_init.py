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
        if timeozc != 12:
            print(' - Using climatology ozone distribution')
            print(' timeozc=',timeozc, ' is not monthly mean',
                  ' - job aborting in subroutin gas_init!!!')

        pkstr = np.zeros(LOZ)
        o3r = np.zeros((JMR, LOZ, 12))

        if LOZ == 17:       # For the operational ozone climatology
         do k = 1, LOZ
           read (NIO3CLM,15) pstr4(k)
  15       format(f10.3)
         enddo

         do imo = 1, 12
           do j = 1, JMR
             read (NIO3CLM,16) imond(imo), ilat(j,imo),                &
    &                          (o3clim4(j,k,imo),k=1,10)
  16         format(i2,i4,10f6.2)
             read (NIO3CLM,20) (o3clim4(j,k,imo),k=11,LOZ)
  20         format(6x,10f6.2)
           enddo
         enddo
       else                      ! For newer ozone climatology
         read (NIO3CLM)
         do k = 1, LOZ
           read (NIO3CLM) pstr4(k)
         enddo

         do imo = 1, 12
           do k = 1, LOZ
             read (NIO3CLM) (o3clim4(j,k,imo),j=1,JMR)
           enddo
         enddo
       endif   ! end if_LOZ_block
!
       do imo = 1, 12
         do k = 1, LOZ
           do j = 1, JMR
             o3r(j,k,imo) = o3clim4(j,k,imo) * 1.655e-6
           enddo
         enddo
       enddo

       do k = 1, LOZ
         pstr(k) = pstr4(k)
       enddo

       if ( me == 0 ) then
         print *,' - Using climatology ozone distribution'
         print *,'   Found ozone data for levels pstr=',               &
    &            (pstr(k),k=1,LOZ)
!         print *,' O3=',(o3r(15,k,1),k=1,LOZ)
       endif

       do k = 1, LOZ
         pkstr(k) = fpkapx(pstr(k)*100.0)
       enddo
     endif   ! end if_ioznflg_block

!  --- ...  co2 data section

     co2_glb = co2vmr_def

     lab_ico2 : if ( ico2flg == 0 ) then

       if ( me == 0 ) then
         print *,' - Using prescribed co2 global mean value=',         &
    &              co2vmr_def
       endif

     else  lab_ico2

       lab_ictm : if ( ictmflg == -1 ) then      ! input user provided data

         inquire (file=co2usr_file, exist=file_exist)
         if ( .not. file_exist ) then
           print *,'   Can not find user CO2 data file: ',co2usr_file, &
    &              ' - Stopped in subroutine gas_init !!'
           stop
         else
           close (NICO2CN)
           open(NICO2CN,file=co2usr_file,form='formatted',status='old')
           rewind NICO2CN
           read (NICO2CN, 25) iyr, cline, co2g1, co2g2
 25        format(i4,a94,f7.2,16x,f5.2)
           co2_glb = co2g1 * 1.0e-6

           if ( ico2flg == 1 ) then
             if ( me == 0 ) then
               print *,' - Using co2 global annual mean value from',   &
    &                  ' user provided data set:',co2usr_file
               print *, iyr,cline(1:94),co2g1,'  GROWTH RATE =', co2g2
             endif
           elseif ( ico2flg == 2 ) then
             allocate ( co2vmr_sav(IMXCO2,JMXCO2,12) )

             do imo = 1, 12
               read (NICO2CN,cform) co2dat
!check          print cform, co2dat

               do j = 1, JMXCO2
                 do i = 1, IMXCO2
                   co2vmr_sav(i,j,imo) = co2dat(i,j) * 1.0e-6
                 enddo
               enddo
             enddo

             if ( me == 0 ) then
               print *,' - Using co2 monthly 2-d data from user',      &
    &                ' provided data set:',co2usr_file
               print *, iyr,cline(1:94),co2g1,'  GROWTH RATE =', co2g2

               print *,' CHECK: Sample of selected months of CO2 data'
               do imo = 1, 12, 3
                 print *,'        Month =',imo
                 print *, (co2vmr_sav(1,j,imo),j=1,jmxco2)
               enddo
             endif
           else
             print *,' ICO2=',ico2flg,' is not a valid selection',     &
    &                ' - Stoped in subroutine gas_init!!!'
             stop
           endif    ! endif_ico2flg_block

           close (NICO2CN)
         endif    ! endif_file_exist_block

       else   lab_ictm                           ! input from observed data

         if ( ico2flg == 1 ) then
           if ( me == 0 ) then
             print *,' - Using observed co2 global annual mean value'
           endiF
         elseif ( ico2flg == 2 ) then
           allocate ( co2vmr_sav(IMXCO2,JMXCO2,12) )

           if ( me == 0 ) then
             print *,' - Using observed co2 monthly 2-d data'
           endif
         else
           print *,' ICO2=',ico2flg,' is not a valid selection',       &
    &              ' - Stoped in subroutine gas_init!!!'
           stop
         endif

         if ( ictmflg == -2 ) then
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