subroutine sfc_init                                               &
    &     ( me )!  ---  inputs:
!  ---  outputs: ( none )

!  ===================================================================  !
!                                                                       !
!  this program is the initialization program for surface radiation     !
!  related quantities (albedo, emissivity, etc.)                        !
!                                                                       !
! usage:         call sfc_init                                          !
!                                                                       !
! subprograms called:  none                                             !
!                                                                       !
!  ====================  defination of variables  ====================  !
!                                                                       !
!  inputs:                                                              !
!      me           - print control flag                                !
!                                                                       !
!  outputs: (none) to module variables only                             !
!                                                                       !
!  external module variables:                                           !
!     ialbflg       - control flag for surface albedo schemes           !
!                     =0: climatology, based on surface veg types       !
!                     =1:                                               !
!     iemsflg       - control flag for sfc emissivity schemes (ab:2-dig)!
!                     a:=0 set sfc air/ground t same for lw radiation   !
!                       =1 set sfc air/ground t diff for lw radiation   !
!                     b:=0 use fixed sfc emissivity=1.0 (black-body)    !
!                       =1 use varying climtology sfc emiss (veg based) !
!                                                                       !
!  ====================    end of description    =====================  !
!
     implicit none

!  ---  inputs:
     integer, intent(in) :: me

!  ---  outputs: ( none )

!  ---  locals:
     integer    :: i, k
!     integer    :: ia, ja
     logical    :: file_exist
     character  :: cline*80
!
!===> ...  begin here
!
     if ( me == 0 ) print *, VTAGSFC   ! print out version tag

!> - Initialization of surface albedo section
!! \n physparam::ialbflg 
!!  - = 0: using climatology surface albedo scheme for SW
!!  - = 1: using MODIS based land surface albedo for SW

     if ( ialbflg == 0 ) then

       if ( me == 0 ) then
         print *,' - Using climatology surface albedo scheme for sw'
       endif

     else if ( ialbflg == 1 ) then

       if ( me == 0 ) then
         print *,' - Using MODIS based land surface albedo for sw'
       endif

     else
       print *,' !! ERROR in Albedo Scheme Setting, IALB=',ialbflg
       stop
     endif    ! end if_ialbflg_block

!> - Initialization of surface emissivity section
!! \n physparam::iemsflg
!!  - = 0: fixed SFC emissivity at 1.0
!!  - = 1: input SFC emissivity type map from "semis_file"

     iemslw = mod(iemsflg, 10)          ! emissivity control
     if ( iemslw == 0 ) then            ! fixed sfc emis at 1.0

       if ( me == 0 ) then
         print *,' - Using Fixed Surface Emissivity = 1.0 for lw'
       endif

     elseif ( iemslw == 1 ) then        ! input sfc emiss type map

!  ---  allocate data space
       if ( .not. allocated(idxems) ) then
         allocate ( idxems(IMXEMS,JMXEMS)    )
       endif

!  ---  check to see if requested emissivity data file existed

       inquire (file=semis_file, exist=file_exist)

       if ( .not. file_exist ) then
         if ( me == 0 ) then
           print *,' - Using Varying Surface Emissivity for lw'
           print *,'   Requested data file "',semis_file,'" not found!'
           print *,'   Change to fixed surface emissivity = 1.0 !'
         endif

         iemslw = 0
       else
         close(NIRADSF)
         open (NIRADSF,file=semis_file,form='formatted',status='old')
         rewind NIRADSF

         read (NIRADSF,12) cline
 12      format(a80)

         read (NIRADSF,14) idxems
 14      format(80i1)

         if ( me == 0 ) then
           print *,' - Using Varying Surface Emissivity for lw'
           print *,'   Opened data file: ',semis_file
           print *, cline
!check      print *,' CHECK: Sample emissivity index data'
!           ia = IMXEMS / 5
!           ja = JMXEMS / 5
!           print *, idxems(1:IMXEMS:ia,1:JMXEMS:ja)
         endif

         close(NIRADSF)
       endif    ! end if_file_exist_block

     else
       print *,' !! ERROR in Emissivity Scheme Setting, IEMS=',iemsflg
       stop
     endif   ! end if_iemslw_block

!
     return
!...................................
     end subroutine sfc_init
!-----------------------------------