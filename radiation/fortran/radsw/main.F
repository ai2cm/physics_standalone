program test
      use m_countdown, only : cdstart, cdshow, cdstop
      use machine, only : kind_phys
      use module_radlw_parameters
      use module_radsw_parameters
      use module_radiation_gases, only : NF_VGAS
      use module_radiation_clouds, only : NF_CLDS
      use module_radiation_aerosols, only : NF_AESW
      use physcons, only : con_g, con_cp, con_avgd, con_amd, &
      con_amw, con_amo3
      use module_radsw_main, only : swrad
      use module_radiation_driver, only : radupdate

      integer :: im, lmk, lmp, npts, levr, imp_physics
      
      real (kind=kind_phys), parameter :: eps     = 1.0e-6
      real (kind=kind_phys), parameter :: oneminus= 1.0-eps
      real (kind=kind_phys), parameter :: bpade   = 1.0/0.278  ! pade approx constant
      real (kind=kind_phys), parameter :: stpfac  = 296.0/1013.0
      real (kind=kind_phys), parameter :: ftiny   = 1.0e-12
      real (kind=kind_phys), parameter :: flimit  = 1.0e-20
      real (kind=kind_phys), parameter :: s0      = 1368.22
      real (kind=kind_phys), parameter :: f_zero  = 0.0
      real (kind=kind_phys), parameter :: f_one   = 1.0

!  ...  atomic weights for conversion from mass to volume mixing ratios
      real (kind=kind_phys), parameter :: amdw    = con_amd/con_amw
      real (kind=kind_phys), parameter :: amdo3   = con_amd/con_amo3
  
!  .  ..  band indices
      integer, dimension(nblow:nbhgh) :: nspa, nspb
      integer, dimension(nblow:nbhgh) :: idxsfc
      integer, dimension(nblow:nbhgh) :: idxebc
  
      data nspa(:) /  9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1 /
      data nspb(:) /  1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1 /

      data idxsfc(:) / 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1 /  ! band index for sfc flux
      data idxebc(:) / 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5 /  ! band index for cld prop
  
      integer, parameter :: nuvb = 27

!! ---  logical flags for optional output fields

      logical :: lhswb  = .false.
      logical :: lhsw0  = .false.
      logical :: lflxprf= .false.
      logical :: lfdncmp= .false.

!  ---  those data will be set up only once by "rlwinit"

!  ...  fluxfac, heatfac are factors for fluxes (in w/m**2) and heating
!       rates (in k/day, or k/sec set by subroutine 'rlwinit')
!       semiss0 are default surface emissivity for each bands

      real (kind=kind_phys) :: heatfac

      real (kind=kind_phys) :: exp_tbl(0:NTBMX)  !transmittance lookup table

!> initial permutation seed used for sub-column cloud scheme
      integer, parameter :: ipsdsw0 = 1

!  ---  the following variables are used for sub-column cloud scheme

      logical :: lprnt
      logical, dimension(:), allocatable :: flag_iter, flag
  
      integer :: iter, i
      integer :: tile, num_tiles
      integer :: ser_count, ser_count_max
      character(len=6) :: ser_count_str
  
      real(8) :: time1, time2

      real(kind=kind_phys), dimension(:, :), allocatable :: plyr, tlyr, qlyr, olyr, &
      dz, delp, htswc, htsw0, htswc_ref
      real(kind=kind_phys), dimension(:, :), allocatable :: plvl, tlvl, sfcalb
      real(kind=kind_phys), dimension(:, :, :), allocatable :: gasvmr
      real(kind=kind_phys), dimension(:, :, :), allocatable :: clouds
      real(kind=kind_phys), dimension(:, :, :, :), allocatable :: faersw
      real(kind=kind_phys), dimension(:, :), allocatable :: cldtausw, cldtausw_ref
      real(kind=kind_phys), dimension(:), allocatable :: de_lgth, coszen
      integer, dimension(:), allocatable :: icsdsw, idxday
      real(kind=kind_phys), dimension(:), allocatable :: si

      type(topfsw_type), dimension(:), allocatable :: topfsw
      type(sfcfsw_type), dimension(:), allocatable :: sfcfsw
      type(cmpfsw_type), dimension(:), allocatable :: scmpsw

      real(kind=kind_phys), dimension(:), allocatable :: upfxc_t_ref, upfx0_t_ref, upfxc_s_ref, upfx0_s_ref
      real(kind=kind_phys), dimension(:), allocatable :: dnfxc_t_ref, dnfxc_s_ref, dnfx0_s_ref
      real(kind=kind_phys), dimension(:), allocatable :: uvbfc_ref, uvbf0_ref, nirbm_ref, nirdf_ref, visbm_ref, visdf_ref 

      integer, parameter :: me = 0
      integer :: ictm, isol, ico2, iaer, num_p2d,    &
     &       ntcw, ialb, iems, num_p3d, npdf3d, ntoz, iovr_sw, iovr_lw, &
     &       isubc_sw, isubc_lw, icliq_sw, iflip, idate(4), jdate(4), nday
      logical :: crick_proof, ccnorm, norad_precip

      integer:: idat(8), jdat(8)
      logical :: lsswr

      real (kind=kind_phys) :: fhswr, dtf

!  ---  outputs:
      real (kind=kind_phys) :: slag, sdec, cdec, solcon


      !$ser init directory='./dump' prefix='Serialized' directory_ref='../data' prefix_ref='Generator' mpi_rank=1
      !$ser mode read
      !$ser savepoint "swrad-in-000000"
      !$ser data im=im levr=levr 

      write(*,*) 'im = ', im
      
      allocate(plyr(im, levr), tlyr(im, levr), qlyr(im, levr), olyr(im, levr), dz(im, levr), &
               delp(im, levr), htswc(im, levr), htsw0(im, levr), htswc_ref(im, levr))
      allocate(plvl(im, levr+1), tlvl(im, levr+1))
      allocate(coszen(im), idxday(im))
      allocate(sfcalb(im, 4))
      allocate(gasvmr(im, levr, NF_VGAS))
      allocate(clouds(im, levr, NF_CLDS))
      allocate(faersw(im, levr, NBDSW, NF_AESW))
      allocate(de_lgth(im))
      allocate(cldtausw(im, levr), cldtausw_ref(im, levr))
      allocate(icsdsw(im), topfsw(im), sfcfsw(im), scmpsw(im), upfxc_t_ref(im), upfx0_t_ref(im), upfxc_s_ref(im), upfx0_s_ref(im))
      allocate(dnfxc_t_ref(im), dnfxc_s_ref(im), dnfx0_s_ref(im))
      allocate(uvbfc_ref(im), uvbf0_ref(im), nirbm_ref(im), nirdf_ref(im), visbm_ref(im), visdf_ref(im))
      allocate(si(levr+1))

      !$ser mode read
      !$ser savepoint "rad-initialize"
      !$ser data si=si ictm=ictm isol=isol imp_physics=imp_physics
        !$ser data ico2=ico2 iaer=iaer ialb=ialb iems=iems
        !$ser data ntcw=ntcw num_p2d=num_p2d num_p3d=num_p3d npdf3d=npdf3d
        !$ser data ntoz=ntoz iovr_sw=iovr_sw iovr_lw=iovr_lw isubc_sw=isubc_sw
        !$ser data isubc_lw=isubc_lw icliq_sw=icliq_sw crick_proof=crick_proof
        !$ser data ccnorm=ccnorm norad_precip=norad_precip idate=idate iflip=iflip
      ! Initialization

      write(*,*) 'isubc_lw = ',isubc_lw

      call rad_initialize(si,  levr,         ictm,    isol,      &
                          ico2,        iaer,         ialb,    iems,      &
                          ntcw,        num_p2d,      num_p3d, npdf3d,    &
                          ntoz,        iovr_sw,      iovr_lw, isubc_sw,  &
                          isubc_lw,    icliq_sw,     crick_proof, ccnorm,&
                          imp_physics, norad_precip, idate,   iflip,  me)

      ser_count_max = 6
      num_tiles = 0

      call cdstart(num_tiles * ser_count_max * 2)
      write(*,*) "Running swrad . . ."

      do tile = 1, 5

      !$ser init directory='./dump' prefix='Serialized' directory_ref='../data' prefix_ref='Generator' mpi_rank=tile

      do ser_count = 0, ser_count_max-1
  
          call cdshow(tile * ser_count_max + ser_count * 2 + iter - 1)
          
          write(ser_count_str, '(i6.6)') ser_count
          write(*,*) 'ser_count = ' // ser_count_str

!$ser mode read
!$ser verbatim write(ser_count_str, '(i6.6)') ser_count
!$ser savepoint "swrad-in-"//trim(ser_count_str)
!$ser data plyr=plyr plvl=plvl tlyr=tlyr tlvl=tlvl qlyr=qlyr olyr=olyr
!$ser data gasvmr=gasvmr clouds=clouds icsdsw=icsdsw faersw=faersw
!$ser data sfcalb=sfcalb dz=dz delp=delp de_lgth=de_lgth
!$ser data coszen=coszen solcon=solcon nday=nday idxday=idxday
!$ser data im=im lmk=lmk lmp=lmp lprnt=lprnt

          call swrad(ser_count, plyr, plvl, tlyr, tlvl, qlyr, olyr, gasvmr,  &        !  ---  inputs
                     clouds, icsdsw, faersw, sfcalb,   &
                     dz, delp, de_lgth, coszen, solcon, nday, idxday,        &
                     im, lmk, lmp, lprnt,                   &
                     htswc, topfsw, sfcfsw, cldtausw, HSW0=htsw0, FDNCMP=scmpsw)        !  ---  outputs

!$ser mode read
!$ser savepoint "swrad-out-"//trim(ser_count_str)
! outputs
!$ser data htswc=htswc_ref upfxc_t=upfxc_t_ref upfx0_t=upfx0_t_ref
!$ser data dnfxc_t=dnfxc_t_ref upfxc_s=upfxc_s_ref upfx0_s=upfx0_s_ref
!$ser data dnfxc_s=dnfxc_s_ref dnfx0_s=dnfx0_s_ref
!$ser data uvbfc=uvbfc_ref uvbf0=uvbf0_ref nirbm=nirbm_ref nirdf=nirdf_ref
!$ser data visbm=visbm_ref visdf=visdf_ref 
!$ser data cldtausw=cldtausw_ref

      call check_r_2d(htswc, htswc_ref, "htswc")
      call check_r_1d(topfsw%upfxc, upfxc_t_ref, "upfxc_t")
      call check_r_1d(topfsw%upfx0, upfx0_t_ref, "upfx0_t")
      call check_r_1d(topfsw%dnfxc, dnfxc_t_ref, "dnfxc_t")
      call check_r_1d(sfcfsw%upfxc, upfxc_s_ref, "upfxc_s")
      call check_r_1d(sfcfsw%upfx0, upfx0_s_ref, "upfx0_s")
      call check_r_1d(sfcfsw%dnfxc, dnfxc_s_ref, "dnfxc_s")
      call check_r_1d(sfcfsw%dnfx0, dnfx0_s_ref, "dnfx0_s")
      call check_r_1d(scmpsw%uvbfc, uvbfc_ref, "uvbfc")
      call check_r_1d(scmpsw%uvbf0, uvbf0_ref, "uvbf0")
      call check_r_1d(scmpsw%nirbm, nirbm_ref, "nirbm")
      call check_r_1d(scmpsw%nirdf, nirdf_ref, "nirdf")
      call check_r_1d(scmpsw%visbm, visbm_ref, "visbm")
      call check_r_1d(scmpsw%visdf, visdf_ref, "visdf")
      call check_r_2d(cldtausw, cldtausw_ref, "cldtausw")
      end do
      

      end do

      !$ser init directory='./dump' prefix='Serialized' directory_ref='../data' prefix_ref='Generator' mpi_rank=0
      !$ser mode read
      !$ser savepoint "rad-update"
      !$ser data idat=idat jdat=jdat fhswr=fhswr dtf=dtf lsswr=lsswr
      !$ser data slag=slag sdec=sdec cdec=cdec solcon=solcon
      

      write(*,*) 'Running update'

      call radupdate (ser_count, idat, jdat, fhswr, dtf,  lsswr, &
                      me,   slag, sdec,  cdec, solcon)
        
      !$ser cleanup

      call cdstop()

      write(*,*) 'FINISHED!'

      !write(*,*) 'htlwc = ', htlwc
      !write(*,*) 'htlwc_ref = ', htlwc_ref

contains

      subroutine check_r_2d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        else
            write(*,*) trim(name) // ' validates!'
        end if

      end subroutine check_r_2d

      subroutine check_r_1d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:), b(:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        else
            write(*,*) trim(name) // ' validates!'
        end if

      end subroutine check_r_1d

      elemental logical function isclose_r(a, b, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a, b
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        real (kind=kind_phys) :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 1.0d-30
        end if
        if (present(rtol)) then
            rtol_local = rtol
        else
            rtol_local = 1.0d-11
        end if

        isclose_r = abs(a - b) <= (atol_local + rtol_local * abs(b))

      end function isclose_r

      elemental logical function isclose_i(a, b, atol)
        implicit none
        integer, intent(in) :: a, b
        integer, intent(in), optional :: atol

        integer :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 0
        end if

        isclose_i = abs(a - b) <= atol_local

      end function isclose_i

      subroutine tic(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t1)
      end subroutine tic

      subroutine toc(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t2)
        write(*,'(a,f5.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
      end subroutine toc

end program test
