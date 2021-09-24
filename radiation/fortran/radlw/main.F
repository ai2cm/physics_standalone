program test
      use m_countdown, only : cdstart, cdshow, cdstop
      use machine, only : kind_phys
      use module_radlw_parameters
      use module_radiation_gases, only : NF_VGAS
      use module_radiation_clouds, only : NF_CLDS
      use module_radiation_aerosols, only : NF_AELW
      use physcons, only : con_g, con_cp, con_avgd, con_amd, &
      con_amw, con_amo3
      use module_radlw_main, only : lwrad
      use module_radiation_driver, only : radupdate

      integer :: im, lmk, lmp, npts, levr, imp_physics
      
      real (kind=kind_phys), parameter :: eps     = 1.0e-6
      real (kind=kind_phys), parameter :: oneminus= 1.0-eps
      real (kind=kind_phys), parameter :: cldmin  = 1.0e-80
      real (kind=kind_phys), parameter :: bpade   = 1.0/0.278  ! pade approx constant
      real (kind=kind_phys), parameter :: stpfac  = 296.0/1013.0
      real (kind=kind_phys), parameter :: wtdiff  = 0.5        ! weight for radiance to flux conversion
      real (kind=kind_phys), parameter :: tblint  = ntbl       ! lookup table conversion factor
      real (kind=kind_phys), parameter :: f_zero  = 0.0
      real (kind=kind_phys), parameter :: f_one   = 1.0

!  ...  atomic weights for conversion from mass to volume mixing ratios
      real (kind=kind_phys), parameter :: amdw    = con_amd/con_amw
      real (kind=kind_phys), parameter :: amdo3   = con_amd/con_amo3
  
!  .  ..  band indices
      integer, dimension(nbands) :: nspa, nspb
  
      data nspa / 1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9 /
      data nspb / 1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0 /
  
      real (kind=kind_phys), dimension(nbands) :: a0, a1, a2
  
      data a0 / 1.66,  1.55,  1.58,  1.66,  1.54, 1.454,  1.89,  1.33,  &
      &         1.668,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66 /
      data a1 / 0.00,  0.25,  0.22,  0.00,  0.13, 0.446, -0.10,  0.40,  &
      &        -0.006,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00 /
      data a2 / 0.00, -12.0, -11.7,  0.00, -0.72,-0.243,  0.19,-0.062,  &
      &         0.414,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00 /

!! ---  logical flags for optional output fields

      logical :: lhlwb  = .false.
      logical :: lhlw0  = .false.
      logical :: lflxprf= .false.

!  ---  those data will be set up only once by "rlwinit"

!  ...  fluxfac, heatfac are factors for fluxes (in w/m**2) and heating
!       rates (in k/day, or k/sec set by subroutine 'rlwinit')
!       semiss0 are default surface emissivity for each bands

      real (kind=kind_phys) :: fluxfac, heatfac, semiss0(nbands)
      data semiss0(:) / nbands*1.0 /

      real (kind=kind_phys) :: tau_tbl(0:ntbl)  !clr-sky opt dep (for cldy transfer)
      real (kind=kind_phys) :: exp_tbl(0:ntbl)  !transmittance lookup table
      real (kind=kind_phys) :: tfn_tbl(0:ntbl)  !tau transition function; i.e. the
                                                !transition of planck func from mean lyr
                                                !temp to lyr boundary temp as a func of
                                                !opt dep. "linear in tau" method is used.

!  ---  the following variables are used for sub-column cloud scheme

      integer, parameter :: ipsdlw0 = ngptlw     ! initial permutation seed

      logical :: lprnt
      logical, dimension(:), allocatable :: flag_iter, flag
  
      integer :: iter, i
      integer :: tile, num_tiles
      integer :: ser_count, ser_count_max
      character(len=6) :: ser_count_str
      character(len=2) :: tile_str
  
      real(8) :: time1, time2

      real(kind=kind_phys), dimension(:, :), allocatable :: plyr, tlyr, qlyr, olyr, &
      dz, delp, htlwc, htlw0, htlwc_ref
      real(kind=kind_phys), dimension(:, :), allocatable :: plvl, tlvl
      real(kind=kind_phys), dimension(:, :, :), allocatable :: gasvmr
      real(kind=kind_phys), dimension(:, :, :), allocatable :: clouds
      real(kind=kind_phys), dimension(:, :, :, :), allocatable :: faerlw
      real(kind=kind_phys), dimension(:, :), allocatable :: cldtaulw, cldtaulw_ref
      real(kind=kind_phys), dimension(:), allocatable :: semis, tsfg, &
                                             de_lgth
      integer, dimension(:), allocatable :: icsdlw
      real(kind=kind_phys), dimension(:), allocatable :: si

      type(topflw_type), dimension(:), allocatable :: topflw
      type(sfcflw_type), dimension(:), allocatable :: sfcflw

      real(kind=kind_phys), dimension(:), allocatable :: upfxc_t_ref, upfx0_t_ref, upfxc_s_ref, upfx0_s_ref

      integer, parameter :: me = 0
      integer :: ictm, isol, ico2, iaer, num_p2d,    &
     &       ntcw, ialb, iems, num_p3d, npdf3d, ntoz, iovr_sw, iovr_lw, &
     &       isubc_sw, isubc_lw, icliq_sw, iflip, idate(4), jdate(4)
      logical :: crick_proof, ccnorm, norad_precip

      integer:: idat(8), jdat(8)
      logical :: lsswr

      real (kind=kind_phys) :: fhswr, dtf

!  ---  outputs:
      real (kind=kind_phys) :: slag, sdec, cdec, solcon

      !$ser init directory='./dump' prefix='Init' directory_ref='../data/LW' prefix_ref='Generator' mpi_rank=0
      !$ser mode read
      !$ser savepoint "lwrad-in-000000"
      !$ser data im=im levr=levr 

      write(*,*) 'im = ', im
      
      allocate(plyr(im, levr), tlyr(im, levr), qlyr(im, levr), olyr(im, levr), dz(im, levr), &
               delp(im, levr), htlwc(im, levr), htlw0(im, levr), htlwc_ref(im, levr))
      allocate(plvl(im, levr+1), tlvl(im, levr+1))
      allocate(gasvmr(im, levr, NF_VGAS))
      allocate(clouds(im, levr, NF_CLDS))
      allocate(faerlw(im, levr, NBDLW, NF_AELW))
      allocate(semis(im), tsfg(im), de_lgth(im))
      allocate(cldtaulw(im, levr), cldtaulw_ref(im, levr))
      allocate(icsdlw(im), topflw(im), sfcflw(im), upfxc_t_ref(im), upfx0_t_ref(im), upfxc_s_ref(im), upfx0_s_ref(im))
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

      call rad_initialize(si,  levr,         ictm,    isol,      &
      ico2,        iaer,         ialb,    iems,      &
      ntcw,        num_p2d,      num_p3d, npdf3d,    &
      ntoz,        iovr_sw,      iovr_lw, isubc_sw,  &
      isubc_lw,    icliq_sw,     crick_proof, ccnorm,&
      imp_physics, norad_precip, idate,   iflip,  me)

      !$ser mode read
      !$ser savepoint "rad-update"
      !$ser data idat=idat jdat=jdat fhswr=fhswr dtf=dtf lsswr=lsswr
      !$ser data slag=slag sdec=sdec cdec=cdec solcon=solcon
      

      write(*,*) 'Running update'

      call radupdate (ser_count, idat, jdat, fhswr, dtf,  lsswr, &
                      me,   slag, sdec,  cdec, solcon)

!$ser cleanup

      ser_count_max = 6
      num_tiles = 6

      call cdstart(num_tiles * ser_count_max * 2)

      do tile = 0, num_tiles-1

!$ser init directory='./dump' prefix='Serialized' directory_ref='../data/LW' prefix_ref='Generator' mpi_rank=tile

      do ser_count = 0, ser_count_max-1
  
          call cdshow(tile * ser_count_max + ser_count * 2 + iter - 1)
          
          write(ser_count_str, '(i6.6)') ser_count
          write(*,*) 'ser_count = ' // ser_count_str

!$ser mode read
!$ser verbatim write(ser_count_str, '(i6.6)') ser_count
!$ser savepoint "lwrad-in-"//trim(ser_count_str)
!$ser data plyr=plyr plvl=plvl tlyr=tlyr tlvl=tlvl qlyr=qlyr olyr=olyr
!$ser data gasvmr=gasvmr clouds=clouds icsdlw=icsdlw faerlw=faerlw
!$ser data semis=semis tsfg=tsfg dz=dz delp=delp de_lgth=de_lgth
!$ser data im=im lmk=lmk lmp=lmp lprnt=lprnt

          call tic(time1, time2)  
          call lwrad(ser_count, plyr, plvl, tlyr, tlvl, qlyr, olyr, gasvmr,  &        !  ---  inputs
                     clouds, icsdlw, faerlw, semis,   &
                     tsfg, dz, delp, de_lgth,                     &
                     im, lmk, lmp, lprnt,                   &
                     htlwc, topflw, sfcflw, cldtaulw, htlw0)        !  ---  outputs
          call toc(time1, time2)

!$ser mode write           
!$ser savepoint "lwrad-check-"//trim(ser_count_str)
!$ser data htlwc=htlwc upfxc_t=topflw%upfxc upfx0_t=topflw%upfx0
!$ser data upfxc_s=sfcflw%upfxc upfx0_s=sfcflw%upfx0
!$ser data dnfxc_s=sfcflw%dnfxc dnfx0_s=sfcflw%dnfx0
!$ser data cldtaulw=cldtaulw htlw0=htlw0

!$ser mode read
!$ser savepoint "lwrad-out-"//trim(ser_count_str)
! outputs
!$ser data htlwc=htlwc_ref upfxc_t=upfxc_t_ref upfx0_t=upfx0_t_ref
!$ser data upfxc_s=upfxc_s_ref upfx0_s=upfx0_s_ref cldtaulw=cldtaulw_ref

      call check_r_2d(htlwc, htlwc_ref, "htlwc")
      call check_r_1d(topflw%upfxc, upfxc_t_ref, "upfxc_t")
      call check_r_1d(topflw%upfx0, upfx0_t_ref, "upfx0_t")
      call check_r_1d(sfcflw%upfxc, upfxc_s_ref, "upfxc_s")
      call check_r_1d(sfcflw%upfx0, upfx0_s_ref, "upfx0_s")
      call check_r_2d(cldtaulw, cldtaulw_ref, "cldtaulw")

      end do
      
!$ser cleanup

      end do

      call cdstop()

      write(*,*) 'FINISHED!'

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
        write(*,'(a,f9.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
      end subroutine toc

end program test
