program test

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_create_savepoint, &
  fs_read_field, &
  fs_write_field, &
  fs_add_savepoint_metainfo
USE utils_ppser, ONLY:  &
  ppser_set_mode, &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_savepoint, &
  ppser_serializer, &
  ppser_serializer_ref, &
  ppser_intlength, &
  ppser_reallength, &
  ppser_realtype, &
  ppser_zrperturb, &
  ppser_get_mode
#endif

      use m_countdown, only : cdstart, cdshow, cdstop
      use machine, only : kind_phys
      use module_radlw_parameters
      use module_radiation_gases, only : NF_VGAS
      use module_radiation_clouds, only : NF_CLDS
      use module_radiation_aerosols, only : NF_AELW
      use physcons, only : con_g, con_cp, con_avgd, con_amd, &
      con_amw, con_amo3
      use module_radlw_main, only : lwrad

      integer, parameter :: im, lmk, lmp, npts, levr
      
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
  
      real(8) :: time1, time2

#ifdef SERIALIZE
! file: main.F lineno: #79
PRINT *, '>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<'
PRINT *, '>>> WARNING: SERIALIZATION IS ON <<<'
PRINT *, '>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<'

! setup serialization environment
call ppser_initialize( &
           directory='./dump', &
           prefix='Serialized', &
           directory_ref='../data', &
           prefix_ref='Generator', &
           mpi_rank=0)
! file: main.F lineno: #80
call ppser_set_mode(1)
! file: main.F lineno: #81
call fs_create_savepoint("lwrad-in-000000", ppser_savepoint)
! file: main.F lineno: #82
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'im', im)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'levr', levr)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'im', im)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'levr', levr)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'im', im, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'levr', levr, ppser_zrperturb)
END SELECT
#endif

      real(kind=kind_phys), dimension(im, levr) :: plyr, tlyr, qlyr, olyr, &
      dz, delp, htlwc, htlw0
      real(kind=kind_phys), dimension(im, levr+1) :: plvl, tlvl
      real(kind=kind_phys), dimension(im, levr, NF_VGAS) :: gasvmr
      real(kind=kind_phys), dimension(im, levr, NF_CLDS) :: clouds
      real(kind=kind_phys), dimension(im, levr, NBDLW, NF_AELW) :: faerlw
      real(kind=kind_phys), dimension(im, levr) :: cldtaulw
      real(kind=kind_phys), dimension(im) :: semis, tsfg, &
                                             de_lgth
      integer, dimension(im) :: icsdlw

      type(topflw_type), dimension(im) :: topflw
      type(sfcflw_type), dimension(im) :: sfcflw

      ser_count_max = 5
      num_tiles = 6

      call cdstart(num_tiles * ser_count_max * 2)

      do tile = 0, num_tiles - 1

#ifdef SERIALIZE
! file: main.F lineno: #105
PRINT *, '>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<'
PRINT *, '>>> WARNING: SERIALIZATION IS ON <<<'
PRINT *, '>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<'

! setup serialization environment
call ppser_initialize( &
           directory='./dump', &
           prefix='Serialized', &
           directory_ref='../data', &
           prefix_ref='Generator', &
           mpi_rank=tile)
#endif

      do ser_count = 0, ser_count_max
  
          call cdshow(tile * ser_count_max + ser_count * 2 + iter - 1)
          
          write(ser_count_str, '(i6.6)') ser_count

#ifdef SERIALIZE
! file: main.F lineno: #113
call ppser_set_mode(1)
! file: main.F lineno: #114
call fs_create_savepoint("lwrad-in-"//trim(ser_count_str), ppser_savepoint)
! file: main.F lineno: #115
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'plyr', plyr)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'plvl', plvl)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'tlyr', tlyr)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'tlvl', tlvl)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'qlyr', qlyr)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'olyr', olyr)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'plyr', plyr)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'plvl', plvl)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tlyr', tlyr)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tlvl', tlvl)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'qlyr', qlyr)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'olyr', olyr)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'plyr', plyr, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'plvl', plvl, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tlyr', tlyr, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tlvl', tlvl, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'qlyr', qlyr, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'olyr', olyr, ppser_zrperturb)
END SELECT
! file: main.F lineno: #116
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'gasvmr', gasvmr)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'clouds', clouds)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'icsdlw', icsdlw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'faerlw', faerlw)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'gasvmr', gasvmr)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'clouds', clouds)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'icsdlw', icsdlw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'faerlw', faerlw)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'gasvmr', gasvmr, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'clouds', clouds, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'icsdlw', icsdlw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'faerlw', faerlw, ppser_zrperturb)
END SELECT
! file: main.F lineno: #117
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'semis', semis)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'tsfg', tsfg)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'dz', dz)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'delp', delp)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'de_lgth', de_lgth)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'semis', semis)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tsfg', tsfg)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dz', dz)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'delp', delp)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'de_lgth', de_lgth)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'semis', semis, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'tsfg', tsfg, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dz', dz, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'delp', delp, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'de_lgth', de_lgth, ppser_zrperturb)
END SELECT
#endif
!     $ser data im=im lmk=lmk lmp=lmp lprnt=lprnt

          call lwrad(plyr, plvl, tlyr, tlvl, qlyr, olyr, gasvmr,  &        !  ---  inputs
                     clouds, icsdlw, faerlw, semis,   &
                     tsfg, dz, delp, de_lgth,                     &
                     im, lmk, lmp, lprnt,                   &
                     htlwc, topflw, sfcflw, cldtaulw,&        !  ---  outputs
                     hlw0=htlw0)
      end do
  
#ifdef SERIALIZE
! file: main.F lineno: #128
! cleanup serialization environment
call ppser_finalize()
#endif

      end do

      call cdstop()

      write(*,*) 'FINISHED!'

contains

      subroutine check_r_2d(a, b, name, atol, rtol)

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_create_savepoint, &
  fs_read_field, &
  fs_write_field, &
  fs_add_savepoint_metainfo
USE utils_ppser, ONLY:  &
  ppser_set_mode, &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_savepoint, &
  ppser_serializer, &
  ppser_serializer_ref, &
  ppser_intlength, &
  ppser_reallength, &
  ppser_realtype, &
  ppser_zrperturb, &
  ppser_get_mode
#endif

        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

      end subroutine check_r_2d

      subroutine check_r_1d(a, b, name, atol, rtol)

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_create_savepoint, &
  fs_read_field, &
  fs_write_field, &
  fs_add_savepoint_metainfo
USE utils_ppser, ONLY:  &
  ppser_set_mode, &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_savepoint, &
  ppser_serializer, &
  ppser_serializer_ref, &
  ppser_intlength, &
  ppser_reallength, &
  ppser_realtype, &
  ppser_zrperturb, &
  ppser_get_mode
#endif

        implicit none
        real (kind=kind_phys), intent(in) :: a(:), b(:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
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

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_create_savepoint, &
  fs_read_field, &
  fs_write_field, &
  fs_add_savepoint_metainfo
USE utils_ppser, ONLY:  &
  ppser_set_mode, &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_savepoint, &
  ppser_serializer, &
  ppser_serializer_ref, &
  ppser_intlength, &
  ppser_reallength, &
  ppser_realtype, &
  ppser_zrperturb, &
  ppser_get_mode
#endif

        implicit none
        real(8) :: t1, t2
        call cpu_time(t1)
      end subroutine tic

      subroutine toc(t1, t2)

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_create_savepoint, &
  fs_read_field, &
  fs_write_field, &
  fs_add_savepoint_metainfo
USE utils_ppser, ONLY:  &
  ppser_set_mode, &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_savepoint, &
  ppser_serializer, &
  ppser_serializer_ref, &
  ppser_intlength, &
  ppser_reallength, &
  ppser_realtype, &
  ppser_zrperturb, &
  ppser_get_mode
#endif

        implicit none
        real(8) :: t1, t2
        call cpu_time(t2)
        write(*,'(a,f5.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
      end subroutine toc

end program test
