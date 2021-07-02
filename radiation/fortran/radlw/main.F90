program test

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_read_field, &
  fs_add_savepoint_metainfo, &
  fs_write_field, &
  fs_create_savepoint
USE utils_ppser, ONLY:  &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_set_mode, &
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

      use module_radiation_driver,  only: radupdate

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
     &       isubc_sw, isubc_lw, icliq_sw, iflip, idate(4)
      logical :: crick_proof, ccnorm, norad_precip

      integer:: idat(8), jdat(8)
      logical :: lsswr

      real (kind=kind_phys) :: deltsw, deltim

!  ---  outputs:
      real (kind=kind_phys) :: slag, sdec, cdec, solcon


#ifdef SERIALIZE
! file: main.F lineno: #113
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
! file: main.F lineno: #114
call ppser_set_mode(1)
! file: main.F lineno: #115
call fs_create_savepoint("lwrad-in-000000", ppser_savepoint)
! file: main.F lineno: #116
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

#ifdef SERIALIZE
! file: main.F lineno: #131
call ppser_set_mode(1)
! file: main.F lineno: #132
call fs_create_savepoint("rad-initialize", ppser_savepoint)
! file: main.F lineno: #133
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'si', si)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ictm', ictm)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'isol', isol)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'imp_physics', imp_physics)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'si', si)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ictm', ictm)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isol', isol)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'imp_physics', imp_physics)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'si', si, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ictm', ictm, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isol', isol, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'imp_physics', imp_physics, ppser_zrperturb)
END SELECT
! file: main.F lineno: #134
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ico2', ico2)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'iaer', iaer)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ialb', ialb)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'iems', iems)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ico2', ico2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iaer', iaer)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ialb', ialb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iems', iems)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ico2', ico2, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iaer', iaer, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ialb', ialb, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iems', iems, ppser_zrperturb)
END SELECT
! file: main.F lineno: #135
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ntcw', ntcw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'num_p2d', num_p2d)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'num_p3d', num_p3d)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'npdf3d', npdf3d)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ntcw', ntcw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'num_p2d', num_p2d)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'num_p3d', num_p3d)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'npdf3d', npdf3d)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ntcw', ntcw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'num_p2d', num_p2d, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'num_p3d', num_p3d, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'npdf3d', npdf3d, ppser_zrperturb)
END SELECT
! file: main.F lineno: #136
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ntoz', ntoz)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'iovr_sw', iovr_sw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'iovr_lw', iovr_lw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'isubc_sw', isubc_sw)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ntoz', ntoz)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iovr_sw', iovr_sw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iovr_lw', iovr_lw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isubc_sw', isubc_sw)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ntoz', ntoz, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iovr_sw', iovr_sw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iovr_lw', iovr_lw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isubc_sw', isubc_sw, ppser_zrperturb)
END SELECT
! file: main.F lineno: #137
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'isubc_lw', isubc_lw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'icliq_sw', icliq_sw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'crick_proof', crick_proof)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isubc_lw', isubc_lw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'icliq_sw', icliq_sw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'crick_proof', crick_proof)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'isubc_lw', isubc_lw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'icliq_sw', icliq_sw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'crick_proof', crick_proof, ppser_zrperturb)
END SELECT
! file: main.F lineno: #138
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'ccnorm', ccnorm)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'norad_precip', norad_precip)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'idate', idate)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'iflip', iflip)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ccnorm', ccnorm)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'norad_precip', norad_precip)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'idate', idate)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iflip', iflip)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'ccnorm', ccnorm, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'norad_precip', norad_precip, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'idate', idate, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'iflip', iflip, ppser_zrperturb)
END SELECT
#endif
      ! Initialization

      call rad_initialize(si,  levr,         ictm,    isol,      &
                          ico2,        iaer,         ialb,    iems,      &
                          ntcw,        num_p2d,      num_p3d, npdf3d,    &
                          ntoz,        iovr_sw,      iovr_lw, isubc_sw,  &
                          isubc_lw,    icliq_sw,     crick_proof, ccnorm,&
                          imp_physics, norad_precip, idate,   iflip,  me)

#ifdef SERIALIZE
! file: main.F lineno: #148
call ppser_set_mode(1)
! file: main.F lineno: #149
call fs_create_savepoint("rad-update", ppser_savepoint)
! file: main.F lineno: #150
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'idat', idat)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'jdat', jdat)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'fhswr', deltsw)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'dtf', deltim)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'idat', idat)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'jdat', jdat)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'fhswr', deltsw)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dtf', deltim)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'idat', idat, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'jdat', jdat, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'fhswr', deltsw, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dtf', deltim, ppser_zrperturb)
END SELECT
! file: main.F lineno: #151
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'lsswr', lsswr)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'slag', slag)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'sdec', sdec)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'cdec', cdec)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lsswr', lsswr)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'slag', slag)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'sdec', sdec)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cdec', cdec)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lsswr', lsswr, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'slag', slag, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'sdec', sdec, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cdec', cdec, ppser_zrperturb)
END SELECT
! file: main.F lineno: #152
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'solcon', solcon)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'solcon', solcon)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'solcon', solcon, ppser_zrperturb)
END SELECT
#endif
      
      !call radupdate(idat,jdat,deltsw,deltim,lsswr, me,        &
      !               slag,sdec,cdec,solcon)

      ser_count_max = 6
      num_tiles = 0

      call cdstart(num_tiles * ser_count_max * 2)

      do tile = 0, 0

#ifdef SERIALIZE
! file: main.F lineno: #164
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

      do ser_count = 0, ser_count_max-1
  
          call cdshow(tile * ser_count_max + ser_count * 2 + iter - 1)
          
          write(ser_count_str, '(i6.6)') ser_count
          write(*,*) 'ser_count = ' // ser_count_str

#ifdef SERIALIZE
! file: main.F lineno: #173
call ppser_set_mode(1)
write(ser_count_str, '(i6.6)') ser_count
! file: main.F lineno: #175
call fs_create_savepoint("lwrad-in-"//trim(ser_count_str), ppser_savepoint)
! file: main.F lineno: #176
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
! file: main.F lineno: #177
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
! file: main.F lineno: #178
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
! file: main.F lineno: #179
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'im', im)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'lmk', lmk)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'lmp', lmp)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'lprnt', lprnt)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'im', im)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lmk', lmk)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lmp', lmp)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lprnt', lprnt)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'im', im, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lmk', lmk, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lmp', lmp, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'lprnt', lprnt, ppser_zrperturb)
END SELECT
#endif

          call lwrad(plyr, plvl, tlyr, tlvl, qlyr, olyr, gasvmr,  &        !  ---  inputs
                     clouds, icsdlw, faerlw, semis,   &
                     tsfg, dz, delp, de_lgth,                     &
                     im, lmk, lmp, lprnt,                   &
                     htlwc, topflw, sfcflw, cldtaulw)        !  ---  outputs

#ifdef SERIALIZE
! file: main.F lineno: #187
call ppser_set_mode(0)
! file: main.F lineno: #188
call fs_create_savepoint("lwrad-check-"//trim(ser_count_str), ppser_savepoint)
! file: main.F lineno: #189
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'htlwc', htlwc)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfxc_t', topflw%upfxc)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfx0_t', topflw%upfx0)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'htlwc', htlwc)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_t', topflw%upfxc)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_t', topflw%upfx0)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'htlwc', htlwc, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_t', topflw%upfxc, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_t', topflw%upfx0, ppser_zrperturb)
END SELECT
! file: main.F lineno: #190
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfxc_s', sfcflw%upfxc)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfx0_s', sfcflw%upfx0)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_s', sfcflw%upfxc)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_s', sfcflw%upfx0)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_s', sfcflw%upfxc, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_s', sfcflw%upfx0, ppser_zrperturb)
END SELECT
! file: main.F lineno: #191
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'dnfxc_s', sfcflw%dnfxc)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'dnfx0_s', sfcflw%dnfx0)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dnfxc_s', sfcflw%dnfxc)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dnfx0_s', sfcflw%dnfx0)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dnfxc_s', sfcflw%dnfxc, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'dnfx0_s', sfcflw%dnfx0, ppser_zrperturb)
END SELECT
! file: main.F lineno: #192
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'cldtaulw', cldtaulw)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cldtaulw', cldtaulw)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cldtaulw', cldtaulw, ppser_zrperturb)
END SELECT
#endif

#ifdef SERIALIZE
! file: main.F lineno: #194
call ppser_set_mode(1)
! file: main.F lineno: #195
call fs_create_savepoint("lwrad-out-"//trim(ser_count_str), ppser_savepoint)
#endif
! outputs
#ifdef SERIALIZE
! file: main.F lineno: #197
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'htlwc', htlwc_ref)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfxc_t', upfxc_t_ref)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfx0_t', upfx0_t_ref)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'htlwc', htlwc_ref)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_t', upfxc_t_ref)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_t', upfx0_t_ref)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'htlwc', htlwc_ref, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_t', upfxc_t_ref, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_t', upfx0_t_ref, ppser_zrperturb)
END SELECT
! file: main.F lineno: #198
SELECT CASE ( ppser_get_mode() )
  CASE(0)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfxc_s', upfxc_s_ref)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'upfx0_s', upfx0_s_ref)
    call fs_write_field(ppser_serializer, ppser_savepoint, 'cldtaulw', cldtaulw_ref)
  CASE(1)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_s', upfxc_s_ref)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_s', upfx0_s_ref)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cldtaulw', cldtaulw_ref)
  CASE(2)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfxc_s', upfxc_s_ref, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'upfx0_s', upfx0_s_ref, ppser_zrperturb)
    call fs_read_field(ppser_serializer_ref, ppser_savepoint, 'cldtaulw', cldtaulw_ref, ppser_zrperturb)
END SELECT
#endif

      call check_r_2d(htlwc, htlwc_ref, "htlwc")
      call check_r_1d(topflw%upfxc, upfxc_t_ref, "upfxc_t")
      call check_r_1d(topflw%upfx0, upfx0_t_ref, "upfx0_t")
      call check_r_1d(sfcflw%upfxc, upfxc_s_ref, "upfxc_s")
      call check_r_1d(sfcflw%upfx0, upfx0_s_ref, "upfx0_s")
      call check_r_2d(cldtaulw, cldtaulw_ref, "cldtaulw")
      end do
#ifdef SERIALIZE
! file: main.F lineno: #207
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
  fs_read_field, &
  fs_add_savepoint_metainfo, &
  fs_write_field, &
  fs_create_savepoint
USE utils_ppser, ONLY:  &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_set_mode, &
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
        else
            write(*,*) trim(name) // ' validates!'
        end if

      end subroutine check_r_2d

      subroutine check_r_1d(a, b, name, atol, rtol)

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_read_field, &
  fs_add_savepoint_metainfo, &
  fs_write_field, &
  fs_create_savepoint
USE utils_ppser, ONLY:  &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_set_mode, &
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

#ifdef SERIALIZE
USE m_serialize, ONLY: &
  fs_read_field, &
  fs_add_savepoint_metainfo, &
  fs_write_field, &
  fs_create_savepoint
USE utils_ppser, ONLY:  &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_set_mode, &
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
  fs_read_field, &
  fs_add_savepoint_metainfo, &
  fs_write_field, &
  fs_create_savepoint
USE utils_ppser, ONLY:  &
  ppser_finalize, &
  ppser_initialize, &
  ppser_get_mode, &
  ppser_set_mode, &
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
