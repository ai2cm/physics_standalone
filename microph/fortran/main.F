program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use gfdl_cloud_microphys_mod, only : gfdl_cloud_microphys_init, &
        gfdl_cloud_microphys_driver

#ifdef USE_MPIF90
    use mpi
    implicit none
#else
    implicit none
    include "mpif.h"
#endif

    integer :: iie, kke, kbot, seconds, rep_idx
    character(len=10) :: file_id
    real (kind=kind_phys) :: dt_in
    real (kind=kind_phys), dimension(:, :), allocatable :: area, land, &
        rain, snow, ice, graupel, rain_ref, snow_ref, ice_ref, graupel_ref
    real (kind=kind_phys), dimension(:, :, :), allocatable :: delp, dz, &
        uin, vin, p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, &
        udt, vdt, w, qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, &
        qi_ref, qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
        qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
        refl_10cm_ref
    
    real (kind=kind_phys), dimension(:, :, :), allocatable :: qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
                                                              pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
                                                              qi_s, qs_s, &
                                                              qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
                                                              pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s
    real (kind=kind_phys), dimension(:, :), allocatable :: area_s, land_s, &
                                                           rain_s, snow_s, ice_s, graupel_s
    logical :: lradar, reset
    integer :: ierror, n_ranks, rank_idx
    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max, dsize_per_rank
    character(len=100) :: nml_file
    character(len=6) :: ser_count_str
    real :: C_orig, CN
    real :: timings(10, 1)
    real :: t_driver_start, t_driver_end, &
            t_tot_start, t_tot_end
    logical :: VALIDATION=.false.
    call MPI_INIT(ierror)
    
    call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank_idx, ierror)
    
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) call cpu_time(t_tot_start)

    timings(:,:) = 0.
    CN = real(CNINPUT)
    dsize_per_rank = ceiling((CN*CN)/n_ranks)
    nunit = 42
    nml_file = 'input.nml'
    ser_count_max = 10
    num_tiles = 6

    ! call cdstart(num_tiles * ser_count_max)

    call gfdl_cloud_microphys_init(0, 0, nunit, (/''/), 6, nml_file)

    do tile = 0, 0 ! num_tiles - 1
    
    !$ser init directory='../data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, 0 !ser_count_max
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser savepoint "cloud_mp-in-"//trim(ser_count_str)
        !$ser data iie=iie kke=kke kbot=kbot

        allocate(area(iie, 1), land(iie, 1), rain(iie, 1), snow(iie, 1), &
            ice(iie, 1), graupel(iie, 1), delp(iie, 1, kke), dz(iie, 1, kke), &
            uin(iie, 1, kke), vin(iie, 1, kke), p(iie, 1, kke), pt(iie, 1, kke), &
            qv(iie, 1, kke), ql(iie, 1, kke), qr(iie, 1, kke), qg(iie, 1, kke), &
            qa(iie, 1, kke), qn(iie, 1, kke), qi(iie, 1, kke), qs(iie, 1, kke), &
            pt_dt(iie, 1, kke), qa_dt(iie, 1, kke), udt(iie, 1, kke), vdt(iie, 1, kke), &
            w(iie, 1, kke), qv_dt(iie, 1, kke), ql_dt(iie, 1, kke), qr_dt(iie, 1, kke), &
            qi_dt(iie, 1, kke), qs_dt(iie, 1, kke), qg_dt(iie, 1, kke), refl_10cm(iie, 1, kke), &
            qi_ref(iie, 1, kke), qs_ref(iie, 1, kke), qv_dt_ref(iie, 1, kke), &
            ql_dt_ref(iie, 1, kke), qr_dt_ref(iie, 1, kke), qi_dt_ref(iie, 1, kke), &
            qs_dt_ref(iie, 1, kke), qg_dt_ref(iie, 1, kke), qa_dt_ref(iie, 1, kke), &
            pt_dt_ref(iie, 1, kke), w_ref(iie, 1, kke), udt_ref(iie, 1, kke), vdt_ref(iie, 1, kke), &
            rain_ref(iie, 1), snow_ref(iie, 1), ice_ref(iie, 1), graupel_ref(iie, 1), &
            refl_10cm_ref(iie, 1, kke))

        ! input
        !$ser data qv=qv ql=ql qr=qr qg=qg qa=qa qn=qn
        !$ser data pt=pt uin=uin vin=vin dz=dz delp=delp
        !$ser data area=area dt_in=dt_in land=land
        !$ser data seconds=seconds p=p lradar=lradar
        !$ser data reset=reset
        ! inout
        !$ser data qi=qi qs=qs qv_dt=qv_dt
        !$ser data ql_dt=ql_dt qr_dt=qr_dt qi_dt=qi_dt qs_dt=qs_dt qg_dt=qg_dt qa_dt=qa_dt
        !$ser data pt_dt=pt_dt w=w udt=udt vdt=vdt
        !$ser data rain=rain snow=snow ice=ice graupel=graupel
        !$ser data refl_10cm=refl_10cm
        allocate( qv_s(dsize_per_rank, 1, kke), ql_s(dsize_per_rank, 1, kke), qr_s(dsize_per_rank, 1, kke), qg_s(dsize_per_rank, 1, kke), qa_s(dsize_per_rank, 1, kke), qn_s(dsize_per_rank, 1, kke), &
                      pt_s(dsize_per_rank, 1, kke), uin_s(dsize_per_rank, 1, kke), vin_s(dsize_per_rank, 1, kke), dz_s(dsize_per_rank, 1, kke), delp_s(dsize_per_rank, 1, kke), p_s(dsize_per_rank, 1, kke), &
                      area_s(dsize_per_rank, 1), land_s(dsize_per_rank, 1), &
                      qi_s(dsize_per_rank, 1, kke), qs_s(dsize_per_rank, 1, kke), &
                      qv_dt_s(dsize_per_rank, 1, kke), ql_dt_s(dsize_per_rank, 1, kke), qr_dt_s(dsize_per_rank, 1, kke), qi_dt_s(dsize_per_rank, 1, kke), qs_dt_s(dsize_per_rank, 1, kke), qg_dt_s(dsize_per_rank, 1, kke), qa_dt_s(dsize_per_rank, 1, kke), &
                      pt_dt_s(dsize_per_rank, 1, kke), w_s(dsize_per_rank, 1, kke), udt_s(dsize_per_rank, 1, kke), vdt_s(dsize_per_rank, 1, kke), refl_10cm_s(dsize_per_rank, 1, kke), &
                      rain_s(dsize_per_rank, 1), snow_s(dsize_per_rank, 1), ice_s(dsize_per_rank, 1), graupel_s(dsize_per_rank, 1) )
        
        call scale_dataset_to_N(CN, dsize_per_rank, kke, iie, &
                      qv, ql, qr, qg, qa, qn, &
                      pt, uin, vin, dz, delp, p, &
                      area, land, qi, qs, &
                      qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, qa_dt, &
                      pt_dt, w, udt, vdt, refl_10cm, &
                      rain, snow, ice, graupel, &
                      qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
                      pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
                      area_s, land_s, qi_s, qs_s, &
                      qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
                      pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s, &
                      rain_s, snow_s, ice_s, graupel_s)
        
        do rep_idx=1, REPS+1
            if (rep_idx .gt. 1) then
                call MPI_BARRIER(MPI_COMM_WORLD, ierror) 
                call cpu_time(t_driver_start)
            endif
            call gfdl_cloud_microphys_driver (qv_s, ql_s, qr_s, qi_s, qs_s, qg_s, qa_s, qn_s,     &
                    qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, pt_dt_s, pt_s, w_s,    &
                    uin_s, vin_s, udt_s, vdt_s, dz_s, delp_s, area_s, dt_in, land_s, rain_s, snow_s, ice_s, &
                    graupel_s, .false., .true., 1, dsize_per_rank, 1, 1, 1,                        &
                    kke, 1, kbot, seconds, p_s, lradar, refl_10cm_s, reset, (rep_idx .gt. 1), timings, 1)
            if (rep_idx .gt. 1) then
                call cpu_time(t_driver_end)
                call MPI_BARRIER(MPI_COMM_WORLD, ierror)
                timings(6, 1) = timings(6, 1) + (t_driver_end - t_driver_start)
            endif
        enddo

    !     !$ser mode read
    !     !$ser savepoint "cloud_mp-out-"//trim(ser_count_str)
    !     ! inout
    !     !$ser data qi=qi_ref qs=qs_ref qv_dt=qv_dt_ref
    !     !$ser data ql_dt=ql_dt_ref qr_dt=qr_dt_ref qi_dt=qi_dt_ref qs_dt=qs_dt_ref qg_dt=qg_dt_ref qa_dt=qa_dt_ref
    !     !$ser data pt_dt=pt_dt_ref w=w_ref udt=udt_ref vdt=vdt_ref
    !     !$ser data rain=rain_ref snow=snow_ref ice=ice_ref graupel=graupel_ref
    !     !$ser data refl_10cm=refl_10cm_ref
    if (VALIDATION) then
        call check_r_3d(qi, qi_ref, "qi")
        call check_r_3d(qs, qs_ref, "qs")
        call check_r_3d(qv_dt, qv_dt_ref, "qv_dt")
        call check_r_3d(ql_dt, ql_dt_ref, "ql_dt")
        call check_r_3d(qr_dt, qr_dt_ref, "qr_dt")
        call check_r_3d(qi_dt, qi_dt_ref, "qi_dt")
        call check_r_3d(qs_dt, qs_dt_ref, "qs_dt")
        call check_r_3d(qg_dt, qg_dt_ref, "qg_dt")
        call check_r_3d(qg_dt, qg_dt_ref, "qg_dt")
        call check_r_3d(qa_dt, qa_dt_ref, "qa_dt")
        call check_r_3d(pt_dt, pt_dt_ref, "pt_dt")
        call check_r_3d(w, w_ref, "w")
        call check_r_3d(udt, udt_ref, "udt")
        call check_r_3d(vdt, vdt_ref, "vdt")
        call check_r_2d(rain, rain_ref, "rain")
        call check_r_2d(snow, snow_ref, "snow")
        call check_r_2d(ice, ice_ref, "ice")
        call check_r_2d(graupel, graupel_ref, "graupel")
        call check_r_3d(refl_10cm, refl_10cm_ref, "refl_10cm")
    endif

        deallocate(area, land, rain, snow, ice, graupel, delp, dz, uin, vin, &
            p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, udt, vdt, w, &
            qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, qi_ref, &
            qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
            qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
            rain_ref, snow_ref, ice_ref, graupel_ref, &
            refl_10cm_ref, &
            qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
            pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
            area_s, land_s, &
            qi_s, qs_s, &
            qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
            pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s, &
            rain_s, snow_s, ice_s, graupel_s)

    end do

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    !$ser cleanup

    end do

    if (BENCHMARK) call MPI_REDUCE(timings(1:6, :), timings(1:6, :), 6*factors_size, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
    if (BENCHMARK) call MPI_REDUCE(timings(8:10, :), timings(8:10, :), 3*factors_size, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) then
        
        call cpu_time(t_tot_end)
        timings(7, 1) = t_tot_end - t_tot_start
        write(*,*) 'FINISHED!'
        write(*,*) CN, dsize_per_rank
        write(file_id, '(i0)') int(CN)
        open(1, file='./out/timings_benchmark_fortran_normal_C'//trim(adjustl(file_id))//'.dat')
        timings(1:6, :) = timings(1:6, :) / REPS
        timings(8:10, :) = timings(8:10, :) / REPS
        timings(5, 1) = timings(1, 1) + timings(2, 1) + timings(3, 1) + timings(4, 1)
        write(*,*) ""
        write(*,'(a, I5, a, I2, a)') "For gridsize ", INT(CN*CN), "x1x", kke, ":"
        write(*,'(a, F7.3, a)') "> Warm rain processes (1st pass) => ", timings(1, 1), " seconds"
        write(*,'(a, F7.3, a)') "> Sedimentation                  => ", timings(2, 1), " seconds"
        write(*,'(a, F7.3, a)') "> Warm rain processes (2nd pass) => ", timings(3, 1), " seconds"
        write(*,'(a, F7.3, a)') "> Ice-phase microphysics         => ", timings(4, 1), " seconds"
        write(*,'(a, F7.3, a)') "> Main loop                      => ", timings(5, 1), " seconds"
        write(*,'(a, F7.3, a)') "> Whole driver                   => ", timings(6, 1), " seconds"
        write(1, *) timings(1, 1), 0., timings(2, 1), &
                            timings(3, 1), timings(4, 1), &
                            timings(5, 1), timings(6, 1), &
                            timings(8, 1), timings(9, 1), timings(10, 1)
        close(1)
        write(*,*) ""
        write(*,'(a, F7.3, a)') ">> Total elapsed time: ", timings(7, 1), " seconds"
    endif

    call MPI_FINALIZE(ierror)

contains

    subroutine check_r_3d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:,:), b(:,:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_3d

    subroutine check_r_2d(a, b, name, atol, rtol)
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
            atol_local = 1.0d-13
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

    subroutine scale_dataset_to_N(CN, dsize_per_rank, kke, iie, &
                      qv, ql, qr, qg, qa, qn, &
                      pt, uin, vin, dz, delp, p, &
                      area, land, qi, qs, &
                      qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, qa_dt, &
                      pt_dt, w, udt, vdt, refl_10cm, &
                      rain, snow, ice, graupel, &
                      qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
                      pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
                      area_s, land_s, qi_s, qs_s, &
                      qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
                      pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s, &
                      rain_s, snow_s, ice_s, graupel_s)
        implicit none
        real (kind=kind_phys), intent(in) :: qv(:,:,:), ql(:,:,:), qr(:,:,:), qg(:,:,:), qa(:,:,:), qn(:,:,:), &
                      pt(:,:,:), uin(:,:,:), vin(:,:,:), dz(:,:,:), delp(:,:,:), p(:,:,:), &
                      area(:,:), land(:,:), qi(:,:,:), qs(:,:,:), &
                      qv_dt(:,:,:), ql_dt(:,:,:), qr_dt(:,:,:), qi_dt(:,:,:), qs_dt(:,:,:), qg_dt(:,:,:), qa_dt(:,:,:), &
                      pt_dt(:,:,:), w(:,:,:), udt(:,:,:), vdt(:,:,:), refl_10cm(:,:,:), &
                      rain(:,:), snow(:,:), ice(:,:), graupel(:,:)

        real (kind=kind_phys), dimension(:, :, :), allocatable :: qv_temp, ql_temp, qr_temp, qg_temp, qa_temp, qn_temp, &
                      pt_temp, uin_temp, vin_temp, dz_temp, delp_temp, p_temp, &
                      qi_temp, qs_temp, &
                      qv_dt_temp, ql_dt_temp, qr_dt_temp, qi_dt_temp, qs_dt_temp, qg_dt_temp, qa_dt_temp, &
                      pt_dt_temp, w_temp, udt_temp, vdt_temp, refl_10cm_temp
                      

        real (kind=kind_phys), dimension(:, :), allocatable :: area_temp, land_temp, rain_temp, snow_temp, ice_temp, graupel_temp

        real (kind=kind_phys), intent(inout) :: qv_s(:,:,:), ql_s(:,:,:), qr_s(:,:,:), qg_s(:,:,:), qa_s(:,:,:), qn_s(:,:,:), &
                      pt_s(:,:,:), uin_s(:,:,:), vin_s(:,:,:), dz_s(:,:,:), delp_s(:,:,:), p_s(:,:,:), &
                      area_s(:,:), land_s(:,:), qi_s(:,:,:), qs_s(:,:,:), &
                      qv_dt_s(:,:,:), ql_dt_s(:,:,:), qr_dt_s(:,:,:), qi_dt_s(:,:,:), qs_dt_s(:,:,:), qg_dt_s(:,:,:), qa_dt_s(:,:,:), &
                      pt_dt_s(:,:,:), w_s(:,:,:), udt_s(:,:,:), vdt_s(:,:,:), refl_10cm_s(:,:,:), &
                      rain_s(:,:), snow_s(:,:), ice_s(:,:), graupel_s(:,:)
        real, intent(in) :: CN
        real :: C_orig
        integer, intent(in):: kke, iie, dsize_per_rank
        integer :: multiplier, part, temp
        C_orig = real(iie)
        multiplier = ceiling(dsize_per_rank/C_orig)
        temp = multiplier*iie
        allocate(qv_temp(temp,1,kke), ql_temp(temp,1,kke), qr_temp(temp,1,kke), qg_temp(temp,1,kke), qa_temp(temp,1,kke), qn_temp(temp,1,kke), &
                      pt_temp(temp,1,kke), uin_temp(temp,1,kke), vin_temp(temp,1,kke), dz_temp(temp,1,kke), delp_temp(temp,1,kke), p_temp(temp,1,kke), &
                      area_temp(temp,1), land_temp(temp,1), qi_temp(temp,1,kke), qs_temp(temp,1,kke), &
                      qv_dt_temp(temp,1,kke), ql_dt_temp(temp,1,kke), qr_dt_temp(temp,1,kke), qi_dt_temp(temp,1,kke), qs_dt_temp(temp,1,kke), qg_dt_temp(temp,1,kke), qa_dt_temp(temp,1,kke), &
                      pt_dt_temp(temp,1,kke), w_temp(temp,1,kke), udt_temp(temp,1,kke), vdt_temp(temp,1,kke), refl_10cm_temp(temp,1,kke), &
                      rain_temp(temp,1), snow_temp(temp,1), ice_temp(temp,1), graupel_temp(temp,1))
        do part=1, multiplier
            qv_temp((part-1)*iie+1:part*iie,:,:) = qv(1:iie,:,:)
            ql_temp((part-1)*iie+1:part*iie,:,:) = ql(1:iie,:,:)
            qr_temp((part-1)*iie+1:part*iie,:,:) = qr(1:iie,:,:)
            qg_temp((part-1)*iie+1:part*iie,:,:) = qg(1:iie,:,:)
            qa_temp((part-1)*iie+1:part*iie,:,:) = qa(1:iie,:,:)
            qn_temp((part-1)*iie+1:part*iie,:,:) = qn(1:iie,:,:)
            pt_temp((part-1)*iie+1:part*iie,:,:) = pt(1:iie,:,:)
            uin_temp((part-1)*iie+1:part*iie,:,:) = uin(1:iie,:,:)
            vin_temp((part-1)*iie+1:part*iie,:,:) = vin(1:iie,:,:)
            dz_temp((part-1)*iie+1:part*iie,:,:) = dz(1:iie,:,:)
            delp_temp((part-1)*iie+1:part*iie,:,:) = delp(1:iie,:,:)
            p_temp((part-1)*iie+1:part*iie,:,:) = p(1:iie,:,:)
            area_temp((part-1)*iie+1:part*iie,:) = area(1:iie,:)
            land_temp((part-1)*iie+1:part*iie,:) = land(1:iie,:)
            qi_temp((part-1)*iie+1:part*iie,:,:) = qi(1:iie,:,:)
            qs_temp((part-1)*iie+1:part*iie,:,:) = qs(1:iie,:,:)
            qv_dt_temp((part-1)*iie+1:part*iie,:,:) = qv_dt(1:iie,:,:) 
            ql_dt_temp((part-1)*iie+1:part*iie,:,:) = ql_dt(1:iie,:,:)
            qr_dt_temp((part-1)*iie+1:part*iie,:,:) = qr_dt(1:iie,:,:)
            qi_dt_temp((part-1)*iie+1:part*iie,:,:) = qi_dt(1:iie,:,:)
            qs_dt_temp((part-1)*iie+1:part*iie,:,:) = qs_dt(1:iie,:,:)
            qg_dt_temp((part-1)*iie+1:part*iie,:,:) = qg_dt(1:iie,:,:)
            qa_dt_temp((part-1)*iie+1:part*iie,:,:) = qa_dt(1:iie,:,:)
            pt_dt_temp((part-1)*iie+1:part*iie,:,:) = pt_dt(1:iie,:,:)
            w_temp((part-1)*iie+1:part*iie,:,:) = w(1:iie,:,:)
            udt_temp((part-1)*iie+1:part*iie,:,:) = udt(1:iie,:,:) 
            vdt_temp((part-1)*iie+1:part*iie,:,:) = vdt(1:iie,:,:)
            refl_10cm_temp((part-1)*iie+1:part*iie,:,:) = refl_10cm(1:iie,:,:)
            rain_temp((part-1)*iie+1:part*iie,:) = rain(1:iie,:)
            snow_temp((part-1)*iie+1:part*iie,:) = snow(1:iie,:)
            ice_temp((part-1)*iie+1:part*iie,:) = ice(1:iie,:)
            graupel_temp((part-1)*iie+1:part*iie,:) = graupel(1:iie,:)
        enddo

        qv_s(1:dsize_per_rank,:,:) = qv_temp(1:dsize_per_rank,:,:)
        ql_s(1:dsize_per_rank,:,:) = ql_temp(1:dsize_per_rank,:,:)
        qr_s(1:dsize_per_rank,:,:) = qr_temp(1:dsize_per_rank,:,:)
        qg_s(1:dsize_per_rank,:,:) = qg_temp(1:dsize_per_rank,:,:)
        qa_s(1:dsize_per_rank,:,:) = qa_temp(1:dsize_per_rank,:,:)
        qn_s(1:dsize_per_rank,:,:) = qn_temp(1:dsize_per_rank,:,:)
        pt_s(1:dsize_per_rank,:,:) = pt_temp(1:dsize_per_rank,:,:)
        uin_s(1:dsize_per_rank,:,:) = uin_temp(1:dsize_per_rank,:,:)
        vin_s(1:dsize_per_rank,:,:) = vin_temp(1:dsize_per_rank,:,:)
        dz_s(1:dsize_per_rank,:,:) = dz_temp(1:dsize_per_rank,:,:)
        delp_s(1:dsize_per_rank,:,:) = delp_temp(1:dsize_per_rank,:,:)
        p_s(1:dsize_per_rank,:,:) = p_temp(1:dsize_per_rank,:,:)
        area_s(1:dsize_per_rank,:) = area_temp(1:dsize_per_rank,:)
        land_s(1:dsize_per_rank,:) = land_temp(1:dsize_per_rank,:)
        qi_s(1:dsize_per_rank,:,:) = qi_temp(1:dsize_per_rank,:,:)
        qs_s(1:dsize_per_rank,:,:) = qs_temp(1:dsize_per_rank,:,:)
        qv_dt_s(1:dsize_per_rank,:,:) = qv_dt_temp(1:dsize_per_rank,:,:)
        ql_dt_s(1:dsize_per_rank,:,:) = ql_dt_temp(1:dsize_per_rank,:,:)
        qr_dt_s(1:dsize_per_rank,:,:) = qr_dt_temp(1:dsize_per_rank,:,:)
        qi_dt_s(1:dsize_per_rank,:,:) = qi_dt_temp(1:dsize_per_rank,:,:)
        qs_dt_s(1:dsize_per_rank,:,:) = qs_dt_temp(1:dsize_per_rank,:,:)
        qg_dt_s(1:dsize_per_rank,:,:) = qg_dt_temp(1:dsize_per_rank,:,:)
        qa_dt_s(1:dsize_per_rank,:,:) = qa_dt_temp(1:dsize_per_rank,:,:)
        pt_dt_s(1:dsize_per_rank,:,:) = pt_dt_temp(1:dsize_per_rank,:,:)
        w_s(1:dsize_per_rank,:,:) = w_temp(1:dsize_per_rank,:,:)
        udt_s(1:dsize_per_rank,:,:) = udt_temp(1:dsize_per_rank,:,:)
        vdt_s(1:dsize_per_rank,:,:) = vdt_temp(1:dsize_per_rank,:,:)
        refl_10cm_s(1:dsize_per_rank,:,:) = refl_10cm_temp(1:dsize_per_rank,:,:)
        rain_s(1:dsize_per_rank,:) = rain_temp(1:dsize_per_rank,:)
        snow_s(1:dsize_per_rank,:) = snow_temp(1:dsize_per_rank,:)
        ice_s(1:dsize_per_rank,:) = ice_temp(1:dsize_per_rank,:)
        graupel_s(1:dsize_per_rank,:) = graupel_temp(1:dsize_per_rank,:)

        deallocate(qv_temp, ql_temp, qr_temp, qg_temp, qa_temp, qn_temp, &
                      pt_temp, uin_temp, vin_temp, dz_temp, delp_temp, p_temp, &
                      area_temp, land_temp, qi_temp, qs_temp, &
                      qv_dt_temp, ql_dt_temp, qr_dt_temp, qi_dt_temp, qs_dt_temp, qg_dt_temp, qa_dt_temp, &
                      pt_dt_temp, w_temp, udt_temp, vdt_temp, refl_10cm_temp, &
                      rain_temp, snow_temp, ice_temp, graupel_temp)
        
    end subroutine scale_dataset_to_N

end program test
