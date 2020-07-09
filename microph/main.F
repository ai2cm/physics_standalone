program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use gfdl_cloud_microphys_mod, only : gfdl_cloud_microphys_init, &
        gfdl_cloud_microphys_driver
    implicit none

    integer :: iie, kke, kbot, seconds

    real (kind=kind_phys) :: dt_in
    real (kind=kind_phys), dimension(:, :), allocatable :: area, land, &
        rain, snow, ice, graupel, rain_ref, snow_ref, ice_ref, graupel_ref
    real (kind=kind_phys), dimension(:, :, :), allocatable :: delp, dz, &
        uin, vin, p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, &
        udt, vdt, w, qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, &
        qi_ref, qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
        qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
        refl_10cm_ref

    logical :: lradar, reset

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=100) :: nml_file
    character(len=6) :: ser_count_str

    nunit = 42
    nml_file = 'input.nml'
    ser_count_max = 10
    num_tiles = 6

    call cdstart(num_tiles * ser_count_max)

    call gfdl_cloud_microphys_init(0, 0, nunit, (/''/), 6, nml_file)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='./data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max

        call cdshow(tile * ser_count_max + ser_count)
        
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

        call gfdl_cloud_microphys_driver (qv, ql, qr, qi, qs, qg, qa, qn,     &
            qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, qa_dt, pt_dt, pt, w,    &
            uin, vin, udt, vdt, dz, delp, area, dt_in, land, rain, snow, ice, &
            graupel, .false., .true., 1, iie, 1, 1, 1,                        &
            kke, 1, kbot, seconds, p, lradar, refl_10cm, reset)

        !$ser mode read
        !$ser savepoint "cloud_mp-out-"//trim(ser_count_str)
        ! inout
        !$ser data qi=qi_ref qs=qs_ref qv_dt=qv_dt_ref
        !$ser data ql_dt=ql_dt_ref qr_dt=qr_dt_ref qi_dt=qi_dt_ref qs_dt=qs_dt_ref qg_dt=qg_dt_ref qa_dt=qa_dt_ref
        !$ser data pt_dt=pt_dt_ref w=w_ref udt=udt_ref vdt=vdt_ref
        !$ser data rain=rain_ref snow=snow_ref ice=ice_ref graupel=graupel_ref
        !$ser data refl_10cm=refl_10cm_ref

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
        
        deallocate(area, land, rain, snow, ice, graupel, delp, dz, uin, vin, &
            p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, udt, vdt, w, &
            qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, qi_ref, &
            qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
            qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
            rain_ref, snow_ref, ice_ref, graupel_ref, &
            refl_10cm_ref)

    end do

    !$ser cleanup

    end do

    call cdstop()

    write(*,*) 'FINISHED!'

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

end program test
