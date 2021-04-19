program test
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use m_countdown, only : cdstart, cdshow, cdstop
    implicit none

    integer :: im, ix, km, itc, ntc, ntk, ntr, ncloud
    integer, dimension(:), allocatable :: islimsk, kcnv, kbot, ktop, kcnv_ref, kbot_ref, ktop_ref

    real (kind=kind_phys) :: clam, c0s, c1, asolfac, pgcon, delt
    real (kind=kind_phys), dimension(:), allocatable :: psp, garea, hpbl, fscav, rn, rn_ref
    real (kind=kind_phys), dimension(:, :), allocatable :: delp, prslp, dot, phil, &
        q1, t1, u1, v1, cnvw, cnvc, ud_mf, dt_mf, q1_ref, t1_ref, u1_ref, v1_ref, &
        cnvw_ref, cnvc_ref, ud_mf_ref, dt_mf_ref
        real (kind=kind_phys), dimension(:, :, :), allocatable :: qtr, qtr_ref

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=100) :: nml_file
    character(len=6) :: ser_count_str

    ser_count_max = 19
    num_tiles = 6

    ! initialization
    call gfuncphys()

    call cdstart(num_tiles * ser_count_max)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='./data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max

        call cdshow(tile * ser_count_max + ser_count)
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser savepoint "samfshalcnv-in-"//trim(ser_count_str)
        !$ser data im=im ix=ix km=km itc=itc ntc=ntc ntk=ntk ntr=ntr ncloud=ncloud
       
        allocate(psp(im), delp(ix,km), prslp(ix,km), garea(im), hpbl(im), dot(ix,km), &
            phil(ix,km), fscav(ntc+1), islimsk(im), kcnv(im), kbot(im), ktop(im), qtr(ix,km,ntr+5), &
            q1(ix,km), t1(ix,km), u1(ix,km), v1(ix,km), rn(im), cnvw(ix,km), cnvc(ix,km), &
            ud_mf(im,km), dt_mf(im,km), kcnv_ref(im), kbot_ref(im), ktop_ref(im), &
            qtr_ref(ix,km,ntr+5), q1_ref(ix,km), t1_ref(ix,km), u1_ref(ix,km), v1_ref(ix,km), &
            rn_ref(im), cnvw_ref(ix,km), cnvc_ref(ix,km), ud_mf_ref(im,km), dt_mf_ref(im,km))

        ! input
        !$ser data clam=clam c0s=c0s c1=c1 asolfac=asolfac pgcon=pgcon delt=delt
        !$ser data islimsk=islimsk psp=psp delp=delp prslp=prslp garea=garea hpbl=hpbl
        !$ser data dot=dot phil=phil fscav=fscav 
        ! inout
        !$ser data kcnv=kcnv kbot=kbot ktop=ktop qtr=qtr q1=q1 t1=t1 u1=u1 v1=v1 rn=rn
        !$ser data cnvw=cnvw cnvc=cnvc ud_mf=ud_mf dt_mf=dt_mf

        call samfshalcnv(im, ix, km, delt, itc, ntc, ntk, ntr, delp,   &
           prslp, psp, phil, qtr, q1, t1, u1, v1, fscav,               &
           rn, kbot, ktop, kcnv, islimsk, garea,                       &
           dot, ncloud, hpbl, ud_mf, dt_mf, cnvw, cnvc,                &
           clam, c0s, c1, pgcon, asolfac)

        !$ser mode read
        !$ser savepoint "samfshalcnv-out-"//trim(ser_count_str)
        ! inout
        !$ser data kcnv=kcnv_ref kbot=kbot_ref ktop=ktop_ref qtr=qtr_ref q1=q1_ref t1=t1_ref
        !$ser data u1=u1_ref v1=v1_ref rn=rn_ref cnvw=cnvw_ref cnvc=cnvc_ref ud_mf=ud_mf_ref
        !$ser data dt_mf=dt_mf_ref

        call check_i_1d(kcnv, kcnv_ref, "kcnv")
        call check_i_1d(kbot, kbot_ref, "kbot")
        call check_i_1d(ktop, ktop_ref, "ktop")
        call check_r_3d(qtr, qtr_ref, "qtr")
        call check_r_2d(q1, q1_ref, "q1")
        call check_r_2d(t1, t1_ref, "t1")
        call check_r_2d(u1, u1_ref, "u1")
        call check_r_2d(v1, v1_ref, "v1")
        call check_r_1d(rn, rn_ref, "rn")
        call check_r_2d(cnvw, cnvw_ref, "cnvw")
        call check_r_2d(cnvc, cnvc_ref, "cnvc")
        call check_r_2d(ud_mf, ud_mf_ref, "ud_mf")
        call check_r_2d(dt_mf, dt_mf_ref, "dt_mf")

        deallocate(psp, delp, prslp, garea, hpbl, dot, &
        phil, fscav, islimsk, kcnv, kbot, ktop, qtr, &
        q1, t1, u1, v1, rn, cnvw, cnvc, &
        ud_mf, dt_mf, kcnv_ref, kbot_ref, ktop_ref, &
        qtr_ref, q1_ref, t1_ref, u1_ref, v1_ref, &
        rn_ref, cnvw_ref, cnvc_ref, ud_mf_ref, dt_mf_ref)

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

    subroutine check_i_1d(a, b, name, atol)
        implicit none
        integer, intent(in) :: a(:), b(:)
        character(len=*), intent(in) :: name
        integer, intent(in), optional :: atol

        logical :: close

        close = all(isclose_i(a, b, atol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_i_1d

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
            rtol_local = 1.0d-12
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

end program test
