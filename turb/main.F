program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use m_satmedmfvdif, only : satmedmfvdif
    implicit none

    integer :: ix, im, km, ntrac, ntcw, ntiw, ntke
    integer, dimension(:), allocatable :: kpbl, kinver, kpbl_ref

    real (kind=kind_phys) :: delt, xkzm_m, xkzm_h, xkzm_s
    real (kind=kind_phys), dimension(:), allocatable :: xmu, garea, &
        psk, rbsoil, zorl, tsea, u10m, v10m, fm, fh, evap, heat, &
        stress, spd1, dusfc, dvsfc, dtsfc, dqsfc, hpbl, &
        dusfc_ref, dvsfc_ref, dtsfc_ref, dqsfc_ref, hpbl_ref
    real (kind=kind_phys), dimension(:, :), allocatable :: dv, du, tdt, &
        u1, v1, t1, swh, hlw, prsi, del, prsl, prslk, phii, phil, &
        dv_ref, du_ref, tdt_ref
    real (kind=kind_phys), dimension(:, :, :), allocatable :: rtg, q1, rtg_ref

    logical :: dspheat

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=6) :: ser_count_str

    nunit = 42
    ser_count_max = 19
    num_tiles = 6

    ! initialize
    call gfuncphys()

    call cdstart(num_tiles * ser_count_max)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='./dump' prefix='Serialized' directory_ref='./data' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max

        call cdshow(tile * ser_count_max + ser_count)
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser savepoint "satmedmfvdif-in-"//trim(ser_count_str)
        !$ser data ix=ix im=im km=km ntrac=ntrac ntcw=ntcw ntiw=ntiw
        !$ser data ntke=ntke

        allocate(kpbl(im), kinver(im), xmu(im), garea(im), psk(ix), &
            rbsoil(im), zorl(im), tsea(im), u10m(im), v10m(im), fm(im), &
            fh(im), evap(im), heat(im), stress(im), spd1(im), dusfc(im), &
            dvsfc(im), dtsfc(im), dqsfc(im), hpbl(im), dv(im, km), &
            du(im, km), tdt(im, km), u1(ix, km), v1(ix, km), t1(ix, km), &
            swh(ix, km), hlw(ix, km), prsi(ix, km+1), del(ix, km), prsl(ix, km), &
            prslk(ix, km), phii(ix, km+1), phil(ix, km), rtg(im, km, ntrac), &
            q1(ix, km, ntrac), &
            dv_ref(im, km), du_ref(im, km), tdt_ref(im, km), rtg_ref(im, km, ntrac), &
            kpbl_ref(im), dusfc_ref(im), dvsfc_ref(im), dtsfc_ref(im), dqsfc_ref(im), &
            hpbl_ref(im))

        ! input
        !$ser data u1=u1 v1=v1
        !$ser data t1=t1 q1=q1 swh=swh hlw=hlw xmu=xmu garea=garea psk=psk
        !$ser data rbsoil=rbsoil zorl=zorl u10m=u10m v10m=v10m fm=fm fh=fh
        !$ser data tsea=tsea heat=heat evap=evap stress=stress spd1=spd1
        !$ser data prsi=prsi del=del prsl=prsl prslk=prslk
        !$ser data phii=phii phil=phil delt=delt dspheat=dspheat
        !$ser data kinver=kinver xkzm_s=xkzm_s 
        ! inout
        !$ser data dv=dv du=du tdt=tdt rtg=rtg kpbl=kpbl 
        !$ser data dusfc=dusfc dvsfc=dvsfc dtsfc=dtsfc dqsfc=dqsfc
        !$ser data xkzm_m=xkzm_m xkzm_h=xkzm_h hpbl=hpbl

        call satmedmfvdif(ix, im, km, ntrac, ntcw, ntiw, ntke, &
            dv, du, tdt, rtg, u1, v1, t1, q1, swh, hlw, xmu, garea, &
            psk, rbsoil, zorl, u10m, v10m, fm, fh, &
            tsea, heat, evap, stress, spd1, kpbl, &
            prsi, del, prsl, prslk, phii, phil, delt,  &
            dspheat, dusfc, dvsfc, dtsfc, dqsfc, hpbl, &
            kinver, xkzm_m, xkzm_h, xkzm_s)

        !$ser mode read
        !$ser savepoint "satmedmfvdif-out-"//trim(ser_count_str)
        ! inout
        !$ser data dv=dv_ref du=du_ref tdt=tdt_ref rtg=rtg_ref kpbl=kpbl_ref
        !$ser data dusfc=dusfc_ref dvsfc=dvsfc_ref dtsfc=dtsfc_ref dqsfc=dqsfc_ref
        !$ser data hpbl=hpbl_ref

        call check_r_2d(dv, dv_ref, "dv")
        call check_r_2d(du, du_ref, "du")
        call check_r_2d(tdt, tdt_ref, "tdt")
        call check_r_3d(rtg, rtg_ref, "rtg")
        call check_i_1d(kpbl, kpbl_ref, "kpbl")
        call check_r_1d(dusfc, dusfc_ref, "dusfc")
        call check_r_1d(dvsfc, dvsfc_ref, "dvsfc")
        call check_r_1d(dtsfc, dtsfc_ref, "dtsfc")
        call check_r_1d(dqsfc, dqsfc_ref, "dqsfc")
        
        deallocate(kpbl, kinver, xmu, garea, psk, &
            rbsoil, zorl, tsea, u10m, v10m, fm, &
            fh, evap, heat, stress, spd1, dusfc, &
            dvsfc, dtsfc, dqsfc, hpbl, dv, &
            du, tdt, u1, v1, t1, &
            swh, hlw, prsi, del, prsl, &
            prslk, phii, phil, rtg, &
            q1, &
            dv_ref, du_ref, tdt_ref, rtg_ref, &
            kpbl_ref, dusfc_ref, dvsfc_ref, dtsfc_ref, dqsfc_ref, &
            hpbl_ref)

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
            rtol_local = 1.0d-6
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
