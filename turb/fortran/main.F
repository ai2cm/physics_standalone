program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use m_satmedmfvdif, only : satmedmfvdif

#ifdef USE_MPIF90
    use mpi
    implicit none
#else
    implicit none
    include "mpif.h"
#endif

    integer :: ierror, n_ranks, rank_idx
    integer :: factors_size, fact_idx, rep_idx
    character(len=10) :: file_id
    real :: t_driver_start, t_driver_end, &
            t_tot_start, t_tot_end
    
    real :: timings(7, 1)
    real :: C_orig, CN
    real, dimension(:), allocatable :: factors

    ! GT4Py development
    logical :: VALIDATION=.false.

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

    integer :: i_dim, ix_s, im_s, st_ix, st_im, en_ix, en_im, part
    real :: factor, divider, multiplier
    integer, dimension(:), allocatable :: kpbl_s, kinver_s
    real (kind=kind_phys), dimension(:), allocatable :: xmu_s, garea_s, &
        psk_s, rbsoil_s, zorl_s, tsea_s, u10m_s, v10m_s, fm_s, fh_s, evap_s, heat_s, &
        stress_s, spd1_s, dusfc_s, dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s
    real (kind=kind_phys), dimension(:, :), allocatable :: dv_s, du_s, tdt_s, &
        u1_s, v1_s, t1_s, swh_s, hlw_s, prsi_s, del_s, prsl_s, prslk_s, phii_s, phil_s
    real (kind=kind_phys), dimension(:, :, :), allocatable :: rtg_s, q1_s

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=6) :: ser_count_str

    call MPI_INIT(ierror)
    
    call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank_idx, ierror)
    
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) call cpu_time(t_tot_start)

    nunit = 42
    ser_count_max = 19
    num_tiles = 6

    timings(:,:) = 0.
    CN = real(CNINPUT)
    
#ifdef PROGRESS_MODE
    call cdstart(num_tiles * ser_count_max)
#endif
    ! initialize
    call gfuncphys(rank_idx, 0, nunit, 6)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='../data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, 0

#ifdef PROGRESS_MODE
        call cdshow(tile * ser_count_max + ser_count)
#endif        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser savepoint "satmedmfvdif-in-000019"
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

        allocate(kpbl_s(INT(CN*CN)), kinver_s(INT(CN*CN)), xmu_s(INT(CN*CN)), garea_s(INT(CN*CN)), psk_s(INT(CN*CN)), &
        rbsoil_s(INT(CN*CN)), zorl_s(INT(CN*CN)), tsea_s(INT(CN*CN)), u10m_s(INT(CN*CN)), v10m_s(INT(CN*CN)), fm_s(INT(CN*CN)), &
        fh_s(INT(CN*CN)), evap_s(INT(CN*CN)), heat_s(INT(CN*CN)), stress_s(INT(CN*CN)), spd1_s(INT(CN*CN)), dusfc_s(INT(CN*CN)), &
        dvsfc_s(INT(CN*CN)), dtsfc_s(INT(CN*CN)), dqsfc_s(INT(CN*CN)), hpbl_s(INT(CN*CN)), dv_s(INT(CN*CN), km), &
        du_s(INT(CN*CN), km), tdt_s(INT(CN*CN), km), u1_s(INT(CN*CN), km), v1_s(INT(CN*CN), km), t1_s(INT(CN*CN), km), &
        swh_s(INT(CN*CN), km), hlw_s(INT(CN*CN), km), prsi_s(INT(CN*CN), km+1), del_s(INT(CN*CN), km), prsl_s(INT(CN*CN), km), &
        prslk_s(INT(CN*CN), km), phii_s(INT(CN*CN), km+1), phil_s(INT(CN*CN), km), rtg_s(INT(CN*CN), km, ntrac), &
        q1_s(INT(CN*CN), km, ntrac))

        call scale_dataset_to_N(CN, km, ix, im, ntrac, &
                                kpbl, kinver, xmu, garea, psk, &
                                rbsoil, zorl, tsea, u10m, v10m, fm, &
                                fh, evap, heat, stress, spd1, dusfc, &
                                dvsfc, dtsfc, dqsfc, hpbl, dv, &
                                du, tdt, u1, v1, t1, &
                                swh, hlw, prsi, del, prsl, &
                                prslk, phii, phil, rtg, &
                                q1, &
                                kpbl_s, kinver_s, xmu_s, garea_s, psk_s, &
                                rbsoil_s, zorl_s, tsea_s, u10m_s, v10m_s, fm_s, &
                                fh_s, evap_s, heat_s, stress_s, spd1_s, dusfc_s, &
                                dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, dv_s, &
                                du_s, tdt_s, u1_s, v1_s, t1_s, &
                                swh_s, hlw_s, prsi_s, del_s, prsl_s, &
                                prslk_s, phii_s, phil_s, rtg_s, &
                                q1_s)

        do rep_idx=1, REPS+1

            if (rep_idx .gt. 1) call cpu_time(t_driver_start)

            call satmedmfvdif(INT(CN*CN), INT(CN*CN), km, ntrac, ntcw, ntiw, ntke, &
                dv_s, du_s, tdt_s, rtg_s, u1_s, v1_s, t1_s, q1_s, swh_s, hlw_s, xmu_s, garea_s, &
                psk_s, rbsoil_s, zorl_s, u10m_s, v10m_s, fm_s, fh_s, &
                tsea_s, heat_s, evap_s, stress_s, spd1_s, kpbl_s, &
                prsi_s, del_s, prsl_s, prslk_s, phii_s, phil_s, delt,  &
                dspheat, dusfc_s, dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, &
                kinver_s, xkzm_m, xkzm_h, xkzm_s)

            if (rep_idx .gt. 1) then
                        call cpu_time(t_driver_end)
                        timings(6, 1) = timings(6, 1) + (t_driver_end - t_driver_start)
            endif

        enddo
        
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
            hpbl_ref, &
            kpbl_s, kinver_s, xmu_s, garea_s, psk_s, &
            rbsoil_s, zorl_s, tsea_s, u10m_s, v10m_s, fm_s, &
            fh_s, evap_s, heat_s, stress_s, spd1_s, dusfc_s, &
            dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, dv_s, &
            du_s, tdt_s, u1_s, v1_s, t1_s, &
            swh_s, hlw_s, prsi_s, del_s, prsl_s, &
            prslk_s, phii_s, phil_s, rtg_s, &
            q1_s)

    end do

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    !$ser cleanup

#ifdef PROGRESS_MODE
    call cdstop()
#endif

    end do

    call MPI_REDUCE(timings(1:4, :), timings(1:4, :), 4, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)

    if (rank_idx .eq. 0) then
        
        call cpu_time(t_tot_end)
        
        timings(7, 1) = t_tot_end - t_tot_start
        
        write(*,*) 'FINISHED!'
            
        write(file_id, '(i0)') int(CN)
        open(1, file='./out/timings_benchmark_fortran_normal_C'//trim(adjustl(file_id))//'.dat')

        timings(1:6, :) = timings(1:6, :) / REPS        
        timings(5, 1) = timings(1, 1) + timings(2, 1) + timings(3, 1) + timings(4, 1)
        write(1, *) timings(1, 1), timings(2, 1), &
                    timings(3, 1), timings(4, 1), &
                    timings(5, 1), timings(6, 1)
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

    subroutine scale_dataset_to_N(CN, km, ix, im, ntrac, &
                                kpbl, kinver, xmu, garea, psk, &
                                rbsoil, zorl, tsea, u10m, v10m, fm, &
                                fh, evap, heat, stress, spd1, dusfc, &
                                dvsfc, dtsfc, dqsfc, hpbl, dv, &
                                du, tdt, u1, v1, t1, &
                                swh, hlw, prsi, del, prsl, &
                                prslk, phii, phil, rtg, &
                                q1, &
                                kpbl_s, kinver_s, xmu_s, garea_s, psk_s, &
                                rbsoil_s, zorl_s, tsea_s, u10m_s, v10m_s, fm_s, &
                                fh_s, evap_s, heat_s, stress_s, spd1_s, dusfc_s, &
                                dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, dv_s, &
                                du_s, tdt_s, u1_s, v1_s, t1_s, &
                                swh_s, hlw_s, prsi_s, del_s, prsl_s, &
                                prslk_s, phii_s, phil_s, rtg_s, &
                                q1_s)
        implicit none
        integer, intent(in) :: kpbl(:), kinver(:)
        real (kind=kind_phys), intent(in) :: xmu(:), garea(:), psk(:), &
            rbsoil(:), zorl(:), tsea(:), u10m(:), v10m(:), fm(:), &
            fh(:), evap(:), heat(:), stress(:), spd1(:), dusfc(:), &
            dvsfc(:), dtsfc(:), dqsfc(:), hpbl(:), dv(:, :), &
            du(:, :), tdt(:, :), u1(:, :), v1(:, :), t1(:, :), &
            swh(:, :), hlw(:, :), prsi(:, :), del(:, :), prsl(:, :), &
            prslk(:, :), phii(:, :), phil(:, :), rtg(:, :, :), &
            q1(:, :, :)

        real (kind=kind_phys), dimension(:, :, :), allocatable :: rtg_temp, q1_temp
        real (kind=kind_phys), dimension(:, :), allocatable :: dv_temp, &
            du_temp, tdt_temp, u1_temp, v1_temp, t1_temp, &
            swh_temp, hlw_temp, prsi_temp, del_temp, prsl_temp, &
            prslk_temp, phii_temp, phil_temp
        integer, dimension(:), allocatable :: kpbl_temp, kinver_temp
        real (kind=kind_phys), dimension(:), allocatable :: xmu_temp, garea_temp, psk_temp, &
            rbsoil_temp, zorl_temp, tsea_temp, u10m_temp, v10m_temp, fm_temp, &
            fh_temp, evap_temp, heat_temp, stress_temp, spd1_temp, dusfc_temp, &
            dvsfc_temp, dtsfc_temp, dqsfc_temp, hpbl_temp
        integer, intent(inout) :: kpbl_s(:), kinver_s(:)
        real (kind=kind_phys), intent(inout) :: xmu_s(:), garea_s(:), psk_s(:), &
                                rbsoil_s(:), zorl_s(:), tsea_s(:), u10m_s(:), v10m_s(:), fm_s(:), &
                                fh_s(:), evap_s(:), heat_s(:), stress_s(:), spd1_s(:), dusfc_s(:), &
                                dvsfc_s(:), dtsfc_s(:), dqsfc_s(:), hpbl_s(:), dv_s(:, :), &
                                du_s(:, :), tdt_s(:, :), u1_s(:, :), v1_s(:, :), t1_s(:, :), &
                                swh_s(:, :), hlw_s(:, :), prsi_s(:, :), del_s(:, :), prsl_s(:, :), &
                                prslk_s(:, :), phii_s(:, :), phil_s(:, :), rtg_s(:, :, :), &
                                q1_s(:, :, :)
        real, intent(in) :: CN
        real :: C_orig
        integer, intent(in):: km, ix, im, ntrac
        integer :: multiplier_ix, multiplier_im, part, temp_ix, temp_im
        C_orig = real(ix)
        multiplier_ix = ceiling((CN*CN)/C_orig)
        temp_ix = multiplier_ix*ix
        multiplier_im = ceiling((CN*CN)/real(im))
        temp_im = multiplier_im*im

        allocate(kpbl_temp(temp_im), kinver_temp(temp_im), xmu_temp(temp_im), garea_temp(temp_im), psk_temp(temp_ix), &
            rbsoil_temp(temp_im), zorl_temp(temp_im), tsea_temp(temp_im), u10m_temp(temp_im), v10m_temp(temp_im), fm_temp(temp_im), &
            fh_temp(temp_im), evap_temp(temp_im), heat_temp(temp_im), stress_temp(temp_im), spd1_temp(temp_im), dusfc_temp(temp_im), &
            dvsfc_temp(temp_im), dtsfc_temp(temp_im), dqsfc_temp(temp_im), hpbl_temp(temp_im), &
            dv_temp(temp_im, km), &
            du_temp(temp_im, km), tdt_temp(temp_im, km), u1_temp(temp_ix, km), v1_temp(temp_ix, km), t1_temp(temp_ix, km), &
            swh_temp(temp_ix, km), hlw_temp(temp_ix, km), prsi_temp(temp_ix, km+1), del_temp(temp_ix, km), prsl_temp(temp_ix, km), &
            prslk_temp(temp_ix, km), phii_temp(temp_ix, km+1), phil_temp(temp_ix, km), &
            rtg_temp(temp_im, km, ntrac), q1_temp(temp_ix, km, ntrac))

        do part=1, multiplier_ix
            kpbl_temp((part-1)*im+1:part*im) = kpbl(1:im)
            kinver_temp((part-1)*im+1:part*im) = kinver(1:im)
            xmu_temp((part-1)*im+1:part*im) = xmu(1:im)
            garea_temp((part-1)*im+1:part*im) = garea(1:im)
            rbsoil_temp((part-1)*im+1:part*im) = rbsoil(1:im)
            zorl_temp((part-1)*im+1:part*im) = zorl(1:im)
            tsea_temp((part-1)*im+1:part*im) = tsea(1:im)
            u10m_temp((part-1)*im+1:part*im) = u10m(1:im)
            v10m_temp((part-1)*im+1:part*im) = v10m(1:im)
            fm_temp((part-1)*im+1:part*im) = fm(1:im)
            fh_temp((part-1)*im+1:part*im) = fh(1:im)
            evap_temp((part-1)*im+1:part*im) = evap(1:im)
            heat_temp((part-1)*im+1:part*im) = heat(1:im)
            stress_temp((part-1)*im+1:part*im) = stress(1:im)
            spd1_temp((part-1)*im+1:part*im) = spd1(1:im)
            dusfc_temp((part-1)*im+1:part*im) = dusfc(1:im)
            dvsfc_temp((part-1)*im+1:part*im) = dvsfc(1:im)
            dtsfc_temp((part-1)*im+1:part*im) = dtsfc(1:im)
            dqsfc_temp((part-1)*im+1:part*im) = dqsfc(1:im)
            hpbl_temp((part-1)*im+1:part*im) = hpbl(1:im)
            du_temp((part-1)*im+1:part*im, :) = du(1:im,:) 
            tdt_temp((part-1)*im+1:part*im, :) = tdt(1:im,:)
            rtg_temp((part-1)*im+1:part*im, :, :) = rtg(1:im, :, :)
        enddo

        do part=1, multiplier_ix
            psk_temp((part-1)*ix+1:part*ix) = psk(1:ix)
            u1_temp((part-1)*ix+1:part*ix, :) = u1(1:ix, :)
            v1_temp((part-1)*ix+1:part*ix, :) = v1(1:ix, :)
            t1_temp((part-1)*ix+1:part*ix, :) = t1(1:ix, :)
            swh_temp((part-1)*ix+1:part*ix, :) = swh(1:ix, :)
            hlw_temp((part-1)*ix+1:part*ix, :) = hlw(1:ix, :)
            prsi_temp((part-1)*ix+1:part*ix, :) = prsi(1:ix, :)
            del_temp((part-1)*ix+1:part*ix, :) = del(1:ix, :)
            prsl_temp((part-1)*ix+1:part*ix, :) = prsl(1:ix, :)
            prslk_temp((part-1)*ix+1:part*ix, :) = prslk(1:ix, :)
            phii_temp((part-1)*ix+1:part*ix, :) = phii(1:ix, :)
            phil_temp((part-1)*ix+1:part*ix, :) = phil(1:ix, :)
            q1_temp((part-1)*ix+1:part*ix, :, :) = q1(1:ix, :, :)
        enddo

        kpbl_s(1:INT(CN*CN)) = kpbl_temp(1:INT(CN*CN))
        kinver_s(1:INT(CN*CN)) = kinver_temp(1:INT(CN*CN))
        xmu_s(1:INT(CN*CN)) = xmu_temp(1:INT(CN*CN))
        garea_s(1:INT(CN*CN)) = garea_temp(1:INT(CN*CN))
        psk_s(1:INT(CN*CN)) = psk_temp(1:INT(CN*CN))
        rbsoil_s(1:INT(CN*CN)) = rbsoil_temp(1:INT(CN*CN))
        zorl_s(1:INT(CN*CN)) = zorl_temp(1:INT(CN*CN))
        tsea_s(1:INT(CN*CN)) = tsea_temp(1:INT(CN*CN))
        u10m_s(1:INT(CN*CN)) = u10m_temp(1:INT(CN*CN))
        v10m_s(1:INT(CN*CN)) = v10m_temp(1:INT(CN*CN))
        fm_s(1:INT(CN*CN)) = fm_temp(1:INT(CN*CN))
        fh_s(1:INT(CN*CN)) = fh_temp(1:INT(CN*CN))
        evap_s(1:INT(CN*CN)) = evap_temp(1:INT(CN*CN))
        heat_s(1:INT(CN*CN)) = heat_temp(1:INT(CN*CN))
        stress_s(1:INT(CN*CN)) = stress_temp(1:INT(CN*CN))
        spd1_s(1:INT(CN*CN)) = spd1_temp(1:INT(CN*CN))
        dusfc_s(1:INT(CN*CN)) = dusfc_temp(1:INT(CN*CN))
        dvsfc_s(1:INT(CN*CN)) = dvsfc_temp(1:INT(CN*CN))
        dtsfc_s(1:INT(CN*CN)) = dtsfc_temp(1:INT(CN*CN))
        dqsfc_s(1:INT(CN*CN)) = dqsfc_temp(1:INT(CN*CN))
        hpbl_s(1:INT(CN*CN)) = hpbl_temp(1:INT(CN*CN))
        dv_s(1:INT(CN*CN), :) = dv_temp(1:INT(CN*CN), :)
        du_s(1:INT(CN*CN), :) = du_temp(1:INT(CN*CN), :)
        tdt_s(1:INT(CN*CN), :) = tdt_temp(1:INT(CN*CN), :)
        u1_s(1:INT(CN*CN), :) = u1_temp(1:INT(CN*CN), :)
        v1_s(1:INT(CN*CN), :) = v1_temp(1:INT(CN*CN), :)
        t1_s(1:INT(CN*CN), :) = t1_temp(1:INT(CN*CN), :)
        swh_s(1:INT(CN*CN), :) = swh_temp(1:INT(CN*CN), :)
        hlw_s(1:INT(CN*CN), :) = hlw_temp(1:INT(CN*CN), :)
        prsi_s(1:INT(CN*CN), :) = prsi_temp(1:INT(CN*CN), :)
        del_s(1:INT(CN*CN), :) = del_temp(1:INT(CN*CN), :)
        prsl_s(1:INT(CN*CN), :) = prsl_temp(1:INT(CN*CN), :)
        prslk_s(1:INT(CN*CN), :) = prslk_temp(1:INT(CN*CN), :)
        phii_s(1:INT(CN*CN), :) = phii_temp(1:INT(CN*CN), :)
        phil_s(1:INT(CN*CN), :) = phil_temp(1:INT(CN*CN), :)
        rtg_s(1:INT(CN*CN), :, :) = rtg_temp(1:INT(CN*CN), :, :)
        q1_s(1:INT(CN*CN), :, :) = q1_temp(1:INT(CN*CN), :, :)

        deallocate(kpbl_temp, kinver_temp, xmu_temp, garea_temp, psk_temp, &
            rbsoil_temp, zorl_temp, tsea_temp, u10m_temp, v10m_temp, fm_temp, &
            fh_temp, evap_temp, heat_temp, stress_temp, spd1_temp, dusfc_temp, &
            dvsfc_temp, dtsfc_temp, dqsfc_temp, hpbl_temp, &
            dv_temp, &
            du_temp, tdt_temp, u1_temp, v1_temp, t1_temp, &
            swh_temp, hlw_temp, prsi_temp, del_temp, prsl_temp, &
            prslk_temp, phii_temp, phil_temp, &
            rtg_temp, q1_temp)

    end subroutine scale_dataset_to_N

end program test
