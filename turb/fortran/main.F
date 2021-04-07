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

    real :: t_driver_start, t_driver_end, &
            t_tot_start, t_tot_end
    
    real, dimension(:, :), allocatable :: timings
    
    real, dimension(:), allocatable :: factors

    ! GT4Py development
    logical :: VALIDATION=.false., BENCHMARK=.false., &
               NORMAL=.false., WEAK=.false., STRONG=.false.

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

#ifdef RUN_MODE
    if (RUN_MODE .eq. 0) then
        VALIDATION = .true.
    elseif (RUN_MODE .eq. 1) then
        BENCHMARK = .true.
        NORMAL    = .true.
    elseif (RUN_MODE .eq. 2) then
        BENCHMARK = .true.
        WEAK      = .true.
    elseif (RUN_MODE .eq. 3) then
        BENCHMARK = .true.
        STRONG    = .true.
    else
        VALIDATION = .true.
    endif
#endif

    if (NORMAL) then
        allocate(factors(12), timings(7, 12))
        factors = (/ 1./12., 1./6., 0.25, 1./3., 5./12., 0.5, 7./12., 2./3., 0.75, 10./12., 11./12., 1. /)
    elseif (WEAK) then
        allocate(factors(1), timings(7, 1))
        factors = (/ 1. /)
    elseif (STRONG) then
        allocate(factors(5), timings(7, 1))
        factors = (/ 12./n_ranks /)
    else
        allocate(factors(1), timings(7, 1))
        factors = (/ 1. /)
    endif
    
    timings = 0.
    
    factors_size = size(factors)

#ifdef PROGRESS_MODE
    call cdstart(num_tiles * ser_count_max)
#endif
    ! initialize
    call gfuncphys(rank_idx, 0, nunit, 6)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='./dump' prefix='Serialized' directory_ref='./data' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max

#ifdef PROGRESS_MODE
        call cdshow(tile * ser_count_max + ser_count)
#endif        
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

        do fact_idx=1, factors_size

            factor = factors(fact_idx)
            ix_s = ix * factor
            im_s = im * factor

            allocate(kpbl_s(im_s), kinver_s(im_s), xmu_s(im_s), garea_s(im_s), psk_s(ix_s), &
            rbsoil_s(im_s), zorl_s(im_s), tsea_s(im_s), u10m_s(im_s), v10m_s(im_s), fm_s(im_s), &
            fh_s(im_s), evap_s(im_s), heat_s(im_s), stress_s(im_s), spd1_s(im_s), dusfc_s(im_s), &
            dvsfc_s(im_s), dtsfc_s(im_s), dqsfc_s(im_s), hpbl_s(im_s), dv_s(im_s, km), &
            du_s(im_s, km), tdt_s(im_s, km), u1_s(ix_s, km), v1_s(ix_s, km), t1_s(ix_s, km), &
            swh_s(ix_s, km), hlw_s(ix_s, km), prsi_s(ix_s, km+1), del_s(ix_s, km), prsl_s(ix_s, km), &
            prslk_s(ix_s, km), phii_s(ix_s, km+1), phil_s(ix_s, km), rtg_s(im_s, km, ntrac), &
            q1_s(ix_s, km, ntrac))

            ! variables not written only needs to reset once
            if (factor .le. 1.) then
                u1_s(1:ix_s,:) = u1(1:ix_s,:)
                v1_s(1:ix_s,:) = v1(1:ix_s,:)
                t1_s(1:ix_s,:) = t1(1:ix_s,:)
                q1_s(1:ix_s,:,:) = q1(1:ix_s,:,:)
                swh_s(1:ix_s,:) = swh(1:ix_s,:)
                hlw_s(1:ix_s,:) = hlw(1:ix_s,:)
                xmu_s(1:im_s) = xmu(1:im_s)
                garea_s(1:im_s) = garea(1:im_s)
                psk_s(1:ix_s) = psk(1:ix_s)
                rbsoil_s(1:im_s) = rbsoil(1:im_s)
                zorl_s(1:im_s) = zorl(1:im_s)
                u10m_s(1:im_s) = u10m(1:im_s)
                v10m_s(1:im_s) = v10m(1:im_s)
                fm_s(1:im_s) = fm(1:im_s)
                fh_s(1:im_s) = fh(1:im_s)
                tsea_s(1:im_s) = tsea(1:im_s)
                heat_s(1:im_s) = heat(1:im_s)
                evap_s(1:im_s) = evap(1:im_s)
                stress_s(1:im_s) = stress(1:im_s)
                spd1_s(1:im_s) = spd1(1:im_s)
                prsi_s(1:ix_s,:) = prsi(1:ix_s,:)
                del_s(1:ix_s,:) = del(1:ix_s,:)
                prsl_s(1:ix_s,:) = prsl(1:ix_s,:)
                prslk_s(1:ix_s,:) = prslk(1:ix_s,:)
                phii_s(1:ix_s,:) = phii(1:ix_s,:)
                phil_s(1:ix_s,:) = phil(1:ix_s,:)
                kinver_s(1:im_s) = kinver(1:im_s)
            else

                do part=1, int(factor)

                    st_im = (part-1)*im + 1
                    st_ix = (part-1)*ix + 1
                    en_im = part*im
                    en_ix = part*ix

                    u1_s(st_ix:en_ix,:) = u1(1:ix,:)
                    v1_s(st_ix:en_ix,:) = v1(1:ix,:)
                    t1_s(st_ix:en_ix,:) = t1(1:ix,:)
                    q1_s(st_ix:en_ix,:,:) = q1(1:ix,:,:)
                    swh_s(st_ix:en_ix,:) = swh(1:ix,:)
                    hlw_s(st_ix:en_ix,:) = hlw(1:ix,:)
                    xmu_s(st_im:en_im) = xmu(1:im)
                    garea_s(st_im:en_im) = garea(1:im)
                    psk_s(st_ix:en_ix) = psk(1:ix)
                    rbsoil_s(st_im:en_im) = rbsoil(1:im)
                    zorl_s(st_im:en_im) = zorl(1:im)
                    u10m_s(st_im:en_im) = u10m(1:im)
                    v10m_s(st_im:en_im) = v10m(1:im)
                    fm_s(st_im:en_im) = fm(1:im)
                    fh_s(st_im:en_im) = fh(1:im)
                    tsea_s(st_im:en_im) = tsea(1:im)
                    heat_s(st_im:en_im) = heat(1:im)
                    evap_s(st_im:en_im) = evap(1:im)
                    stress_s(st_im:en_im) = stress(1:im)
                    spd1_s(st_im:en_im) = spd1(1:im)
                    prsi_s(st_ix:en_ix,:) = prsi(1:ix,:)
                    del_s(st_ix:en_ix,:) = del(1:ix,:)
                    prsl_s(st_ix:en_ix,:) = prsl(1:ix,:)
                    prslk_s(st_ix:en_ix,:) = prslk(1:ix,:)
                    phii_s(st_ix:en_ix,:) = phii(1:ix,:)
                    phil_s(st_ix:en_ix,:) = phil(1:ix,:)
                    kinver_s(st_im:en_im) = kinver(1:im)
                    
                enddo
            endif

            do rep_idx=1, REPS+1

                if (factor .le. 1.) then

                    dv_s(1:im_s,:) = dv(1:im_s,:)
                    du_s(1:im_s,:) = du(1:im_s,:)
                    tdt_s(1:im_s,:) = tdt(1:im_s,:)
                    rtg_s(1:im_s,:,:) = rtg(1:im_s,:,:)
                    kpbl_s(1:im_s) = kpbl(1:im_s)
                    dusfc_s(1:im_s) = dusfc(1:im_s)
                    dvsfc_s(1:im_s) = dvsfc(1:im_s)
                    dtsfc_s(1:im_s) = dtsfc(1:im_s)
                    dqsfc_s(1:im_s) = dqsfc(1:im_s)

                else

                    do part=1, int(factor)

                        st_im = (part-1)*im + 1
                        en_im = part*im

                        dv_s(st_im:en_im,:) = dv(1:im,:)
                        du_s(st_im:en_im,:) = du(1:im,:)
                        tdt_s(st_im:en_im,:) = tdt(1:im,:)
                        rtg_s(st_im:en_im,:,:) = rtg(1:im,:,:)
                        kpbl_s(st_im:en_im) = kpbl(1:im)
                        dusfc_s(st_im:en_im) = dusfc(1:im)
                        dvsfc_s(st_im:en_im) = dvsfc(1:im)
                        dtsfc_s(st_im:en_im) = dtsfc(1:im)
                        dqsfc_s(st_im:en_im) = dqsfc(1:im)
                    enddo
                
                endif

                if (BENCHMARK .and. (rep_idx .gt. 1)) call cpu_time(t_driver_start)

                call satmedmfvdif(ix_s, im_s, km, ntrac, ntcw, ntiw, ntke, &
                    dv_s, du_s, tdt_s, rtg_s, u1_s, v1_s, t1_s, q1_s, swh_s, hlw_s, xmu_s, garea_s, &
                    psk_s, rbsoil_s, zorl_s, u10m_s, v10m_s, fm_s, fh_s, &
                    tsea_s, heat_s, evap_s, stress_s, spd1_s, kpbl_s, &
                    prsi_s, del_s, prsl_s, prslk_s, phii_s, phil_s, delt,  &
                    dspheat, dusfc_s, dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, &
                    kinver_s, xkzm_m, xkzm_h, xkzm_s)

                if (BENCHMARK .and. (rep_idx .gt. 1)) then
                            call cpu_time(t_driver_end)
                            timings(6, fact_idx) = timings(6, fact_idx) + (t_driver_end - t_driver_start)
                endif

            enddo

            if (fact_idx .ne. factors_size) then
                deallocate(kpbl_s, kinver_s, xmu_s, garea_s, psk_s, &
                        rbsoil_s, zorl_s, tsea_s, u10m_s, v10m_s, fm_s, &
                        fh_s, evap_s, heat_s, stress_s, spd1_s, dusfc_s, &
                        dvsfc_s, dtsfc_s, dqsfc_s, hpbl_s, dv_s, &
                        du_s, tdt_s, u1_s, v1_s, t1_s, &
                        swh_s, hlw_s, prsi_s, del_s, prsl_s, &
                        prslk_s, phii_s, phil_s, rtg_s, &
                        q1_s)

            endif

        enddo

        dv = dv_s
        du = du_s
        tdt = tdt_s
        rtg = rtg_s
        kpbl = kpbl_s
        dusfc = dusfc_s
        dvsfc = dvsfc_s
        dtsfc = dtsfc_s
        dqsfc = dqsfc_s

        !$ser mode read
        !$ser savepoint "satmedmfvdif-out-"//trim(ser_count_str)
        ! inout
        !$ser data dv=dv_ref du=du_ref tdt=tdt_ref rtg=rtg_ref kpbl=kpbl_ref
        !$ser data dusfc=dusfc_ref dvsfc=dvsfc_ref dtsfc=dtsfc_ref dqsfc=dqsfc_ref
        !$ser data hpbl=hpbl_ref
        
        if (VALIDATION) then
            call check_r_2d(dv, dv_ref, "dv")
            call check_r_2d(du, du_ref, "du")
            call check_r_2d(tdt, tdt_ref, "tdt")
            call check_r_3d(rtg, rtg_ref, "rtg")
            call check_i_1d(kpbl, kpbl_ref, "kpbl")
            call check_r_1d(dusfc, dusfc_ref, "dusfc")
            call check_r_1d(dvsfc, dvsfc_ref, "dvsfc")
            call check_r_1d(dtsfc, dtsfc_ref, "dtsfc")
            call check_r_1d(dqsfc, dqsfc_ref, "dqsfc")
        endif
        
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

    if (BENCHMARK) call MPI_REDUCE(timings(1:4, :), timings(1:4, :), 4*factors_size, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) then
        
        call cpu_time(t_tot_end)
        
        timings(7, 1) = t_tot_end - t_tot_start
        
        write(*,*) 'FINISHED!'
        
        if (BENCHMARK) then
            
            if (NORMAL) then
                open(1, file="../out/timings_benchmark_fortran_normal.dat")
            else if (WEAK) then
                open(1, file="../out/timings_weak_fortran_normal.dat", status="old", position="append")
            else if (STRONG) then
                open(1, file="../out/timings_strong_fortran_normal.dat", status="old", position="append")
            endif
            
            timings(1:6, :) = timings(1:6, :) / REPS
            
            do fact_idx=1, factors_size
                
                i_dim = int(im * factors(fact_idx) * n_ranks)
                
                timings(5, fact_idx) = timings(1, fact_idx) + timings(2, fact_idx) + timings(3, fact_idx) + timings(4, fact_idx)
                
                if (factors(fact_idx) .le. 1.) then
                    divider    = factors(fact_idx)
                    multiplier = 1.
                else
                    divider    = 1.
                    multiplier = factors(fact_idx)
                endif
                
                write(1, *) divider, multiplier, i_dim * km, n_ranks, &
                            0., timings(1, fact_idx), 0., timings(2, fact_idx), &
                            0., timings(3, fact_idx), 0., timings(4, fact_idx), &
                            0., timings(5, fact_idx), timings(6, fact_idx)
                
            enddo
            
            close(1)
            
        endif
    
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

end program test
