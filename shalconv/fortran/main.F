program test
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use m_countdown, only : cdstart, cdshow, cdstop

#ifdef USE_MPIF90
    use mpi
    implicit none
#else
    implicit none
    include "mpif.h"
#endif

    integer :: ierror, n_ranks, rank_idx
    integer :: factors_size, fact_idx, rep_idx, dsize_per_rank
    character(len=10) :: file_id
    real :: t_driver_start, t_driver_end, &
            t_tot_start, t_tot_end
    
    real :: timings(7, 1)
    real :: C_orig, CN
    integer :: im, ix, km, itc, ntc, ntk, ntr, ncloud
    integer, dimension(:), allocatable :: islimsk, kcnv, kbot, ktop, kcnv_ref, kbot_ref, ktop_ref

    real (kind=kind_phys) :: clam, c0s, c1, asolfac, pgcon, delt
    real (kind=kind_phys), dimension(:), allocatable :: psp, garea, hpbl, fscav, rn, rn_ref
    real (kind=kind_phys), dimension(:, :), allocatable :: delp, prslp, dot, phil, &
        q1, t1, u1, v1, cnvw, cnvc, ud_mf, dt_mf, q1_ref, t1_ref, u1_ref, v1_ref, &
        cnvw_ref, cnvc_ref, ud_mf_ref, dt_mf_ref
    real (kind=kind_phys), dimension(:, :, :), allocatable :: qtr, qtr_ref
    integer, dimension(:), allocatable :: islimsk_s, kcnv_s, kbot_s, ktop_s
    real (kind=kind_phys), dimension(:), allocatable :: psp_s, garea_s, hpbl_s, fscav_s, rn_s
    real (kind=kind_phys), dimension(:, :), allocatable :: delp_s, prslp_s, dot_s, phil_s, &
        q1_s, t1_s, u1_s, v1_s, cnvw_s, cnvc_s, ud_mf_s, dt_mf_s
    real (kind=kind_phys), dimension(:, :, :), allocatable :: qtr_s

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=100) :: nml_file
    character(len=6) :: ser_count_str

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank_idx, ierror)
    
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) call cpu_time(t_tot_start)

    ser_count_max = 19
    num_tiles = 6
    timings(:,:) = 0.
    CN = real(CNINPUT)
    dsize_per_rank = ceiling((CN*CN)/n_ranks)

    ! initialization
    call gfuncphys()

    call cdstart(num_tiles * ser_count_max)

    do tile = 0, 0
    
    !$ser init directory='../data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, 0

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

        allocate(delp_s(dsize_per_rank, km), prslp_s(dsize_per_rank, km), psp_s(dsize_per_rank), &
                phil_s(dsize_per_rank, km), qtr_s(dsize_per_rank,km,ntr+5), q1_s(dsize_per_rank, km), &
                t1_s(dsize_per_rank, km), u1_s(dsize_per_rank, km), v1_s(dsize_per_rank, km), &
                rn_s(dsize_per_rank), kbot_s(dsize_per_rank), ktop_s(dsize_per_rank), &
                kcnv_s(dsize_per_rank), islimsk_s(dsize_per_rank), garea_s(dsize_per_rank), &
                dot_s(dsize_per_rank, km), hpbl_s(dsize_per_rank), &
                ud_mf_s(dsize_per_rank, km), dt_mf_s(dsize_per_rank, km), &
                cnvw_s(dsize_per_rank, km), cnvc_s(dsize_per_rank, km))
        ! input
        !$ser data clam=clam c0s=c0s c1=c1 asolfac=asolfac pgcon=pgcon delt=delt
        !$ser data islimsk=islimsk psp=psp delp=delp prslp=prslp garea=garea hpbl=hpbl
        !$ser data dot=dot phil=phil fscav=fscav 
        ! inout
        !$ser data kcnv=kcnv kbot=kbot ktop=ktop qtr=qtr q1=q1 t1=t1 u1=u1 v1=v1 rn=rn
        !$ser data cnvw=cnvw cnvc=cnvc ud_mf=ud_mf dt_mf=dt_mf

        call scale_dataset_to_N(CN, dsize_per_rank, im, ix, km, delt, itc, ntc, ntk, ntr, &
                                delp, prslp, psp, phil, qtr, q1, t1, u1, v1,         &
                                rn, kbot, ktop, kcnv, islimsk, garea,                       &
                                dot, hpbl, ud_mf, dt_mf, cnvw, cnvc,                &
                                clam, c0s, c1, pgcon, asolfac,                              &
                                delp_s, prslp_s, psp_s, phil_s, qtr_s, q1_s, t1_s, u1_s, v1_s, &
                                rn_s, kbot_s, ktop_s, kcnv_s, islimsk_s, garea_s,                       &
                                dot_s, hpbl_s, ud_mf_s, dt_mf_s, cnvw_s, cnvc_s)
        do rep_idx=1, REPS+1

            if (rep_idx .gt. 1) then
                call MPI_BARRIER(MPI_COMM_WORLD, ierror) 
                call cpu_time(t_driver_start)
            endif

            call samfshalcnv(dsize_per_rank, dsize_per_rank, km, delt, itc, ntc, ntk, ntr, delp_s,   &
            prslp_s, psp_s, phil_s, qtr_s, q1_s, t1_s, u1_s, v1_s, fscav,               &
            rn_s, kbot_s, ktop_s, kcnv_s, islimsk_s, garea_s,                       &
            dot_s, ncloud, hpbl_s, ud_mf_s, dt_mf_s, cnvw_s, cnvc_s,                &
            clam, c0s, c1, pgcon, asolfac)

            if (rep_idx .gt. 1) then
                call cpu_time(t_driver_end)
                call MPI_BARRIER(MPI_COMM_WORLD, ierror)
                timings(6, 1) = timings(6, 1) + (t_driver_end - t_driver_start)
            endif

        enddo

        !$ser mode read
        !$ser savepoint "samfshalcnv-out-"//trim(ser_count_str)
        ! inout
        !$ser data kcnv=kcnv_ref kbot=kbot_ref ktop=ktop_ref qtr=qtr_ref q1=q1_ref t1=t1_ref
        !$ser data u1=u1_ref v1=v1_ref rn=rn_ref cnvw=cnvw_ref cnvc=cnvc_ref ud_mf=ud_mf_ref
        !$ser data dt_mf=dt_mf_ref

        deallocate(psp, delp, prslp, garea, hpbl, dot, &
        phil, fscav, islimsk, kcnv, kbot, ktop, qtr, &
        q1, t1, u1, v1, rn, cnvw, cnvc, &
        ud_mf, dt_mf, kcnv_ref, kbot_ref, ktop_ref, &
        qtr_ref, q1_ref, t1_ref, u1_ref, v1_ref, &
        rn_ref, cnvw_ref, cnvc_ref, ud_mf_ref, dt_mf_ref)

    end do
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    !$ser cleanup

    end do

    call MPI_REDUCE(timings(1:6, :), timings(1:6, :), 6, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) then
        
        call cpu_time(t_tot_end)
        
        timings(7, 1) = t_tot_end - t_tot_start
        
        write(*,*) 'FINISHED!'
        write(*,*) CN, dsize_per_rank
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

    subroutine scale_dataset_to_N(CN, dsize_per_rank, im, ix, km, delt, itc, ntc, ntk, ntr, &
        delp, prslp, psp, phil, qtr, q1, t1, u1, v1,         &
        rn, kbot, ktop, kcnv, islimsk, garea,                       &
        dot, hpbl, ud_mf, dt_mf, cnvw, cnvc,                &
        clam, c0s, c1, pgcon, asolfac,                              &
        delp_s, prslp_s, psp_s, phil_s, qtr_s, q1_s, t1_s, u1_s, v1_s, &
        rn_s, kbot_s, ktop_s, kcnv_s, islimsk_s, garea_s,                       &
        dot_s, hpbl_s, ud_mf_s, dt_mf_s, cnvw_s, cnvc_s)
        implicit none

        integer, intent(in):: dsize_per_rank, im, ix, km, itc, ntc, ntk, ntr, kbot(:), ktop(:), islimsk(:), kcnv(:)
        real (kind=kind_phys), intent(in) :: CN, psp(:), delp(:,:), prslp(:,:), garea(:), hpbl(:), dot(:,:), &
            phil(:,:), qtr(:,:,:), &
            q1(:,:), t1(:,:), u1(:,:), v1(:,:), rn(:), cnvw(:,:), cnvc(:,:), &
            ud_mf(:,:), dt_mf(:,:), clam, c0s, c1, asolfac, pgcon, delt
        integer, intent(inout) :: kbot_s(:), ktop_s(:), islimsk_s(:), kcnv_s(:)
        real (kind=kind_phys), intent(inout) :: psp_s(:), delp_s(:,:), prslp_s(:,:), garea_s(:), hpbl_s(:), dot_s(:,:), &
            phil_s(:,:), qtr_s(:,:,:), &
            q1_s(:,:), t1_s(:,:), u1_s(:,:), v1_s(:,:), rn_s(:), cnvw_s(:,:), cnvc_s(:,:), &
            ud_mf_s(:,:), dt_mf_s(:,:)

        real (kind=kind_phys), dimension(:), allocatable :: psp_temp, garea_temp, hpbl_temp, rn_temp, islimsk_temp, &
                                                            kcnv_temp, kbot_temp, ktop_temp
        real (kind=kind_phys), dimension(:, :), allocatable :: delp_temp, prslp_temp, dot_temp, phil_temp, &
            q1_temp, t1_temp, u1_temp, v1_temp, cnvw_temp, cnvc_temp, ud_mf_temp, dt_mf_temp
        real (kind=kind_phys), dimension(:, :, :), allocatable :: qtr_temp
        real :: C_orig
        integer :: multiplier_ix, multiplier_im, part, temp_ix, temp_im

        C_orig = real(ix)
        multiplier_ix = ceiling(dsize_per_rank/C_orig)
        temp_ix = multiplier_ix*ix
        multiplier_im = ceiling(dsize_per_rank/real(im))
        temp_im = multiplier_im*im

        allocate(psp_temp(temp_im), delp_temp(temp_ix,km), prslp_temp(temp_ix,km), garea_temp(temp_im), hpbl_temp(temp_im), dot_temp(temp_ix,km), &
        phil_temp(temp_ix,km), islimsk_temp(temp_im), kcnv_temp(temp_im), kbot_temp(temp_im), ktop_temp(temp_im), qtr_temp(temp_ix,km,ntr+5), &
        q1_temp(temp_ix,km), t1_temp(temp_ix,km), u1_temp(temp_ix,km), v1_temp(temp_ix,km), rn_temp(temp_im), cnvw_temp(temp_ix,km), cnvc_temp(temp_ix,km), &
        ud_mf_temp(temp_im,km), dt_mf_temp(temp_im,km))

        do part=1, multiplier_ix
            psp_temp((part-1)*im+1:part*im) = psp(1:im)
            delp_temp((part-1)*ix+1:part*ix,:) = delp(1:ix,:)
            prslp_temp((part-1)*ix+1:part*ix,:) = prslp(1:ix,:) 
            garea_temp((part-1)*im+1:part*im) = garea(1:im) 
            hpbl_temp((part-1)*im+1:part*im) = hpbl(1:im)
            dot_temp((part-1)*ix+1:part*ix,:) = dot(1:ix,:) 
            phil_temp((part-1)*ix+1:part*ix,:) = phil(1:ix,:)
            islimsk_temp((part-1)*im+1:part*im) = islimsk(1:im)
            kcnv_temp((part-1)*im+1:part*im) = kcnv(1:im)
            kbot_temp((part-1)*im+1:part*im) = kbot_temp(1:im)
            ktop_temp((part-1)*im+1:part*im) = ktop(1:im)
            qtr_temp((part-1)*ix+1:part*ix,:,:) = qtr(1:ix,:,:)
            q1_temp((part-1)*ix+1:part*ix,:) = q1(1:ix,:)
            t1_temp((part-1)*ix+1:part*ix,:) = t1(1:ix,:)
            u1_temp((part-1)*ix+1:part*ix,:) = u1(1:ix,:)
            v1_temp((part-1)*ix+1:part*ix,:) = v1(1:ix,:)
            rn_temp((part-1)*im+1:part*im) = rn(1:ix)
            cnvw_temp((part-1)*ix+1:part*ix,:) = cnvw(1:ix,:)
            cnvc_temp((part-1)*ix+1:part*ix,:) = cnvc(1:ix,:)
            ud_mf_temp((part-1)*im+1:part*im,:) = ud_mf(1:ix,:)
            dt_mf_temp((part-1)*im+1:part*im,:) = dt_mf(1:ix,:)
        enddo

        psp_s(1:dsize_per_rank) = psp_temp(1:dsize_per_rank)
        delp_s(1:dsize_per_rank,:) = delp_temp(1:dsize_per_rank,:)
        prslp_s(1:dsize_per_rank,:) = prslp_temp(1:dsize_per_rank,:)
        garea_s(1:dsize_per_rank) = garea_temp(1:dsize_per_rank)
        hpbl_s(1:dsize_per_rank) = hpbl_temp(1:dsize_per_rank)
        dot_s(1:dsize_per_rank,:) = dot_temp(1:dsize_per_rank,:)
        phil_s(1:dsize_per_rank,:) = phil_temp(1:dsize_per_rank,:)
        islimsk_s(1:dsize_per_rank) = islimsk_temp(1:dsize_per_rank)
        kcnv_s(1:dsize_per_rank) = kcnv_temp(1:dsize_per_rank)
        kbot_s(1:dsize_per_rank) = kbot_temp(1:dsize_per_rank)
        ktop_s(1:dsize_per_rank) = ktop(1:dsize_per_rank)
        qtr_s(1:dsize_per_rank,:,:) = qtr_temp(1:dsize_per_rank,:,:)
        q1_s(1:dsize_per_rank,:) = q1_temp(1:dsize_per_rank,:)
        t1_s(1:dsize_per_rank,:) = t1_temp(1:dsize_per_rank,:)
        u1_s(1:dsize_per_rank,:) = u1_temp(1:dsize_per_rank,:)
        v1_s(1:dsize_per_rank,:) = v1_temp(1:dsize_per_rank,:)
        rn_s(1:dsize_per_rank) = rn_temp(1:dsize_per_rank)
        cnvw_s(1:dsize_per_rank,:) = cnvw_temp(1:dsize_per_rank,:)
        cnvc_s(1:dsize_per_rank,:) = cnvc_temp(1:dsize_per_rank,:)
        ud_mf_s(1:dsize_per_rank,:) = ud_mf_temp(1:dsize_per_rank,:)
        dt_mf_s(1:dsize_per_rank,:) = dt_mf_temp(1:dsize_per_rank,:)

        deallocate(psp_temp, delp_temp, prslp_temp, garea_temp, hpbl_temp, dot_temp, &
        phil_temp, islimsk_temp, kcnv_temp, kbot_temp, ktop_temp, qtr_temp, &
        q1_temp, t1_temp, u1_temp, v1_temp, rn_temp, cnvw_temp, cnvc_temp, &
        ud_mf_temp, dt_mf_temp)
    end subroutine scale_dataset_to_N

end program test
