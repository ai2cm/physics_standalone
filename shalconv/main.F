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

    ser_count_max = 10
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

        if (any(kcnv /= kcnv_ref)) write(*,*) 'ERROR: kcnv does not validate', tile, ser_count
        if (any(kbot /= kbot_ref)) write(*,*) 'ERROR: kbot does not validate', tile, ser_count
        if (any(ktop /= ktop_ref)) write(*,*) 'ERROR: XXktopX does not validate', tile, ser_count
        if (any(qtr /= qtr_ref)) write(*,*) 'ERROR: qtr does not validate', tile, ser_count
        if (any(q1 /= q1_ref)) write(*,*) 'ERROR: q1 does not validate', tile, ser_count
        if (any(t1 /= t1_ref)) write(*,*) 'ERROR: t1 does not validate', tile, ser_count
        if (any(u1 /= u1_ref)) write(*,*) 'ERROR: u1 does not validate', tile, ser_count
        if (any(v1 /= v1_ref)) write(*,*) 'ERROR: v1 does not validate', tile, ser_count
        if (any(rn /= rn_ref)) write(*,*) 'ERROR: rn does not validate', tile, ser_count
        if (any(cnvw /= cnvw_ref)) write(*,*) 'ERROR: cnvw does not validate', tile, ser_count
        if (any(cnvc /= cnvc_ref)) write(*,*) 'ERROR: cnvc does not validate', tile, ser_count
        if (any(ud_mf /= ud_mf_ref)) write(*,*) 'ERROR: ud_mf does not validate', tile, ser_count
        if (any(dt_mf /= dt_mf_ref)) write(*,*) 'ERROR: dt_mf does not validate', tile, ser_count

        deallocate(psp, delp, prslp, garea, hpbl, dot, &
        phil, fscav, islimsk, kcnv, kbot, ktop, qtr, &
        q1, t1, u1, v1, rn, cnvw, cnvc, &
        ud_mf, dt_mf, kcnv_ref, kbot_ref, ktop_ref, &
        qtr_ref, q1_ref, t1_ref, u1_ref, v1_ref, &
        rn_ref, cnvw_ref, cnvc_ref, ud_mf_ref, dt_mf_ref)

    end do

    !ser cleanup

    end do

    call cdstop()

    write(*,*) 'FINISHED!'

end program test
