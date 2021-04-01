program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use module_sfc_drv, only : sfc_drv
    implicit none

!  ---  input:
    integer, intent(in) :: im, km, isot, ivegsrc
    real (kind=kind_phys), dimension(5) :: pertvegf

    integer, allocatable, dimension(:) :: soiltyp, vegtype, slopetyp

    real (kind=kind_phys), allocatable, dimension(:) :: ps,             &
        t1, q1, sigmaf, sfcemis, dlwflx, dswsfc, snet, tg3, cm,         &
        ch, prsl1, prslki, wind, shdmin, shdmax,                        &
        snoalb, sfalb, zf, bexppert, xlaipert, vegfpert

    real (kind=kind_phys) :: delt

    logical, allocatable, dimension(:) :: flag_iter, flag_guess, land

    logical :: lheatstrg

!  ---  in/out:
    real (kind=kind_phys), allocatable, dimension(:) :: weasd,           &
        snwdph, tskin, tprcp, srflag, canopy, trans, tsurf, zorl,        &
        weasd_ref, snwdph_ref, tskin_ref, tprcp_ref, srflag_ref,         &
        canopy_ref, trans_ref, tsurf_ref, zorl_ref

    real (kind=kind_phys), allocatable, dimension(:,:) :: smc, stc, slc, &
        smc_ref, stc_ref, slc_ref

!  ---  output:
    real (kind=kind_phys), allocatable, dimension(:) :: sncovr1,         &
        qsurf, gflux, drain, evap, hflx, ep, runoff, cmm, chh,           &
        evbs, evcw, sbsno, snowc, stm, snohf, smcwlt2, smcref2,          &
        wet1,                                                            &
        sncovr1_ref, qsurf_ref, gflux_ref, drain_ref, evap_ref,          &
        hflx_ref, ep_ref, runoff_ref, cmm_ref, chh_ref, evbs_ref,        &
        evcw_ref, sbsno_ref, snowc_ref, stm_ref, snohf_ref, smcwlt2_ref, &
        smcref2_ref, wet1_ref

    integer :: iter, i, nlunit, me
    integer :: tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=6) :: ser_count_str

    real(8) :: time1, time2

    ser_count_max = 15
    num_tiles = 6

    ! initialize
    call gfuncphys()
    me = 0
    isot = 1
    ivegsrc = 1
    nlunit = 0
    call set_soilveg(me, isot, ivegsrc, nlunit)

    call cdstart(num_tiles * ser_count_max * 2)

    do tile = 0, num_tiles - 1
    
    !$ser init directory='./dump' prefix='Serialized' directory_ref='./data' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max
    do iter = 1, 2

        call cdshow(tile * ser_count_max * 2 + ser_count * 2 + iter - 1)
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser verbatim if (iter == 1) then
        !$ser savepoint "sfc_drv-in-iter1-"//trim(ser_count_str)
        !$ser verbatim else
        !$ser savepoint "sfc_drv-in-iter2-"//trim(ser_count_str)
        !$ser verbatim end if
        !$ser data im=im km=km

        allocate(                                                                                         &
            ! inputs
            ps(im), t1(im), q1(im), soiltyp(im), vegtype(im), sigmaf(im), sfcemis(im),                    &
            dlwflx(im), dswsfc(im), snet(im), tg3(im), cm(im), ch(im), prsl1(im),                         &
            prslki(im), zf(im), land(im), wind(im), slopetyp(im), shdmin(im), shdmax(im),                 &
            snoalb(im), sfalb(im), flag_iter(im), flag_guess(im),                                         &
            bexppert(im), xlaipert(im), vegfpert(im),                                                     &
            ! in/outs
            weasd(im), snwdph(im), tskin(im), tprcp(im), srflag(im), smc(im,km), stc(im,km),              &
            slc(im,km), canopy(im), trans(im), tsurf(im), zorl(im),                                       &
            ! outputs
            sncovr1(im), qsurf(im), gflux(im), drain(im), evap(im), hflx(im), ep(im),                     &
            runoff(im), cmm(im), chh(im), evbs(im), evcw(im), sbsno(im), snowc(im), stm(im),              &
            snohf(im), smcwlt2(im), smcref2(im), wet1(im),                                                &
            ! in/outs ref
            weasd_ref(im), snwdph_ref(im), tskin_ref(im), tprcp_ref(im), srflag_ref(im),                  &
            smc_ref(im,km), stc_ref(im,km), slc_ref(im,km), canopy_ref(im), trans_ref(im), tsurf_ref(im), &
            zorl_ref(im),                                                                                 &
            ! outputs ref
            sncovr1_ref(im), qsurf_ref(im), gflux_ref(im), drain_ref(im), evap_ref(im),                   &
            hflx_ref(im), ep_ref(im), runoff_ref(im), cmm_ref(im), chh_ref(im), evbs_ref(im),             &
            evcw_ref(im), sbsno_ref(im), snowc_ref(im), stm_ref(im), snohf_ref(im), smcwlt2_ref(im),      &
            smcref2_ref(im), wet1_ref(im)                                                                 &
        )

        ! -- inputs
        !$ser data ps=ps t1=t1 q1=q1 soiltyp=soiltyp vegtype=vegtype sigmaf=sigmaf
        !$ser data sfcemis=sfcemis dlwflx=dlwflx dswsfc=dswsfc snet=snet delt=delt tg3=tg3
        !$ser data cm=cm ch=ch prsl1=prsl1 prslki=prslki zf=zf land=land wind=wind slopetyp=slopetyp
        !$ser data shdmin=shdmin shdmax=shdmax snoalb=snoalb sfalb=sfalb flag_iter=flag_iter
        !$ser data flag_guess=flag_guess lheatstrg=lheatstrg isot=isot ivegsrc=ivegsrc bexppert=bexppert
        !$ser data xlaipert=xlaipert vegfpert=vegfpert pertvegf=pertvegf
        ! -- in/outs
        !$ser data weasd=weasd snwdph=snwdph tskin=tskin tprcp=tprcp srflag=srflag smc=smc stc=stc
        !$ser data slc=slc canopy=canopy trans=trans tsurf=tsurf zorl=zorl
        ! -- outputs
        !$ser data sncovr1=sncovr1 qsurf=qsurf gflux=gflux drain=drain evap=evap hflx=hflx ep=ep
        !$ser data runoff=runoff cmm=cmm chh=chh evbs=evbs evcw=evcw sbsno=sbsno snowc=snowc stm=stm
        !$ser data snohf=snohf smcwlt2=smcwlt2 smcref2=smcref2 wet1=wet1

        call tic(time1, time2)

        call sfc_drv                                                     &
        !  ---  inputs:
            ( im, km, ps, t1, q1, soiltyp, vegtype, sigmaf,              &
              sfcemis, dlwflx, dswsfc, snet, delt, tg3, cm, ch,          &
              prsl1, prslki, zf, land, wind, slopetyp,                   &
              shdmin, shdmax, snoalb, sfalb, flag_iter, flag_guess,      &
              lheatstrg, isot, ivegsrc,                                  &
              bexppert, xlaipert, vegfpert,pertvegf,                     &
        !  ---  in/outs:
              weasd, snwdph, tskin, tprcp, srflag, smc, stc, slc,        &
              canopy, trans, tsurf, zorl,                                &
        !  ---  outputs:
              sncovr1, qsurf, gflux, drain, evap, hflx, ep, runoff,      &
              cmm, chh, evbs, evcw, sbsno, snowc, stm, snohf,            &
              smcwlt2, smcref2, wet1,                                    &
              iter, ser_count_str                                        &
            )

        call toc(time1, time2)

        !$ser mode read
        !$ser verbatim if (iter == 1) then
        !$ser savepoint "sfc_drv-out-iter1-"//trim(ser_count_str)
        !$ser verbatim else
        !$ser savepoint "sfc_drv-out-iter2-"//trim(ser_count_str)
        !$ser verbatim end if

        ! -- in/outs ref
        !$ser data weasd=weasd_ref snwdph=snwdph_ref tskin=tskin_ref tprcp=tprcp_ref srflag=srflag_ref smc=smc_ref stc=stc_ref
        !$ser data slc=slc_ref canopy=canopy_ref trans=trans_ref tsurf=tsurf_ref zorl=zorl_ref
        ! -- outputs ref
        !$ser data sncovr1=sncovr1_ref qsurf=qsurf_ref gflux=gflux_ref drain=drain_ref evap=evap_ref hflx=hflx_ref ep=ep_ref
        !$ser data runoff=runoff_ref cmm=cmm_ref chh=chh_ref evbs=evbs_ref evcw=evcw_ref sbsno=sbsno_ref snowc=snowc_ref stm=stm_ref
        !$ser data snohf=snohf_ref smcwlt2=smcwlt2_ref smcref2=smcref2_ref wet1=wet1_ref

        call check_r_1d(weasd, weasd_ref, "weasd")
        call check_r_1d(snwdph, snwdph_ref, "snwdph")
        call check_r_1d(tskin, tskin_ref, "tskin")
        call check_r_1d(tprcp, tprcp_ref, "tprcp")
        call check_r_1d(srflag, srflag_ref, "srflag")
        call check_r_2d(smc, smc_ref, "smc")
        call check_r_2d(stc, stc_ref, "stc")
        call check_r_2d(slc, slc_ref, "slc")
        call check_r_1d(canopy, canopy_ref, "canopy")
        call check_r_1d(trans, trans_ref, "trans")
        call check_r_1d(tsurf, tsurf_ref, "tsurf")
        call check_r_1d(zorl, zorl_ref, "zorl")
        call check_r_1d(sncovr1, sncovr1_ref, "sncovr1")
        call check_r_1d(qsurf, qsurf_ref, "qsurf")
        call check_r_1d(gflux, gflux_ref, "gflux")
        call check_r_1d(drain, drain_ref, "drain")
        call check_r_1d(evap, evap_ref, "evap")
        call check_r_1d(hflx, hflx_ref, "hflx")
        call check_r_1d(ep, ep_ref, "ep")
        call check_r_1d(runoff, runoff_ref, "runoff")
        call check_r_1d(cmm, cmm_ref, "cmm")
        call check_r_1d(chh, chh_ref, "chh")
        call check_r_1d(evbs, evbs_ref, "evbs")
        call check_r_1d(evcw, evcw_ref, "evcw")
        call check_r_1d(sbsno, sbsno_ref, "sbsno")
        call check_r_1d(snowc, snowc_ref, "snowc")
        call check_r_1d(stm, stm_ref, "stm")
        call check_r_1d(snohf, snohf_ref, "snohf")
        call check_r_1d(smcwlt2, smcwlt2_ref, "smcwlt2")
        call check_r_1d(smcref2, smcref2_ref, "smcref2")
        call check_r_1d(wet1, wet1_ref, "wet1")

        deallocate(                                                         &
            ! inputs
            ps, t1, q1, soiltyp, vegtype, sigmaf, sfcemis,                  &
            dlwflx, dswsfc, snet, tg3, cm, ch, prsl1,                       &
            prslki, zf, land, wind, slopetyp, shdmin, shdmax,               &
            snoalb, sfalb, flag_iter, flag_guess,                           &
            bexppert, xlaipert, vegfpert,                                   &
            ! in/outs
            weasd, snwdph, tskin, tprcp, srflag, smc, stc,                  &
            slc, canopy, trans, tsurf, zorl,                                &
            ! outputs
            sncovr1, qsurf, gflux, drain, evap, hflx, ep,                   &
            runoff, cmm, chh, evbs, evcw, sbsno, snowc, stm,                &
            snohf, smcwlt2, smcref2, wet1,                                  &
            ! in/outs ref
            weasd_ref, snwdph_ref, tskin_ref, tprcp_ref, srflag_ref,        &
            smc_ref, stc_ref, slc_ref, canopy_ref, trans_ref, tsurf_ref,    &
            zorl_ref,                                                       &
            ! outputs ref
            sncovr1_ref, qsurf_ref, gflux_ref, drain_ref, evap_ref,         &
            hflx_ref, ep_ref, runoff_ref, cmm_ref, chh_ref, evbs_ref,       &
            evcw_ref, sbsno_ref, snowc_ref, stm_ref, snohf_ref, smcwlt2_ref,&
            smcref2_ref, wet1_ref                                           &
        )

    end do
    end do

    !$ser cleanup

    end do

    call cdstop()

    write(*,*) 'FINISHED!'

contains

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
