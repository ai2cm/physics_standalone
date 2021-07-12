from radphysparam import iovrsw, iovrlw, ivflip, icldflg

def cld_init(si, NLAY, imp_physics, me ):
    #  ===================================================================  !
    #                                                                       !
    # abstract: cld_init is an initialization program for cloud-radiation   !
    #   calculations. it sets up boundary layer cloud top.                  !
    #                                                                       !
    #                                                                       !
    # inputs:                                                               !
    #   si (L+1)        : model vertical sigma layer interface              !
    #   NLAY            : vertical layer number                             !
    #   imp_physics     : MP identifier                                     !
    #   me              : print control flag                                !
    #                                                                       !
    #  outputs: (none)                                                      !
    #           to module variables                                         !
    #                                                                       !
    #  external module variables: (in physparam)                            !
    #   icldflg         : cloud optical property scheme control flag        !
    #                     =0: abort! diagnostic cloud method discontinued   !
    #                     =1: model use prognostic cloud method             !
    #   imp_physics         : cloud microphysics scheme control flag        !
    #                     =99: zhao/carr/sundqvist microphysics cloud       !
    #                     =98: zhao/carr/sundqvist microphysics cloud+pdfcld!
    #                     =11: GFDL microphysics cloud                      !
    #                     =8: Thompson microphysics                         !
    #                     =6: WSM6 microphysics                             !
    #                     =10: MG microphysics                              !
    #   iovrsw/iovrlw   : sw/lw control flag for cloud overlapping scheme   !
    #                     =0: random overlapping clouds                     !
    #                     =1: max/ran overlapping clouds                    !
    #                     =2: maximum overlap clouds       (mcica only)     !
    #                     =3: decorrelation-length overlap (mcica only)     !
    #   ivflip          : control flag for direction of vertical index      !
    #                     =0: index from toa to surface                     !
    #                     =1: index from surface to toa                     !
    #  usage:       call cld_init                                           !
    #                                                                       !
    #  subroutines called:    rhtable                                       !
    #                                                                       !
    #  ===================================================================  !

    #===> ...  begin here
    #
    #  ---  set up module variables
    VTAGCLD = 'NCEP-Radiation_clouds    v5.1  Nov 2012'
    iovr = max(iovrsw, iovrlw)    # cld ovlp used for diag HML cld output

    if me == 0:
        print(VTAGCLD)      # print out version tag

    if icldflg == 0:
        print(' - Diagnostic Cloud Method has been discontinued')
    else:
        if me == 0:
            print(' - Using Prognostic Cloud Method')
            if imp_physics == 99:
                print('--- Zhao/Carr/Sundqvist microphysics')
            elif imp_physics == 98:
                print('--- zhao/carr/sundqvist + pdf cloud')
            elif imp_physics == 11:
                print('--- GFDL Lin cloud microphysics')
            elif imp_physics == 8:
                print('--- Thompson cloud microphysics')
            elif imp_physics == 6:
                print('--- WSM6 cloud microphysics')
            elif imp_physics == 10:
                print('--- MG cloud microphysics')
            else:
                print('!!! ERROR in cloud microphysc specification!!!',
                      f'imp_physics (NP3D) = {imp_physics}')
    # Compute the top of BL cld (llyr), which is the topmost non 
    # cld(low) layer for stratiform (at or above lowest 0.1 of the
    # atmosphere).

    if ivflip == 0:    # data from toa to sfc
        for k in range(NLAY-1, 0, -1):
            kl = k
            if si[k] < 0.9e0:
                break

        llyr = kl
    else:                      # data from sfc to top
        for k in range(1, NLAY):
            kl = k
            if si[k] < 0.9e0:
                break

        llyr = kl - 1

    return llyr