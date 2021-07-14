import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import solar_file
from phys_const import con_solr, con_solr_old

def sol_init(me, isolar):
    #  ===================================================================  !
    #                                                                       !
    #  initialize astronomy process, set up module constants.               !
    #                                                                       !
    #  inputs:                                                              !
    #     me      - print message control flag                              !
    #                                                                       !
    #  outputs:  (to module variable)                                       !
    #     ( none )                                                          !
    #                                                                       !
    #  external module variable: (in physparam)                             !
    #   isolar    - = 0: use the old fixed solar constant in "physcon"      !
    #               =10: use the new fixed solar constant in "physcon"      !
    #               = 1: use noaa ann-mean tsi tbl abs-scale with cyc apprx !
    #               = 2: use noaa ann-mean tsi tbl tim-scale with cyc apprx !
    #               = 3: use cmip5 ann-mean tsi tbl tim-scale with cyc apprx!
    #               = 4: use cmip5 mon-mean tsi tbl tim-scale with cyc apprx!
    #   solar_file- external solar constant data table                      !
    #                                                                       !
    #  internal module variable:                                            !
    #   isolflg   - internal solar constant scheme control flag             !
    #   solc0     - solar constant  (w/m**2)                                !
    #   solar_fname-file name for solar constant table assigned based on    !
    #               the scheme control flag, isolflg.                       !
    #                                                                       !
    #  usage:    call sol_init                                              !
    #                                                                       !
    #  subprograms called:  none                                            !
    #                                                                       !
    #  ===================================================================  !

    VTAGAST = 'NCEP-Radiation_astronomy v5.2  Jan 2013'

    if me == 0:
        print(VTAGAST)    # print out version tag

    #  ---  initialization
    # global isolflg = isolar
    # global solc0 = con_solr
    # global solar_fname = solar_file
    # global iyr_sav = 0
    # global nstp = 6
    solar_fname = solar_file
    solc0 = con_solr

    if isolar == 0:
        solc0 = con_solr_old
        if me == 0:
            print(f'- Using old fixed solar constant = {solc0}')
    elif isolar == 10:
        if me == 0:
            print(f'- Using new fixed solar constant = {solc0}')
    elif isolar == 1:        # noaa ann-mean tsi in absolute scale
        solar_fname = solar_file[:14] + 'noaa_a0.txt' + solar_file[26:]

        if me == 0:
            print('- Using NOAA annual mean TSI table in ABS scale',
                  ' with cycle approximation (old values)!')

        file_exist = os.path.isfile(solar_fname)
        if not file_exist:
            isolflg = 10

            if me == 0:
                print(f'Requested solar data file "{solar_fname}" not found!')
                print(f'Using the default solar constant value = {solc0}',
                      f' reset control flag isolflg={isolflg}')

    elif isolar == 2:        # noaa ann-mean tsi in tim scale
        solar_fname = solar_file[:14] + 'noaa_an.txt' + solar_file[26:]

        if me == 0:
            print(' - Using NOAA annual mean TSI table in TIM scale',
                  ' with cycle approximation (new values)!')

        file_exist = os.path.isfile(solar_fname)
        if not file_exist:
            isolflg = 10

            if me == 0:
                print(f'Requested solar data file "{solar_fname}" not found!')
                print(f'Using the default solar constant value = {solc0}',
                      f' reset control flag isolflg={isolflg}')

    elif isolar == 3:        # cmip5 ann-mean tsi in tim scale
        solar_fname = solar_file[:14] + 'cmip_an.txt' + solar_file[26:]

        if me == 0:
            print('- Using CMIP5 annual mean TSI table in TIM scale',
                  ' with cycle approximation')

        file_exist = os.path.isfile(solar_fname)
        if not file_exist:
            isolflg = 10

            if me == 0:
                print(f'Requested solar data file "{solar_fname}" not found!')
                print(f'Using the default solar constant value = {solc0}',
                      f' reset control flag isolflg={isolflg}')

    elif isolar == 4:      # cmip5 mon-mean tsi in tim scale
        solar_fname = solar_file[:14] + 'cmip_mn.txt' + solar_file[26:]

        if me == 0:
            print('- Using CMIP5 monthly mean TSI table in TIM scale',
                  ' with cycle approximation')


        file_exist = os.path.isfile(solar_fname)
        if not file_exist:
            isolflg = 10

            if me == 0:
                print(f'Requested solar data file "{solar_fname}" not found!')
                print(f'Using the default solar constant value = {solc0}',
                      f' reset control flag isolflg={isolflg}')
    else:                               # selection error
        isolflg = 10

        if me == 0:
            print('- !!! ERROR in selection of solar constant data',
                  f' source, ISOL = {isolar}')
            print(f'Using the default solar constant value = {solc0}',
                  f' reset control flag isolflg={isolflg}')

    sol_dict = dict()
    sol_dict['solar_fname'] = solar_fname

    return sol_dict