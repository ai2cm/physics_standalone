import sys
import numpy as np
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_solr

from radiation_astronomy import AstronomyClass
from radiation_aerosols import AerosolClass
from radiation_gases import GasClass

def radupdate(indict):
    # =================   subprogram documentation block   ================ !
    #                                                                       !
    # subprogram:   radupdate   calls many update subroutines to check and  !
    #   update radiation required but time varying data sets and module     !
    #   variables.                                                          !
    #                                                                       !
    # usage:        call radupdate                                          !
    #                                                                       !
    # attributes:                                                           !
    #   language:  fortran 90                                               !
    #   machine:   ibm sp                                                   !
    #                                                                       !
    #  ====================  definition of variables  ====================  !
    #                                                                       !
    # input parameters:                                                     !
    #   idate(8)       : ncep absolute date and time of initial condition   !
    #                    (yr, mon, day, t-zone, hr, min, sec, mil-sec)      !
    #   jdate(8)       : ncep absolute date and time at fcst time           !
    #                    (yr, mon, day, t-zone, hr, min, sec, mil-sec)      !
    #   deltsw         : sw radiation calling frequency in seconds          !
    #   deltim         : model timestep in seconds                          !
    #   lsswr          : logical flags for sw radiation calculations        !
    #   me             : print control flag                                 !
    #                                                                       !
    #  outputs:                                                             !
    #   slag           : equation of time in radians                        !
    #   sdec, cdec     : sin and cos of the solar declination angle         !
    #   solcon         : sun-earth distance adjusted solar constant (w/m2)  !
    #                                                                       !
    #  external module variables:                                           !
    #   isolar   : solar constant cntrl  (in module physparam)               !
    #              = 0: use the old fixed solar constant in "physcon"       !
    #              =10: use the new fixed solar constant in "physcon"       !
    #              = 1: use noaa ann-mean tsi tbl abs-scale with cycle apprx!
    #              = 2: use noaa ann-mean tsi tbl tim-scale with cycle apprx!
    #              = 3: use cmip5 ann-mean tsi tbl tim-scale with cycl apprx!
    #              = 4: use cmip5 mon-mean tsi tbl tim-scale with cycl apprx!
    #   ictmflg  : =yyyy#, external data ic time/date control flag          !
    #              =   -2: same as 0, but superimpose seasonal cycle        !
    #                      from climatology data set.                       !
    #              =   -1: use user provided external data for the          !
    #                      forecast time, no extrapolation.                 !
    #              =    0: use data at initial cond time, if not            !
    #                      available, use latest, no extrapolation.         !
    #              =    1: use data at the forecast time, if not            !
    #                      available, use latest and extrapolation.         !
    #              =yyyy0: use yyyy data for the forecast time,             !
    #                      no further data extrapolation.                   !
    #              =yyyy1: use yyyy data for the fcst. if needed, do        !
    #                      extrapolation to match the fcst time.            !
    #                                                                       !
    #  module variables:                                                    !
    #   loz1st   : first-time clim ozone data read flag                     !
    #                                                                       !
    #  subroutines called: sol_update, aer_update, gas_update               !
    #                                                                       !
    #  ===================================================================  !
    #

    idate = indict['idat']
    jdate = indict['jdat']
    deltsw = indict['fhswr']
    deltim = indict['dtf']
    lsswr = indict['lsswr']
    me = indict['me']
    ictmflg = indict['ictm']
    isolar = indict['isol']
    iaerflg = indict['iaer']
    ioznflg = indict['ntoz']
    ico2flg = indict['ico2']
    month0 = indict['month0']
    iyear0 = indict['iyear0']
    monthd = indict['monthd']
    loz1st = indict['loz1st']

    # -# Set up time stamp at fcst time and that for green house gases
    # (currently co2 only)
    # --- ...  time stamp at fcst time

    iyear = jdate[0]
    imon  = jdate[1]
    iday  = jdate[2]
    ihour = jdate[4]

    #  --- ...  set up time stamp used for green house gases (** currently co2 only)

    if ictmflg == 0 or ictmflg == -2:  # get external data at initial condition time
        kyear = idate[0]
        kmon  = idate[1]
        kday  = idate[2]
        khour = idate[4]
    else:                        # get external data at fcst or specified time
        kyear = iyear
        kmon  = imon
        kday  = iday
        khour = ihour

    if month0 != imon:
        lmon_chg = True
        month0 = imon
    else:
        lmon_chg = False

    # -# Call module_radiation_astronomy::sol_update(), yearly update, no
    # time interpolation.
    if lsswr:
        if isolar == 0 or isolar == 10:
            lsol_chg = False
        elif iyear0 != iyear:
            lsol_chg = True
        else:
            lsol_chg = isolar == 4 and lmon_chg

        iyear0 = iyear

        print(f'lsol_chg = {lsol_chg}')

        sol = AstronomyClass(me, isolar)

        slag, sdec, cdec, solcon = sol.sol_update(jdate,
                                                  kyear,
                                                  deltsw,
                                                  deltim,
                                                  lsol_chg, me)
        soldict = {'slag': slag, 'sdec': sdec, 'cdec': cdec, 'solcon': solcon}
#
    # -# Call module_radiation_aerosols::aer_update(), monthly update, no
    # time interpolation
    if lmon_chg:

        NLAY = 63

        aer = AerosolClass(NLAY, me, iaerflg)
        aer.aer_update(iyear, imon, me)
        aerdict = aer.return_updatedata()
#
    # -# Call co2 and other gases update routine:
    # module_radiation_gases::gas_update()
    if monthd != kmon:
        monthd = kmon
        lco2_chg = True
    else:
        lco2_chg = False

    gas = GasClass(me, ioznflg, ico2flg, ictmflg)
    gas.gas_update(kyear,
                   kmon,
                   kday,
                   khour,
                   loz1st,
                   lco2_chg,
                   me)
    gasdict = gas.return_updatedata()

    if loz1st:
        loz1st = False

    return soldict, aerdict, gasdict, loz1st