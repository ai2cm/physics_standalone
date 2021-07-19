import numpy as np
import xarray as xr
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_pi

def sol_update(jdate, kyear, deltsw, deltim, lsol_chg, me, isolflg, solar_fname):
    #  ===================================================================  !
    #                                                                       !
    #  sol_update computes solar parameters at forecast time                !
    #                                                                       !
    #  inputs:                                                              !
    #     jdate(8)- ncep absolute date and time at fcst time                !
    #                (yr, mon, day, t-zone, hr, min, sec, mil-sec)          !
    #     kyear   - usually kyear=jdate(1). if not, it is for hindcast mode,!
    #               and it is usually the init cond time and serves as the  !
    #               upper limit of data can be used.                        !
    #     deltsw  - time duration in seconds per sw calculation             !
    #     deltim  - timestep in seconds                                     !
    #     lsol_chg- logical flags for change solar constant                 !
    #     me      - print message control flag                              !
    #                                                                       !
    #  outputs:                                                             !
    #    slag          - equation of time in radians                        !
    #    sdec, cdec    - sin and cos of the solar declination angle         !
    #    solcon        - sun-earth distance adjusted solar constant (w/m2)  !
    #                                                                       !
    #                                                                       !
    #  module variable:                                                     !
    #   solc0   - solar constant  (w/m**2) not adjusted by earth-sun dist   !
    #   isolflg - solar constant control flag                               !
    #             = 0: use the old fixed solar constant                     !
    #             =10: use the new fixed solar constant                     !
    #             = 1: use noaa ann-mean tsi tbl abs-scale with cycle apprx !
    #             = 2: use noaa ann-mean tsi tbl tim-scale with cycle apprx !
    #             = 3: use cmip5 ann-mean tsi tbl tim-scale with cycle apprx!
    #             = 4: use cmip5 mon-mean tsi tbl tim-scale with cycle apprx!
    #   solar_fname-external solar constant data table                      !
    #   sindec  - sine of the solar declination angle                       !
    #   cosdec  - cosine of the solar declination angle                     !
    #   anginc  - solar angle increment per iteration for cosz calc         !
    #   nstp    - total number of zenith angle iterations                   !
    #   smon_sav- saved monthly solar constants (isolflg=4 only)            !
    #   iyr_sav - saved year  of data previously used                       !
    #                                                                       !
    #  usage:    call sol_update                                            !
    #                                                                       !
    #  subprograms called:  solar, prtime                                   !
    #                                                                       !
    #  external functions called: iw3jdn                                    !
    #                                                                       !
    #  ===================================================================  !
    #

    #  ---  locals:
    hrday = 1.0/24.0    # frc day/hour
    minday = 1.0/1440.0  # frc day/minute
    secday = 1.0/86400.0 # frc day/second

    f3600 = 3600.0
    pid12 = con_pi/12.
    #
    #===>  ...  begin here
    #
    #  --- ...  forecast time
    iyear = jdate[0]
    imon  = jdate[1]
    iday  = jdate[2]
    ihr   = jdate[4]
    imin  = jdate[5]
    isec  = jdate[6]

    if lsol_chg:   # get solar constant from data table
        if iyr_sav == iyear:   # same year, no new reading necessary
            if isolflg == 4:
                solc0 = smon_sav[imon]
            else:                           # need to read in new data
                iyr_sav = iyear
                #  --- ...  check to see if the solar constant data file existed
                file_exist = os.path.isfile(solar_fname)

                if not file_exist:
                    print(' !!! ERROR! Can not find solar constant file!!!')
                else:
                    iyr = iyear

                    ds = xr.open_dataset(solar_fname)

                    iyr1 = ds['iyr1']
                    iyr2 = ds['iyr2']
                    icy1 = ds['icy1']
                    icy2 = ds['icy2']
                    smean = ds['smean']
                    cline = ds['cline']

                    if me == 0:
                        print('Updating solar constant with cycle approx')
                        print(f'Opened solar constant data file: {solar_fname}')

                    #  --- ...  check if there is a upper year limit put on the data table

                    if iyr < iyr1:
                        icy = icy1 - iyr1 + 1    # range of the earlest cycle in data table
                        while iyr < iyr1:
                            iyr += icy

                        if me == 0:
                            print(f'*** Year {iyear} out of table range!')
                            print(f'{iyr1}, {iyr2}')
                            print("Using the closest-cycle year ('{iyr}')")

                    elif iyr > iyr2:
                        icy = iyr2 - icy2 + 1   # range of the latest cycle in data table
                        while iyr > iyr2:
                            iyr -= icy

                        if me == 0:
                            print(f'*** Year {iyear} out of table range!')
                            print(f'{iyr1}, {iyr2}')
                            print(f"Using the closest-cycle year ('{iyr}')")


                    #  --- ...  locate the right record for the year of data

                    if isolflg < 4:        # use annual mean data tables
                        i = iyr2
                        while i >= iyr1:
                            jyr = ds['jyr']
                            solc1 = ds['solc1']

                            if i == iyr and iyr == jyr:
                                solc0  = smean + solc1

                                if me == 0:
                                    print('CHECK: Solar constant data used for year')
                                    print(f'{iyr}, {solc1}, {solc0}')
                            else:
                                i -= 1
                    elif isolflg == 4:   # use monthly mean data tables
                        i = iyr2
                        while i >= iyr1:
                            jyr = ds['jyr']
                            smon = ds['smon']

                            if i == iyr and iyr == jyr:
                                for nn in range(12):
                                    smon_sav[nn] = smean + smon[nn]
                                solc0 = smean + smon[imon]

                                if me == 0:
                                    print('CHECK: Solar constant data used for year')
                                    print(f'{iyr} and month {imon}')

                            else:                               
                                i -= 1

    #  --- ...  calculate forecast julian day and fraction of julian day

    jd1 = iw3jdn(iyear, imon, iday)

    #  --- ...  unlike in normal applications, where day starts from 0 hr,
    #           in astronomy applications, day stats from noon.

    if ihr < 12:
        jd1 -= 1
        fjd1= 0.5 + float(ihr)*hrday + float(imin)*minday + float(isec)*secday
    else:
        fjd1= float(ihr - 12)*hrday + float(imin)*minday + float(isec)*secday

    fjd1 += jd1

    jd  = int(fjd1)
    fjd -= jd

    # -# Call solar()
    r1, dlt, alp = solar(jd, fjd)

    #  --- ...  calculate sun-earth distance adjustment factor appropriate to date
    solcon = solc0 / (r1*r1)

    slag = sollag
    sdec = sindec
    cdec = cosdec

    #  --- ...  diagnostic print out

    if me == 0:
        prtime(jd, fjd, dlt, alp, r1, solcon)

    #  --- ...  setting up calculation parameters used by subr coszmn

    nswr  = max(1, round(deltsw/deltim))   # number of mdl t-step per sw call
    dtswh = deltsw / f3600                 # time length in hours

    nstp = max(6, nswr)
    anginc = pid12 * dtswh / float(nstp)

    if me == 0:
        print('for cosz calculations: nswr,deltim,deltsw,dtswh =',
              f'{nswr}, {deltim}, {deltsw}, {dtswh}, anginc, nstp =',
              f'{anginc}, {nstp}')

    return slag, sdec, cdec, solcon


def prtime(jd, fjd, dlt, alp, r1, solc):
    #  ===================================================================  !
    #                                                                       !
    #  prtime prints out forecast date, time, and astronomy quantities.     !
    #                                                                       !
    #  inputs:                                                              !
    #    jd       - forecast julian day                                     !
    #    fjd      - forecast fraction of julian day                         !
    #    dlt      - declination angle of sun in radians                     !
    #    alp      - right ascension of sun in radians                       !
    #    r1       - earth-sun radius vector in meter                        !
    #    solc     - solar constant in w/m^2                                 !
    #                                                                       !
    #  outputs:   ( none )                                                  !
    #                                                                       !
    #  module variables:                                                    !
    #    sollag   - equation of time in radians                             !
    #                                                                       !
    #  usage:    call prtime                                                !
    #                                                                       !
    #  external subroutines called: w3fs26                                  !
    #                                                                       !
    #  ===================================================================  !
    #

    #  ---  locals:
    sixty  = 60.0

    hpi = 0.5 * con_pi

    sign = '-'
    sigb = ' '
    
    month = ['JAN.', 'FEB.', 'MAR.', 'APR.', 'MAY ', 'JUNE',
             'JULY', 'AUG.', 'SEP.', 'OCT.', 'NOV ', 'DEC.']

    #===>  ...  begin here

    #  --- ...  get forecast hour and minute from fraction of julian day

    if fjd >= 0.5:
        jda = jd + 1
        mfjd= round(fjd*1440.0)
        ihr = mfjd / 60 - 12
        xmin= float(mfjd) - (ihr + 12)*sixty
    else:
        jda = jd
        mfjd= round(fjd*1440.0)
        ihr = mfjd / 60 + 12
        xmin= float(mfjd) - (ihr - 12)*sixty

    #  --- ...  get forecast year, month, and day from julian day

    iyear, imon, iday, idaywk, idayyr = w3fs26(jda)

    #  -- ...  compute solar parameters

    dltd = np.rad2deg(dlt)
    ltd  = dltd
    dltm = sixty * (np.abs(dltd) - abs(float(ltd)))
    ltm  = dltm
    dlts = sixty * (dltm - float(ltm))

    if ((dltd < 0.0) and (ltd == 0.0)):
        dsig = sign
    else:
        dsig = sigb

    halp = 6.0 * alp / hpi
    ihalp= halp
    ymin = np.abs(halp - float(ihalp)) * sixty
    iyy  = ymin
    asec = (ymin - float(iyy)) * sixty

    eqt  = 228.55735 * sollag
    eqsec= sixty * eqt

    print(f'0 FORECAST DATE {iday},{imon},{iyear} AT {ihr} HRS, {xmin} MINS',
          f'JULIAN DAY {jd} PLUS {fjd}')

    print(f'RADIUS VECTOR {r1} RIGHT ASCENSION OF SUN',
          f'{halp} HRS, OR {ihalp} HRS {iyy} MINS {asec} SECS')

    print(f'DECLINATION OF THE SUN {dltd} DEGS, OR {dsig}',
          f'{ltd} DEGS {ltm} MINS {dlts} SECS/  EQUATION OF TIME',
          f'{eqt} MINS, OR {eqsec} SECS, OR {sollag} RADIANS',
          f'SOLAR CONSTANT {solc} (DISTANCE AJUSTED)')


def solar(jd, fjd):
    #  ===================================================================  !
    #                                                                       !
    #  solar computes radius vector, declination and right ascension of     !
    #  sun, and equation of time.                                           !
    #                                                                       !
    #  inputs:                                                              !
    #    jd       - julian day                                              !
    #    fjd      - fraction of the julian day                              !
    #                                                                       !
    #  outputs:                                                             !
    #    r1       - earth-sun radius vector                                 !
    #    dlt      - declination of sun in radians                           !
    #    alp      - right ascension of sun in radians                       !
    #                                                                       !
    #  module variables:                                                    !
    #    sollag   - equation of time in radians                             !
    #    sindec   - sine of declination angle                               !
    #    cosdec   - cosine of declination angle                             !
    #                                                                       !
    #  usage:    call solar                                                 !
    #                                                                       !
    #  external subroutines called: none                                    !
    #                                                                       !
    #  ===================================================================  !
    #

    #  ---  locals:
    parcyear = 365.25   # days of year
    parccr   = 1.3e-6   # iteration limit
    partpp   = 1.55     # days between epoch and
    parsvt6  = 78.035   # days between perihelion passage
    parjdor  = 2415020  # jd of epoch which is january

    tpi = 2. * con_pi

    #===>  ...  begin here

    # --- ...  computes time in julian centuries after epoch

    t1 = float(jd - jdor) / 36525.0

    # --- ...  computes length of anomalistic and tropical years (minus 365 days)

    year = 0.25964134 + 0.304e-5 * t1
    tyear= 0.24219879 - 0.614e-5 * t1

    # --- ...  computes orbit eccentricity and angle of earth's inclination from t

    ec   = 0.01675104 - (0.418e-4 + 0.126e-6 * t1) * t1
    angin= 23.452294 - (0.0130125 + 0.164e-5 * t1) * t1

    ador = jdor
    doe = ador + (svt6 * cyear) / (year - tyear)

    # --- ...  deleqn is updated svt6 for current date

    deleqn= float(jdoe - jd) * (year - tyear) / cyear
    year  = year + 365.0
    sni   = np.sin( np.deg2rad(angin))
    tini  = 1.0 / np.tan(np.deg2rad(angin))
    er    = np.sqrt( (1.0 + ec) / (1.0 - ec) )
    qq    = deleqn * tpi / year

    # --- ...  determine true anomaly at equinox

    e1    = 1.0
    cd    = 1.0
    iter  = 0

    while cd > ccr:
        ep   = e1 - (e1 - ec*np.sin(e1) - qq) / (1.0 - ec*np.cos(e1))
        cd   = np.abs(e1 - ep)
        e1   = ep
        iter += 1

        if iter > 10:
            print(f'ITERATION COUNT FOR LOOP 32 = {iter}')
            print(f'E, EP, CD = {e1}, {ep}, {cd}')
            break

    eq = 2.0 * np.arctan(er * np.tan(0.5*e1))

    # --- ...  date is days since last perihelion passage

    dat  = float(jd - jdor) - tpp + fjd
    date = dat % year

    # --- ...  solve orbit equations by newton's method

    em   = tpi * date / year
    e1   = 1.0
    cr   = 1.0
    iter = 0

    while cr > ccr:
        ep   = e1 - (e1 - ec*np.sin(e1) - em) / (1.0 - ec*np.cos(e1))
        cr   = np.abs(e1 - ep)
        e1   = ep
        iter += 1

        if iter > 10:
            print(f'ITERATION COUNT FOR LOOP 31 = {iter}')
            break

    w1 = 2.0 * np.arctan(er * np.tan(0.5*e1))

    r1 = 1.0 - ec*np.cos(e1)

    sindec = sni * np.sin(w1 - eq)
    cosdec = np.sqrt(1.0 - sindec*sindec)

    dlt = np.arcsin(sindec)
    alp = np.arcsin(np.tan(dlt)*tini)

    tst = np.cos(w1 - eq)
    if tst < 0.0:
        alp = con_pi - alp
    if alp < 0.0:
        alp = alp + tpi

    sun = tpi * (date - deleqn) / year
    if sun < 0.0:
         sun = sun + tpi
    sollag = sun - alp - 0.03255

    return r1, dlt, alp


def iw3jdn(iyear, month, iday):

    iw3jdn = iday - 32075 + 1461 * (iyear + 4800 + (month - 14) / 12) / 4 + \
        367 * (month - 2 - (month -14) / 12 * 12) / 12 - \
        3 * ((iyear + 4900 + (month - 14) / 12) / 100) / 4

    return iw3jdn


def w3fs26(JLDAYN):
    L = JLDAYN + 68569
    N = 4 * L / 146097
    L = L - (146097 * N + 3) / 4
    I = 4000 * (L + 1) / 1461001
    L = L - 1461 * I / 4 + 31
    J = 80 * L / 2447
    IDAY   = L - 2447 * J / 80
    L      = J / 11
    MONTH  = J + 2 - 12 * L
    IYEAR  = 100 * (N - 49) + I + L
    IDAYWK = ((JLDAYN + 1) % 7) + 1
    IDAYYR = JLDAYN - \
        (-31739 +1461 * (IYEAR+4799) / 4 - 3 * ((IYEAR+4899)/100)/4)

    return IYEAR, MONTH, IDAY, IDAYWK, IDAYYR