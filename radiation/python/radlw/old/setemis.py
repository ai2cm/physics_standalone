import numpy as np

def setemis(xlon, xlat, slmsk, snowf, sncovr, zorlf, tsknf, tairf, hprif,
            IMAX, iemslw, idxems, ialbflg):
    #  ===================================================================  !
    #                                                                       !
    #  this program computes surface emissivity for lw radiation.           !
    #                                                                       !
    #  usage:         call setemis                                          !
    #                                                                       !
    #  subprograms called:  none                                            !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                              !
    #     xlon  (IMAX)  - longitude in radiance, ok for both 0->2pi or      !
    #                     -pi -> +pi ranges                                 !
    #     xlat  (IMAX)  - latitude  in radiance, default to pi/2 -> -pi/2   !
    #                     range, otherwise see in-line comment              !
    #     slmsk (IMAX)  - sea(0),land(1),ice(2) mask on fcst model grid     !
    #     snowf (IMAX)  - snow depth water equivalent in mm                 !
    #     sncovr(IMAX)  - ialbflg=1: snow cover over land in fraction       !
    #     zorlf (IMAX)  - surface roughness in cm                           !
    #     tsknf (IMAX)  - ground surface temperature in k                   !
    #     tairf (IMAX)  - lowest model layer air temperature in k           !
    #     hprif (IMAX)  - topographic sdv in m                              !
    #     IMAX          - array horizontal dimension                        !
    #                                                                       !
    #  outputs:                                                             !
    #     sfcemis(IMAX) - surface emissivity                                !
    #                                                                       !
    #  -------------------------------------------------------------------  !
    #                                                                       !
    #  surface type definations:                                            !
    #     1. open water                   2. grass/wood/shrub land          !
    #     3. tundra/bare soil             4. sandy desert                   !
    #     5. rocky desert                 6. forest                         !
    #     7. ice                          8. snow                           !
    #                                                                       !
    #  input index data lon from 0 towards east, lat from n to s            !
    #                                                                       !
    #  ====================    end of description    =====================  !

    #  ---  reference emiss value for diff surface emiss index
    #       1-open water, 2-grass/shrub land, 3-bare soil, tundra,
    #       4-sandy desert, 5-rocky desert, 6-forest, 7-ice, 8-snow

    emsref = [0.97, 0.95, 0.94, 0.90, 0.93, 0.96, 0.96, 0.99]
    IMXEMS = 360
    JMXEMS = 180
    sfcemis = np.zeros(IMAX)

    # Set sfcemis default to 1.0 or by surface type and condition.
    if iemslw == 0:        # sfc emiss default to 1.0
        sfcemis = np.ones(IMAX)
        return sfcemis
    else:                           # emiss set by sfc type and condition
       dltg = 360.0 / float(IMXEMS)
       hdlt = 0.5 * dltg

    #  --- ...  mapping input data onto model grid
    #           note: this is a simple mapping method, an upgrade is needed if
    #           the model grid is much corcer than the 1-deg data resolution

    for i in range(IMAX):

        if round(slmsk[i]) == 0:          # sea point
            sfcemis[i] = emsref[0]

        elif round(slmsk[i]) == 2:        # sea-ice
            sfcemis[i] = emsref[6]

        else:                             # land

            #  ---  map grid in longitude direction
            i2 = 1
            j2 = 1

            tmp1 = np.rad2deg(xlon[i])
            if tmp1 < 0.0:
                tmp1 = tmp1 + 360.0

            for i1 in range(IMXEMS):
                tmp2 = dltg * (i1 - 1) + hdlt
                if abs(tmp1-tmp2) <= hdlt:
                    i2 = i1
                    break

            #  ---  map grid in latitude direction
            tmp1 = np.rad2deg(xlat[i])        # if xlat in pi/2 -> -pi/2 range

            for j1 in range(JMXEMS):
                tmp2 = 90.0 - dltg * (j1 - 1)
                if abs(tmp1-tmp2) <= hdlt:
                    j2 = j1
                    break

            idx = max(2, idxems[i2, j2])
            if idx >= 7:
                idx = 2
            sfcemis[i] = emsref[idx]

        # -# Check for snow covered area.

        if ialbflg == 1 and round(slmsk[i]) == 1: # input land area snow cover
            fsno0 = sncovr[i]
            fsno1 = 1.0 - fsno0
            sfcemis[i] = sfcemis[i]*fsno1 + emsref[7]*fsno0
        else:               # compute snow cover from snow depth
            if snowf[i] > 0.0:
                asnow = 0.02*snowf[i]
                argh  = min(0.50, max(.025, 0.01*zorlf[i]))
                hrgh  = min(1.0, max(0.20, 1.0577-1.1538e-3*hprif[i]))
                fsno0 = asnow / (argh + asnow) * hrgh
                if round(slmsk[i]) == 0 and tsknf[i] > 271.2:
                    fsno0 = 0.0
                fsno1 = 1.0 - fsno0
                sfcemis[i] = sfcemis[i]*fsno1 + emsref[7]*fsno0

    return sfcemis