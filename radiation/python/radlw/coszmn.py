import numpy as np
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_pi

def coszmn(xlon, sinlat, coslat, solhr, IM, me,
           anginc, sollag, sindec, cosdec):
    #  ===================================================================  !
    #                                                                       !
    #  coszmn computes mean cos solar zenith angle over sw calling interval !
    #                                                                       !
    #  inputs:                                                              !
    #    xlon  (IM)    - grids' longitudes in radians, work both on zonal   !
    #                    0->2pi and -pi->+pi arrangements                   !
    #    sinlat(IM)    - sine of the corresponding latitudes                !
    #    coslat(IM)    - cosine of the corresponding latitudes              !
    #    solhr         - time after 00z in hours                            !
    #    IM            - num of grids in horizontal dimension               !
    #    me            - print message control flag                         !
    #                                                                       !
    #  outputs:                                                             !
    #    coszen(IM)    - average of cosz for daytime only in sw call interval
    #    coszdg(IM)    - average of cosz over entire sw call interval       !
    #                                                                       !
    #  module variables:                                                    !
    #    sollag        - equation of time                                   !
    #    sindec        - sine of the solar declination angle                !
    #    cosdec        - cosine of the solar declination angle              !
    #    anginc        - solar angle increment per iteration for cosz calc  !
    #    nstp          - total number of zenith angle iterations            !
    #                                                                       !
    #  usage:    call comzmn                                                !
    #                                                                       !
    #  external subroutines called: none                                    !
    #                                                                       !
    #  ===================================================================  !
    #
    pid12 = con_pi/12.0
    nstp = 6
    czlimt = 0.0001

    coszen = np.zeros(IM)
    coszdg = np.zeros(IM)
    istsun = np.zeros(IM)

    solang = pid12 * (solhr - 12.0)         # solar angle at present time
    rstp = 1.0 / float(nstp)

    for i in range(IM):
       coszen[i] = 0.0
       istsun[i] = 0

    for it in range(nstp):
        cns = solang + (float[it]-0.5)*anginc + sollag
        for i in range(IM):
            coszn     = sindec * sinlat[i] + cosdec * coslat[i] * \
                np.cos(cns+xlon[i])
            coszen[i] = coszen[i] + max(0.0, coszn)
            if coszn > czlimt:
                istsun[i] = istsun[i] + 1

    #  --- ...  compute time averages

    for i in range(IM):
        coszdg[i] = coszen[i] * rstp
        if istsun(i) > 0:
            coszen[i] = coszen[i] / istsun[i]

    return coszen, coszdg