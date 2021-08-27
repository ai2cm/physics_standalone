import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import stpfac
from util import compare_data

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

invars = ["pavel", "tavel", "h2ovmr", "nlay", "nlp1"]

indict = dict()
for var in invars:
    indict[var] = serializer.read(
        var, serializer.savepoint["swrad-setcoef-input-000000"]
    )


# This subroutine computes various coefficients needed in radiative
# transfer calculation.
# \param pavel           layer pressure (mb)
# \param tavel           layer temperature (k)
# \param h2ovmr          layer w.v. volumn mixing ratio (kg/kg)
# \param nlay            total number of vertical layers
# \param nlp1            total number of vertical levels
# \param laytrop         tropopause layer index (unitless)
# \param jp              indices of lower reference pressure
# \param jt,jt1          indices of lower reference temperatures at
#                       levels of jp and jp+1
# \param facij           factors mltiply the reference ks,i,j=0/1 for
#                       lower/higher of the 2 appropriate temperature
#                       and altitudes.
# \param selffac         scale factor for w. v. self-continuum equals
#                       (w.v. density)/(atmospheric density at 296k
#                       and 1013 mb)
# \param seffrac         factor for temperature interpolation of
#                       reference w.v. self-continuum data
# \param indself         index of lower ref temp for selffac
# \param forfac          scale factor for w. v. foreign-continuum
# \param forfrac         factor for temperature interpolation of
#                       reference w.v. foreign-continuum data
# \param indfor          index of lower ref temp for forfac


def setcoef(pavel, tavel, h2ovmr, nlay, nlp1):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    # purpose:  compute various coefficients needed in radiative transfer   !
    #    calculations.                                                      !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                       -size- !
    #   pavel     - real, layer pressures (mb)                         nlay !
    #   tavel     - real, layer temperatures (k)                       nlay !
    #   h2ovmr    - real, layer w.v. volum mixing ratio (kg/kg)        nlay !
    #   nlay/nlp1 - integer, total number of vertical layers, levels    1   !
    #                                                                       !
    #  outputs:                                                             !
    #   laytrop   - integer, tropopause layer index (unitless)          1   !
    #   jp        - real, indices of lower reference pressure          nlay !
    #   jt, jt1   - real, indices of lower reference temperatures      nlay !
    #                 at levels of jp and jp+1                              !
    #   facij     - real, factors multiply the reference ks,           nlay !
    #                 i,j=0/1 for lower/higher of the 2 appropriate         !
    #                 temperatures and altitudes.                           !
    #   selffac   - real, scale factor for w. v. self-continuum        nlay !
    #                 equals (w. v. density)/(atmospheric density           !
    #                 at 296k and 1013 mb)                                  !
    #   selffrac  - real, factor for temperature interpolation of      nlay !
    #                 reference w. v. self-continuum data                   !
    #   indself   - integer, index of lower ref temp for selffac       nlay !
    #   forfac    - real, scale factor for w. v. foreign-continuum     nlay !
    #   forfrac   - real, factor for temperature interpolation of      nlay !
    #                 reference w.v. foreign-continuum data                 !
    #   indfor    - integer, index of lower ref temp for forfac        nlay !
    #                                                                       !
    #  ======================    end of definitions    ===================  !   #

    #  ---  outputs:
    indself = np.zeros(nlay, dtype=np.int32)
    indfor = np.zeros(nlay, dtype=np.int32)
    jp = np.zeros(nlay, dtype=np.int32)
    jt = np.zeros(nlay, dtype=np.int32)
    jt1 = np.zeros(nlay, dtype=np.int32)

    fac00 = np.zeros(nlay)
    fac01 = np.zeros(nlay)
    fac10 = np.zeros(nlay)
    fac11 = np.zeros(nlay)
    selffac = np.zeros(nlay)
    selffrac = np.zeros(nlay)
    forfac = np.zeros(nlay)
    forfrac = np.zeros(nlay)

    ds = xr.open_dataset("../lookupdata/radsw_ref_data.nc")
    preflog = ds["preflog"].data
    tref = ds["tref"].data

    laytrop = nlay

    for k in range(nlay):

        forfac[k] = pavel[k] * stpfac / (tavel[k] * (1.0 + h2ovmr[k]))

        #  --- ...  find the two reference pressures on either side of the
        #           layer pressure.  store them in jp and jp1.  store in fp the
        #           fraction of the difference (in ln(pressure)) between these
        #           two values that the layer pressure lies.

        plog = np.log(pavel[k])
        jp[k] = max(1, min(58, int(36.0 - 5.0 * (plog + 0.04)))) - 1
        jp1 = jp[k] + 1
        fp = 5.0 * (preflog[jp[k]] - plog)

        #  --- ...  determine, for each reference pressure (jp and jp1), which
        #          reference temperature (these are different for each reference
        #          pressure) is nearest the layer temperature but does not exceed it.
        #          store these indices in jt and jt1, resp. store in ft (resp. ft1)
        #          the fraction of the way between jt (jt1) and the next highest
        #          reference temperature that the layer temperature falls.

        tem1 = (tavel[k] - tref[jp[k]]) / 15.0
        tem2 = (tavel[k] - tref[jp1]) / 15.0
        jt[k] = max(1, min(4, int(3.0 + tem1))) - 1
        jt1[k] = max(1, min(4, int(3.0 + tem2))) - 1
        ft = tem1 - float(jt[k] - 2)
        ft1 = tem2 - float(jt1[k] - 2)

        #  --- ...  we have now isolated the layer ln pressure and temperature,
        #           between two reference pressures and two reference temperatures
        #           (for each reference pressure).  we multiply the pressure
        #           fraction fp with the appropriate temperature fractions to get
        #           the factors that will be needed for the interpolation that yields
        #           the optical depths (performed in routines taugbn for band n).

        fp1 = 1.0 - fp
        fac10[k] = fp1 * ft
        fac00[k] = fp1 * (1.0 - ft)
        fac11[k] = fp * ft1
        fac01[k] = fp * (1.0 - ft1)

        #  --- ...  if the pressure is less than ~100mb, perform a different
        #           set of species interpolations.

        if plog > 4.56:

            laytrop = k + 1

            #  --- ...  set up factors needed to separately include the water vapor
            #           foreign-continuum in the calculation of absorption coefficient.

            tem1 = (332.0 - tavel[k]) / 36.0
            indfor[k] = min(2, max(1, int(tem1)))
            forfrac[k] = tem1 - float(indfor[k])

            #  --- ...  set up factors needed to separately include the water vapor
            #           self-continuum in the calculation of absorption coefficient.

            tem2 = (tavel[k] - 188.0) / 7.2
            indself[k] = min(9, max(1, int(tem2) - 7))
            selffrac[k] = tem2 - float(indself[k] + 7)
            selffac[k] = h2ovmr[k] * forfac[k]

        else:

            #  --- ...  set up factors needed to separately include the water vapor
            #           foreign-continuum in the calculation of absorption coefficient.

            tem1 = (tavel[k] - 188.0) / 36.0
            indfor[k] = 3
            forfrac[k] = tem1 - 1.0

            indself[k] = 0
            selffrac[k] = 0.0
            selffac[k] = 0.0

    jp += 1
    jt += 1
    jt1 += 1

    return (
        laytrop,
        jp,
        jt,
        jt1,
        fac00,
        fac01,
        fac10,
        fac11,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
    )


(
    laytrop,
    jp,
    jt,
    jt1,
    fac00,
    fac01,
    fac10,
    fac11,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
) = setcoef(
    indict["pavel"],
    indict["tavel"],
    indict["h2ovmr"],
    indict["nlay"][0],
    indict["nlp1"][0],
)

outdict = dict()

outdict["laytrop"] = laytrop
outdict["jp"] = jp
outdict["jt"] = jt
outdict["jt1"] = jt1
outdict["fac00"] = fac00
outdict["fac01"] = fac01
outdict["fac10"] = fac10
outdict["fac11"] = fac11
outdict["selffac"] = selffac
outdict["selffrac"] = selffrac
outdict["indself"] = indself
outdict["forfac"] = forfac
outdict["forfrac"] = forfrac
outdict["indfor"] = indfor

outvars = [
    "laytrop",
    "jp",
    "jt",
    "jt1",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "selffac",
    "selffrac",
    "indself",
    "forfac",
    "forfrac",
    "indfor",
]

valdict = dict()
for var in outvars:
    valdict[var] = serializer.read(
        var, serializer.savepoint["swrad-setcoef-output-000000"]
    )

compare_data(outdict, valdict)
