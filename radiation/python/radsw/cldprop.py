import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import stpfac, nbands, nbdsw, ngptsw, ftiny, idxebc, nblow
from radphysparam import iswcliq, iswcice
from util import compare_data

isubcsw = 2
iovrsw = 1

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()


# This subroutine computes the cloud optical properties for each
# cloudy layer and g-point interval.
# \param cfrac          layer cloud fraction
# \n for  physparam::iswcliq > 0 (prognostic cloud scheme)  - - -
# \param cliqp          layer in-cloud liq water path (\f$g/m^2\f$)
# \param reliq          mean eff radius for liq cloud (micron)
# \param cicep          layer in-cloud ice water path (\f$g/m^2\f$)
# \param reice          mean eff radius for ice cloud (micron)
# \param cdat1          layer rain drop water path (\f$g/m^2\f$)
# \param cdat2          effective radius for rain drop (micron)
# \param cdat3          layer snow flake water path(\f$g/m^2\f$)
# \param cdat4          mean eff radius for snow flake(micron)
# \n for physparam::iswcliq = 0  (diagnostic cloud scheme)  - - -
# \param cliqp          not used
# \param cicep          not used
# \param reliq          not used
# \param reice          not used
# \param cdat1          layer cloud optical depth
# \param cdat2          layer cloud single scattering albedo
# \param cdat3          layer cloud asymmetry factor
# \param cdat4          optional use
# \param cf1            effective total cloud cover at surface
# \param nlay           vertical layer number
# \param ipseed         permutation seed for generating random numbers
#                      (isubcsw>0)
# \param taucw          cloud optical depth, w/o delta scaled
# \param ssacw          weighted cloud single scattering albedo
#                      (ssa = ssacw / taucw)
# \param asycw          weighted cloud asymmetry factor
#                      (asy = asycw / ssacw)
# \param cldfrc         cloud fraction of grid mean value
# \param cldfmc         cloud fraction for each sub-column
# \section General_cldprop General Algorithm


def cldprop(
    cfrac,
    cliqp,
    reliq,
    cicep,
    reice,
    cdat1,
    cdat2,
    cdat3,
    cdat4,
    cf1,
    nlay,
    ipseed,
    dz,
    delgth,
    ipt,
):

    #  ===================  program usage description  ===================  !
    #                                                                       !
    # Purpose: Compute the cloud optical properties for each cloudy layer   !
    # and g-point interval.                                                 !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                        size  !
    #    cfrac - real, layer cloud fraction                            nlay !
    #        .....  for  iswcliq > 0 (prognostic cloud sckeme)  - - -       !
    #    cliqp - real, layer in-cloud liq water path (g/m**2)          nlay !
    #    reliq - real, mean eff radius for liq cloud (micron)          nlay !
    #    cicep - real, layer in-cloud ice water path (g/m**2)          nlay !
    #    reice - real, mean eff radius for ice cloud (micron)          nlay !
    #    cdat1 - real, layer rain drop water path (g/m**2)             nlay !
    #    cdat2 - real, effective radius for rain drop (micron)         nlay !
    #    cdat3 - real, layer snow flake water path(g/m**2)             nlay !
    #    cdat4 - real, mean eff radius for snow flake(micron)          nlay !
    #        .....  for iswcliq = 0  (diagnostic cloud sckeme)  - - -       !
    #    cdat1 - real, layer cloud optical depth                       nlay !
    #    cdat2 - real, layer cloud single scattering albedo            nlay !
    #    cdat3 - real, layer cloud asymmetry factor                    nlay !
    #    cdat4 - real, optional use                                    nlay !
    #    cliqp - real, not used                                        nlay !
    #    cicep - real, not used                                        nlay !
    #    reliq - real, not used                                        nlay !
    #    reice - real, not used                                        nlay !
    #                                                                       !
    #    cf1   - real, effective total cloud cover at surface           1   !
    #    nlay  - integer, vertical layer number                         1   !
    #    ipseed- permutation seed for generating random numbers (isubcsw>0) !
    #    dz    - real, layer thickness (km)                            nlay !
    #    delgth- real, layer cloud decorrelation length (km)            1   !
    #                                                                       !
    #  outputs:                                                             !
    #    taucw  - real, cloud optical depth, w/o delta scaled    nlay*nbdsw !
    #    ssacw  - real, weighted cloud single scattering albedo  nlay*nbdsw !
    #                             (ssa = ssacw / taucw)                     !
    #    asycw  - real, weighted cloud asymmetry factor          nlay*nbdsw !
    #                             (asy = asycw / ssacw)                     !
    #    cldfrc - real, cloud fraction of grid mean value              nlay !
    #    cldfmc - real, cloud fraction for each sub-column       nlay*ngptsw!
    #                                                                       !
    #                                                                       !
    #  explanation of the method for each value of iswcliq, and iswcice.    !
    #  set up in module "physparam"                                         !
    #                                                                       !
    #     iswcliq=0  : input cloud optical property (tau, ssa, asy).        !
    #                  (used for diagnostic cloud method)                   !
    #     iswcliq>0  : input cloud liq/ice path and effective radius, also  !
    #                  require the user of 'iswcice' to specify the method  !
    #                  used to compute aborption due to water/ice parts.    !
    #  ...................................................................  !
    #                                                                       !
    #     iswcliq=1  : liquid water cloud optical properties are computed   !
    #                  as in hu and stamnes (1993), j. clim., 6, 728-742.   !
    #     iswcliq=2  : updated coeffs for hu and stamnes (1993) by aer      !
    #                  w v3.9-v4.0.                                         !
    #                                                                       !
    #     iswcice used only when iswcliq > 0                                !
    #                  the cloud ice path (g/m2) and ice effective radius   !
    #                  (microns) are inputs.                                !
    #     iswcice=1  : ice cloud optical properties are computed as in      !
    #                  ebert and curry (1992), jgr, 97, 3831-3836.          !
    #     iswcice=2  : ice cloud optical properties are computed as in      !
    #                  streamer v3.0 (2001), key, streamer user's guide,    !
    #                  cooperative institude for meteorological studies,95pp!
    #     iswcice=3  : ice cloud optical properties are computed as in      !
    #                  fu (1996), j. clim., 9.                              !
    #                                                                       !
    #  other cloud control module variables:                                !
    #     isubcsw =0: standard cloud scheme, no sub-col cloud approximation !
    #             >0: mcica sub-col cloud scheme using ipseed as permutation!
    #                 seed for generating rundom numbers                    !
    #                                                                       !
    #  ======================  end of description block  =================  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_cldprtb_data.nc")
    extliq1 = ds["extliq1"].data
    extliq2 = ds["extliq2"].data
    ssaliq1 = ds["ssaliq1"].data
    ssaliq2 = ds["ssaliq2"].data
    asyliq1 = ds["asyliq1"].data
    asyliq2 = ds["asyliq2"].data
    extice2 = ds["extice2"].data
    ssaice2 = ds["ssaice2"].data
    asyice2 = ds["asyice2"].data
    extice3 = ds["extice3"].data
    ssaice3 = ds["ssaice3"].data
    asyice3 = ds["asyice3"].data
    abari = ds["abari"].data
    bbari = ds["bbari"].data
    cbari = ds["cbari"].data
    dbari = ds["dbari"].data
    ebari = ds["ebari"].data
    fbari = ds["fbari"].data
    b0s = ds["b0s"].data
    b1s = ds["b1s"].data
    b0r = ds["b0r"].data
    c0s = ds["c0s"].data
    c0r = ds["c0r"].data
    a0r = ds["a0r"].data
    a1r = ds["a1r"].data
    a0s = ds["a0s"].data
    a1s = ds["a1s"].data

    #  ---  outputs:
    cldfmc = np.zeros((nlay, ngptsw))
    taucw = np.zeros((nlay, nbdsw))
    ssacw = np.ones((nlay, nbdsw))
    asycw = np.zeros((nlay, nbdsw))
    cldfrc = np.zeros(nlay)

    #  ---  locals:
    tauliq = np.zeros(nbands)
    tauice = np.zeros(nbands)
    ssaliq = np.zeros(nbands)
    ssaice = np.zeros(nbands)
    ssaran = np.zeros(nbands)
    ssasnw = np.zeros(nbands)
    asyliq = np.zeros(nbands)
    asyice = np.zeros(nbands)
    asyran = np.zeros(nbands)
    asysnw = np.zeros(nbands)
    cldf = np.zeros(nlay)

    lcloudy = np.zeros((nlay, ngptsw), dtype=bool)

    # Compute cloud radiative properties for a cloudy column.

    if iswcliq > 0:

        for k in range(nlay):
            if cfrac[k] > ftiny:

                #  - Compute optical properties for rain and snow.
                #    For rain: tauran/ssaran/asyran
                #    For snow: tausnw/ssasnw/asysnw
                #  - Calculation of absorption coefficients due to water clouds
                #    For water clouds: tauliq/ssaliq/asyliq
                #  - Calculation of absorption coefficients due to ice clouds
                #    For ice clouds: tauice/ssaice/asyice
                #  - For Prognostic cloud scheme: sum up the cloud optical property:
                #     \f$ taucw=tauliq+tauice+tauran+tausnw \f$
                #     \f$ ssacw=ssaliq+ssaice+ssaran+ssasnw \f$
                #     \f$ asycw=asyliq+asyice+asyran+asysnw \f$

                cldran = cdat1[k]
                cldsnw = cdat3[k]
                refsnw = cdat4[k]
                dgesnw = 1.0315 * refsnw  # for fu's snow formula

                tauran = cldran * a0r

                #  ---  if use fu's formula it needs to be normalized by snow/ice density
                #       !not use snow density = 0.1 g/cm**3 = 0.1 g/(mu * m**2)
                #       use ice density = 0.9167 g/cm**3 = 0.9167 g/(mu * m**2)
                #       1/0.9167 = 1.09087
                #       factor 1.5396=8/(3*sqrt(3)) converts reff to generalized ice particle size
                #       use newer factor value 1.0315
                if cldsnw > 0.0 and refsnw > 10.0:
                    tausnw = cldsnw * 1.09087 * (a0s + a1s / dgesnw)  # fu's formula
                else:
                    tausnw = 0.0

                for ib in range(nbands):
                    ssaran[ib] = tauran * (1.0 - b0r[ib])
                    ssasnw[ib] = tausnw * (1.0 - (b0s[ib] + b1s[ib] * dgesnw))
                    asyran[ib] = ssaran[ib] * c0r[ib]
                    asysnw[ib] = ssasnw[ib] * c0s[ib]

                cldliq = cliqp[k]
                cldice = cicep[k]
                refliq = reliq[k]
                refice = reice[k]

                #  --- ...  calculation of absorption coefficients due to water clouds.

                if cldliq <= 0.0:
                    for ib in range(nbands):
                        tauliq[ib] = 0.0
                        ssaliq[ib] = 0.0
                        asyliq[ib] = 0.0
                else:
                    factor = refliq - 1.5
                    index = max(1, min(57, int(factor))) - 1
                    fint = factor - float(index + 1)

                    if iswcliq == 1:
                        for ib in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq1[index, ib]
                                + fint * (extliq1[index + 1, ib] - extliq1[index, ib]),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq1[index, ib]
                                    + fint
                                    * (ssaliq1[index + 1, ib] - ssaliq1[index, ib]),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq1[index, ib]
                                    + fint
                                    * (asyliq1[index + 1, ib] - asyliq1[index, ib]),
                                ),
                            )

                            tauliq[ib] = cldliq * extcoliq
                            ssaliq[ib] = tauliq[ib] * ssacoliq
                            asyliq[ib] = ssaliq[ib] * asycoliq
                    elif iswcliq == 2:  # use updated coeffs
                        for ib in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq2[index, ib]
                                + fint * (extliq2[index + 1, ib] - extliq2[index, ib]),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq2[index, ib]
                                    + fint
                                    * (ssaliq2[index + 1, ib] - ssaliq2[index, ib]),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq2[index, ib]
                                    + fint
                                    * (asyliq2[index + 1, ib] - asyliq2[index, ib]),
                                ),
                            )

                            tauliq[ib] = cldliq * extcoliq
                            ssaliq[ib] = tauliq[ib] * ssacoliq
                            asyliq[ib] = ssaliq[ib] * asycoliq

                #  --- ...  calculation of absorption coefficients due to ice clouds.

                if cldice <= 0.0:
                    for ib in range(nbands):
                        tauice[ib] = 0.0
                        ssaice[ib] = 0.0
                        asyice[ib] = 0.0
                else:

                    #  --- ...  ebert and curry approach for all particle sizes though somewhat
                    #           unjustified for large ice particles

                    if iswcice == 1:
                        refice = min(130.0, max(13.0, refice))

                        for ib in range(nbands):
                            ia = idxebc[ib] - 1  # eb_&_c band index for ice cloud coeff

                            extcoice = max(0.0, abari[ia] + bbari[ia] / refice)
                            ssacoice = max(
                                0.0, min(1.0, 1.0 - cbari[ia] - dbari[ia] * refice)
                            )
                            asycoice = max(
                                0.0, min(1.0, ebari[ia] + fbari[ia] * refice)
                            )

                            tauice[ib] = cldice * extcoice
                            ssaice[ib] = tauice[ib] * ssacoice
                            asyice[ib] = ssaice[ib] * asycoice

                    #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                    elif iswcice == 2:
                        refice = min(131.0, max(5.0, refice))

                        factor = (refice - 2.0) / 3.0
                        index = max(1, min(42, int(factor))) - 1
                        fint = factor - float(index + 1)

                        for ib in range(nbands):
                            extcoice = max(
                                0.0,
                                extice2[index, ib]
                                + fint * (extice2[index + 1, ib] - extice2[index, ib]),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice2[index, ib]
                                    + fint
                                    * (ssaice2[index + 1, ib] - ssaice2[index, ib]),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice2[index, ib]
                                    + fint
                                    * (asyice2[index + 1, ib] - asyice2[index, ib]),
                                ),
                            )

                            tauice[ib] = cldice * extcoice
                            ssaice[ib] = tauice[ib] * ssacoice
                            asyice[ib] = ssaice[ib] * asycoice

                    #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                    #           (generalized effective size from 5 to 140 microns)
                    elif iswcice == 3:
                        dgeice = max(5.0, min(140.0, 1.0315 * refice))

                        factor = (dgeice - 2.0) / 3.0
                        index = max(1, min(45, int(factor))) - 1
                        fint = factor - float(index + 1)

                        for ib in range(nbands):
                            extcoice = max(
                                0.0,
                                extice3[index, ib]
                                + fint * (extice3[index + 1, ib] - extice3[index, ib]),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice3[index, ib]
                                    + fint
                                    * (ssaice3[index + 1, ib] - ssaice3[index, ib]),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice3[index, ib]
                                    + fint
                                    * (asyice3[index + 1, ib] - asyice3[index, ib]),
                                ),
                            )

                            tauice[ib] = cldice * extcoice
                            ssaice[ib] = tauice[ib] * ssacoice
                            asyice[ib] = ssaice[ib] * asycoice

                for ib in range(nbdsw):
                    jb = nblow + ib - 2
                    taucw[k, ib] = tauliq[jb] + tauice[jb] + tauran + tausnw
                    ssacw[k, ib] = ssaliq[jb] + ssaice[jb] + ssaran[jb] + ssasnw[jb]
                    asycw[k, ib] = asyliq[jb] + asyice[jb] + asyran[jb] + asysnw[jb]

    else:  #  lab_if_iswcliq

        for k in range(nlay):
            if cfrac[k] > ftiny:
                for ib in range(nbdsw):
                    taucw[k, ib] = cdat1[k]
                    ssacw[k, ib] = cdat1[k] * cdat2[k]
                    asycw[k, ib] = ssacw[k, ib] * cdat3[k]

    # -# if physparam::isubcsw > 0, call mcica_subcol() to distribute
    #    cloud properties to each g-point.

    if isubcsw > 0:  # mcica sub-col clouds approx
        cldf = cfrac
        cldf = np.where(cldf < ftiny, 0.0, cldf)

        #  --- ...  call sub-column cloud generator

        lcloudy = mcica_subcol(cldf, nlay, ipseed, dz, delgth, ipt)

        for ig in range(ngptsw):
            for k in range(nlay):
                if lcloudy[k, ig]:
                    cldfmc[k, ig] = 1.0
                else:
                    cldfmc[k, ig] = 0.0

    else:  # non-mcica, normalize cloud
        for k in range(nlay):
            cldfrc[k] = cfrac[k] / cf1

    return taucw, ssacw, asycw, cldfrc, cldfmc


#  This subroutine computes the sub-colum cloud profile flag array.
# \param cldf        layer cloud fraction
# \param nlay        number of model vertical layers
# \param ipseed      permute seed for random num generator
# \param lcloudy     sub-colum cloud profile flag array


def mcica_subcol(cldf, nlay, ipseed, dz, de_lgth, ipt):
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  input variables:                                                size !
    #   cldf    - real, layer cloud fraction                           nlay !
    #   nlay    - integer, number of model vertical layers               1  !
    #   ipseed  - integer, permute seed for random num generator         1  !
    #    ** note : if the cloud generator is called multiple times, need    !
    #              to permute the seed between each call; if between calls  !
    #              for lw and sw, use values differ by the number of g-pts. !
    #    dz    - real, layer thickness (km)                            nlay !
    #    de_lgth-real, layer cloud decorrelation length (km)            1   !
    #                                                                       !
    #  output variables:                                                    !
    #   lcloudy - logical, sub-colum cloud profile flag array    nlay*ngptsw!
    #                                                                       !
    #  other control flags from module variables:                           !
    #     iovrsw    : control flag for cloud overlapping method             !
    #                 =0: random                                            !
    #                 =1: maximum/random overlapping clouds                 !
    #                 =2: maximum overlap cloud                             !
    #                 =3: cloud decorrelation-length overlap method         !
    #                                                                       !
    #  =====================    end of definitions    ====================  !

    ds = xr.open_dataset("../lookupdata/rand2d_sw.nc")
    rand2d = ds["rand2d"][ipt, :].data

    #  ---  outputs:
    lcloudy = np.zeros((nlay, ngptsw))

    #  ---  locals:
    cdfunc = np.zeros((nlay, ngptsw))

    #  --- ...  sub-column set up according to overlapping assumption

    if iovrsw == 1:  # max-ran overlap

        k1 = 0
        for n in range(ngptsw):
            for k in range(nlay):
                cdfunc[k, n] = rand2d[k1]
                k1 = k1 + 1

        #  ---  first pick a random number for bottom/top layer.
        #       then walk up the column: (aer's code)
        #       if layer below is cloudy, use the same rand num in the layer below
        #       if layer below is clear,  use a new random number

        #  ---  from bottom up
        for k in range(1, nlay):
            k1 = k - 1
            tem1 = 1.0 - cldf[k1]

            for n in range(ngptsw):
                if cdfunc[k1, n] > tem1:
                    cdfunc[k, n] = cdfunc[k1, n]
                else:
                    cdfunc[k, n] = cdfunc[k, n] * tem1

    #  --- ...  generate subcolumns for homogeneous clouds

    for k in range(nlay):
        tem1 = 1.0 - cldf[k]

        for n in range(ngptsw):
            lcloudy[k, n] = cdfunc[k, n] >= tem1

    return lcloudy
