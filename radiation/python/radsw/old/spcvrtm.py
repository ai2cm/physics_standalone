import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radsw_param import (
    nbdsw,
    ngptsw,
    NGB,
    nblow,
    idxsfc,
    ftiny,
    flimit,
    oneminus,
    ntbmx,
    bpade,
    eps,
    nuvb,
)
from radphysparam import iswmode
from util import compare_data

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

invars = [
    "ssolar",
    "cosz1",
    "sntz1",
    "albbm",
    "albdf",
    "sfluxzen",
    "cldfmc",
    "zcf1",
    "zcf0",
    "taug",
    "taur",
    "tauae",
    "ssaae",
    "asyae",
    "taucw",
    "ssacw",
    "asycw",
    "nlay",
    "nlp1",
    "exp_tbl",
]

indict = dict()
for var in invars:
    indict[var] = serializer.read(
        var, serializer.savepoint["swrad-spcvrtm-input-000000"]
    )


def spcvrtm(
    ssolar,
    cosz,
    sntz,
    albbm,
    albdf,
    sfluxzen,
    cldfmc,
    cf1,
    cf0,
    taug,
    taur,
    tauae,
    ssaae,
    asyae,
    taucw,
    ssacw,
    asycw,
    nlay,
    nlp1,
    exp_tbl,
):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    #   purpose:  computes the shortwave radiative fluxes using two-stream  !
    #             method of h. barker and mcica, the monte-carlo independent!
    #             column approximation, for the representation of sub-grid  !
    #             cloud variability (i.e. cloud overlap).                   !
    #                                                                       !
    #   subprograms called:  vrtqdr                                         !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                        size  !
    #    ssolar  - real, incoming solar flux at top                    1    !
    #    cosz    - real, cosine solar zenith angle                     1    !
    #    sntz    - real, secant solar zenith angle                     1    !
    #    albbm   - real, surface albedo for direct beam radiation      2    !
    #    albdf   - real, surface albedo for diffused radiation         2    !
    #    sfluxzen- real, spectral distribution of incoming solar flux ngptsw!
    #    cldfmc  - real, layer cloud fraction for g-point        nlay*ngptsw!
    #    cf1     - real, >0: cloudy sky, otherwise: clear sky          1    !
    #    cf0     - real, =1-cf1                                        1    !
    #    taug    - real, spectral optical depth for gases        nlay*ngptsw!
    #    taur    - real, optical depth for rayleigh scattering   nlay*ngptsw!
    #    tauae   - real, aerosols optical depth                  nlay*nbdsw !
    #    ssaae   - real, aerosols single scattering albedo       nlay*nbdsw !
    #    asyae   - real, aerosols asymmetry factor               nlay*nbdsw !
    #    taucw   - real, weighted cloud optical depth            nlay*nbdsw !
    #    ssacw   - real, weighted cloud single scat albedo       nlay*nbdsw !
    #    asycw   - real, weighted cloud asymmetry factor         nlay*nbdsw !
    #    nlay,nlp1 - integer,  number of layers/levels                 1    !
    #                                                                       !
    #  output variables:                                                    !
    #    fxupc   - real, tot sky upward flux                     nlp1*nbdsw !
    #    fxdnc   - real, tot sky downward flux                   nlp1*nbdsw !
    #    fxup0   - real, clr sky upward flux                     nlp1*nbdsw !
    #    fxdn0   - real, clr sky downward flux                   nlp1*nbdsw !
    #    ftoauc  - real, tot sky toa upwd flux                         1    !
    #    ftoau0  - real, clr sky toa upwd flux                         1    !
    #    ftoadc  - real, toa downward (incoming) solar flux            1    !
    #    fsfcuc  - real, tot sky sfc upwd flux                         1    !
    #    fsfcu0  - real, clr sky sfc upwd flux                         1    !
    #    fsfcdc  - real, tot sky sfc dnwd flux                         1    !
    #    fsfcd0  - real, clr sky sfc dnwd flux                         1    !
    #    sfbmc   - real, tot sky sfc dnwd beam flux (nir/uv+vis)       2    !
    #    sfdfc   - real, tot sky sfc dnwd diff flux (nir/uv+vis)       2    !
    #    sfbm0   - real, clr sky sfc dnwd beam flux (nir/uv+vis)       2    !
    #    sfdf0   - real, clr sky sfc dnwd diff flux (nir/uv+vis)       2    !
    #    suvbfc  - real, tot sky sfc dnwd uv-b flux                    1    !
    #    suvbf0  - real, clr sky sfc dnwd uv-b flux                    1    !
    #                                                                       !
    #  internal variables:                                                  !
    #    zrefb   - real, direct beam reflectivity for clear/cloudy    nlp1  !
    #    zrefd   - real, diffuse reflectivity for clear/cloudy        nlp1  !
    #    ztrab   - real, direct beam transmissivity for clear/cloudy  nlp1  !
    #    ztrad   - real, diffuse transmissivity for clear/cloudy      nlp1  !
    #    zldbt   - real, layer beam transmittance for clear/cloudy    nlp1  !
    #    ztdbt   - real, lev total beam transmittance for clr/cld     nlp1  !
    #                                                                       !
    #  control parameters in module "physparam"                             !
    #    iswmode - control flag for 2-stream transfer schemes               !
    #              = 1 delta-eddington    (joseph et al., 1976)             !
    #              = 2 pifm               (zdunkowski et al., 1980)         !
    #              = 3 discrete ordinates (liou, 1973)                      !
    #                                                                       !
    #  *******************************************************************  !
    #  original code description                                            !
    #                                                                       !
    #  method:                                                              !
    #  -------                                                              !
    #     standard delta-eddington, p.i.f.m., or d.o.m. layer calculations. !
    #     kmodts  = 1 eddington (joseph et al., 1976)                       !
    #             = 2 pifm (zdunkowski et al., 1980)                        !
    #             = 3 discrete ordinates (liou, 1973)                       !
    #                                                                       !
    #  modifications:                                                       !
    #  --------------                                                       !
    #   original: h. barker                                                 !
    #   revision: merge with rrtmg_sw: j.-j.morcrette, ecmwf, feb 2003      !
    #   revision: add adjustment for earth/sun distance:mjiacono,aer,oct2003!
    #   revision: bug fix for use of palbp and palbd: mjiacono, aer, nov2003!
    #   revision: bug fix to apply delta scaling to clear sky: aer, dec2004 !
    #   revision: code modified so that delta scaling is not done in cloudy !
    #             profiles if routine cldprop is used; delta scaling can be !
    #             applied by swithcing code below if cldprop is not used to !
    #             get cloud properties. aer, jan 2005                       !
    #   revision: uniform formatting for rrtmg: mjiacono, aer, jul 2006     !
    #   revision: use exponential lookup table for transmittance: mjiacono, !
    #             aer, aug 2007                                             !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !

    #  ---  constant parameters:
    zcrit = 0.9999995  # thresold for conservative scattering
    zsr3 = np.sqrt(3.0)
    od_lo = 0.06
    eps1 = 1.0e-8

    #  ---  outputs:
    fxupc = np.zeros((nlp1, nbdsw))
    fxdnc = np.zeros((nlp1, nbdsw))
    fxup0 = np.zeros((nlp1, nbdsw))
    fxdn0 = np.zeros((nlp1, nbdsw))

    sfbmc = np.zeros(2)
    sfdfc = np.zeros(2)
    sfbm0 = np.zeros(2)
    sfdf0 = np.zeros(2)

    #  ---  locals:
    ztaus = np.zeros(nlay)
    zssas = np.zeros(nlay)
    zasys = np.zeros(nlay)
    zldbt0 = np.zeros(nlay)

    zrefb = np.zeros(nlp1)
    zrefd = np.zeros(nlp1)
    ztrab = np.zeros(nlp1)
    ztrad = np.zeros(nlp1)
    ztdbt = np.zeros(nlp1)
    zldbt = np.zeros(nlp1)
    zfu = np.zeros(nlp1)
    zfd = np.zeros(nlp1)

    #  --- ...  loop over all g-points in each band

    for jg in range(ngptsw):
        jb = NGB[jg] - 1
        ib = jb + 1 - nblow
        ibd = idxsfc[jb - 15] - 1  # spectral band index

        if jg == 0:
            print(f"jb = {jb}")
            print(f"ib = {ib}")
            print(f"ibd = {ibd}")

        zsolar = ssolar * sfluxzen[jg]

        #  --- ...  set up toa direct beam and surface values (beam and diff)

        ztdbt[nlp1 - 1] = 1.0
        ztdbt0 = 1.0

        zldbt[0] = 0.0
        if ibd != -1:
            zrefb[0] = albbm[ibd]
            zrefd[0] = albdf[ibd]
        else:
            zrefb[0] = 0.5 * (albbm[0] + albbm[1])
            zrefd[0] = 0.5 * (albdf[0] + albdf[1])

        ztrab[0] = 0.0
        ztrad[0] = 0.0

        # -# Compute clear-sky optical parameters, layer reflectance and
        #    transmittance.
        #    - Set up toa direct beam and surface values (beam and diff)
        #    - Delta scaling for clear-sky condition
        #    - General two-stream expressions for physparam::iswmode
        #    - Compute homogeneous reflectance and transmittance for both
        #      conservative and non-conservative scattering
        #    - Pre-delta-scaling clear and cloudy direct beam transmittance
        #    - Call swflux() to compute the upward and downward radiation fluxes

        for k in range(nlay - 1, -1, -1):
            kp = k + 1

            ztau0 = max(ftiny, taur[k, jg] + taug[k, jg] + tauae[k, ib])
            zssa0 = taur[k, jg] + tauae[k, ib] * ssaae[k, ib]
            zasy0 = asyae[k, ib] * ssaae[k, ib] * tauae[k, ib]
            zssaw = min(oneminus, zssa0 / ztau0)
            zasyw = zasy0 / max(ftiny, zssa0)

            #  --- ...  saving clear-sky quantities for later total-sky usage
            ztaus[k] = ztau0
            zssas[k] = zssa0
            zasys[k] = zasy0

            #  --- ...  delta scaling for clear-sky condition
            za1 = zasyw * zasyw
            za2 = zssaw * za1

            ztau1 = (1.0 - za2) * ztau0
            zssa1 = (zssaw - za2) / (1.0 - za2)
            zasy1 = zasyw / (1.0 + zasyw)  # to reduce truncation error
            zasy3 = 0.75 * zasy1

            #  --- ...  general two-stream expressions
            if iswmode == 1:
                zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                zgam3 = 0.5 - zasy3 * cosz
            elif iswmode == 2:  # pifm
                zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                zgam3 = 0.5 - zasy3 * cosz
            elif iswmode == 3:  # discrete ordinates
                zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

            zgam4 = 1.0 - zgam3

            #  --- ...  compute homogeneous reflectance and transmittance

            if zssaw >= zcrit:  # for conservative scattering
                za1 = zgam1 * cosz - zgam3
                za2 = zgam1 * ztau1

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1 = min(ztau1 * sntz, 500.0)
                if zb1 <= od_lo:
                    zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                else:
                    ftind = zb1 / (bpade + zb1)
                    itind = int(ftind * ntbmx + 0.5)
                    zb2 = exp_tbl[itind]

                #      ...  collimated beam
                zrefb[kp] = max(0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2)))
                ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                #      ...      isotropic incidence
                zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd[kp]))

            else:  # for non-conservative scattering
                za1 = zgam1 * zgam4 + zgam2 * zgam3
                za2 = zgam1 * zgam3 + zgam2 * zgam4
                zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                zrk2 = 2.0 * zrk

                zrp = zrk * cosz
                zrp1 = 1.0 + zrp
                zrm1 = 1.0 - zrp
                zrpp1 = 1.0 - zrp * zrp
                zrpp = np.copysign(
                    max(flimit, abs(zrpp1)), zrpp1
                )  # avoid numerical singularity
                zrkg1 = zrk + zgam1
                zrkg3 = zrk * zgam3
                zrkg4 = zrk * zgam4

                zr1 = zrm1 * (za2 + zrkg3)
                zr2 = zrp1 * (za2 - zrkg3)
                zr3 = zrk2 * (zgam3 - za2 * cosz)
                zr4 = zrpp * zrkg1
                zr5 = zrpp * (zrk - zgam1)

                zt1 = zrp1 * (za1 + zrkg4)
                zt2 = zrm1 * (za1 - zrkg4)
                zt3 = zrk2 * (zgam4 + za1 * cosz)

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1 = min(zrk * ztau1, 500.0)
                if zb1 <= od_lo:
                    zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                else:
                    ftind = zb1 / (bpade + zb1)
                    itind = int(ftind * ntbmx + 0.5)
                    zexm1 = exp_tbl[itind]

                zexp1 = 1.0 / zexm1

                zb2 = min(sntz * ztau1, 500.0)
                if zb2 <= od_lo:
                    zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                else:
                    ftind = zb2 / (bpade + zb2)
                    itind = int(ftind * ntbmx + 0.5)
                    zexm2 = exp_tbl[itind]

                zexp2 = 1.0 / zexm2
                ze1r45 = zr4 * zexp1 + zr5 * zexm1

                #      ...      collimated beam
                if ze1r45 >= -eps1 and ze1r45 <= eps1:
                    zrefb[kp] = eps1
                    ztrab[kp] = zexm2
                else:
                    zden1 = zssa1 / ze1r45
                    zrefb[kp] = max(
                        0.0, min(1.0, (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1)
                    )
                    ztrab[kp] = max(
                        0.0,
                        min(
                            1.0,
                            zexm2
                            * (1.0 - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2) * zden1),
                        ),
                    )

                #      ...      diffuse beam
                zden1 = zr4 / (ze1r45 * zrkg1)
                zrefd[kp] = max(0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1))
                ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

            #  --- ...  direct beam transmittance. use exponential lookup table
            #           for transmittance, or expansion of exponential for low
            #           optical depth

            zr1 = ztau1 * sntz
            if zr1 <= od_lo:
                zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
            else:
                ftind = zr1 / (bpade + zr1)
                itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                zexp3 = exp_tbl[itind]

            ztdbt[k] = zexp3 * ztdbt[kp]
            zldbt[kp] = zexp3

            #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
            #           (must use 'orig', unscaled cloud optical depth)

            zr1 = ztau0 * sntz
            if zr1 <= od_lo:
                zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
            else:
                ftind = zr1 / (bpade + zr1)
                itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                zexp4 = exp_tbl[itind]

            zldbt0[k] = zexp4
            ztdbt0 = zexp4 * ztdbt0

        if jg == 0:
            print(f"zrefb = {zrefb}")
            print(f"zrefd = {zrefd}")
            print(f"ztrab = {ztrab}")
            print(f"ztrad = {ztrad}")
            print(f"zldbt = {zldbt}")
            print(f"ztdbt = {ztdbt}")

        zfu, zfd = vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1)
        if jg == 0:
            print(f"zfu = {zfu}")
            print(f"zfd = {zfd}")

        #  --- ...  compute upward and downward fluxes at levels
        for k in range(nlp1):
            fxup0[k, ib] = fxup0[k, ib] + zsolar * zfu[k]
            fxdn0[k, ib] = fxdn0[k, ib] + zsolar * zfd[k]

        # --- ...  surface downward beam/diffuse flux components
        zb1 = zsolar * ztdbt0
        zb2 = zsolar * (zfd[0] - ztdbt0)

        if ibd != -1:
            sfbm0[ibd] = sfbm0[ibd] + zb1
            sfdf0[ibd] = sfdf0[ibd] + zb2
        else:
            zf1 = 0.5 * zb1
            zf2 = 0.5 * zb2
            sfbm0[0] = sfbm0[0] + zf1
            sfdf0[0] = sfdf0[0] + zf2
            sfbm0[1] = sfbm0[1] + zf1
            sfdf0[1] = sfdf0[1] + zf2

        # -# Compute total sky optical parameters, layer reflectance and
        #    transmittance.
        #    - Set up toa direct beam and surface values (beam and diff)
        #    - Delta scaling for total-sky condition
        #    - General two-stream expressions for physparam::iswmode
        #    - Compute homogeneous reflectance and transmittance for
        #      conservative scattering and non-conservative scattering
        #    - Pre-delta-scaling clear and cloudy direct beam transmittance
        #    - Call swflux() to compute the upward and downward radiation fluxes

        if cf1 > eps:

            #  --- ...  set up toa direct beam and surface values (beam and diff)
            ztdbt0 = 1.0
            zldbt[0] = 0.0

            for k in range(nlay - 1, -1, -1):
                kp = k + 1
                if cldfmc[k, jg] > ftiny:  # it is a cloudy-layer

                    ztau0 = ztaus[k] + taucw[k, ib]
                    zssa0 = zssas[k] + ssacw[k, ib]
                    zasy0 = zasys[k] + asycw[k, ib]
                    zssaw = min(oneminus, zssa0 / ztau0)
                    zasyw = zasy0 / max(ftiny, zssa0)

                    #  --- ...  delta scaling for total-sky condition
                    za1 = zasyw * zasyw
                    za2 = zssaw * za1

                    ztau1 = (1.0 - za2) * ztau0
                    zssa1 = (zssaw - za2) / (1.0 - za2)
                    zasy1 = zasyw / (1.0 + zasyw)
                    zasy3 = 0.75 * zasy1

                    #  --- ...  general two-stream expressions
                    if iswmode == 1:
                        zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                        zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                        zgam3 = 0.5 - zasy3 * cosz
                    elif iswmode == 2:  # pifm
                        zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                        zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                        zgam3 = 0.5 - zasy3 * cosz
                    elif iswmode == 3:  # discrete ordinates
                        zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                        zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                        zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                    zgam4 = 1.0 - zgam3

                    #  --- ...  compute homogeneous reflectance and transmittance

                    if zssaw >= zcrit:  # for conservative scattering
                        za1 = zgam1 * cosz - zgam3
                        za2 = zgam1 * ztau1

                        #  --- ...  use exponential lookup table for transmittance, or expansion
                        #           of exponential for low optical depth

                        zb1 = min(ztau1 * sntz, 500.0)
                        if zb1 <= od_lo:
                            zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                        else:
                            ftind = zb1 / (bpade + zb1)
                            itind = int(ftind * ntbmx + 0.5)
                            zb2 = exp_tbl[itind]

                        #      ...  collimated beam
                        zrefb[kp] = max(
                            0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                        )
                        ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                        #      ...  isotropic incidence
                        zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                        ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd(kp)))

                    else:  # for non-conservative scattering
                        za1 = zgam1 * zgam4 + zgam2 * zgam3
                        za2 = zgam1 * zgam3 + zgam2 * zgam4
                        zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                        zrk2 = 2.0 * zrk

                        zrp = zrk * cosz
                        zrp1 = 1.0 + zrp
                        zrm1 = 1.0 - zrp
                        zrpp1 = 1.0 - zrp * zrp
                        zrpp = np.copysign(
                            max(flimit, abs(zrpp1)), zrpp1
                        )  # avoid numerical singularity
                        zrkg1 = zrk + zgam1
                        zrkg3 = zrk * zgam3
                        zrkg4 = zrk * zgam4

                        zr1 = zrm1 * (za2 + zrkg3)
                        zr2 = zrp1 * (za2 - zrkg3)
                        zr3 = zrk2 * (zgam3 - za2 * cosz)
                        zr4 = zrpp * zrkg1
                        zr5 = zrpp * (zrk - zgam1)

                        zt1 = zrp1 * (za1 + zrkg4)
                        zt2 = zrm1 * (za1 - zrkg4)
                        zt3 = zrk2 * (zgam4 + za1 * cosz)

                        #  --- ...  use exponential lookup table for transmittance, or expansion
                        #           of exponential for low optical depth

                        zb1 = min(zrk * ztau1, 500.0)
                        if zb1 <= od_lo:
                            zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                        else:
                            ftind = zb1 / (bpade + zb1)
                            itind = int(ftind * ntbmx + 0.5)
                            zexm1 = exp_tbl[itind]

                        zexp1 = 1.0 / zexm1

                        zb2 = min(ztau1 * sntz, 500.0)
                        if zb2 <= od_lo:
                            zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                        else:
                            ftind = zb2 / (bpade + zb2)
                            itind = int(ftind * ntbmx + 0.5)
                            zexm2 = exp_tbl[itind]

                        zexp2 = 1.0 / zexm2
                        ze1r45 = zr4 * zexp1 + zr5 * zexm1

                        #      ...  collimated beam
                        if ze1r45 >= -eps1 and ze1r45 <= eps1:
                            zrefb[kp] = eps1
                            ztrab[kp] = zexm2
                        else:
                            zden1 = zssa1 / ze1r45
                            zrefb[kp] = max(
                                0.0,
                                min(
                                    1.0,
                                    (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1,
                                ),
                            )
                            ztrab[kp] = max(
                                0.0,
                                min(
                                    1.0,
                                    zexm2
                                    * (
                                        1.0
                                        - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2)
                                        * zden1
                                    ),
                                ),
                            )

                        #      ...  diffuse beam
                        zden1 = zr4 / (ze1r45 * zrkg1)
                        zrefd[kp] = max(0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1))
                        ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

                    #  --- ...  direct beam transmittance. use exponential lookup table
                    #           for transmittance, or expansion of exponential for low
                    #           optical depth

                    zr1 = ztau1 * sntz
                    if zr1 <= od_lo:
                        zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
                    else:
                        ftind = zr1 / (bpade + zr1)
                        itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                        zexp3 = exp_tbl[itind]

                    zldbt[kp] = zexp3
                    ztdbt[k] = zexp3 * ztdbt[kp]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    #           (must use 'orig', unscaled cloud optical depth)

                    zr1 = ztau0 * sntz
                    if zr1 <= od_lo:
                        zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                    else:
                        ftind = zr1 / (bpade + zr1)
                        itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                        zexp4 = exp_tbl[itind]

                    ztdbt0 = zexp4 * ztdbt0

                else:  # if_cldfmc_block  ---  it is a clear layer

                    #  --- ...  direct beam transmittance
                    ztdbt[k] = zldbt[kp] * ztdbt[kp]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    ztdbt0 = zldbt0[k] * ztdbt0

            #  --- ...  perform vertical quadrature

            zfu, zfd = vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1)

            #  --- ...  compute upward and downward fluxes at levels
            for k in range(nlp1):
                fxupc[k, ib] = fxupc[k, ib] + zsolar * zfu[k]
                fxdnc[k, ib] = fxdnc[k, ib] + zsolar * zfd[k]

            #  -# Process and save outputs.
            # --- ...  surface downward beam/diffused flux components
            zb1 = zsolar * ztdbt0
            zb2 = zsolar * (zfd[0] - ztdbt0)

            if ibd != -1:
                sfbmc[ibd] = sfbmc[ibd] + zb1
                sfdfc[ibd] = sfdfc[ibd] + zb2
            else:
                zf1 = 0.5 * zb1
                zf2 = 0.5 * zb2
                sfbmc[0] = sfbmc[0] + zf1
                sfdfc[0] = sfdfc[0] + zf2
                sfbmc[1] = sfbmc[1] + zf1
                sfdfc[1] = sfdfc[1] + zf2

    #  --- ...  end of g-point loop
    ftoadc = 0
    ftoauc = 0
    ftoau0 = 0
    fsfcu0 = 0
    fsfcuc = 0
    fsfcd0 = 0
    fsfcdc = 0

    for ib in range(nbdsw):
        ftoadc = ftoadc + fxdn0[nlp1 - 1, ib]
        ftoau0 = ftoau0 + fxup0[nlp1 - 1, ib]
        fsfcu0 = fsfcu0 + fxup0[0, ib]
        fsfcd0 = fsfcd0 + fxdn0[0, ib]

    # --- ...  uv-b surface downward flux
    ibd = nuvb - nblow
    suvbf0 = fxdn0[0, ibd]

    if cf1 <= eps:  # clear column, set total-sky=clear-sky fluxes
        for ib in range(nbdsw):
            for k in range(nlp1):
                fxupc[k, ib] = fxup0[k, ib]
                fxdnc[k, ib] = fxdn0[k, ib]

        ftoauc = ftoau0
        fsfcuc = fsfcu0
        fsfcdc = fsfcd0

        # --- ...  surface downward beam/diffused flux components
        sfbmc[0] = sfbm0[0]
        sfdfc[0] = sfdf0[0]
        sfbmc[1] = sfbm0[1]
        sfdfc[1] = sfdf0[1]

        # --- ...  uv-b surface downward flux
        suvbfc = suvbf0
    else:  # cloudy column, compute total-sky fluxes
        for ib in range(nbdsw):
            ftoauc = ftoauc + fxupc[nlp1 - 1, ib]
            fsfcuc = fsfcuc + fxupc[0, ib]
            fsfcdc = fsfcdc + fxdnc[0, ib]

        # --- ...  uv-b surface downward flux
        suvbfc = fxdnc[0, ibd]

    return (
        fxupc,
        fxdnc,
        fxup0,
        fxdn0,
        ftoauc,
        ftoau0,
        ftoadc,
        fsfcuc,
        fsfcu0,
        fsfcdc,
        fsfcd0,
        sfbmc,
        sfdfc,
        sfbm0,
        sfdf0,
        suvbfc,
        suvbf0,
    )


# This subroutine is called by spcvrtc() and spcvrtm(), and computes
# the upward and downward radiation fluxes.
# \param zrefb           layer direct beam reflectivity
# \param zrefd           layer diffuse reflectivity
# \param ztrab           layer direct beam transmissivity
# \param ztrad           layer diffuse transmissivity
# \param zldbt           layer mean beam transmittance
# \param ztdbt           total beam transmittance at levels
# \param NLAY, NLP1      number of layers/levels
# \param zfu             upward flux at layer interface
# \param zfd             downward flux at layer interface
# \section General_swflux General Algorithm
def vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    #   purpose:  computes the upward and downward radiation fluxes         !
    #                                                                       !
    #   interface:  "vrtqdr" is called by "spcvrc" and "spcvrm"             !
    #                                                                       !
    #   subroutines called : none                                           !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  input variables:                                                     !
    #    zrefb(NLP1)     - layer direct beam reflectivity                   !
    #    zrefd(NLP1)     - layer diffuse reflectivity                       !
    #    ztrab(NLP1)     - layer direct beam transmissivity                 !
    #    ztrad(NLP1)     - layer diffuse transmissivity                     !
    #    zldbt(NLP1)     - layer mean beam transmittance                    !
    #    ztdbt(NLP1)     - total beam transmittance at levels               !
    #    NLAY, NLP1      - number of layers/levels                          !
    #                                                                       !
    #  output variables:                                                    !
    #    zfu  (NLP1)     - upward flux at layer interface                   !
    #    zfd  (NLP1)     - downward flux at layer interface                 !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !

    #  ---  outputs:
    zfu = np.zeros(nlp1)
    zfd = np.zeros(nlp1)

    #  ---  locals:
    zrupb = np.zeros(nlp1)
    zrupd = np.zeros(nlp1)
    zrdnd = np.zeros(nlp1)
    ztdn = np.zeros(nlp1)

    # -# Link lowest layer with surface.
    zrupb[0] = zrefb[0]  # direct beam
    zrupd[0] = zrefd[0]  # diffused

    # -# Pass from bottom to top.
    for k in range(nlay):
        kp = k + 1

        zden1 = 1.0 / (1.0 - zrupd[k] * zrefd[kp])
        zrupb[kp] = (
            zrefb[kp]
            + (ztrad[kp] * ((ztrab[kp] - zldbt[kp]) * zrupd[k] + zldbt[kp] * zrupb[k]))
            * zden1
        )
        zrupd[kp] = zrefd[kp] + ztrad[kp] * ztrad[kp] * zrupd[k] * zden1

    # -# Upper boundary conditions
    ztdn[nlp1 - 1] = 1.0
    zrdnd[nlp1 - 1] = 0.0
    ztdn[nlay - 1] = ztrab[nlp1 - 1]
    zrdnd[nlay - 1] = zrefd[nlp1 - 1]

    # -# Pass from top to bottom
    for k in range(nlay - 1, 0, -1):
        zden1 = 1.0 / (1.0 - zrefd[k] * zrdnd[k])
        ztdn[k - 1] = (
            ztdbt[k] * ztrab[k]
            + (ztrad[k] * ((ztdn[k] - ztdbt[k]) + ztdbt[k] * zrefb[k] * zrdnd[k]))
            * zden1
        )
        zrdnd[k - 1] = zrefd[k] + ztrad[k] * ztrad[k] * zrdnd[k] * zden1

    # -# Up and down-welling fluxes at levels.
    for k in range(nlp1):
        zden1 = 1.0 / (1.0 - zrdnd[k] * zrupd[k])
        zfu[k] = (ztdbt[k] * zrupb[k] + (ztdn[k] - ztdbt[k]) * zrupd[k]) * zden1
        zfd[k] = (
            ztdbt[k] + (ztdn[k] - ztdbt[k] + ztdbt[k] * zrupb[k] * zrdnd[k]) * zden1
        )

    return zfu, zfd


(
    fxupc,
    fxdnc,
    fxup0,
    fxdn0,
    ftoauc,
    ftoau0,
    ftoadc,
    fsfcuc,
    fsfcu0,
    fsfcdc,
    fsfcd0,
    sfbmc,
    sfdfc,
    sfbm0,
    sfdf0,
    suvbfc,
    suvbf0,
) = spcvrtm(
    indict["ssolar"],
    indict["cosz1"],
    indict["sntz1"],
    indict["albbm"],
    indict["albdf"],
    indict["sfluxzen"],
    indict["cldfmc"],
    indict["zcf1"],
    indict["zcf0"],
    indict["taug"],
    indict["taur"],
    indict["tauae"],
    indict["ssaae"],
    indict["asyae"],
    indict["taucw"],
    indict["ssacw"],
    indict["asycw"],
    indict["nlay"][0],
    indict["nlp1"][0],
    indict["exp_tbl"],
)

outdict = dict()
outdict["fxupc"] = fxupc
outdict["fxdnc"] = fxdnc
outdict["fxup0"] = fxup0
outdict["fxdn0"] = fxdn0
outdict["ftoauc"] = ftoauc
outdict["ftoau0"] = ftoau0
outdict["ftoadc"] = ftoadc
outdict["fsfcuc"] = fsfcuc
outdict["fsfcu0"] = fsfcu0
outdict["fsfcdc"] = fsfcdc
outdict["fsfcd0"] = fsfcd0
outdict["sfbmc"] = sfbmc
outdict["sfdfc"] = sfdfc
outdict["sfbm0"] = sfbm0
outdict["sfdf0"] = sfdf0
outdict["suvbfc"] = suvbfc
outdict["suvbf0"] = suvbf0

outvars = [
    "fxupc",
    "fxdnc",
    "fxup0",
    "fxdn0",
    "ftoauc",
    "ftoau0",
    "ftoadc",
    "fsfcuc",
    "fsfcu0",
    "fsfcdc",
    "fsfcd0",
    "sfbmc",
    "sfdfc",
    "sfbm0",
    "sfdf0",
    "suvbfc",
    "suvbf0",
]

valdict = dict()
for var in outvars:
    valdict[var] = serializer.read(
        var, serializer.savepoint["swrad-spcvrtm-output-000000"]
    )

compare_data(outdict, valdict)
