import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "..")
from radsw_param import (
    ngptsw,
    nblow,
    nbhgh,
    nspa,
    nspb,
    ng,
    ngs,
    oneminus,
    NG16,
    NG17,
    NG18,
    NG19,
    NG20,
    NG21,
    NG22,
    NG23,
    NG24,
    NG25,
    NG26,
    NG27,
    NG28,
    NG29,
    NS16,
    NS17,
    NS18,
    NS19,
    NS20,
    NS21,
    NS22,
    NS23,
    NS24,
    NS25,
    NS26,
    NS27,
    NS28,
    NS29,
)
from util import compare_data
from config import *

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

invars = [
    "colamt",
    "colmol",
    "fac00",
    "fac01",
    "fac10",
    "fac11",
    "jp",
    "jt",
    "jt1",
    "laytrop",
    "forfac",
    "forfrac",
    "indfor",
    "selffac",
    "selffrac",
    "indself",
    "NLAY",
]

indict = dict()
for var in invars:
    if var == "NLAY":
        indict[var] = serializer.read(
            var, serializer.savepoint["swrad-taumol-input-000000"]
        )[0]
    else:
        indict[var] = serializer.read(
            var, serializer.savepoint["swrad-taumol-input-000000"]
        )[10, ...]

ibx = [1, 1, 1, 2, 2, 3, 4, 3, 5, 4, 5, 6, 2, 7]

ds = xr.open_dataset("../lookupdata/radsw_sflux_data.nc")
strrat = ds["strrat"].data
specwt = ds["specwt"].data
layreffr = ds["layreffr"].data
ix1 = ds["ix1"].data
ix2 = ds["ix2"].data
ibx = ds["ibx"].data
sfluxref01 = ds["sfluxref01"].data
sfluxref02 = ds["sfluxref02"].data
sfluxref03 = ds["sfluxref03"].data
scalekur = ds["scalekur"].data


def taumol(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
):
    #  ==================   program usage description   ==================  !
    #                                                                       !
    #  description:                                                         !
    #    calculate optical depths for gaseous absorption and rayleigh       !
    #    scattering.                                                        !
    #                                                                       !
    #  subroutines called: taugb## (## = 16 - 29)                           !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                         size !
    #    colamt  - real, column amounts of absorbing gases the index        !
    #                    are for h2o, co2, o3, n2o, ch4, and o2,            !
    #                    respectively (molecules/cm**2)          nlay*maxgas!
    #    colmol  - real, total column amount (dry air+water vapor)     nlay !
    #    facij   - real, for each layer, these are factors that are         !
    #                    needed to compute the interpolation factors        !
    #                    that multiply the appropriate reference k-         !
    #                    values.  a value of 0/1 for i,j indicates          !
    #                    that the corresponding factor multiplies           !
    #                    reference k-value for the lower/higher of the      !
    #                    two appropriate temperatures, and altitudes,       !
    #                    respectively.                                 naly !
    #    jp      - real, the index of the lower (in altitude) of the        !
    #                    two appropriate ref pressure levels needed         !
    #                    for interpolation.                            nlay !
    #    jt, jt1 - integer, the indices of the lower of the two approp      !
    #                    ref temperatures needed for interpolation (for     !
    #                    pressure levels jp and jp+1, respectively)    nlay !
    #    laytrop - integer, tropopause layer index                       1  !
    #    forfac  - real, scale factor needed to foreign-continuum.     nlay !
    #    forfrac - real, factor needed for temperature interpolation   nlay !
    #    indfor  - integer, index of the lower of the two appropriate       !
    #                    reference temperatures needed for foreign-         !
    #                    continuum interpolation                       nlay !
    #    selffac - real, scale factor needed to h2o self-continuum.    nlay !
    #    selffrac- real, factor needed for temperature interpolation        !
    #                    of reference h2o self-continuum data          nlay !
    #    indself - integer, index of the lower of the two appropriate       !
    #                    reference temperatures needed for the self-        !
    #                    continuum interpolation                       nlay !
    #    nlay    - integer, number of vertical layers                    1  !
    #                                                                       !
    #  output:                                                              !
    #    sfluxzen- real, spectral distribution of incoming solar flux ngptsw!
    #    taug    - real, spectral optical depth for gases        nlay*ngptsw!
    #    taur    - real, opt depth for rayleigh scattering       nlay*ngptsw!
    #                                                                       !
    #  ===================================================================  !
    #  ************     original subprogram description    ***************  !
    #                                                                       !
    #                  optical depths developed for the                     !
    #                                                                       !
    #                rapid radiative transfer model (rrtm)                  !
    #                                                                       !
    #            atmospheric and environmental research, inc.               !
    #                        131 hartwell avenue                            !
    #                        lexington, ma 02421                            !
    #                                                                       !
    #                                                                       !
    #                           eli j. mlawer                               !
    #                         jennifer delamere                             !
    #                         steven j. taubman                             !
    #                         shepard a. clough                             !
    #                                                                       !
    #                                                                       !
    #                                                                       !
    #                       email:  mlawer@aer.com                          !
    #                       email:  jdelamer@aer.com                        !
    #                                                                       !
    #        the authors wish to acknowledge the contributions of the       !
    #        following people:  patrick d. brown, michael j. iacono,        !
    #        ronald e. farren, luke chen, robert bergstrom.                 !
    #                                                                       !
    #  *******************************************************************  !
    #                                                                       !
    #  taumol                                                               !
    #                                                                       !
    #    this file contains the subroutines taugbn (where n goes from       !
    #    16 to 29).  taugbn calculates the optical depths and Planck        !
    #    fractions per g-value and layer for band n.                        !
    #                                                                       !
    #  output:  optical depths (unitless)                                   !
    #           fractions needed to compute planck functions at every layer !
    #           and g-value                                                 !
    #                                                                       !
    #  modifications:                                                       !
    #                                                                       !
    # revised: adapted to f90 coding, j.-j.morcrette, ecmwf, feb 2003       !
    # revised: modified for g-point reduction, mjiacono, aer, dec 2003      !
    # revised: reformatted for consistency with rrtmg_lw, mjiacono, aer,    !
    #          jul 2006                                                     !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !

    id0 = np.zeros((nlay, nbhgh), dtype=np.int32)
    id1 = np.zeros((nlay, nbhgh), dtype=np.int32)
    sfluxzen = np.zeros(ngptsw)

    taug = np.zeros((nlay, ngptsw))
    taur = np.zeros((nlay, ngptsw))

    for b in range(nbhgh - nblow + 1):
        jb = nblow + b - 1

        #  --- ...  indices for layer optical depth

        for k in range(laytrop):
            id0[k, jb] = ((jp[k] - 1) * 5 + (jt[k] - 1)) * nspa[b] - 1
            id1[k, jb] = (jp[k] * 5 + (jt1[k] - 1)) * nspa[b] - 1

        for k in range(laytrop, nlay):
            id0[k, jb] = ((jp[k] - 13) * 5 + (jt[k] - 1)) * nspb[b] - 1
            id1[k, jb] = ((jp[k] - 12) * 5 + (jt1[k] - 1)) * nspb[b] - 1

        #  --- ...  calculate spectral flux at toa
        ibd = ibx[b] - 1
        njb = ng[b]
        ns = ngs[b]

        if jb in [15, 19, 22, 24, 25, 28]:
            for j in range(njb):
                sfluxzen[ns + j] = sfluxref01[j, 0, ibd]
        elif jb == 26:
            for j in range(njb):
                sfluxzen[ns + j] = scalekur * sfluxref01[j, 0, ibd]
        else:
            if jb == 16 or jb == 27:
                ks = nlay - 1
                for k in range(laytrop - 1, nlay - 1):
                    if (jp[k] < layreffr[b]) and jp[k + 1] >= layreffr[b]:
                        ks = k + 1
                        break

                colm1 = colamt[ks, ix1[b] - 1]
                colm2 = colamt[ks, ix2[b] - 1]

                speccomb = colm1 + strrat[b] * colm2
                specmult = specwt[b] * min(oneminus, colm1 / speccomb)
                js = 1 + int(specmult) - 1
                fs = np.mod(specmult, 1.0)

                for j in range(njb):
                    sfluxzen[ns + j] = sfluxref02[j, js, ibd] + fs * (
                        sfluxref02[j, js + 1, ibd] - sfluxref02[j, js, ibd]
                    )
            else:
                ks = laytrop - 1
                for k in range(laytrop - 1):
                    if jp[k] < layreffr[b] and jp[k + 1] >= layreffr[b]:
                        ks = k + 1
                        break
                colm1 = colamt[ks, ix1[b] - 1]
                colm2 = colamt[ks, ix2[b] - 1]
                speccomb = colm1 + strrat[b] * colm2
                specmult = specwt[b] * min(oneminus, colm1 / speccomb)
                js = 1 + int(specmult) - 1
                fs = np.mod(specmult, 1.0)

                for j in range(njb):
                    sfluxzen[ns + j] = sfluxref03[j, js, ibd] + fs * (
                        sfluxref03[j, js + 1, ibd] - sfluxref03[j, js, ibd]
                    )

    taug, taur = taumol16(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol17(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol18(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol19(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol20(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol21(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol22(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol23(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol24(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol25(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol26(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol27(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol28(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    taug, taur = taumol29(
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    )

    return sfluxzen, taug, taur, id0, id1


def taumol16(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    ds = xr.open_dataset("../lookupdata/radsw_kgb16_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ... compute the optical depth by interpolating in ln(pressure),
    #          temperature, and appropriate species.  below laytrop, the water
    #          vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG16):
            taur[k, NS16 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[0] * colamt[k, 4]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 15] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 15] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10
        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG16):
            taug[k, NS16 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            ) + colamt[k, 0] * (
                selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 15] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 15] + 1
            ind12 = ind11 + 1

            for j in range(NG16):
                taug[k, NS16 + j] = colamt[k, 4] * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )
    return taug, taur


# The subroutine computes the optical depth in band 17:  3250-4000
# cm-1 (low - h2o,co2; high - h2o,co2)


def taumol17(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 17:  3250-4000 cm-1 (low - h2o,co2; high - h2o,co2)         !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb17_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG17):
            taur[k, NS17 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[1] * colamt[k, 1]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 16] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 16] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG17):
            taug[k, NS17 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            ) + colamt[k, 0] * (
                selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 0] + strrat[1] * colamt[k, 1]
        specmult = 4.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 16] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 5
        ind04 = ind01 + 6
        ind11 = id1[k, 16] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 5
        ind14 = ind11 + 6

        indf = indfor[k] - 1
        indfp = indf + 1

        for j in range(NG17):
            taug[k, NS17 + j] = speccomb * (
                fac000 * absb[ind01, j]
                + fac100 * absb[ind02, j]
                + fac010 * absb[ind03, j]
                + fac110 * absb[ind04, j]
                + fac001 * absb[ind11, j]
                + fac101 * absb[ind12, j]
                + fac011 * absb[ind13, j]
                + fac111 * absb[ind14, j]
            ) + colamt[k, 0] * forfac[k] * (
                forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j])
            )

    return taug, taur


# The subroutine computes the optical depth in band 18:  4000-4650
# cm-1 (low - h2o,ch4; high - ch4)


def taumol18(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)             !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb18_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG18):
            taur[k, NS18 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[2] * colamt[k, 4]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 17] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 17] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG18):
            taug[k, NS18 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            ) + colamt[k, 0] * (
                selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 17] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 17] + 1
        ind12 = ind11 + 1

        for j in range(NG18):
            taug[k, NS18 + j] = colamt[k, 4] * (
                fac00[k] * absb[ind01, j]
                + fac10[k] * absb[ind02, j]
                + fac01[k] * absb[ind11, j]
                + fac11[k] * absb[ind12, j]
            )

    return taug, taur


# The subroutine computes the optical depth in band 19:  4650-5150
# cm-1 (low - h2o,co2; high - co2)


def taumol19(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 19:  4650-5150 cm-1 (low - h2o,co2; high - co2)             !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb19_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG19):
            taur[k, NS19 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[3] * colamt[k, 1]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 18] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 18] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG19):
            taug[k, NS19 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            ) + colamt[k, 0] * (
                selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 18] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 18] + 1
        ind12 = ind11 + 1

        for j in range(NG19):
            taug[k, NS19 + j] = colamt[k, 1] * (
                fac00[k] * absb[ind01, j]
                + fac10[k] * absb[ind02, j]
                + fac01[k] * absb[ind11, j]
                + fac11[k] * absb[ind12, j]
            )

    return taug, taur


# The subroutine computes the optical depth in band 20:  5150-6150
# cm-1 (low - h2o; high - h2o)


def taumol20(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 20:  5150-6150 cm-1 (low - h2o; high - h2o)                 !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb20_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    absch4 = ds["absch4"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG20):
            taur[k, NS20 + j] = tauray

    for k in range(laytrop):
        ind01 = id0[k, 19] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 19] + 1
        ind12 = ind11 + 1

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG20):
            taug[k, NS20 + j] = (
                colamt[k, 0]
                * (
                    (
                        fac00[k] * absa[ind01, j]
                        + fac10[k] * absa[ind02, j]
                        + fac01[k] * absa[ind11, j]
                        + fac11[k] * absa[ind12, j]
                    )
                    + selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )
                + colamt[k, 4] * absch4[j]
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 19] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 19] + 1
        ind12 = ind11 + 1

        indf = indfor[k] - 1
        indfp = indf + 1

        for j in range(NG20):
            taug[k, NS20 + j] = (
                colamt[k, 0]
                * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )
                + colamt[k, 4] * absch4[j]
            )

    return taug, taur


# The subroutine computes the optical depth in band 21:  6150-7700
# cm-1 (low - h2o,co2; high - h2o,co2)


def taumol21(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 21:  6150-7700 cm-1 (low - h2o,co2; high - h2o,co2)         !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb21_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG21):
            taur[k, NS21 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[5] * colamt[k, 1]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 20] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 20] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG21):
            taug[k, NS21 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            ) + colamt[k, 0] * (
                selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 0] + strrat[5] * colamt[k, 1]
        specmult = 4.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 20] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 5
        ind04 = ind01 + 6
        ind11 = id1[k, 20] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 5
        ind14 = ind11 + 6

        indf = indfor[k] - 1
        indfp = indf + 1

        for j in range(NG21):
            taug[k, NS21 + j] = speccomb * (
                fac000 * absb[ind01, j]
                + fac100 * absb[ind02, j]
                + fac010 * absb[ind03, j]
                + fac110 * absb[ind04, j]
                + fac001 * absb[ind11, j]
                + fac101 * absb[ind12, j]
                + fac011 * absb[ind13, j]
                + fac111 * absb[ind14, j]
            ) + colamt[k, 0] * forfac[k] * (
                forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j])
            )

    return taug, taur


# The subroutine computes the optical depth in band 22:  7700-8050
# cm-1 (low - h2o,o2; high - o2)


def taumol22(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 22:  7700-8050 cm-1 (low - h2o,o2; high - o2)               !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb22_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  the following factor is the ratio of total o2 band intensity (lines
    #           and mate continuum) to o2 band intensity (line only). it is needed
    #           to adjust the optical depths since the k's include only lines.

    o2adj = 1.6
    o2tem = 4.35e-4 / (350.0 * 2.0)

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG22):
            taur[k, NS22 + j] = tauray

    for k in range(laytrop):
        o2cont = o2tem * colamt[k, 5]
        speccomb = colamt[k, 0] + strrat[6] * colamt[k, 5]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 21] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 21] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG22):
            taug[k, NS22 + j] = (
                speccomb
                * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                )
                + colamt[k, 0]
                * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )
                + o2cont
            )

    for k in range(laytrop, nlay):
        o2cont = o2tem * colamt[k, 5]

        ind01 = id0[k, 21] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 21] + 1
        ind12 = ind11 + 1

        for j in range(NG22):
            taug[k, NS22 + j] = (
                colamt[k, 5]
                * o2adj
                * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )
                + o2cont
            )

    return taug, taur


# The subroutine computes the optical depth in band 23:  8050-12850
# cm-1 (low - h2o; high - nothing)


def taumol23(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 23:  8050-12850 cm-1 (low - h2o; high - nothing)            !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb23_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    rayl = ds["rayl"].data
    givfac = ds["givfac"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        for j in range(NG23):
            taur[k, NS23 + j] = colmol[k] * rayl[j]

    for k in range(laytrop):
        ind01 = id0[k, 22] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 22] + 1
        ind12 = ind11 + 1

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG23):
            taug[k, NS23 + j] = colamt[k, 0] * (
                givfac
                * (
                    fac00[k] * absa[ind01, j]
                    + fac10[k] * absa[ind02, j]
                    + fac01[k] * absa[ind11, j]
                    + fac11[k] * absa[ind12, j]
                )
                + selffac[k]
                * (
                    selfref[inds, j]
                    + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                )
                + forfac[k]
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        for j in range(NG23):
            taug[k, NS23 + j] = 0.0

    return taug, taur


# The subroutine computes the optical depth in band 24:  12850-16000
# cm-1 (low - h2o,o2; high - o2)


def taumol24(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 24:  12850-16000 cm-1 (low - h2o,o2; high - o2)             !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb24_data.nc")
    selfref = ds["selfref"].data
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    abso3a = ds["abso3a"].data
    abso3b = ds["abso3b"].data
    rayla = ds["rayla"].data
    raylb = ds["raylb"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[8] * colamt[k, 5]
        specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 23] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 23] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG24):
            taug[k, NS24 + j] = (
                speccomb
                * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                )
                + colamt[k, 2] * abso3a[j]
                + colamt[k, 0]
                * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )
            )

            taur[k, NS24 + j] = colmol[k] * (
                rayla[j, js - 1] + fs * (rayla[j, js] - rayla[j, js - 1])
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 23] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 23] + 1
        ind12 = ind11 + 1

        for j in range(NG24):
            taug[k, NS24 + j] = (
                colamt[k, 5]
                * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )
                + colamt[k, 2] * abso3b[j]
            )

            taur[k, NS24 + j] = colmol[k] * raylb[j]

    return taug, taur


# The subroutine computes the optical depth in band 25:  16000-22650
# cm-1 (low - h2o; high - nothing)


def taumol25(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 25:  16000-22650 cm-1 (low - h2o; high - nothing)           !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb25_data.nc")
    absa = ds["absa"].data
    abso3a = ds["abso3a"].data
    abso3b = ds["abso3b"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        for j in range(NG25):
            taur[k, NS25 + j] = colmol[k] * rayl[j]

    for k in range(laytrop):
        ind01 = id0[k, 24] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 24] + 1
        ind12 = ind11 + 1

        for j in range(NG25):
            taug[k, NS25 + j] = (
                colamt[k, 0]
                * (
                    fac00[k] * absa[ind01, j]
                    + fac10[k] * absa[ind02, j]
                    + fac01[k] * absa[ind11, j]
                    + fac11[k] * absa[ind12, j]
                )
                + colamt[k, 2] * abso3a[j]
            )

    for k in range(laytrop, nlay):
        for j in range(NG25):
            taug[k, NS25 + j] = colamt[k, 2] * abso3b[j]

    return taug, taur


# The subroutine computes the optical depth in band 26:  22650-29000
# cm-1 (low - nothing; high - nothing)


def taumol26(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 26:  22650-29000 cm-1 (low - nothing; high - nothing)       !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb26_data.nc")
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        for j in range(NG26):
            taug[k, NS26 + j] = 0.0
            taur[k, NS26 + j] = colmol[k] * rayl[j]

    return taug, taur


# The subroutine computes the optical depth in band 27:  29000-38000
# cm-1 (low - o3; high - o3)


def taumol27(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 27:  29000-38000 cm-1 (low - o3; high - o3)                 !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb27_data.nc")
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        for j in range(NG27):
            taur[k, NS27 + j] = colmol[k] * rayl[j]

    for k in range(laytrop):
        ind01 = id0[k, 26] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 26] + 1
        ind12 = ind11 + 1

        for j in range(NG27):
            taug[k, NS27 + j] = colamt[k, 2] * (
                fac00[k] * absa[ind01, j]
                + fac10[k] * absa[ind02, j]
                + fac01[k] * absa[ind11, j]
                + fac11[k] * absa[ind12, j]
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 26] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 26] + 1
        ind12 = ind11 + 1

        for j in range(NG27):
            taug[k, NS27 + j] = colamt[k, 2] * (
                fac00[k] * absb[ind01, j]
                + fac10[k] * absb[ind02, j]
                + fac01[k] * absb[ind11, j]
                + fac11[k] * absb[ind12, j]
            )

    return taug, taur


# The subroutine computes the optical depth in band 28:  38000-50000
# cm-1 (low - o3,o2; high - o3,o2)


def taumol28(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 28:  38000-50000 cm-1 (low - o3,o2; high - o3,o2)           !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb28_data.nc")
    absa = ds["absa"].data
    absb = ds["absb"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG28):
            taur[k, NS28 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 2] + strrat[12] * colamt[k, 5]
        specmult = 8.0 * min(oneminus, colamt[k, 2] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 27] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 9
        ind04 = ind01 + 10
        ind11 = id1[k, 27] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 9
        ind14 = ind11 + 10

        for j in range(NG28):
            taug[k, NS28 + j] = speccomb * (
                fac000 * absa[ind01, j]
                + fac100 * absa[ind02, j]
                + fac010 * absa[ind03, j]
                + fac110 * absa[ind04, j]
                + fac001 * absa[ind11, j]
                + fac101 * absa[ind12, j]
                + fac011 * absa[ind13, j]
                + fac111 * absa[ind14, j]
            )

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 2] + strrat[12] * colamt[k, 5]
        specmult = 4.0 * min(oneminus, colamt[k, 2] / speccomb)

        js = 1 + int(specmult)
        fs = np.mod(specmult, 1.0)
        fs1 = 1.0 - fs
        fac000 = fs1 * fac00[k]
        fac010 = fs1 * fac10[k]
        fac100 = fs * fac00[k]
        fac110 = fs * fac10[k]
        fac001 = fs1 * fac01[k]
        fac011 = fs1 * fac11[k]
        fac101 = fs * fac01[k]
        fac111 = fs * fac11[k]

        ind01 = id0[k, 27] + js
        ind02 = ind01 + 1
        ind03 = ind01 + 5
        ind04 = ind01 + 6
        ind11 = id1[k, 27] + js
        ind12 = ind11 + 1
        ind13 = ind11 + 5
        ind14 = ind11 + 6

        for j in range(NG28):
            taug[k, NS28 + j] = speccomb * (
                fac000 * absb[ind01, j]
                + fac100 * absb[ind02, j]
                + fac010 * absb[ind03, j]
                + fac110 * absb[ind04, j]
                + fac001 * absb[ind11, j]
                + fac101 * absb[ind12, j]
                + fac011 * absb[ind13, j]
                + fac111 * absb[ind14, j]
            )

    return taug, taur


# The subroutine computes the optical depth in band 29:  820-2600
# cm-1 (low - h2o; high - co2)


def taumol29(
    colamt,
    colmol,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    laytrop,
    forfac,
    forfrac,
    indfor,
    selffac,
    selffrac,
    indself,
    nlay,
    id0,
    id1,
    taug,
    taur,
):

    #  ------------------------------------------------------------------  !
    #     band 29:  820-2600 cm-1 (low - h2o; high - co2)                  !
    #  ------------------------------------------------------------------  !
    #

    ds = xr.open_dataset("../lookupdata/radsw_kgb29_data.nc")
    forref = ds["forref"].data
    absa = ds["absa"].data
    absb = ds["absb"].data
    selfref = ds["selfref"].data
    absh2o = ds["absh2o"].data
    absco2 = ds["absco2"].data
    rayl = ds["rayl"].data

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG29):
            taur[k, NS29 + j] = tauray

    for k in range(laytrop):
        ind01 = id0[k, 28] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 28] + 1
        ind12 = ind11 + 1

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1

        for j in range(NG29):
            taug[k, NS29 + j] = (
                colamt[k, 0]
                * (
                    (
                        fac00[k] * absa[ind01, j]
                        + fac10[k] * absa[ind02, j]
                        + fac01[k] * absa[ind11, j]
                        + fac11[k] * absa[ind12, j]
                    )
                    + selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )
                + colamt[k, 1] * absco2[j]
            )

    for k in range(laytrop, nlay):
        ind01 = id0[k, 28] + 1
        ind02 = ind01 + 1
        ind11 = id1[k, 28] + 1
        ind12 = ind11 + 1

        for j in range(NG29):
            taug[k, NS29 + j] = (
                colamt[k, 1]
                * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )
                + colamt[k, 0] * absh2o[j]
            )

    return taug, taur


sfluxzen, taug, taur, id0, id1 = taumol(
    indict["colamt"],
    indict["colmol"],
    indict["fac00"],
    indict["fac01"],
    indict["fac10"],
    indict["fac11"],
    indict["jp"],
    indict["jt"],
    indict["jt1"],
    indict["laytrop"],
    indict["forfac"],
    indict["forfrac"],
    indict["indfor"],
    indict["selffac"],
    indict["selffrac"],
    indict["indself"],
    indict["NLAY"],
)

outdict = dict()
outdict["sfluxzen"] = sfluxzen
outdict["taug"] = taug
outdict["taur"] = taur

valvars = ["sfluxzen", "taug", "taur"]
valdict = dict()
for var in valvars:
    valdict[var] = serializer.read(
        var, serializer.savepoint["swrad-taumol-output-000000"]
    )[10, ...]

compare_data(outdict, valdict)
