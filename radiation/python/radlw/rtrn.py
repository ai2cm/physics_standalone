import os
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import (nbands, ngptlw, ngb, eps, ntbl, bpade, wtdiff, fluxfac,
                         heatfac)

def rtrn(semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs,
         secdif, nlay,nlp1, lhlw0, lhlwb, exp_tbl, tau_tbl, tfn_tbl):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    # purpose:  compute the upward/downward radiative fluxes, and heating   !
    # rates for both clear or cloudy atmosphere.  clouds are assumed as     !
    # randomly overlaping in a vertical colum.                              !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                     -size-   !
    #   semiss  - real, lw surface emissivity                         nbands!
    #   delp    - real, layer pressure thickness (mb)                  nlay !
    #   cldfrc  - real, layer cloud fraction                         0:nlp1 !
    #   taucld  - real, layer cloud opt depth                    nbands,nlay!
    #   tautot  - real, total optical depth (gas+aerosols)       ngptlw,nlay!
    #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
    #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
    #   fracs   - real, planck fractions                         ngptlw,nlay!
    #   secdif  - real, secant of diffusivity angle                   nbands!
    #   nlay    - integer, number of vertical layers                    1   !
    #   nlp1    - integer, number of vertical levels (interfaces)       1   !
    #                                                                       !
    #  outputs:                                                             !
    #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
    #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
    #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
    #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
    #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
    #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
    #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
    #                                                                       !
    #  module veriables:                                                    !
    #   ngb     - integer, band index for each g-value                ngptlw!
    #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
    #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
    #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
    #   bpade   - real, pade approx constant (1/0.278)                   1  !
    #   wtdiff  - real, weight for radiance to flux conversion           1  !
    #   ntbl    - integer, dimension of look-up tables                   1  !
    #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
    #   exp_tbl - real, transmittance lookup table                   0:ntbl !
    #   tfn_tbl - real, tau transition function                      0:ntbl !
    #                                                                       !
    #  local variables:                                                     !
    #    itgas  - integer, index for gases contribution look-up table    1  !
    #    ittot  - integer, index for gases plus clouds  look-up table    1  !
    #    reflct - real, surface reflectance                              1  !
    #    atrgas - real, gaseous absorptivity                             1  !
    #    atrtot - real, gaseous and cloud absorptivity                   1  !
    #    odcld  - real, cloud optical depth                              1  !
    #    efclrfr- real, effective clear sky fraction (1-efcldfr)       nlay !
    #    odepth - real, optical depth of gaseous only                    1  !
    #    odtot  - real, optical depth of gas and cloud                   1  !
    #    gasfac - real, gas-only pade factor, used for planck fn         1  !
    #    totfac - real, gas+cld pade factor, used for planck fn          1  !
    #    bbdgas - real, gas-only planck function for downward rt         1  !
    #    bbugas - real, gas-only planck function for upward rt           1  !
    #    bbdtot - real, gas and cloud planck function for downward rt    1  !
    #    bbutot - real, gas and cloud planck function for upward rt      1  !
    #    gassrcu- real, upwd source radiance due to gas only            nlay!
    #    totsrcu- real, upwd source radiance due to gas+cld             nlay!
    #    gassrcd- real, dnwd source radiance due to gas only             1  !
    #    totsrcd- real, dnwd source radiance due to gas+cld              1  !
    #    radtotu- real, spectrally summed total sky upwd radiance        1  !
    #    radclru- real, spectrally summed clear sky upwd radiance        1  !
    #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
    #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
    #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
    #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
    #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
    #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
    #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
    #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
    #                                                                       !
    #                                                                       !
    #  *******************************************************************  !
    #  original code description                                            !
    #                                                                       !
    #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
    #  revision for gcms:  michael j. iacono; october, 2002                 !
    #  revision for f90:   michael j. iacono; june, 2006                    !
    #                                                                       !
    #  this program calculates the upward fluxes, downward fluxes, and      !
    #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
    #  to this program is the atmospheric profile, all Planck function      !
    #  information, and the cloud fraction by layer.  a variable diffusivity!
    #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
    #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
    #  of the column water vapor, and other bands use a value of 1.66.  the !
    #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
    #  here.  note that use of the emissivity angle for the flux integration!
    #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
    #  clouds are treated with a random cloud overlap method.               !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !

    #  ---  outputs:
    htr = np.zeros(nlay)
    htrcl = np.zeros(nlay)

    htrb = np.zeros((nlay, nbands))

    totuflux = np.zeros(nlp1)
    totdflux = np.zeros(nlp1)
    totuclfl = np.zeros(nlp1)
    totdclfl = np.zeros(nlp1)

    #  ---  locals:
    rec_6 = 0.166667

    clrurad = np.zeros((nlp1, nbands))
    clrdrad = np.zeros((nlp1, nbands))
    toturad = np.zeros((nlp1, nbands))
    totdrad = np.zeros((nlp1, nbands))

    gassrcu = np.zeros(nlay)
    totsrcu = np.zeros(nlay)
    trngas = np.zeros(nlay)
    efclrfr = np.zeros(nlay)
    rfdelp = np.zeros(nlay)

    fnet = np.zeros(nlp1)
    fnetc = np.zeros(nlp1)

    f_zero = 0.0
    f_one = 1.0
    tblint = ntbl
    
    #
    #===> ...  begin here
    #
    for ib in range(nbands):
        for k in range(nlp1):
            toturad[k, ib] = f_zero
            totdrad[k, ib] = f_zero
            clrurad[k, ib] = f_zero
            clrdrad[k, ib] = f_zero

    for k in range(nlp1):
        totuflux[k] = f_zero
        totdflux[k] = f_zero
        totuclfl[k] = f_zero
        totdclfl[k] = f_zero

    #  --- ...  loop over all g-points

    for ig in range(ngptlw):
        ib = ngb(ig)

        radtotd = f_zero
        radclrd = f_zero

        #> -# Downward radiative transfer loop.

        for k in range(nlay-1, -1, -1):

            #  - clear sky, gases contribution

            odepth = max(f_zero, secdif[ib]*tautot[ig, k])
            if odepth <= 0.06:
                atrgas = odepth - 0.5*odepth*odepth
                trng   = f_one - atrgas
                gasfac = rec_6 * odepth
            else:
                tblind = odepth / (bpade + odepth)
                itgas = tblint*tblind + 0.5
                trng  = exp_tbl[itgas]
                atrgas = f_one - trng
                gasfac = tfn_tbl[itgas]
                odepth = tau_tbl[itgas]

            plfrac = fracs[ig, k]
            blay = pklay[ib, k]

            dplnku = pklev[ib, k] - blay
            dplnkd = pklev[ib, k-1] - blay
            bbdgas = plfrac * (blay + dplnkd*gasfac)
            bbugas = plfrac * (blay + dplnku*gasfac)
            gassrcd= bbdgas * atrgas
            gassrcu[k] = bbugas * atrgas
            trngas[k] = trng

            # - total sky, gases+clouds contribution

            clfr = cldfrc[k]
            if clfr >= eps:
                #\n  - cloudy layer

                odcld = secdif[ib] * taucld[ib, k]
                efclrfr[k] = f_one-(f_one - np.exp(-odcld))*clfr
                odtot = odepth + odcld
                if odtot < 0.06:
                    totfac = rec_6 * odtot
                    atrtot = odtot - 0.5*odtot*odtot
                else:
                    tblind = odtot / (bpade + odtot)
                    ittot  = tblint*tblind + 0.5
                    totfac = tfn_tbl[ittot]
                    atrtot = f_one - exp_tbl[ittot]

                bbdtot = plfrac * (blay + dplnkd*totfac)
                bbutot = plfrac * (blay + dplnku*totfac)
                totsrcd= bbdtot * atrtot
                totsrcu[k] = bbutot * atrtot

                #  --- ...  total sky radiance
                radtotd = radtotd*trng*efclrfr[k] + gassrcd + \
                            clfr*(totsrcd - gassrcd)
                totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                #  --- ...  clear sky radiance
                radclrd = radclrd*trng + gassrcd
                clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd

            else:
                #  --- ...  clear layer

                #  --- ...  total sky radiance
                radtotd = radtotd*trng + gassrcd
                totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                #  --- ...  clear sky radiance
                radclrd = radclrd*trng + gassrcd
                clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd

        #> -# Compute spectral emissivity & reflectance, include the
        #!    contribution of spectrally varying longwave emissivity and
        #!     reflection from the surface to the upward radiative transfer.

        #     note: spectral and Lambertian reflection are identical for the
        #           diffusivity angle flux integration used here.

        reflct = f_one - semiss[ib]
        rad0 = semiss[ib] * fracs[ig, 0] * pklay[ib, 0]

        # -# Compute total sky radiance.
        radtotu = rad0 + reflct*radtotd
        toturad[0, ib] = toturad[0, ib] + radtotu

        # -# Compute clear sky radiance
        radclru = rad0 + reflct*radclrd
        clrurad[0, ib] = clrurad[0, ib] + radclru

        # -# Upward radiative transfer loop.

        for k in range(nlay):
            clfr = cldfrc[k+1]
            trng = trngas[k]
            gasu = gassrcu(k)

            if clfr >= eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu = radtotu*trng*efclrfr(k) + gasu + \
                    + clfr*(totsrcu(k) - gasu)
                toturad[k, ib] = toturad[k, ib] + radtotu

                #  --- ... clear sky radiance
                radclru = radclru*trng + gasu
                clrurad[k, ib] = clrurad[k, ib] + radclru

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu = radtotu*trng + gasu
                toturad[k, ib] = toturad[k, ib] + radtotu

                #  --- ... clear sky radiance
                radclru = radclru*trng + gasu
                clrurad[k, ib] = clrurad[k, ib] + radclru

    # -    # Process longwave output from band for total and clear streams. 
    #      Calculate upward, downward, and net flux.

    flxfac = wtdiff * fluxfac

    for k in range(nlp1):
        for ib in range(nbands):
            totuflux[k] = totuflux[k] + toturad[k, ib]
            totdflux[k] = totdflux[k] + totdrad[k, ib]
            totuclfl[k] = totuclfl[k] + clrurad[k, ib]
            totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

        totuflux[k] = totuflux[k] * flxfac
        totdflux[k] = totdflux[k] * flxfac
        totuclfl[k] = totuclfl[k] * flxfac
        totdclfl[k] = totdclfl[k] * flxfac

    #  --- ...  calculate net fluxes and heating rates
    fnet[0] = totuflux[0] - totdflux[0]

    for k in range(nlay):
        rfdelp(k) = heatfac / delp[k]
        fnet[k] = totuflux[k] - totdflux[k]
        htr[k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

    # --- ...  optional clear sky heating rates
    if lhlw0:
        fnetc[0] = totuclfl[0] - totdclfl[0]

        for k in range(nlay):
            fnetc[k] = totuclfl[k] - totdclfl[k]
            htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

    # --- ...  optional spectral band heating rates
    if lhlwb:
        for ib in range(nbands):
            fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

            for k in range(nlay):
                fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

    return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb


def rtrnmr(semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs,
           secdif, nlay, nlp1, lhlw0, lhlwb, exp_tbl, tau_tbl, tfn_tbl):
    #  ===================  program usage description  ===================  !
    #                                                                       !
    # purpose:  compute the upward/downward radiative fluxes, and heating   !
    # rates for both clear or cloudy atmosphere.  clouds are assumed as in  !
    # maximum-randomly overlaping in a vertical colum.                      !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                     -size-   !
    #   semiss  - real, lw surface emissivity                         nbands!
    #   delp    - real, layer pressure thickness (mb)                  nlay !
    #   cldfrc  - real, layer cloud fraction                         0:nlp1 !
    #   taucld  - real, layer cloud opt depth                    nbands,nlay!
    #   tautot  - real, total optical depth (gas+aerosols)       ngptlw,nlay!
    #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
    #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
    #   fracs   - real, planck fractions                         ngptlw,nlay!
    #   secdif  - real, secant of diffusivity angle                   nbands!
    #   nlay    - integer, number of vertical layers                    1   !
    #   nlp1    - integer, number of vertical levels (interfaces)       1   !
    #                                                                       !
    #  outputs:                                                             !
    #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
    #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
    #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
    #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
    #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
    #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
    #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
    #                                                                       !
    #  module veriables:                                                    !
    #   ngb     - integer, band index for each g-value                ngptlw!
    #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
    #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
    #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
    #   bpade   - real, pade approx constant (1/0.278)                   1  !
    #   wtdiff  - real, weight for radiance to flux conversion           1  !
    #   ntbl    - integer, dimension of look-up tables                   1  !
    #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
    #   exp_tbl - real, transmittance lookup table                   0:ntbl !
    #   tfn_tbl - real, tau transition function                      0:ntbl !
    #                                                                       !
    #  local variables:                                                     !
    #    itgas  - integer, index for gases contribution look-up table    1  !
    #    ittot  - integer, index for gases plus clouds  look-up table    1  !
    #    reflct - real, surface reflectance                              1  !
    #    atrgas - real, gaseous absorptivity                             1  !
    #    atrtot - real, gaseous and cloud absorptivity                   1  !
    #    odcld  - real, cloud optical depth                              1  !
    #    odepth - real, optical depth of gaseous only                    1  !
    #    odtot  - real, optical depth of gas and cloud                   1  !
    #    gasfac - real, gas-only pade factor, used for planck fn         1  !
    #    totfac - real, gas+cld pade factor, used for planck fn          1  !
    #    bbdgas - real, gas-only planck function for downward rt         1  !
    #    bbugas - real, gas-only planck function for upward rt           1  !
    #    bbdtot - real, gas and cloud planck function for downward rt    1  !
    #    bbutot - real, gas and cloud planck function for upward rt      1  !
    #    gassrcu- real, upwd source radiance due to gas only            nlay!
    #    totsrcu- real, upwd source radiance due to gas + cld           nlay!
    #    gassrcd- real, dnwd source radiance due to gas only             1  !
    #    totsrcd- real, dnwd source radiance due to gas + cld            1  !
    #    radtotu- real, spectrally summed total sky upwd radiance        1  !
    #    radclru- real, spectrally summed clear sky upwd radiance        1  !
    #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
    #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
    #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
    #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
    #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
    #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
    #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
    #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
    #                                                                       !
    #                                                                       !
    #  *******************************************************************  !
    #  original code description                                            !
    #                                                                       !
    #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
    #  revision for gcms:  michael j. iacono; october, 2002                 !
    #  revision for f90:   michael j. iacono; june, 2006                    !
    #                                                                       !
    #  this program calculates the upward fluxes, downward fluxes, and      !
    #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
    #  to this program is the atmospheric profile, all Planck function      !
    #  information, and the cloud fraction by layer.  a variable diffusivity!
    #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
    #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
    #  of the column water vapor, and other bands use a value of 1.66.  the !
    #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
    #  here.  note that use of the emissivity angle for the flux integration!
    #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
    #  clouds are treated with a maximum-random cloud overlap method.       !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !   #

    #  ---  outputs:
    htr = np.zeros(nlay)
    htrcl = np.zeros(nlay)
    htrb = np.zeros((nlay, nbands))

    totuflux = np.zeros(nlp1)
    totdflux = np.zeros(nlp1)
    totuclfl = np.zeros(nlp1)
    totdclfl = np.zeros(nlp1)

    #  ---  locals:
    rec_6 = 0.166667

    clrurad = np.zeros((nlp1, nbands))
    clrdrad = np.zeros((nlp1, nbands))
    toturad = np.zeros((nlp1, nbands))
    totdrad = np.zeros((nlp1, nbands))

    gassrcu = np.zeros(nlay)
    totsrcu = np.zeros(nlay)
    trngas = np.zeros(nlay)
    trntot = np.zeros(nlay)
    rfdelp = np.zeros(nlay)

    fnet = np.zeros(nlp1)
    fnetc = np.zeros(nlp1)

    faccld1u = np.zeros(nlp1)
    faccld2u = np.zeros(nlp1)
    facclr1u = np.zeros(nlp1)
    facclr2u = np.zeros(nlp1)
    faccmb1u = np.zeros(nlp1)
    faccmb2u = np.zeros(nlp1)

    faccld1d = np.zeros(nlp1)
    faccld2d = np.zeros(nlp1)
    facclr1d = np.zeros(nlp1)
    facclr2d = np.zeros(nlp1)
    faccmb1d = np.zeros(nlp1)
    faccmb2d = np.zeros(nlp1)

    lstcldu = np.zeros(nlay)
    lstcldd = np.zeros(nlay)

    tblint = ntbl
    #
    #===> ...  begin here
    #

    lstcldu[0] = cldfrc[0] > eps
    rat1 = 0.0
    rat2 = 0.0

    for k in range(nlay-1):

        lstcldu[k+1] = cldfrc[k+1] > eps and cldfrc[k] <= eps

        if cldfrc[k] > eps:
            # Setup maximum/random cloud overlap.

            if cldfrc[k+1] >= cldfrc[k]:
                if lstcldu[k]:
                    if cldfrc[k] < 1.0:
                        facclr2u[k+1] = (cldfrc[k+1] - cldfrc[k]) / \
                            (1.0 - cldfrc[k])
                    facclr2u[k] = 0.0
                    faccld2u[k] = 0.0
                else:
                    fmax = max(cldfrc[k], cldfrc[k-1])
                    if cldfrc[k+1] > fmax:
                        facclr1u[k+1] = rat2
                        facclr2u[k+1] = (cldfrc[k+1] - fmax)/(1.0 - fmax)
                    elif cldfrc[k+1] < fmax:
                        facclr1u[k+1] = (cldfrc[k+1] - cldfrc[k]) / \
                            (cldfrc[k-1] - cldfrc[k])
                    else:
                        facclr1u[k+1] = rat2

                if facclr1u[k+1] > 0.0 or facclr2u[k+1] > 0.0:
                    rat1 = 1.0
                    rat2 = 0.0
                else:
                    rat1 = 0.0
                    rat2 = 0.0
            else:
                if lstcldu[k]:
                    faccld2u[k+1] = (cldfrc[k] - cldfrc[k+1]) / cldfrc[k]
                    facclr2u[k] = 0.0
                    faccld2u[k] = 0.0
                else:
                    fmin = min(cldfrc[k], cldfrc[k-1])
                    if cldfrc[k+1] <= fmin:
                        faccld1u[k+1] = rat1
                        faccld2u[k+1] = (fmin - cldfrc[k+1]) / fmin
                    else:
                        faccld1u[k+1] = (cldfrc[k] - cldfrc[k+1]) / \
                            (cldfrc[k] - fmin)

                if faccld1u[k+1] > 0.0 or faccld2u[k+1] > 0.0:
                    rat1 = 0.0
                    rat2 = 1.0
                else:
                    rat1 = 0.0
                    rat2 = 0.0

            faccmb1u[k+1] = facclr1u[k+1] * faccld2u[k] * cldfrc[k-1]
            faccmb2u[k+1] = faccld1u[k+1] * facclr2u[k] * (1.0 - cldfrc[k-1])

    for k in range(nlp1):
        faccld1d[k] = 0.0
        faccld2d[k] = 0.0
        facclr1d[k] = 0.0
        facclr2d[k] = 0.0
        faccmb1d[k] = 0.0
        faccmb2d[k] = 0.0

    lstcldd[nlay] = cldfrc[nlay] > eps
    rat1 = 0.0
    rat2 = 0.0

    for k in range(nlay-1, 0, -1):
        lstcldd[k-1] = cldfrc[k-1] > eps and cldfrc[k] <= eps

        if cldfrc[k] > eps:
            if cldfrc[k-1] >= cldfrc[k]:
                if lstcldd[k]:
                    if cldfrc[k] < 1.0:
                        facclr2d[k-1] = (cldfrc[k-1] - cldfrc[k]) / \
                            (1.0 - cldfrc[k])

                    facclr2d[k] = 0.0
                    faccld2d[k] = 0.0
                else:
                    fmax = max(cldfrc[k], cldfrc[k+1])

                    if cldfrc[k-1] > fmax:
                        facclr1d[k-1] = rat2
                        facclr2d[k-1] = (cldfrc[k-1] - fmax) / (1.0 - fmax)
                    elif cldfrc[k-1] < fmax:
                        facclr1d[k-1] = (cldfrc[k-1] - cldfrc[k]) / \
                            (cldfrc[k+1] - cldfrc[k])
                    else:
                        facclr1d[k-1] = rat2

                if facclr1d[k-1] > 0.0 or facclr2d[k-1] > 0.0:
                    rat1 = 1.0
                    rat2 = 0.0
                else:
                    rat1 = 0.0
                    rat2 = 0.0
            else:
                if lstcldd[k]:
                    faccld2d[k-1] = (cldfrc[k] - cldfrc[k-1]) / cldfrc[k]
                    facclr2d[k] = 0.0
                    faccld2d[k] = 0.0
                else:
                    fmin = min(cldfrc[k], cldfrc[k+1])

                    if cldfrc[k-1] <= fmin:
                        faccld1d[k-1] = rat1
                        faccld2d[k-1] = (fmin - cldfrc[k-1]) / fmin
                    else:
                        faccld1d[k-1] = (cldfrc[k] - cldfrc[k-1]) / \
                            (cldfrc[k] - fmin)

                if faccld1d[k-1] > 0.0 or faccld2d[k-1] > 0.0:
                    rat1 = 0.0
                    rat2 = 1.0
                else:
                    rat1 = 0.0
                    rat2 = 0.0

            faccmb1d[k-1] = facclr1d[k-1] * faccld2d[k] * cldfrc[k+1]
            faccmb2d[k-1] = faccld1d[k-1] * facclr2d[k] * \
                (1.0 - cldfrc[k+1])

    # Initialize for radiative transfer

    for ib in range(nbands):
        for k in range(nlp1):
            toturad[k, ib] = 0.0
            totdrad[k, ib] = 0.0
            clrurad[k, ib] = 0.0
            clrdrad[k, ib] = 0.0

        for k in range(nlp1):
            totuflux[k] = 0.0
            totdflux[k] = 0.0
            totuclfl[k] = 0.0
            totdclfl[k] = 0.0

        #  --- ...  loop over all g-points

        for ig in range(ngptlw):
            ib = ngb[ig]

            radtotd = 0.0
            radclrd = 0.0

            # Downward radiative transfer loop:

            for k in range(nlay-1, None, -1):

                #  --- ...  clear sky, gases contribution

                odepth = max(0.0, secdif[ib]*tautot[ig, k])
                if odepth <= 0.06:
                    atrgas = odepth - 0.5*odepth*odepth
                    trng   = 1.0 - atrgas
                    gasfac = rec_6 * odepth
                else:
                    tblind = odepth / (bpade + odepth)
                    itgas = tblint*tblind + 0.5
                    trng  = exp_tbl[itgas]
                    atrgas = 1.0 - trng
                    gasfac = tfn_tbl[itgas]
                    odepth = tau_tbl[itgas]

                plfrac = fracs[ig, k]
                blay = pklay[ib, k]

                dplnku = pklev[ib, k  ] - blay
                dplnkd = pklev[ib, k-1] - blay
                bbdgas = plfrac * (blay + dplnkd*gasfac)
                bbugas = plfrac * (blay + dplnku*gasfac)
                gassrcd   = bbdgas * atrgas
                gassrcu[k] = bbugas * atrgas
                trngas[k] = trng

                #  --- ...  total sky, gases+clouds contribution

                clfr = cldfrc[k]
                if lstcldd[k]:
                    totradd = clfr * radtotd
                    clrradd = radtotd - totradd
                    rad = 0.0

                if clfr >= eps:
                    #  - cloudy layer

                    odcld = secdif[ib] * taucld[ib, k]
                    odtot = odepth + odcld
                    if odtot < 0.06:
                        totfac = rec_6 * odtot
                        atrtot = odtot - 0.5*odtot*odtot
                        trnt   = 1.0 - atrtot
                    else:
                        tblind = odtot / (bpade + odtot)
                        ittot  = tblint*tblind + 0.5
                        totfac = tfn_tbl[ittot]
                        trnt   = exp_tbl[ittot]
                        atrtot = 1.0 - trnt

                    bbdtot = plfrac * (blay + dplnkd*totfac)
                    bbutot = plfrac * (blay + dplnku*totfac)
                    totsrcd   = bbdtot * atrtot
                    totsrcu[k] = bbutot * atrtot
                    trntot[k] = trnt

                    totradd = totradd*trnt + clfr*totsrcd
                    clrradd = clrradd*trng + (1.0 - clfr)*gassrcd

                    #  - total sky radiance
                    radtotd = totradd + clrradd
                    totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                    #  - clear sky radiance
                    radclrd = radclrd*trng + gassrcd
                    clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd

                    radmod = rad*(facclr1d[k-1]*trng + faccld1d[k-1]*trnt) - \
                        faccmb1d[k-1]*gassrcd + faccmb2d[k-1]*totsrcd

                    rad = -radmod + facclr2d[k-1]*(clrradd + radmod) - \
                        faccld2d[k-1]*(totradd - radmod)
                    totradd = totradd + rad
                    clrradd = clrradd - rad

                else:
                    #  --- ...  clear layer

                    #  --- ...  total sky radiance
                    radtotd = radtotd*trng + gassrcd
                    totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                    #  --- ...  clear sky radiance
                    radclrd = radclrd*trng + gassrcd
                    clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd


            # Compute spectral emissivity & reflectance, include the
            #    contribution of spectrally varying longwave emissivity and
            #    reflection from the surface to the upward radiative transfer.

            #    note: spectral and Lambertian reflection are identical for the
            #          diffusivity angle flux integration used here.

            reflct = 1.0 - semiss[ib]
            rad0 = semiss[ib]* fracs[ig, 0] * pklay[ib, 0]

            # -# Compute total sky radiance.
            radtotu = rad0 + reflct*radtotd
            toturad[0, ib] = toturad[0, ib] + radtotu

            # Compute clear sky radiance.
            radclru = rad0 + reflct*radclrd
            clrurad[0, ib] = clrurad[0, ib] + radclru

            # Upward radiative transfer loop:
            for k in range(nlay):
                clfr = cldfrc[k]
                trng = trngas[k]
                gasu = gassrcu[k]

                if lstcldu[k]:
                    totradu = clfr * radtotu
                    clrradu = radtotu - totradu
                    rad = 0.0

                if clfr >= eps:
                    #  - cloudy layer radiance
                    trnt = trntot[k]
                    totu = totsrcu[k]
                    totradu = totradu*trnt + clfr*totu
                    clrradu = clrradu*trng + (1.0 - clfr)*gasu

                    #  - total sky radiance
                    radtotu = totradu + clrradu
                    toturad[k, ib] = toturad[k, ib] + radtotu

                    #  - clear sky radiance
                    radclru = radclru*trng + gasu
                    clrurad[k, ib] = clrurad[k, ib] + radclru

                    radmod = rad*(facclr1u[k+1]*trng + faccld1u[k+1]*trnt) - \
                        faccmb1u[k+1]*gasu + faccmb2u[k+1]*totu
                    rad = -radmod + facclr2u[k+1]*(clrradu + radmod) - \
                        faccld2u[k+1]*(totradu - radmod)
                    totradu += rad
                    clrradu -= rad
                else:
                    #  --- ...  clear layer

                    #  --- ...  total sky radiance
                    radtotu = radtotu*trng + gasu
                    toturad[k, ib] = toturad[k, ib] + radtotu

                    #  --- ...  clear sky radiance
                    radclru = radclru*trng + gasu
                    clrurad[k, ib] = clrurad[k, ib] + radclru

        # -# Process longwave output from band for total and clear streams.
        # calculate upward, downward, and net flux.

        flxfac = wtdiff * fluxfac

        for k in range(nlp1):
            for ib in range(nbands):
                totuflux[k] = totuflux[k] + toturad[k, ib]
                totdflux[k] = totdflux[k] + totdrad[k, ib]
                totuclfl[k] = totuclfl[k] + clrurad[k, ib]
                totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

            totuflux[k] = totuflux[k] * flxfac
            totdflux[k] = totdflux[k] * flxfac
            totuclfl[k] = totuclfl[k] * flxfac
            totdclfl[k] = totdclfl[k] * flxfac

        #  --- ...  calculate net fluxes and heating rates
        fnet[0] = totuflux[0] - totdflux[0]

        for k in range(nlay):
            rfdelp[k] = heatfac / delp[k]
            fnet[k] = totuflux[k] - totdflux[k]
            htr [k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        # --- ...  optional clear sky heating rates
        if lhlw0:
            fnetc[0] = totuclfl[0] - totdclfl[0]

            for k in range(nlay):
                fnetc[k] = totuclfl[k] - totdclfl[k]
                htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

        # --- ...  optional spectral band heating rates
        if lhlwb:
            for ib in range(nbands):
                fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                for k in range(nlay):
                    fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                    htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

    return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb


def rtrnmc(semiss, delp, cldfmc, taucld, tautot, pklay, pklev, fracs, secdif,
           nlay, nlp1, lhlw0, lhlwb, exp_tbl, tau_tbl, tfn_tbl):
    #  ===================  program usage description  ===================  !
    # purpose:  compute the upward/downward radiative fluxes, and heating   !
    # rates for both clear or cloudy atmosphere.  clouds are treated with   !
    # the mcica stochastic approach.                                        !
    #                                                                       !
    # subprograms called:  none                                             !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                     -size-   !
    #   semiss  - real, lw surface emissivity                         nbands!
    #   delp    - real, layer pressure thickness (mb)                  nlay !
    #   cldfmc  - real, layer cloud fraction (sub-column)        ngptlw*nlay!
    #   taucld  - real, layer cloud opt depth                    nbands*nlay!
    #   tautot  - real, total optical depth (gas+aerosols)       ngptlw*nlay!
    #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
    #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
    #   fracs   - real, planck fractions                         ngptlw*nlay!
    #   secdif  - real, secant of diffusivity angle                   nbands!
    #   nlay    - integer, number of vertical layers                    1   !
    #   nlp1    - integer, number of vertical levels (interfaces)       1   !
    #                                                                       !
    #  outputs:                                                             !
    #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
    #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
    #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
    #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
    #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
    #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
    #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
    #                                                                       !
    #  module veriables:                                                    !
    #   ngb     - integer, band index for each g-value                ngptlw!
    #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
    #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
    #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
    #   bpade   - real, pade approx constant (1/0.278)                   1  !
    #   wtdiff  - real, weight for radiance to flux conversion           1  !
    #   ntbl    - integer, dimension of look-up tables                   1  !
    #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
    #   exp_tbl - real, transmittance lookup table                   0:ntbl !
    #   tfn_tbl - real, tau transition function                      0:ntbl !
    #                                                                       !
    #  local variables:                                                     !
    #    itgas  - integer, index for gases contribution look-up table    1  !
    #    ittot  - integer, index for gases plus clouds  look-up table    1  !
    #    reflct - real, surface reflectance                              1  !
    #    atrgas - real, gaseous absorptivity                             1  !
    #    atrtot - real, gaseous and cloud absorptivity                   1  !
    #    odcld  - real, cloud optical depth                              1  !
    #    efclrfr- real, effective clear sky fraction (1-efcldfr)        nlay!
    #    odepth - real, optical depth of gaseous only                    1  !
    #    odtot  - real, optical depth of gas and cloud                   1  !
    #    gasfac - real, gas-only pade factor, used for planck function   1  !
    #    totfac - real, gas and cloud pade factor, used for planck fn    1  !
    #    bbdgas - real, gas-only planck function for downward rt         1  !
    #    bbugas - real, gas-only planck function for upward rt           1  !
    #    bbdtot - real, gas and cloud planck function for downward rt    1  !
    #    bbutot - real, gas and cloud planck function for upward rt      1  !
    #    gassrcu- real, upwd source radiance due to gas                 nlay!
    #    totsrcu- real, upwd source radiance due to gas+cld             nlay!
    #    gassrcd- real, dnwd source radiance due to gas                  1  !
    #    totsrcd- real, dnwd source radiance due to gas+cld              1  !
    #    radtotu- real, spectrally summed total sky upwd radiance        1  !
    #    radclru- real, spectrally summed clear sky upwd radiance        1  !
    #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
    #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
    #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
    #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
    #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
    #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
    #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
    #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
    #                                                                       !
    #                                                                       !
    #  *******************************************************************  !
    #  original code description                                            !
    #                                                                       !
    #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
    #  revision for gcms:  michael j. iacono; october, 2002                 !
    #  revision for f90:   michael j. iacono; june, 2006                    !
    #                                                                       !
    #  this program calculates the upward fluxes, downward fluxes, and      !
    #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
    #  to this program is the atmospheric profile, all Planck function      !
    #  information, and the cloud fraction by layer.  a variable diffusivity!
    #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
    #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
    #  of the column water vapor, and other bands use a value of 1.66.  the !
    #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
    #  here.  note that use of the emissivity angle for the flux integration!
    #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
    #  clouds are treated with the mcica stochastic approach and            !
    #  maximum-random cloud overlap.                                        !
    #                                                                       !
    #  *******************************************************************  !
    #  ======================  end of description block  =================  !

    #  ---  outputs:
    htr = np.zeros(nlay)
    htrcl = np.zeros(nlay)
    htrb = np.zeros((nlay, nbands))

    totuflux = np.zeros(nlp1)
    totdflux = np.zeros(nlp1)
    totuclfl = np.zeros(nlp1)
    totdclfl = np.zeros(nlp1)

    #  ---  locals:
    rec_6 = 0.166667

    clrurad = np.zeros((nlp1, nbands))
    clrdrad = np.zeros((nlp1, nbands))
    toturad = np.zeros((nlp1, nbands))
    totdrad = np.zeros((nlp1, nbands))

    gassrcu = np.zeros(nlay)
    totsrcu = np.zeros(nlay)
    trngas = np.zeros(nlay)
    efclrfr = np.zeros(nlay)
    rfdelp = np.zeros(nlay)

    fnet = np.zeros(nlp1)
    fnetc = np.zeros(nlp1)

    tblint = ntbl

    #
    #===> ...  begin here
    #

    #  --- ...  loop over all g-points

    for ig in range(ngptlw):
        ib = ngb[ig]

        radtotd = 0.0
        radclrd = 0.0

        # Downward radiative transfer loop.
        # - Clear sky, gases contribution
        # - Total sky, gases+clouds contribution
        # - Cloudy layer
        # - Total sky radiance
        # - Clear sky radiance

        for k in range(nlay-1, None, -1):

            #  --- ...  clear sky, gases contribution

            odepth = max(0.0, secdif[ib]*tautot[ig, k])
            if odepth <= 0.06:
                atrgas = odepth - 0.5*odepth*odepth
                trng   = 1.0 - atrgas
                gasfac = rec_6 * odepth
            else:
                tblind = odepth / (bpade + odepth)
                itgas = tblint*tblind + 0.5
                trng  = exp_tbl(itgas)
                atrgas = 1.0 - trng
                gasfac = tfn_tbl(itgas)
                odepth = tau_tbl(itgas)

            plfrac = fracs[ig, k]
            blay = pklay[ib, k]

            dplnku = pklev[ib, k  ] - blay
            dplnkd = pklev[ib, k-1] - blay
            bbdgas = plfrac * (blay + dplnkd*gasfac)
            bbugas = plfrac * (blay + dplnku*gasfac)
            gassrcd= bbdgas * atrgas
            gassrcu[k] = bbugas * atrgas
            trngas[k] = trng

            #  --- ...  total sky, gases+clouds contribution

            clfm = cldfmc[ig, k]
            if clfm >= eps:
                #  --- ...  cloudy layer
                odcld = secdif[ib] * taucld[ib, k]
                efclrfr[k] = 1.0 - (1.0 - np.exp(-odcld))*clfm
                odtot = odepth + odcld
                if odtot < 0.06:
                    totfac = rec_6 * odtot
                    atrtot = odtot - 0.5*odtot*odtot
                else:
                    tblind = odtot / (bpade + odtot)
                    ittot  = tblint*tblind + 0.5
                    totfac = tfn_tbl[ittot]
                    atrtot = 1.0 - exp_tbl[ittot]

                bbdtot = plfrac * (blay + dplnkd*totfac)
                bbutot = plfrac * (blay + dplnku*totfac)
                totsrcd= bbdtot * atrtot
                totsrcu[k] = bbutot * atrtot

                #  --- ...  total sky radiance
                radtotd = radtotd*trng*efclrfr[k] + gassrcd + \
                    clfm*(totsrcd - gassrcd)
                totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                #  --- ...  clear sky radiance
                radclrd = radclrd*trng + gassrcd
                clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd

            else:
                #  --- ...  clear layer

                #  --- ...  total sky radiance
                radtotd = radtotd*trng + gassrcd
                totdrad[k-1, ib] = totdrad[k-1, ib] + radtotd

                #  --- ...  clear sky radiance
                radclrd = radclrd*trng + gassrcd
                clrdrad[k-1, ib] = clrdrad[k-1, ib] + radclrd

        #    Compute spectral emissivity & reflectance, include the
        #    contribution of spectrally varying longwave emissivity and
        #    reflection from the surface to the upward radiative transfer.

        #     note: spectral and Lambertian reflection are identical for the
        #           diffusivity angle flux integration used here.

        reflct = 1.0 - semiss[ib]
        rad0 = semiss[ib] * fracs[ig, 0] * pklay[ib, 0]

        # Compute total sky radiance
        radtotu = rad0 + reflct*radtotd
        toturad[0, ib] = toturad[0, ib] + radtotu

        # Compute clear sky radiance
        radclru = rad0 + reflct*radclrd
        clrurad[0, ib] = clrurad[0, ib] + radclru

        # Upward radiative transfer loop
        # - Compute total sky radiance
        # - Compute clear sky radiance

        # toturad holds summed radiance for total sky stream
        # clrurad holds summed radiance for clear sky stream

        for k in range(nlay):
            clfm = cldfmc[ig, k]
            trng = trngas[k]
            gasu = gassrcu[k]

            if clfm > eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu = radtotu*trng*efclrfr[k] + gasu + \
                    clfm*(totsrcu[k] - gasu)
                toturad[k, ib] = toturad[k, ib] + radtotu

                #  --- ... clear sky radiance
                radclru = radclru*trng + gasu
                clrurad[k, ib] = clrurad[k, ib] + radclru

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu = radtotu*trng + gasu
                toturad[k, ib] = toturad[k, ib] + radtotu

                #  --- ... clear sky radiance
                radclru = radclru*trng + gasu
                clrurad[k, ib] = clrurad[k, ib] + radclru

    # Process longwave output from band for total and clear streams.
    # Calculate upward, downward, and net flux.

    flxfac = wtdiff * fluxfac

    for k in range(nlp1):
        for ib in range(nbands):
            totuflux[k] = totuflux[k] + toturad[k, ib]
            totdflux[k] = totdflux[k] + totdrad[k, ib]
            totuclfl[k] = totuclfl[k] + clrurad[k, ib]
            totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

        totuflux[k] = totuflux[k] * flxfac
        totdflux[k] = totdflux[k] * flxfac
        totuclfl[k] = totuclfl[k] * flxfac
        totdclfl[k] = totdclfl[k] * flxfac

    #  --- ...  calculate net fluxes and heating rates
    fnet[0] = totuflux[0] - totdflux[0]

    for k in range(nlay):
        rfdelp[k] = heatfac / delp[k]
        fnet[k] = totuflux[k] - totdflux[k]
        htr [k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

    # --- ...  optional clear sky heating rates
    if lhlw0:
        fnetc[0] = totuclfl[0] - totdclfl[0]

        for k in range(nlay):
            fnetc[k] = totuclfl[k] - totdclfl[k]
            htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

    # --- ...  optional spectral band heating rates
    if lhlwb:
        for ib in range(nbands):
            fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

            for k in range(nlay):
                fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

    return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb