import numpy as np
import xarray as xr
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import (ilwrgas as ilwrgas,
                          icldflg as icldflg,
                          ilwcliq as ilwcliq,
                          ilwrate as ilwrate,
                          ilwcice as ilwcice)
from radlw_param import ntbl, nbands, nrates, delwave, ngptlw, ngb, absrain, abssnow0, ipat
from phys_const import (con_g as g,
                        con_cp as cp,
                        con_amd,
                        con_amw,
                        con_amo3)

class RadLWClass():
    VTAGLW='NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82'
    expeps = 1.e-20

    bpade   = 1.0/0.278
    eps = 1.0e-6
    oneminus = 1.0-eps
    cldmin = 1.0e-80
    stpfac = 296.0/1013.0
    wtdiff = 0.5
    tblint = ntbl
    
    amdw = con_amd/con_amw
    amdo3 = con_amd/con_amo3

    nspa = [1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9]
    nspb = [1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

    a0 = [1.66,  1.55,  1.58,  1.66,  1.54, 1.454,  1.89,  1.33,
          1.668,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66]
    a1 = [0.00,  0.25,  0.22,  0.00,  0.13, 0.446, -0.10,  0.40,
          -0.006,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]
    a2 = [0.00, -12.0, -11.7,  0.00, -0.72,-0.243,  0.19,-0.062,
          0.414,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]

    def __init__(self, me, iovrlw, isubclw):
        self.lhlwb = False
        self.lhlw0 = False
        self.lflxprf = False

        self.semiss0 = np.ones(nbands)

        self.iovrlw = iovrlw
        self.isubclw = isubclw

        self.exp_tbl = np.zeros(ntbl+1)
        self.tau_tbl = np.zeros(ntbl+1)
        self.tfn_tbl = np.zeros(ntbl+1)


        expeps = 1e-20

        if self.iovrlw < 0 or self.iovrlw > 3:
            print(f'  *** Error in specification of cloud overlap flag',
                    f' IOVRLW={self.iovrlw}, in RLWINIT !!')
        elif iovrlw >= 2 and isubclw == 0:
            if me == 0:
                print(f'  *** IOVRLW={self.iovrlw} is not available for',
                        ' ISUBCLW=0 setting!!')
                print('      The program uses maximum/random overlap instead.')

        if me == 0:
            print(f'- Using AER Longwave Radiation, Version: {self.VTAGLW}')

            if ilwrgas > 0:
                print('   --- Include rare gases N2O, CH4, O2, CFCs ',
                        'absorptions in LW')
            else:
                print('   --- Rare gases effect is NOT included in LW')

            if self.isubclw == 0:
                print('   --- Using standard grid average clouds, no ',
                        '   sub-column clouds approximation applied')
            elif self.isubclw == 1:
                print('   --- Using MCICA sub-colum clouds approximation ',
                        '   with a prescribed sequence of permutaion seeds')
            elif self.isubclw == 2:
                print('   --- Using MCICA sub-colum clouds approximation ',
                        '   with provided input array of permutation seeds')
            else:
                print(f'  *** Error in specification of sub-column cloud ',
                        f' control flag isubclw = {self.isubclw}!!')

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
            print('*** Model cloud scheme inconsistent with LW',
                    'radiation cloud radiative property setup !!')

        #  --- ...  setup constant factors for flux and heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        pival = 2.0 * np.arcsin(1.0)
        self.fluxfac = pival * 2.0e4

        if ilwrate == 1:
            self.heatfac = g * 864.0 / cp            #   (in k/day)
        else:
            self.heatfac = g * 1.0e-2 / cp           #   (in k/second)

        #  --- ...  compute lookup tables for transmittance, tau transition
        #           function, and clear sky tau (for the cloudy sky radiative
        #           transfer).  tau is computed as a function of the tau
        #           transition function, transmittance is calculated as a
        #           function of tau, and the tau transition function is
        #           calculated using the linear in tau formulation at values of
        #           tau above 0.01.  tf is approximated as tau/6 for tau < 0.01.
        #           all tables are computed at intervals of 0.001.  the inverse
        #           of the constant used in the pade approximation to the tau
        #           transition function is set to b.

        self.tau_tbl[0] = 0.0
        self.exp_tbl[0] = 1.0
        self.tfn_tbl[0] = 0.0

        self.tau_tbl[ntbl] = 1.e10
        self.exp_tbl[ntbl] = expeps
        self.tfn_tbl[ntbl] = 1.0

        explimit = int(np.floor(-np.log(np.finfo(float).tiny)))

        for i in range(1, ntbl):
            tfn = (i) / (ntbl-i)
            self.tau_tbl[i] = self.bpade * tfn
            if self.tau_tbl[i] >= explimit:
                self.exp_tbl[i] = expeps
            else:
                self.exp_tbl[i] = np.exp(-self.tau_tbl[i])

            if self.tau_tbl[i] < 0.06:
                self.tfn_tbl[i] = self.tau_tbl[i] / 6.0
            else:
                self.tfn_tbl[i] = 1.0 - 2.0*((1.0 / self.tau_tbl[i]) - (self.exp_tbl[i]/(1.0 - self.exp_tbl[i])))

    def return_initdata(self):
        outdict = {'semiss0': self.semiss0,
                   'fluxfac': self.fluxfac,
                   'heatfac': self.heatfac,
                   'exp_tbl': self.exp_tbl,
                   'tau_tbl': self.tau_tbl,
                   'tfn_tbl': self.tfn_tbl}
        return outdict

    def setcoef(self, pavel, tavel, tz, stemp, h2ovmr, colamt, coldry, colbrd,
                nlay, nlp1):

        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                       -size- !
        #   pavel     - real, layer pressures (mb)                         nlay !
        #   tavel     - real, layer temperatures (k)                       nlay !
        #   tz        - real, level (interface) temperatures (k)         0:nlay !
        #   stemp     - real, surface ground temperature (k)                1   !
        #   h2ovmr    - real, layer w.v. volum mixing ratio (kg/kg)        nlay !
        #   colamt    - real, column amounts of absorbing gases      nlay*maxgas!
        #                 2nd indices range: 1-maxgas, for watervapor,          !
        #                 carbon dioxide, ozone, nitrous oxide, methane,        !
        #                 oxigen, carbon monoxide,etc. (molecules/cm**2)        !
        #   coldry    - real, dry air column amount                        nlay !
        #   colbrd    - real, column amount of broadening gases            nlay !
        #   nlay/nlp1 - integer, total number of vertical layers, levels    1   !
        #                                                                       !
        #  outputs:                                                             !
        #   laytrop   - integer, tropopause layer index (unitless)          1   !
        #   pklay     - real, integrated planck func at lay temp   nbands*0:nlay!
        #   pklev     - real, integrated planck func at lev temp   nbands*0:nlay!
        #   jp        - real, indices of lower reference pressure          nlay !
        #   jt, jt1   - real, indices of lower reference temperatures      nlay !
        #   rfrate    - real, ref ratios of binary species param   nlay*nrates*2!
        #     (:,m,:)m=1-h2o/co2,2-h2o/o3,3-h2o/n2o,4-h2o/ch4,5-n2o/co2,6-o3/co2!
        #     (:,:,n)n=1,2: the rates of ref press at the 2 sides of the layer  !
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
        #   minorfrac - real, factor for minor gases                       nlay !
        #   scaleminor,scaleminorn2                                             !
        #             - real, scale factors for minor gases                nlay !
        #   indminor  - integer, index of lower ref temp for minor gases   nlay !
        #                                                                       !
        #  ======================    end of definitions    ===================  !


        #===> ... begin here
        #
        #  --- ...  calculate information needed by the radiative transfer routine
        #           that is specific to this atmosphere, especially some of the
        #           coefficients and indices needed to compute the optical depths
        #           by interpolating data from stored reference atmospheres.

        dfile = '../lookupdata/totplnk.nc'
        pfile = '../lookupdata/radlw_ref_data.nc'
        totplnk = xr.open_dataset(dfile)['totplnk'].data
        preflog = xr.open_dataset(pfile)['preflog'].data
        tref = xr.open_dataset(pfile)['tref'].data
        chi_mls = xr.open_dataset(pfile)['chi_mls'].data

        pklay = np.zeros((nbands, nlp1))
        pklev = np.zeros((nbands, nlp1))

        jp = np.zeros(nlay, dtype=np.int32)
        jt = np.zeros(nlay, dtype=np.int32)
        jt1 = np.zeros(nlay, dtype=np.int32)
        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)
        selffac = np.zeros(nlay)
        scaleminor = np.zeros(nlay)
        scaleminorn2 = np.zeros(nlay)
        indminor = np.zeros(nlay)
        minorfrac = np.zeros(nlay)
        indfor = np.zeros(nlay)
        indself = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        rfrate = np.zeros((nlay, nrates, 2))



        indlay = np.minimum(180, np.maximum(1, int(stemp-159.0)))
        indlev = np.minimum(180, np.maximum(1, int(tz[0]-159.0) ))
        tlyrfr = stemp - int(stemp)
        tlvlfr = tz[0] - int(tz[0])

        for i in range(nbands):
            tem1 = totplnk[indlay, i] - totplnk[indlay-1, i]
            tem2 = totplnk[indlev, i] - totplnk[indlev-1, i]
            pklay[i, 0] = delwave[i] * (totplnk[indlay-1, i] + tlyrfr*tem1)
            pklev[i, 0] = delwave[i] * (totplnk[indlev-1, i] + tlvlfr*tem2)


        #  --- ...  begin layer loop
        #           calculate the integrated Planck functions for each band at the
        #           surface, level, and layer temperatures.

        laytrop = 0

        for k in range(nlay):
            indlay = np.minimum(180, np.maximum(1, int(tavel[k]-159.0)))
            tlyrfr = tavel[k] - int(tavel[k])

            indlev = np.minimum(180, np.maximum(1, int(tz[k+1]-159.0)))
            tlvlfr = tz[k+1] - int(tz[k+1])

            #  --- ...  begin spectral band loop

            for i in range(nbands):
                pklay[i, k+1] = delwave[i] * (totplnk[indlay-1, i] + tlyrfr
                                            * (totplnk[indlay, i] - \
                    totplnk[indlay-1, i]))
                pklev[i, k+1] = delwave[i] * (totplnk[indlev-1, i] + tlvlfr
                                            * (totplnk[indlev, i] - \
                    totplnk[indlev-1, i]))

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure. store them in jp and jp1. store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = np.log(pavel[k])
            jp[k] = np.maximum(1, np.minimum(58, int(36.0 - 5.0*(plog+0.04))))-1
            jp1 = jp[k] + 1
            #  --- ...  limit pressure extrapolation at the top
            fp = np.maximum(0.0, np.minimum(1.0, 5.0*(preflog[jp[k]]-plog)))

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #           reference temperature (these are different for each
            #           reference pressure) is nearest the layer temperature but does
            #           not exceed it. store these indices in jt and jt1, resp.
            #           store in ft (resp. ft1) the fraction of the way between jt
            #           (jt1) and the next highest reference temperature that the
            #           layer temperature falls.

            tem1 = (tavel[k]-tref[jp[k]]) / 15.0
            tem2 = (tavel[k]-tref[jp1]) / 15.0
            jt[k] = np.maximum(1, np.minimum(4, int(3.0 + tem1)))-1
            jt1[k] = np.maximum(1, np.minimum(4, int(3.0 + tem2)))-1
            #  --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
            ft  = np.maximum(-0.5, np.minimum(1.5, tem1 - float(jt[k] - 2)))
            ft1 = np.maximum(-0.5, np.minimum(1.5, tem2 - float(jt1[k] - 2)))

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n)

            tem1 = 1.0- fp
            fac10[k] = tem1 * ft
            fac00[k] = tem1 * (1.0 - ft)
            fac11[k] = fp * ft1
            fac01[k] = fp * (1.0 - ft1)

            forfac[k] = pavel[k]*self.stpfac / (tavel[k]*(1.0 + h2ovmr[k]))
            selffac[k] = h2ovmr[k] * forfac[k]

            #  --- ...  set up factors needed to separately include the minor gases
            #           in the calculation of absorption coefficient

            scaleminor[k] = pavel[k] / tavel[k]
            scaleminorn2[k] = (pavel[k] / tavel[k]) * \
                (colbrd[k]/(coldry[k] + colamt[k, 0]))
            tem1 = (tavel[k] - 180.8) / 7.2
            indminor[k] = np.minimum(18, np.maximum(1, int(tem1)))
            minorfrac[k] = tem1 - float(indminor[k])

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            if plog > 4.56:
                laytrop =  laytrop + 1

                tem1 = (332.0 - tavel[k]) / 36.0
                indfor[k] = np.minimum(2, np.maximum(1, int(tem1)))
                forfrac[k] = tem1 - float(indfor[k])

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem1 = (tavel[k] - 188.0) / 7.2
                indself[k] = np.minimum(9, np.maximum(1, int(tem1)-7))
                selffrac[k] = tem1 - float(indself[k] + 7)

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in lower atmosphere.

                rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 0, 1] = chi_mls[0, jp[k]+1] / chi_mls[1, jp[k]+1]

                rfrate[k, 1, 0] = chi_mls[0, jp[k]] / chi_mls[2, jp[k]]
                rfrate[k, 1, 1] = chi_mls[0, jp[k]+1] / chi_mls[2, jp[k]+1]

                rfrate[k, 2, 0] = chi_mls[0, jp[k]] / chi_mls[3, jp[k]]
                rfrate[k, 2, 1] = chi_mls[0, jp[k]+1] / chi_mls[3, jp[k]+1]

                rfrate[k, 3, 0] = chi_mls[0, jp[k]] / chi_mls[5, jp[k]]
                rfrate[k, 3, 1] = chi_mls[0, jp[k]+1] / chi_mls[5, jp[k]+1]

                rfrate[k, 4, 0] = chi_mls[3, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 4, 1] = chi_mls[3, jp[k]+1] / chi_mls[1, jp[k]+1]

            else:

                tem1 = (tavel[k] - 188.0) / 36.0
                indfor[k] = 3
                forfrac[k] = tem1 - 1.0

                indself[k] = 0
                selffrac[k] = 0.0

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in upper atmosphere.

                rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 0, 1] = chi_mls[0, jp[k]+1] / chi_mls[1, jp[k]+1]

                rfrate[k, 5, 0] = chi_mls[2, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 5, 1] = chi_mls[2, jp[k]+1] / chi_mls[1, jp[k]+1]

            #  --- ...  rescale selffac and forfac for use in taumol

            selffac[k] = colamt[k, 0] * selffac[k]
            forfac[k]  = colamt[k, 0] * forfac[k]

        return (laytrop, pklay, pklev, jp, jt, jt1, rfrate, fac00, fac01, fac10,
                fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
                minorfrac, scaleminor, scaleminorn2, indminor)

    def rtrn(self, semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs,
             secdif, nlay, nlp1):
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

        #
        #===> ...  begin here
        #

        #  --- ...  loop over all g-points

        for ig in range(ngptlw):
            ib = ngb(ig)

            radtotd = 0.0
            radclrd = 0.0

            #> -# Downward radiative transfer loop.

            for k in range(nlay-1, -1, -1):

                #  - clear sky, gases contribution

                odepth = max(0.0, secdif[ib]*tautot[ig, k])
                if odepth <= 0.06:
                    atrgas = odepth - 0.5*odepth*odepth
                    trng   = 1.0 - atrgas
                    gasfac = rec_6 * odepth
                else:
                    tblind = odepth / (self.bpade + odepth)
                    itgas = self.tblint*tblind + 0.5
                    trng  = self.exp_tbl[itgas]
                    atrgas = 1.0 - trng
                    gasfac = self.tfn_tbl[itgas]
                    odepth = self.tau_tbl[itgas]

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
                if clfr >= self.eps:
                    #\n  - cloudy layer

                    odcld = secdif[ib] * taucld[ib, k]
                    efclrfr[k] = 1.0-(1.0 - np.exp(-odcld))*clfr
                    odtot = odepth + odcld
                    if odtot < 0.06:
                        totfac = rec_6 * odtot
                        atrtot = odtot - 0.5*odtot*odtot
                    else:
                        tblind = odtot / (self.bpade + odtot)
                        ittot  = self.tblint*tblind + 0.5
                        totfac = self.tfn_tbl[ittot]
                        atrtot = 1.0 - self.exp_tbl[ittot]

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

            reflct = 1.0 - semiss[ib]
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

                if clfr >= self.eps:
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

        flxfac = self.wtdiff * self.fluxfac

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
            rfdelp[k] = self.heatfac / delp[k]
            fnet[k] = totuflux[k] - totdflux[k]
            htr[k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        # --- ...  optional clear sky heating rates
        if self.lhlw0:
            fnetc[0] = totuclfl[0] - totdclfl[0]

            for k in range(nlay):
                fnetc[k] = totuclfl[k] - totdclfl[k]
                htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

        # --- ...  optional spectral band heating rates
        if self.lhlwb:
            for ib in range(nbands):
                fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                for k in range(nlay):
                    fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                    htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def rtrnmr(self, semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs,
               secdif, nlay, nlp1):
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

        lstcldu[0] = cldfrc[0] > self.eps
        rat1 = 0.0
        rat2 = 0.0

        for k in range(nlay-1):

            lstcldu[k+1] = cldfrc[k+1] > self.eps and cldfrc[k] <= self.eps

            if cldfrc[k] > self.eps:
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

        lstcldd[nlay] = cldfrc[nlay] > self.eps
        rat1 = 0.0
        rat2 = 0.0

        for k in range(nlay-1, 0, -1):
            lstcldd[k-1] = cldfrc[k-1] > self.eps and cldfrc[k] <= self.eps

            if cldfrc[k] > self.eps:
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
                        tblind = odepth / (self.bpade + odepth)
                        itgas = tblint*tblind + 0.5
                        trng  = self.exp_tbl[itgas]
                        atrgas = 1.0 - trng
                        gasfac = self.tfn_tbl[itgas]
                        odepth = self.tau_tbl[itgas]

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

                    if clfr >= self.eps:
                        #  - cloudy layer

                        odcld = secdif[ib] * taucld[ib, k]
                        odtot = odepth + odcld
                        if odtot < 0.06:
                            totfac = rec_6 * odtot
                            atrtot = odtot - 0.5*odtot*odtot
                            trnt   = 1.0 - atrtot
                        else:
                            tblind = odtot / (self.bpade + odtot)
                            ittot  = tblint*tblind + 0.5
                            totfac = self.tfn_tbl[ittot]
                            trnt   = self.exp_tbl[ittot]
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

                    if clfr >= self.eps:
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

            flxfac = self.wtdiff * self.fluxfac

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
                rfdelp[k] = self.heatfac / delp[k]
                fnet[k] = totuflux[k] - totdflux[k]
                htr [k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

            # --- ...  optional clear sky heating rates
            if self.lhlw0:
                fnetc[0] = totuclfl[0] - totdclfl[0]

                for k in range(nlay):
                    fnetc[k] = totuclfl[k] - totdclfl[k]
                    htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

            # --- ...  optional spectral band heating rates
            if self.lhlwb:
                for ib in range(nbands):
                    fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                    for k in range(nlay):
                        fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                        htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def rtrnmc(self, semiss, delp, cldfmc, taucld, tautot, pklay, pklev, fracs,
               secdif, nlay, nlp1):
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
                    tblind = odepth / (self.bpade + odepth)
                    itgas = tblint*tblind + 0.5
                    trng  = self.exp_tbl(itgas)
                    atrgas = 1.0 - trng
                    gasfac = self.tfn_tbl(itgas)
                    odepth = self.tau_tbl(itgas)

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
                if clfm >= self.eps:
                    #  --- ...  cloudy layer
                    odcld = secdif[ib] * taucld[ib, k]
                    efclrfr[k] = 1.0 - (1.0 - np.exp(-odcld))*clfm
                    odtot = odepth + odcld
                    if odtot < 0.06:
                        totfac = rec_6 * odtot
                        atrtot = odtot - 0.5*odtot*odtot
                    else:
                        tblind = odtot / (self.bpade + odtot)
                        ittot  = tblint*tblind + 0.5
                        totfac = self.tfn_tbl[ittot]
                        atrtot = 1.0 - self.exp_tbl[ittot]

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

                if clfm > self.eps:
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

        flxfac = self.wtdiff * self.fluxfac

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
            rfdelp[k] = self.heatfac / delp[k]
            fnet[k] = totuflux[k] - totdflux[k]
            htr [k] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        # --- ...  optional clear sky heating rates
        if self.lhlw0:
            fnetc[0] = totuclfl[0] - totdclfl[0]

            for k in range(nlay):
                fnetc[k] = totuclfl[k] - totdclfl[k]
                htrcl[k] = (fnetc[k-1] - fnetc[k]) * rfdelp[k]

        # --- ...  optional spectral band heating rates
        if self.lhlwb:
            for ib in range(nbands):
                fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                for k in range(nlay):
                    fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                    htrb[k, ib] = (fnet[k-1] - fnet[k]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def cldprop(self, cfrac, cliqp, reliq, cicep, reice, cdat1, cdat2, cdat3, cdat4,  
                nlay, nlp1, ipseed, dz, de_lgth):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        # purpose:  compute the cloud optical depth(s) for each cloudy layer    !
        # and g-point interval.                                                 !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                       -size- !
        #    cfrac - real, layer cloud fraction                          0:nlp1 !
        #        .....  for ilwcliq > 0  (prognostic cloud sckeme)  - - -       !
        #    cliqp - real, layer in-cloud liq water path (g/m**2)          nlay !
        #    reliq - real, mean eff radius for liq cloud (micron)          nlay !
        #    cicep - real, layer in-cloud ice water path (g/m**2)          nlay !
        #    reice - real, mean eff radius for ice cloud (micron)          nlay !
        #    cdat1 - real, layer rain drop water path  (g/m**2)            nlay !
        #    cdat2 - real, effective radius for rain drop (microm)         nlay !
        #    cdat3 - real, layer snow flake water path (g/m**2)            nlay !
        #    cdat4 - real, effective radius for snow flakes (micron)       nlay !
        #        .....  for ilwcliq = 0  (diagnostic cloud sckeme)  - - -       !
        #    cdat1 - real, input cloud optical depth                       nlay !
        #    cdat2 - real, layer cloud single scattering albedo            nlay !
        #    cdat3 - real, layer cloud asymmetry factor                    nlay !
        #    cdat4 - real, optional use                                    nlay !
        #    cliqp - not used                                              nlay !
        #    reliq - not used                                              nlay !
        #    cicep - not used                                              nlay !
        #    reice - not used                                              nlay !
        #                                                                       !
        #    dz     - real, layer thickness (km)                           nlay !
        #    de_lgth- real, layer cloud decorrelation length (km)             1 !
        #    nlay  - integer, number of vertical layers                      1  !
        #    nlp1  - integer, number of vertical levels                      1  !
        #    ipseed- permutation seed for generating random numbers (isubclw>0) !
        #                                                                       !
        #  outputs:                                                             !
        #    cldfmc - real, cloud fraction for each sub-column       ngptlw*nlay!
        #    taucld - real, cld opt depth for bands (non-mcica)      nbands*nlay!
        #                                                                       !
        #  explanation of the method for each value of ilwcliq, and ilwcice.    !
        #    set up in module "module_radlw_cntr_para"                          !
        #                                                                       !
        #     ilwcliq=0  : input cloud optical property (tau, ssa, asy).        !
        #                  (used for diagnostic cloud method)                   !
        #     ilwcliq>0  : input cloud liq/ice path and effective radius, also  !
        #                  require the user of 'ilwcice' to specify the method  !
        #                  used to compute aborption due to water/ice parts.    !
        #  ...................................................................  !
        #                                                                       !
        #     ilwcliq=1:   the water droplet effective radius (microns) is input!
        #                  and the opt depths due to water clouds are computed  !
        #                  as in hu and stamnes, j., clim., 6, 728-742, (1993). !
        #                  the values for absorption coefficients appropriate for
        #                  the spectral bands in rrtm have been obtained for a  !
        #                  range of effective radii by an averaging procedure   !
        #                  based on the work of j. pinto (private communication).
        #                  linear interpolation is used to get the absorption   !
        #                  coefficients for the input effective radius.         !
        #                                                                       !
        #     ilwcice=1:   the cloud ice path (g/m2) and ice effective radius   !
        #                  (microns) are input and the optical depths due to ice!
        #                  clouds are computed as in ebert and curry, jgr, 97,  !
        #                  3831-3836 (1992).  the spectral regions in this work !
        #                  have been matched with the spectral bands in rrtm to !
        #                  as great an extent as possible:                      !
        #                     e&c 1      ib = 5      rrtm bands 9-16            !
        #                     e&c 2      ib = 4      rrtm bands 6-8             !
        #                     e&c 3      ib = 3      rrtm bands 3-5             !
        #                     e&c 4      ib = 2      rrtm band 2                !
        #                     e&c 5      ib = 1      rrtm band 1                !
        #     ilwcice=2:   the cloud ice path (g/m2) and ice effective radius   !
        #                  (microns) are input and the optical depths due to ice!
        #                  clouds are computed as in rt code, streamer v3.0     !
        #                  (ref: key j., streamer user's guide, cooperative     !
        #                  institute for meteorological satellite studies, 2001,!
        #                  96 pp.) valid range of values for re are between 5.0 !
        #                  and 131.0 micron.                                    !
        #     ilwcice=3:   the ice generalized effective size (dge) is input and!
        #                  the optical properties, are calculated as in q. fu,  !
        #                  j. climate, (1998). q. fu provided high resolution   !
        #                  tales which were appropriately averaged for the bands!
        #                  in rrtm_lw. linear interpolation is used to get the  !
        #                  coeff from the stored tables. valid range of values  !
        #                  for deg are between 5.0 and 140.0 micron.            !
        #                                                                       !
        #  other cloud control module variables:                                !
        #     isubclw =0: standard cloud scheme, no sub-col cloud approximation !
        #             >0: mcica sub-col cloud scheme using ipseed as permutation!
        #                 seed for generating rundom numbers                    !
        #                                                                       !
        #  ======================  end of description block  =================  !
        #
    
        #
        #===> ...  begin here
        #
        cldmin = 1.0e-80

        taucld = np.zeros((nbands, nlay))
        tauice = np.zeros(nbands)
        tauliq = np.zeros(nbands)
        cldfmc = np.zeros((ngptlw, nlay))
        cldf = np.zeros(nlay)

        ds = '../lookupdata/radlw_cldprlw_data.nc'
        absliq1 = ds['absliq1']
        absice0 = ds['absice0']
        absice1 = ds['absice1']
        absice2 = ds['absice2']
        absice3 = ds['absice3']
    
        # Compute cloud radiative properties for a cloudy column:
        # - Compute cloud radiative properties for rain and snow (tauran,tausnw)
        # - Calculation of absorption coefficients due to water clouds(tauliq)
        # - Calculation of absorption coefficients due to ice clouds (tauice).
        # - For prognostic cloud scheme: sum up the cloud optical property:
        #   \f$ taucld=tauice+tauliq+tauran+tausnw \f$
    
        #  --- ...  compute cloud radiative properties for a cloudy column
    
        if ilwcliq > 0:
            for k in range(nlay):
                if cfrac[k] > cldmin:
                    tauran = absrain * cdat1[k]                      # ncar formula

                    #  ---  if use fu's formula it needs to be normalized by snow density
                    #       !not use snow density = 0.1 g/cm**3 = 0.1 g/(mu * m**2)
                    #       use ice density = 0.9167 g/cm**3 = 0.9167 g/(mu * m**2)
                    #       factor 1.5396=8/(3*sqrt(3)) converts reff to generalized ice particle size
                    #       use newer factor value 1.0315
                    #       1/(0.9167*1.0315) = 1.05756
                    if cdat3[k] > 0.0 and cdat4[k] > 10.0:
                        tausnw = abssnow0*1.05756*cdat3[k]/cdat4[k]      # fu's formula
                    else:
                        tausnw = 0.0
    
                    cldliq = cliqp[k]
                    cldice = cicep[k]
                    refliq = reliq[k]
                    refice = reice[k]
    
                    #  --- ...  calculation of absorption coefficients due to water clouds.
    
                    if cldliq <= 0.0:
                        for ib in range(nbands):
                            tauliq[ib] = 0.0
                    else:
                        if ilwcliq == 1:
                            factor = refliq - 1.5
                            index  = max(1, min(57, int(factor)))
                            fint   = factor - float(index)
    
                        for ib in range(nbands):
                            tauliq[ib] = max(0.0, cldliq*(absliq1[index, ib] + \
                                fint*(absliq1[index+1, ib]-absliq1[index, ib])))
    
                    #  --- ...  calculation of absorption coefficients due to ice clouds.
                    if cldice <= 0.0:
                        for ib in range(nbands):
                            tauice[ib] = 0.0
                    else:
                        #  --- ...  ebert and curry approach for all particle sizes though somewhat
                        #           unjustified for large ice particles
                        if ilwcice == 1:
                            refice = min(130.0, max(13.0, np.real(refice)))

                            for ib in range(nbands):
                                ia = ipat[ib]             # eb_&_c band index for ice cloud coeff
                                tauice[ib] = max(0.0, cldice*(absice1[0, ia] + \
                                    absice1[1, ia]/refice))

                            #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                            #           and ebert and curry approach for ice eff radius greater than 131.0 microns.
                            #           no smoothing between the transition of the two methods.

                        elif ilwcice == 2:
                            factor = (refice - 2.0) / 3.0
                            index  = max(1, min(42, int(factor)))
                            fint = factor - float(index)

                            for ib in range(nbands):
                                tauice[ib] = max(0.0, cldice*(absice2[index, ib] + \
                                    fint*(absice2[index+1, ib] - absice2[index, ib])))
    
                        #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                        #           (generalized effective size from 5 to 140 microns)
                
                        elif ilwcice == 3:
                            dgeice = max(5.0, 1.0315*refice)              # v4.71 value
                            factor = (dgeice - 2.0) / 3.0
                            index  = max(1, min(45, int(factor)))
                            fint   = factor - float(index)
                
                            for ib in range(nbands):
                                tauice[ib] = max(0.0, cldice*(absice3[index, ib] + \
                                    fint*(absice3[index+1, ib] - absice3[index, ib])))
                
                            for ib in range(nbands):
                                taucld[ib, k] = tauice[ib] + tauliq[ib] + tauran + tausnw
    
            else:
    
                for k in range(nlay):
                    if cfrac[k] > cldmin:
                        for ib in range(nbands):
                            taucld[ib, k] = cdat1[k]
    
        # -# if physparam::isubclw > 0, call mcica_subcol() to distribute
        #    cloud properties to each g-point.
    
        if self.isubclw > 0:    # mcica sub-col clouds approx
            for k in range(nlay):
                if cfrac[k] < cldmin:
                    cldf[k] = 0.0
                else:
                    cldf[k] = cfrac[k]
    
            #  --- ...  call sub-column cloud generator
    
            lcloudy = self.mcica_subcol(cldf, nlay, ipseed, dz, de_lgth)
    
            for k in range(nlay):
                for ig in range(ngptlw):
                    if lcloudy[ig, k]:
                        cldfmc[ig, k] = 1.0
                    else:
                        cldfmc[ig, k] = 0.0

        return cldfmc, taucld
