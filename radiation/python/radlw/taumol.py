import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import (ngb, ngptlw, nspa, nspb, oneminus,
                        ng01, ng02, ng03, ng04, ng05, ng06, ng07, ng08,
                        ng09, ng10, ng11, ng12, ng13, ng14, ng15, ng16,
                        ns01, ns02, ns03, ns04, ns05, ns06, ns07, ns08,
                        ns09, ns10, ns11, ns12, ns13, ns14, ns15, ns16)


def taumol(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay):

    #  ************    original subprogram description    ***************   !
    #                                                                       !
    #                  optical depths developed for the                     !
    #                                                                       !
    #                rapid radiative transfer model (rrtm)                  !
    #                                                                       !
    #            atmospheric and environmental research, inc.               !
    #                        131 hartwell avenue                            !
    #                        lexington, ma 02421                            !
    #                                                                       !
    #                           eli j. mlawer                               !
    #                         jennifer delamere                             !
    #                         steven j. taubman                             !
    #                         shepard a. clough                             !
    #                                                                       !
    #                       email:  mlawer@aer.com                          !
    #                       email:  jdelamer@aer.com                        !
    #                                                                       !
    #        the authors wish to acknowledge the contributions of the       !
    #        following people:  karen cady-pereira, patrick d. brown,       !
    #        michael j. iacono, ronald e. farren, luke chen,                !
    #        robert bergstrom.                                              !
    #                                                                       !
    #  revision for g-point reduction: michael j. iacono; aer, inc.         !
    #                                                                       !
    #     taumol                                                            !
    #                                                                       !
    #     this file contains the subroutines taugbn (where n goes from      !
    #     1 to 16).  taugbn calculates the optical depths and planck        !
    #     fractions per g-value and layer for band n.                       !
    #                                                                       !
    #  *******************************************************************  !
    #  ==================   program usage description   ==================  !
    #                                                                       !
    #    call  taumol                                                       !
    #       inputs:                                                         !
    #          ( laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,              !
    #            rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,                  !
    #            selffac,selffrac,indself,forfac,forfrac,indfor,            !
    #            minorfrac,scaleminor,scaleminorn2,indminor,                !
    #            nlay,                                                      !
    #       outputs:                                                        !
    #            fracs, tautot )                                            !
    #                                                                       !
    #  subprograms called:  taugb## (## = 01 -16)                           !
    #                                                                       !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                        size  !
    #     laytrop   - integer, tropopause layer index (unitless)        1   !
    #                   layer at which switch is made for key species       !
    #     pavel     - real, layer pressures (mb)                       nlay !
    #     coldry    - real, column amount for dry air (mol/cm2)        nlay !
    #     colamt    - real, column amounts of h2o, co2, o3, n2o, ch4,       !
    #                   o2, co (mol/cm**2)                       nlay*maxgas!
    #     colbrd    - real, column amount of broadening gases          nlay !
    #     wx        - real, cross-section amounts(mol/cm2)      nlay*maxxsec!
    #     tauaer    - real, aerosol optical depth               nbands*nlay !
    #     rfrate    - real, reference ratios of binary species parameter    !
    #     (:,m,:)m=1-h2o/co2,2-h2o/o3,3-h2o/n2o,4-h2o/ch4,5-n2o/co2,6-o3/co2!
    #     (:,:,n)n=1,2: the rates of ref press at the 2 sides of the layer  !
    #                                                          nlay*nrates*2!
    #     facij     - real, factors multiply the reference ks, i,j of 0/1   !
    #                   for lower/higher of the 2 appropriate temperatures  !
    #                   and altitudes                                  nlay !
    #     jp        - real, index of lower reference pressure          nlay !
    #     jt, jt1   - real, indices of lower reference temperatures    nlay !
    #                   for pressure levels jp and jp+1, respectively       !
    #     selffac   - real, scale factor for water vapor self-continuum     !
    #                   equals (water vapor density)/(atmospheric density   !
    #                   at 296k and 1013 mb)                           nlay !
    #     selffrac  - real, factor for temperature interpolation of         !
    #                   reference water vapor self-continuum data      nlay !
    #     indself   - integer, index of lower reference temperature for     !
    #                   the self-continuum interpolation               nlay !
    #     forfac    - real, scale factor for w. v. foreign-continuum   nlay !
    #     forfrac   - real, factor for temperature interpolation of         !
    #                   reference w.v. foreign-continuum data          nlay !
    #     indfor    - integer, index of lower reference temperature for     !
    #                   the foreign-continuum interpolation            nlay !
    #     minorfrac - real, factor for minor gases                     nlay !
    #     scaleminor,scaleminorn2                                           !
    #               - real, scale factors for minor gases              nlay !
    #     indminor  - integer, index of lower reference temperature for     !
    #                   minor gases                                    nlay !
    #     nlay      - integer, total number of layers                   1   !
    #                                                                       !
    #  outputs:                                                             !
    #     fracs     - real, planck fractions                     ngptlw,nlay!
    #     tautot    - real, total optical depth (gas+aerosols)   ngptlw,nlay!
    #                                                                       !
    #  internal variables:                                                  !
    #     ng##      - integer, number of g-values in band ## (##=01-16) 1   !
    #     nspa      - integer, for lower atmosphere, the number of ref      !
    #                   atmos, each has different relative amounts of the   !
    #                   key species for the band                      nbands!
    #     nspb      - integer, same but for upper atmosphere          nbands!
    #     absa      - real, k-values for lower ref atmospheres (no w.v.     !
    #                   self-continuum) (cm**2/molecule)  nspa(##)*5*13*ng##!
    #     absb      - real, k-values for high ref atmospheres (all sources) !
    #                   (cm**2/molecule)               nspb(##)*5*13:59*ng##!
    #     ka_m'mgas'- real, k-values for low ref atmospheres minor species  !
    #                   (cm**2/molecule)                          mmn##*ng##!
    #     kb_m'mgas'- real, k-values for high ref atmospheres minor species !
    #                   (cm**2/molecule)                          mmn##*ng##!
    #     selfref   - real, k-values for w.v. self-continuum for ref atmos  !
    #                   used below laytrop (cm**2/mol)               10*ng##!
    #     forref    - real, k-values for w.v. foreign-continuum for ref atmos
    #                   used below/above laytrop (cm**2/mol)          4*ng##!
    #                                                                       !
    #  ******************************************************************   !

    #
    #===> ...  begin here
    #
    taug, fracs = taugb01(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay)

    taug, fracs, tauself = taugb02(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb03(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs, tauself)
    taug, fracs = taugb04(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb05(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb06(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb07(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb08(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb09(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb10(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb11(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb12(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs, taufor = taugb13(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb14(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs, taufor)
    taug, fracs = taugb15(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)
    taug, fracs = taugb16(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
                          rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
                          selffac,selffrac,indself,forfac,forfrac,indfor,
                          minorfrac,scaleminor,scaleminorn2,indminor,
                          nlay, taug, fracs)

    tautot = np.zeros((ngptlw, nlay))

    #  ---  combine gaseous and aerosol optical depths

    for ig in range(ngptlw):
       ib = ngb[ig]-1

       for k in range(nlay):
           tautot[ig, k] = taug[ig, k] + tauaer[ib, k]

    return fracs, tautot

    # band 1:  10-350 cm-1 (low key - h2o; low minor - n2);
    #  (high key - h2o; high minor - n2)         

def taugb01(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay):
    #  ------------------------------------------------------------------  !
    #  written by eli j. mlawer, atmospheric & environmental research.     !
    #  revised by michael j. iacono, atmospheric & environmental research. !
    #                                                                      !
    #     band 1:  10-350 cm-1 (low key - h2o; low minor - n2)             !
    #                          (high key - h2o; high minor - n2)           !
    #                                                                      !
    #  compute the optical depth by interpolating in ln(pressure) and      !
    #  temperature.  below laytrop, the water vapor self-continuum and     !
    #  foreign continuum is interpolated (in temperature) separately.      !
    #  ------------------------------------------------------------------  !

    #  ---  minor gas mapping levels:
    #     lower - n2, p = 142.5490 mbar, t = 215.70 k
    #     upper - n2, p = 142.5490 mbar, t = 215.70 k

    #  --- ...  lower atmosphere loop

    taug = np.zeros((ngptlw, nlay))
    fracs = np.zeros((ngptlw, nlay))

    ds = xr.open_dataset('../lookupdata/radlw_kgb01_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    ka_mn2 = ds['ka_mn2']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[0]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[0]
        inds = indself[k]-1
        indf = indfor[k]-1
        indm = indminor[k]-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        pp = pavel[k]
        scalen2 = colbrd[k] * scaleminorn2[k]
        if pp < 250.0:
            corradj = 1.0 - 0.15 * (250.0-pp) / 154.4
        else:
            corradj = 1.0

        for ig in range(ng01):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] -  forref[ig, indf])) 
            taun2 = scalen2 * (ka_mn2[ig, indm] + minorfrac[k] * \
                (ka_mn2[ig, indmp] - ka_mn2[ig, indm]))

            taug[ig, k] = corradj * (colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself + taufor + taun2)

            fracs[ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[0]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[0]
        indf = indfor[k]-1
        indm = indminor[k]-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1
        indmp = indm + 1

        scalen2 = colbrd[k] * scaleminorn2[k]
        corradj = 1.0 - 0.15 * (pavel[k] / 95.6)

        for ig in range(ng01):
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] -  forref[ig, indf])) 
            taun2 = scalen2 * (ka_mn2[ig, indm] + minorfrac[k] * \
                (ka_mn2[ig, indmp] - ka_mn2[ig, indm]))

            taug[ig, k] = corradj * (colamt[k, 0] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                taufor + taun2)

            fracs[ig, k] = fracrefb[ig]

    return taug, fracs



# Band 2:  350-500 cm-1 (low key - h2o; high key - h2o)
def taugb02(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 2:  350-500 cm-1 (low key - h2o; high key - h2o)            !
    #  ------------------------------------------------------------------  !
    #
    #===> ...  begin here
    #
    #  --- ...  lower atmosphere loop

    ds = xr.open_dataset('../lookupdata/radlw_kgb02_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[1]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[1]
        inds = indself[k]-1
        indf = indfor[k]-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        corradj = 1.0 - 0.05 * (pavel[k] - 100.0) / 900.0

        for ig in range(ng02):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns02+ig, k] = corradj * (colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                + tauself + taufor)

            fracs[ns02+ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[1]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[1]
        indf = indfor[k]-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1

        for ig in range(ng02):
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns02+ig, k] = colamt[k, 0] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                taufor

            fracs[ns02+ig, k] = fracrefb[ig]

    return taug, fracs, tauself

# Band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o);
#                        (high key - h2o,co2; high minor - n2o)
def taugb03(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs, tauself):
    #  ------------------------------------------------------------------  !
    #     band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o)       !
    #                           (high key - h2o,co2; high minor - n2o)     !
    #  ------------------------------------------------------------------  !

    #
    #===> ...  begin here
    #
    #  --- ...  minor gas mapping levels:
    #     lower - n2o, p = 706.272 mbar, t = 278.94 k
    #     upper - n2o, p = 95.58 mbar, t = 215.7 k

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb03_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    ka_mn2o = ds['ka_mn2o']
    kb_mn2o = ds['kb_mn2o']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    refrat_planck_a = chi_mls[0, 8]/chi_mls[1, 8]    # P = 212.725 mb
    refrat_planck_b = chi_mls[0, 12]/chi_mls[1, 12]  # P = 95.58   mb
    refrat_m_a      = chi_mls[0, 2]/chi_mls[1, 2]    # P = 706.270 mb
    refrat_m_b      = chi_mls[0, 12]/chi_mls[1, 12]  # P = 95.58   mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 0, 0]*colamt[k, 1]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0        
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[2] + js-1

        speccomb1 = colamt[k, 0] + rfrate[k, 0, 1]*colamt[k, 1]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[2] + js1-1

        speccomb_mn2o = colamt[k, 0] + refrat_m_a*colamt[k, 1]
        specparm_mn2o = colamt[k, 0] / speccomb_mn2o
        specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
        jmn2o = 1 + int(specmult_mn2o)-1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 1]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0

        inds = indself[k]-1
        indf = indfor[k]-1
        indm = indminor[k]-1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jmn2op= jmn2o+ 1
        jplp  = jpl  + 1

        #  --- ...  in atmospheres where the amount of n2O is too great to be considered
        #           a minor species, adjust the column amount of n2O by an empirical factor
        #           to obtain the proper contribution.

        p = coldry[k] * chi_mls[3, jp[k]]
        ratn2o = colamt[k, 3] / p
        if ratn2o > 1.5:
            adjfac = 0.5 + (ratn2o - 0.5)**0.65
            adjcoln2o = adjfac * p
        else:
            adjcoln2o = colamt[k, 3]

        if specparm < 0.125:
            p = fs - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p = -fs
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk0 = 1.0 - fs
            fk1 = fs
            fk2 = 0.0
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk0*fac00[k]
        fac100 = fk1*fac00[k]
        fac200 = fk2*fac00[k]
        fac010 = fk0*fac10[k]
        fac110 = fk1*fac10[k]
        fac210 = fk2*fac10[k]

        if specparm1 < 0.125:
            p = fs1 - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p = -fs1
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk0 = 1.0 - fs1
            fk1 = fs1
            fk2 = 0.0
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk0*fac01[k]
        fac101 = fk1*fac01[k]
        fac201 = fk2*fac01[k]
        fac011 = fk0*fac11[k]
        fac111 = fk1*fac11[k]
        fac211 = fk2*fac11[k]

        for ig in range(ng03):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))
            n2om1   = ka_mn2o[ig, jmn2o, indm] + fmn2o * \
                (ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm])
            n2om2   = ka_mn2o[ig, jmn2o, indmp] + fmn2o * \
                (ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp])
            absn2o  = n2om1 + minorfrac[k] * (n2om2 - n2om1)

            tau_major = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                fac200*absa[ig, id200] + fac210*absa[ig, id210])

            tau_major1 = speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                fac201*absa[ig, id201] + fac211*absa[ig, id211])

            taug[ns03+ig, k] = tau_major + tau_major1 + \
                tauself + taufor + adjcoln2o*absn2o

            fracs[ns03+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig,jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 0] + rfrate[k, 0, 0]*colamt[k, 1]
        specparm = colamt[k, 0] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-13)*5 + (jt[k]-1)) * nspb[2] + js-1

        speccomb1 = colamt[k, 0] + rfrate[k, 0, 1]*colamt[k, 1]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[2] + js1-1

        speccomb_mn2o = colamt[k, 0] + refrat_m_b*colamt[k, 1]
        specparm_mn2o = colamt[k, 0] / speccomb_mn2o
        specmult_mn2o = 4.0 * min(specparm_mn2o, oneminus)
        jmn2o = 1 + int(specmult_mn2o)-1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_b*colamt[k, 1]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0

        indf = indfor[k]-1
        indm = indminor[k]-1
        indfp = indf + 1
        indmp = indm + 1
        jmn2op= jmn2o+ 1
        jplp  = jpl  + 1

        id000 = ind0
        id010 = ind0 + 5
        id100 = ind0 + 1
        id110 = ind0 + 6
        id001 = ind1
        id011 = ind1 + 5
        id101 = ind1 + 1
        id111 = ind1 + 6

        #  --- ...  in atmospheres where the amount of n2o is too great to be considered
        #           a minor species, adjust the column amount of N2O by an empirical factor
        #           to obtain the proper contribution.

        p = coldry[k] * chi_mls[3, jp[k]]
        ratn2o = colamt[k, 3] / p
        if ratn2o > 1.5:
            adjfac = 0.5 + (ratn2o - 0.5)**0.65
            adjcoln2o = adjfac * p
        else:
            adjcoln2o = colamt[k, 3]

        fk0 = 1.0 - fs
        fk1 = fs
        fac000 = fk0*fac00[k]
        fac010 = fk0*fac10[k]
        fac100 = fk1*fac00[k]
        fac110 = fk1*fac10[k]

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0*fac01[k]
        fac011 = fk0*fac11[k]
        fac101 = fk1*fac01[k]
        fac111 = fk1*fac11[k]

        for ig in range(ng03):
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            n2om1  = kb_mn2o[ig, jmn2o, indm] + fmn2o * \
                (kb_mn2o[ig, jmn2op, indm] - kb_mn2o[ig, jmn2o, indm])
            n2om2  = kb_mn2o[ig, jmn2o, indmp] + fmn2o * \
                (kb_mn2o[ig, jmn2op, indmp] - kb_mn2o[ig, jmn2o, indmp])
            absn2o = n2om1 + minorfrac[k] * (n2om2 - n2om1)

            tau_major = speccomb * \
                (fac000*absb[ig, id000] + fac010*absb[ig, id010] + \
                fac100*absb[ig, id100] + fac110*absb[ig, id110])

            tau_major1 = speccomb1 * \
                (fac001*absb[ig, id001] + fac011*absb[ig, id011] + \
                fac101*absb[ig, id101] + fac111*absb[ig, id111])

            taug[ns03+ig, k] = tau_major + tau_major1 + \
                taufor + adjcoln2o*absn2o            

            fracs[ns03+ig, k] = fracrefb[ig, jpl] + fpl * \
                (fracrefb[ig, jplp] - fracrefb[ig, jpl])

    return taug, fracs


# Band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)
#----------------------------------
def taugb04(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)     !
    #  ------------------------------------------------------------------  !
    #
    #===> ...  begin here
    #

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb04_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    refrat_planck_a = chi_mls[0, 10]/chi_mls[1, 10]     # P = 142.5940 mb
    refrat_planck_b = chi_mls[2, 12]/chi_mls[1, 12]     # P = 95.58350 mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 0, 0]*colamt[k, 1]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[3] + js-1

        speccomb1 = colamt[k, 0] + rfrate[k, 0, 1]*colamt[k, 1]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = ( jp[k]*5 + (jt1[k]-1)) * nspa[3] + js1-1

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 1]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0

        inds = indself[k]-1
        indf = indfor[k]-1
        indsp = inds + 1
        indfp = indf + 1
        jplp  = jpl  + 1

        if specparm < 0.125:
            p = fs - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p = -fs
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk0 = 1.0 - fs
            fk1 = fs
            fk2 = 0.0
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk0*fac00[k]
        fac100 = fk1*fac00[k]
        fac200 = fk2*fac00[k]
        fac010 = fk0*fac10[k]
        fac110 = fk1*fac10[k]
        fac210 = fk2*fac10[k]

        if specparm1 < 0.125:
            p = fs1 - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p = -fs1
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk0 = 1.0 - fs1
            fk1 = fs1
            fk2 = 0.0
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk0*fac01[k]
        fac101 = fk1*fac01[k]
        fac201 = fk2*fac01[k]
        fac011 = fk0*fac11[k]
        fac111 = fk1*fac11[k]
        fac211 = fk2*fac11[k]

        for ig in range(ng04):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            tau_major = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                fac200*absa[ig, id200] + fac210*absa[ig, id210])

            tau_major1 = speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                fac201*absa[ig, id201] + fac211*absa[ig, id211])

            taug[ns04+ig, k] = tau_major + tau_major1 + tauself + taufor

            fracs[ns04+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 2] + rfrate[k, 5, 0]*colamt[k, 1]
        specparm = colamt[k, 2] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-13)*5 + (jt[k]-1)) * nspb[3] + js-1

        speccomb1 = colamt[k, 2] + rfrate[k, 5, 1]*colamt[k, 1]
        specparm1 = colamt[k, 2] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[3] + js1-1

        speccomb_planck = colamt[k, 2] + refrat_planck_b*colamt[k, 1]
        specparm_planck = colamt[k, 2] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0
        jplp = jpl + 1

        id000 = ind0
        id010 = ind0 + 5
        id100 = ind0 + 1
        id110 = ind0 + 6
        id001 = ind1
        id011 = ind1 + 5
        id101 = ind1 + 1
        id111 = ind1 + 6

        fk0 = 1.0 - fs
        fk1 = fs
        fac000 = fk0*fac00[k]
        fac010 = fk0*fac10[k]
        fac100 = fk1*fac00[k]
        fac110 = fk1*fac10[k]

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0*fac01[k]
        fac011 = fk0*fac11[k]
        fac101 = fk1*fac01[k]
        fac111 = fk1*fac11[k]

        for ig in range(ng04):
            tau_major =  speccomb * \
                (fac000*absb[ig, id000] + fac010*absb[ig, id010] + \
                fac100*absb[ig, id100] + fac110*absb[ig, id110])
            tau_major1 = speccomb1 * \
                (fac001*absb[ig, id001] + fac011*absb[ig, id011] + \
                fac101*absb[ig, id101] + fac111*absb[ig, id111])

            taug[ns04+ig, k] = tau_major + tau_major1

            fracs[ns04+ig, k] = fracrefb[ig, jpl] + fpl * \
                (fracrefb[ig, jplp] - fracrefb[ig, jpl])

        #  --- ...  empirical modification to code to improve stratospheric cooling rates
        #           for co2. revised to apply weighting for g-point reduction in this band.

        taug[ns04+ 7, k] = taug[ns04+ 7, k] * 0.92
        taug[ns04+ 8, k] = taug[ns04+ 8, k] * 0.88
        taug[ns04+9, k] = taug[ns04+9, k] * 1.07
        taug[ns04+10, k] = taug[ns04+10, k] * 1.1
        taug[ns04+11, k] = taug[ns04+11, k] * 0.99
        taug[ns04+12, k] = taug[ns04+12, k] * 0.88
        taug[ns04+13, k] = taug[ns04+13, k] * 0.943

    return taug, fracs

# Band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4) 
#                       (high key - o3,co2)                 
def taugb05(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)  !
    #                           (high key - o3,co2)                        !
    #  ------------------------------------------------------------------  !
    #
    #===> ...  begin here
    #
    #  --- ...  minor gas mapping level :
    #     lower - o3, p = 317.34 mbar, t = 240.77 k
    #     lower - ccl4

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb05_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mo3 = ds['ka_mo3']
    ccl4 = ds['ccl4']

    refrat_planck_a = chi_mls[0, 4]/chi_mls[1, 4]      # P = 473.420 mb
    refrat_planck_b = chi_mls[2, 42]/chi_mls[1, 42]    # P = 0.2369  mb
    refrat_m_a = chi_mls[0, 6]/chi_mls[1, 6]           # P = 317.348 mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 0, 0]*colamt[k, 1]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[4] + js-1

        speccomb1 = colamt[k, 0] + rfrate[k, 0, 1]*colamt[k, 1]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[4] + js1-1

        speccomb_mo3 = colamt[k, 0] + refrat_m_a*colamt[k, 1]
        specparm_mo3 = colamt[k, 0] / speccomb_mo3
        specmult_mo3 = 8.0 * min(specparm_mo3, oneminus)
        jmo3 = 1 + int(specmult_mo3)-1
        fmo3 = specmult_mo3 % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 1]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0

        inds = indself[k]-1
        indf = indfor[k]-1
        indm = indminor[k]-1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp  = jpl  + 1
        jmo3p = jmo3 + 1

        if specparm < 0.125:
            p0   = fs - 1.0
            p40  = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0   = -fs
            p40  = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1   = fs1 - 1.0
            p41  = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1   = -fs1
            p41  = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng05):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))
            o3m1    = ka_mo3[ig, jmo3, indm] + fmo3 * \
                (ka_mo3[ig, jmo3p, indm] -  ka_mo3[ig, jmo3, indm])
            o3m2    = ka_mo3[ig, jmo3, indmp] + fmo3 * \
                (ka_mo3[ig, jmo3p, indmp] - ka_mo3[ig, jmo3, indmp])
            abso3   = o3m1 + minorfrac[k]*(o3m2 - o3m1)

            taug[ns05+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor+abso3*colamt[k, 2]+wx[k, 0]*ccl4[ig]

            fracs[ns05+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 2] + rfrate[k, 5, 0]*colamt[k, 1]
        specparm = colamt[k, 2] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-13)*5 + (jt[k]-1)) * nspb[4] + js-1

        speccomb1 = colamt[k, 2] + rfrate[k, 5, 1]*colamt[k, 1]
        specparm1 = colamt[k, 2] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[4] + js1-1

        speccomb_planck = colamt[k, 2] + refrat_planck_b*colamt[k, 1]
        specparm_planck = colamt[k, 2] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0
        jplp= jpl + 1

        id000 = ind0
        id010 = ind0 + 5
        id100 = ind0 + 1
        id110 = ind0 + 6
        id001 = ind1
        id011 = ind1 + 5
        id101 = ind1 + 1
        id111 = ind1 + 6

        fk00 = 1.0 - fs
        fk10 = fs

        fk01 = 1.0 - fs1
        fk11 = fs1

        fac000 = fk00 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac100 = fk10 * fac00[k]
        fac110 = fk10 * fac10[k]

        fac001 = fk01 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac101 = fk11 * fac01[k]
        fac111 = fk11 * fac11[k]

        for ig in range(ng05):
            taug[ns05+ig, k] = speccomb * \
                (fac000*absb[ig, id000] + fac010*absb[ig, id010] + \
                fac100*absb[ig, id100] + fac110*absb[ig, id110]) + \
                speccomb1 * \
                (fac001*absb[ig, id001] + fac011*absb[ig, id011] + \
                 fac101*absb[ig, id101] + fac111*absb[ig, id111]) + \
                wx[k, 0] * ccl4[ig]

            fracs[ns05+ig, k] = fracrefb[ig, jpl] + fpl * \
                (fracrefb[ig, jplp] - fracrefb[ig, jpl])

    return taug, fracs

# Band 6:  820-980 cm-1 (low key - h2o; low minor - co2) 
#                       (high key - none; high minor - cfc11, cfc12)
def taugb06(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 6:  820-980 cm-1 (low key - h2o; low minor - co2)           !
    #                           (high key - none; high minor - cfc11, cfc12)
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level:
    #     lower - co2, p = 706.2720 mb, t = 294.2 k
    #     upper - cfc11, cfc12

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb06_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    fracrefa = ds['fracrefa']
    ka_mco2 = ds['ka_mco2']
    cfc11adj = ds['cfc11adj']
    cfc12 = ds['cfc12']

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[5]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[5]

        inds = indself[k]-1
        indf = indfor[k]-1
        indm = indminor[k]-1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[1, jp[k]+1]
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.77
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        for ig in range(ng06):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))
            absco2  = ka_mco2[ig, indm] + minorfrac[k] * \
                (ka_mco2[ig, indmp] - ka_mco2[ig, indm])

            taug[ns06+ig, k] = colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                 fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself + taufor + adjcolco2*absco2 + \
                wx[k, 1]*cfc11adj[ig] + wx[k, 2]*cfc12[ig]

            fracs[ns06+ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop
    #           nothing important goes on above laytrop in this band.

    for k in range(laytrop, nlay):
        for ig in range(ng06):
            taug[ns06+ig, k] = wx[k, 1]*cfc11adj[ig] + wx[k, 2]*cfc12[ig]
            fracs[ns06+ig, k] = fracrefa[ig]

    return taug, fracs

# Band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)
#                        (high key - o3; high minor - co2)
def taugb07(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)       !
    #                            (high key - o3; high minor - co2)         !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - co2, p = 706.2620 mbar, t= 278.94 k
    #     upper - co2, p = 12.9350 mbar, t = 234.01 k

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb07_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mco2 = ds['ka_mco2']
    kb_mco2 = ds['kb_mco2']

    refrat_planck_a = chi_mls[0, 2]/chi_mls[2, 2]     # P = 706.2620 mb
    refrat_m_a = chi_mls[0, 2]/chi_mls[2, 2]          # P = 706.2720 mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 1, 0]*colamt[k, 2]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[6] + js-1

        speccomb1 = colamt[k, 0] + rfrate[k, 1, 1]*colamt[k, 2]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[6] + js1-1

        speccomb_mco2 = colamt[k, 0] + refrat_m_a*colamt[k, 2]
        specparm_mco2 = colamt[k, 0] / speccomb_mco2
        specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
        jmco2 = 1 + int(specmult_mco2)-1
        fmco2 = specmult_mco2 % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 2]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck)-1
        fpl = specmult_planck % 1.0

        inds = indself[k]-1
        indf = indfor[k]-1
        indm = indminor[k]-1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp  = jpl  + 1
        jmco2p= jmco2+ 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        #  --- ...  in atmospheres where the amount of CO2 is too great to be considered
        #           a minor species, adjust the column amount of CO2 by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[1, jp[k]]
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 3.0 + (ratco2-3.0)**0.79
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng07):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            co2m1   = ka_mco2[ig, jmco2, indm] + fmco2 * \
                (ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm])
            co2m2   = ka_mco2[ig, jmco2, indmp] + fmco2 * \
                (ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp])
            absco2  = co2m1 + minorfrac[k] * (co2m2 - co2m1)

            taug[ns07+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor + adjcolco2*absco2

            fracs[ns07+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    for k in range(laytrop, nlay):
        temp   = coldry[k] * chi_mls[1, jp[k]]
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.79
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[6]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[6]

        indm = indminor[k]-1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng07):
            absco2 = kb_mco2[ig, indm] + minorfrac[k] * \
                (kb_mco2[ig, indmp] - kb_mco2[ig, indm])

            taug[ns07+ig, k] = colamt[k, 2] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                adjcolco2 * absco2

            fracs[ns07+ig, k] = fracrefb[ig]

        #  --- ...  empirical modification to code to improve stratospheric cooling rates
        #           for o3.  revised to apply weighting for g-point reduction in this band.

        taug[ns07+ 5, k] = taug[ns07+ 5, k] * 0.92
        taug[ns07+ 6, k] = taug[ns07+ 6, k] * 0.88
        taug[ns07+ 7, k] = taug[ns07+ 7, k] * 1.07
        taug[ns07+ 8, k] = taug[ns07+ 8, k] * 1.1
        taug[ns07+ 9, k] = taug[ns07+ 9, k] * 0.99
        taug[ns07+10, k] = taug[ns07+10, k] * 0.855

    return taug, fracs


# Band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o) 
#                         (high key - o3; high minor - co2, n2o) 
def taugb08(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)  !
    #                             (high key - o3; high minor - co2, n2o)   !
    #  ------------------------------------------------------------------  !
    #  --- ...  minor gas mapping level:
    #     lower - co2, p = 1053.63 mb, t = 294.2 k
    #     lower - o3,  p = 317.348 mb, t = 240.77 k
    #     lower - n2o, p = 706.2720 mb, t= 278.94 k
    #     lower - cfc12,cfc11
    #     upper - co2, p = 35.1632 mb, t = 223.28 k
    #     upper - n2o, p = 8.716e-2 mb, t = 226.03 k

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb08_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mo3 = ds['ka_mo3']
    ka_mco2 = ds['ka_mco2']
    kb_mco2 = ds['kb_mco2']
    cfc12 = ds['cfc12']
    ka_mn2o = ds['ka_mn2o']
    kb_mn2o = ds['kb_mn2o']
    cfc22adj = ds['cfc22adj']

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[7]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[7]

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indm = indminor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[1, jp[k]]
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.65
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        for ig in range(ng08):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))
            absco2  = (ka_mco2[ig, indm] + minorfrac[k] * \
                (ka_mco2[ig, indmp] - ka_mco2[ig, indm]))
            abso3   = (ka_mo3[ig, indm] + minorfrac[k] * \
                (ka_mo3[ig, indmp] - ka_mo3[ig, indm]))
            absn2o  = (ka_mn2o[ig, indm] + minorfrac[k] * \
                (ka_mn2o[ig, indmp] - ka_mn2o[ig, indm]))

            taug[ns08+ig, k] = colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself+taufor + adjcolco2*absco2 + \
                colamt[k, 2]*abso3 + colamt[k, 3]*absn2o + \
                wx[k, 2]*cfc12[ig] + wx[k, 3]*cfc22adj[ig]

            fracs[ns08+ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[7]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[7]

        indm = indminor[k]-1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[1, jp[k]]
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.65
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        for ig in range(ng08):
            absco2 = (kb_mco2[ig, indm] + minorfrac[k] * \
                (kb_mco2[ig, indmp] - kb_mco2[ig, indm]))
            absn2o = (kb_mn2o[ig, indm]+ minorfrac[k] * \
                (kb_mn2o[ig, indmp] - kb_mn2o[ig, indm]))

            taug[ns08+ig, k] = colamt[k, 2] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                adjcolco2*absco2 + colamt[k, 3]*absn2o + \
                wx[k, 2]*cfc12[ig] + wx[k, 3]*cfc22adj[ig]

            fracs[ns08+ig, k] = fracrefb[ig]

    return taug, fracs


# Band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)
#                         (high key - ch4; high minor - n2o)  
def taugb09(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)     !
    #                             (high key - ch4; high minor - n2o)       !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - n2o, p = 706.272 mbar, t = 278.94 k
    #     upper - n2o, p = 95.58 mbar, t = 215.7 k

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb09_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mn2o = ds['ka_mn2o']
    kb_mn2o = ds['kb_mn2o']

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 8]/chi_mls[5, 8]       # P = 212 mb
    refrat_m_a = chi_mls[0, 2]/chi_mls[5, 2]            # P = 706.272 mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 3, 0]*colamt[k, 4]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[8] + js - 1

        speccomb1 = colamt[k, 0] + rfrate[k, 3, 1]*colamt[k, 4]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[8] + js1 - 1

        speccomb_mn2o = colamt[k, 0] + refrat_m_a*colamt[k, 4]
        specparm_mn2o = colamt[k, 0] / speccomb_mn2o
        specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
        jmn2o = 1 + int(specmult_mn2o) - 1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 4]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck) - 1
        fpl = specmult_planck % 1.0

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indm = indminor[k] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp  = jpl  + 1
        jmn2op= jmn2o+ 1

        #  --- ...  in atmospheres where the amount of n2o is too great to be considered
        #           a minor species, adjust the column amount of n2o by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[3, jp[k]]
        ratn2o = colamt[k, 3] / temp
        if ratn2o > 1.5:
            adjfac = 0.5 + (ratn2o-0.5)**0.65
            adjcoln2o = adjfac * temp
        else:
            adjcoln2o = colamt[k, 3]

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
          fk00 = 1.0 - fs
          fk10 = fs
          fk20 = 0.0

          id000 = ind0
          id010 = ind0 + 9
          id100 = ind0 + 1
          id110 = ind0 +10
          id200 = ind0
          id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng09):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            n2om1   = ka_mn2o[ig, jmn2o, indm] + fmn2o * \
                (ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm])
            n2om2   = ka_mn2o[ig, jmn2o, indmp] + fmn2o \
                * (ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp])
            absn2o  = n2om1 + minorfrac[k] * (n2om2 - n2om1)

            taug[ns09+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                 fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                 fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor + adjcoln2o*absn2o            

            fracs[ns09+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop
    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[8]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[8]

        indm = indminor[k]-1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of n2o is too great to be considered
        #           a minor species, adjust the column amount of n2o by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * chi_mls[3, jp[k]]
        ratn2o = colamt[k, 3] / temp
        if ratn2o > 1.5:
            adjfac = 0.5 + (ratn2o - 0.5)**0.65
            adjcoln2o = adjfac * temp
        else:
            adjcoln2o = colamt[k, 3]

        for ig in range(ng09):
            absn2o = kb_mn2o[ig, indm] + minorfrac[k] * \
                (kb_mn2o[ig, indmp] - kb_mn2o[ig, indm])

            taug[ns09+ig, k] = colamt[k, 4] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                 fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                adjcoln2o*absn2o

            fracs[ns09+ig, k] = fracrefb[ig]

    return taug, fracs

# Band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)
def taugb10(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)         !
    #  ------------------------------------------------------------------  !

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb10_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[9]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[9]

        inds = indself[k] - 1
        indf = indfor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        for ig in range(ng10):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns10+ig, k] = colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                 fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself + taufor

            fracs[ns10+ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[9]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[9]

        indf = indfor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1

        for ig in range(ng10):
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))

            taug[ns10+ig, k] = colamt[k, 0] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                 fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                taufor

            fracs[ns10+ig, k] = fracrefb[ig]

    return taug, fracs

# Band 11:  1480-1800 cm-1 (low - h2o; low minor - o2) 
#                          (high key - h2o; high minor - o2)   
def taugb11(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)             !
    #                              (high key - h2o; high minor - o2)       !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - o2, p = 706.2720 mbar, t = 278.94 k
    #     upper - o2, p = 4.758820 mbarm t = 250.85 k

    ds = xr.open_dataset('../lookupdata/radlw_kgb11_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mo2 = ds['ka_mo2']
    kb_mo2 = ds['kb_mo2']

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[10]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[10]

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indm = indminor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        scaleo2 = colamt[k, 5] * scaleminor[k]

        for ig in range(ng11):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf]))
            tauo2   = scaleo2 * (ka_mo2[ig, indm] + minorfrac[k] * \
                (ka_mo2[ig, indmp] - ka_mo2[ig, indm]))

            taug[ns11+ig, k] = colamt[k, 0] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                 fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself + taufor + tauo2

            fracs[ns11+ig, k] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[10]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[10]

        indf = indfor[k] - 1
        indm = indminor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1
        indmp = indm + 1

        scaleo2 = colamt[k, 5] * scaleminor[k]

        for ig in range(ng11):
            taufor = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            tauo2  = scaleo2 * (kb_mo2[ig, indm] + minorfrac[k] * \
                (kb_mo2[ig, indmp] - kb_mo2[ig, indm]))

            taug[ns11+ig, k] = colamt[k, 0] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                 fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p]) + \
                taufor + tauo2

            fracs[ns11+ig, k] = fracrefb[ig]

    return taug, fracs

# Band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
def taugb12(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)         !
    #  ------------------------------------------------------------------  !

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb12_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    fracrefa = ds['fracrefa']

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 9]/chi_mls[1, 9]      # P =   174.164 mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 0, 0]*colamt[k, 1]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[11] + js - 1

        speccomb1 = colamt[k, 0] + rfrate[k, 0, 1]*colamt[k, 1]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[11] + js1 - 1

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 1]
        specparm_planck = colamt[k, 0] / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = 8.0 * specparm_planck
        jpl = 1 + int(specmult_planck) - 1
        fpl = specmult_planck % 1.0

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp  = jpl  + 1

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng12):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns12+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                 fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                 fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor

            fracs[ns12+ig, k] = fracrefa[ig, jpl] + fpl*(fracrefa[ig, jplp] - \
                fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        for ig in range(ng12):
            taug[ns12+ig, k] = 0.0
            fracs[ns12+ig, k] = 0.0

    return taug, fracs

# Band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)
def taugb13(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)  !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping levels :
    #     lower - co2, p = 1053.63 mb, t = 294.2 k
    #     lower - co, p = 706 mb, t = 278.94 k
    #     upper - o3, p = 95.5835 mb, t = 215.7 k

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb13_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']
    ka_mco2 = ds['ka_mco2']
    ka_mco = ds['ka_mco']
    kb_mo3 = ds['kb_mo3']

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 4]/chi_mls[3, 4]        # P = 473.420 mb (Level 5)
    refrat_m_a = chi_mls[0, 0]/chi_mls[3, 0]             # P = 1053. (Level 1)
    refrat_m_a3 = chi_mls[0, 2]/chi_mls[3, 2]            # P = 706. (Level 3)

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 2, 0]*colamt[k, 3]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[12] + js - 1

        speccomb1 = colamt[k, 0] + rfrate[k, 2, 1]*colamt[k, 3]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[12] + js1 - 1

        speccomb_mco2 = colamt[k, 0] + refrat_m_a*colamt[k, 3]
        specparm_mco2 = colamt[k, 0] / speccomb_mco2
        specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
        jmco2 = 1 + int(specmult_mco2) - 1
        fmco2 = specmult_mco2 % 1.0

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        speccomb_mco = colamt[k, 0] + refrat_m_a3*colamt[k, 3]
        specparm_mco = colamt[k, 0] / speccomb_mco
        specmult_mco = 8.0 * min(specparm_mco, oneminus)
        jmco = 1 + int(specmult_mco) - 1
        fmco = specmult_mco % 1.0

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 3]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck) - 1
        fpl = specmult_planck % 1.0

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indm = indminor[k] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp  = jpl  + 1
        jmco2p= jmco2+ 1
        jmcop = jmco + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp   = coldry[k] * 3.55e-4
        ratco2 = colamt[k, 1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.68
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[k, 1]

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng13):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            co2m1   = ka_mco2[ig, jmco2, indm] + fmco2 * \
                (ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm])
            co2m2   = ka_mco2[ig, jmco2, indmp] + fmco2 * \
                (ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp])
            absco2  = co2m1 + minorfrac[k] * (co2m2 - co2m1)
            com1    = ka_mco[ig, jmco, indm]+ fmco * \
                (ka_mco[ig, jmcop, indm] - ka_mco[ig, jmco, indm])
            com2    = ka_mco[ig, jmco, indmp] + fmco * \
                (ka_mco[ig, jmcop, indmp] - ka_mco[ig, jmco, indmp])
            absco   = com1 + minorfrac[k] * (com2 - com1)

            taug[ns13+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                 fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                 fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor + adjcolco2*absco2 + \
                colamt[k, 6]*absco

            fracs[ns13+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        indm = indminor[k] - 1
        indmp = indm + 1

        for ig in range(ng13):
            abso3 = kb_mo3[ig, indm] + minorfrac[k] * \
                (kb_mo3[ig, indmp] - kb_mo3[ig, indm])

            taug[ns13+ig, k] = colamt[k, 2]*abso3

            fracs[ns13+ig, k] =  fracrefb[ig]

    return taug, fracs, taufor

# Band 14:  2250-2380 cm-1 (low - co2; high - co2) 
def taugb14(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs, taufor):
    #  ------------------------------------------------------------------  !
    #     band 14:  2250-2380 cm-1 (low - co2; high - co2)                 !
    #  ------------------------------------------------------------------  !

    ds = xr.open_dataset('../lookupdata/radlw_kgb14_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        ind0 = ((jp[k]-1)*5 + (jt [k]-1)) * nspa[13]
        ind1 = ( jp[k]   *5 + (jt1[k]-1)) * nspa[13]

        inds = indself[k] - 1
        indf = indfor[k] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        for ig in range(ng14):
            tauself = selffac[k] * (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns14+ig, k] = colamt[k, 1] * \
                (fac00[k]*absa[ig, ind0] + fac10[k]*absa[ig, ind0p] + \
                 fac01[k]*absa[ig, ind1] + fac11[k]*absa[ig, ind1p]) + \
                tauself + taufor

            fracs[ns14+ig, k] = fracrefa[ig]


    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[13]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[13]

        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng14):
            taug[ns14+ig, k] = colamt[k, 1] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                 fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p])

            fracs[ns14+ig, k] = fracrefb[ig]

    return taug, fracs

# Band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2) 
#                          (high - nothing)     
def taugb15(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)         !
    #                              (high - nothing)                        !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - nitrogen continuum, P = 1053., T = 294.

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb15_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    fracrefa = ds['fracrefa']
    ka_mn2 = ds['ka_mn2']

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    refrat_planck_a = chi_mls[3, 0]/chi_mls[1, 0]      # P = 1053. mb (Level 1)
    refrat_m_a = chi_mls[3, 0]/chi_mls[1, 0]           # P = 1053. mb

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 3] + rfrate[k, 4, 0]*colamt[k, 1]
        specparm = colamt[k, 3] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[14] + js - 1

        speccomb1 = colamt[k, 3] + rfrate[k, 4, 1]*colamt[k, 1]
        specparm1 = colamt[k, 3] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[14] + js1 - 1

        speccomb_mn2 = colamt[k, 3] + refrat_m_a*colamt[k, 1]
        specparm_mn2 = colamt[k, 3] / speccomb_mn2
        specmult_mn2 = 8.0 * min(specparm_mn2, oneminus)
        jmn2 = 1 + int(specmult_mn2) - 1
        fmn2 = specmult_mn2 % 1.0

        speccomb_planck = colamt[k, 3] + refrat_planck_a*colamt[k, 1]
        specparm_planck = colamt[k, 3] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck) - 1
        fpl = specmult_planck % 1.0

        scalen2 = colbrd[k] * scaleminor[k]

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indm = indminor[k] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp  = jpl  + 1
        jmn2p = jmn2 + 1

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
          fk01 = 1.0 - fs1
          fk11 = fs1
          fk21 = 0.0

          id001 = ind1
          id011 = ind1 + 9
          id101 = ind1 + 1
          id111 = ind1 +10
          id201 = ind1
          id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng15):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 
            n2m1    = ka_mn2[ig, jmn2, indm] + fmn2 * \
                (ka_mn2[ig, jmn2p, indm] - ka_mn2[ig, jmn2, indm])
            n2m2    = ka_mn2[ig, jmn2, indmp] + fmn2 * \
                (ka_mn2[ig, jmn2p, indmp] - ka_mn2[ig, jmn2, indmp])
            taun2   = scalen2 * (n2m1 + minorfrac[k] * (n2m2 - n2m1))

            taug[ns15+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                 fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                 fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                     tauself + taufor + taun2

            fracs[ns15+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        for ig in range(ng15):
            taug[ns15+ig, k] = 0.0

            fracs[ns15+ig, k] = 0.0

    return taug, fracs


# Band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)
def taugb16(laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,
           rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           minorfrac,scaleminor,scaleminorn2,indminor,
           nlay, taug, fracs):
    #  ------------------------------------------------------------------  !
    #     band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)      !
    #  ------------------------------------------------------------------  !

    dsc = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    chi_mls = dsc['chi_mls']

    ds = xr.open_dataset('../lookupdata/radlw_kgb16_data.nc')
    selfref = ds['selfref']
    forref = ds['forref']
    absa = ds['absa']
    absb = ds['absb']
    fracrefa = ds['fracrefa']
    fracrefb = ds['fracrefb']

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    refrat_planck_a = chi_mls[0, 5]/chi_mls[5, 5]        # P = 387. mb (Level 6)

    #  --- ...  lower atmosphere loop

    for k in range(laytrop):
        speccomb = colamt[k, 0] + rfrate[k, 3, 0]*colamt[k, 4]
        specparm = colamt[k, 0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + int(specmult)
        fs = specmult % 1.0
        ind0 = ((jp[k]-1)*5 + (jt[k]-1)) * nspa[15] + js - 1

        speccomb1 = colamt[k, 0] + rfrate[k, 3, 1]*colamt[k, 4]
        specparm1 = colamt[k, 0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + int(specmult1)
        fs1 = specmult1 % 1.0
        ind1 = (jp[k]*5 + (jt1[k]-1)) * nspa[15] + js1 - 1

        speccomb_planck = colamt[k, 0] + refrat_planck_a*colamt[k, 4]
        specparm_planck = colamt[k, 0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + int(specmult_planck) - 1
        fpl = specmult_planck % 1.0

        inds = indself[k] - 1
        indf = indfor[k] - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp  = jpl  + 1

        if specparm < 0.125:
          p0 = fs - 1.0
          p40 = p0**4
          fk00 = p40
          fk10 = 1.0 - p0 - 2.0*p40
          fk20 = p0 + p40

          id000 = ind0
          id010 = ind0 + 9
          id100 = ind0 + 1
          id110 = ind0 +10
          id200 = ind0 + 2
          id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00[k]
        fac100 = fk10 * fac00[k]
        fac200 = fk20 * fac00[k]
        fac010 = fk00 * fac10[k]
        fac110 = fk10 * fac10[k]
        fac210 = fk20 * fac10[k]

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01[k]
        fac101 = fk11 * fac01[k]
        fac201 = fk21 * fac01[k]
        fac011 = fk01 * fac11[k]
        fac111 = fk11 * fac11[k]
        fac211 = fk21 * fac11[k]

        for ig in range(ng16):
            tauself = selffac[k]* (selfref[ig, inds] + selffrac[k] * \
                (selfref[ig, indsp] - selfref[ig, inds]))
            taufor  = forfac[k] * (forref[ig, indf] + forfrac[k] * \
                (forref[ig, indfp] - forref[ig, indf])) 

            taug[ns16+ig, k] = speccomb * \
                (fac000*absa[ig, id000] + fac010*absa[ig, id010] + \
                 fac100*absa[ig, id100] + fac110*absa[ig, id110] + \
                 fac200*absa[ig, id200] + fac210*absa[ig, id210]) + \
                speccomb1 * \
                (fac001*absa[ig, id001] + fac011*absa[ig, id011] + \
                 fac101*absa[ig, id101] + fac111*absa[ig, id111] + \
                 fac201*absa[ig, id201] + fac211*absa[ig, id211]) + \
                tauself + taufor

            fracs[ns16+ig, k] = fracrefa[ig, jpl] + fpl * \
                (fracrefa[ig, jplp] - fracrefa[ig, jpl])

    #  --- ...  upper atmosphere loop

    for k in range(laytrop, nlay):
        ind0 = ((jp[k]-13)*5 + (jt [k]-1)) * nspb[15]
        ind1 = ((jp[k]-12)*5 + (jt1[k]-1)) * nspb[15]

        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng16):
            taug[ns16+ig, k] = colamt[k, 4] * \
                (fac00[k]*absb[ig, ind0] + fac10[k]*absb[ig, ind0p] + \
                 fac01[k]*absb[ig, ind1] + fac11[k]*absb[ig, ind1p])

            fracs[ns16+ig, k] = fracrefb[ig]

    return taug, fracs