import numpy as np
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import nbands, ngptlw, absrain, abssnow0, ipat
from radphysparam import ilwcliq, ilwcice, isubclw

def cldprop(cfrac, cliqp, reliq, cicep, reice, cdat1, cdat2, cdat3, cdat4,
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
   
    if isubclw > 0:    # mcica sub-col clouds approx
        for k in range(nlay):
            if cfrac[k] < cldmin:
                cldf[k] = 0.0
            else:
                cldf[k] = cfrac[k]
   
        #  --- ...  call sub-column cloud generator
   
        lcloudy = mcica_subcol(cldf, nlay, ipseed, dz, de_lgth)
   
        for k in range(nlay):
            for ig in range(ngptlw):
                if lcloudy[ig, k]:
                    cldfmc[ig, k] = 1.0
                else:
                    cldfmc[ig, k] = 0.0

    return cldfmc, taucld