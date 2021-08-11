##  \file radlw_main.f
#!!  This file contains NCEP's modifications of the rrtmg-lw radiation
#!!  code from AER.
#
#!!!!!  ==============================================================  !!!!!
#!!!!!               lw-rrtm3 radiation package description             !!!!!
#!!!!!  ==============================================================  !!!!!
#!                                                                          !
#!   this package includes ncep's modifications of the rrtm-lw radiation    !
#!   code from aer inc.                                                     !
#!                                                                          !
#!    the lw-rrtm3 package includes these parts:                            !
#!                                                                          !
#!       'radlw_rrtm3_param.f'                                              !
#!       'radlw_rrtm3_datatb.f'                                             !
#!       'radlw_rrtm3_main.f'                                               !
#!                                                                          !
#!    the 'radlw_rrtm3_param.f' contains:                                   !
#!                                                                          !
#!       'module_radlw_parameters'  -- band parameters set up               !
#!                                                                          !
#!    the 'radlw_rrtm3_datatb.f' contains:                                  !
#!                                                                          !
#!       'module_radlw_avplank'     -- plank flux data                      !
#!       'module_radlw_ref'         -- reference temperature and pressure   !
#!       'module_radlw_cldprlw'     -- cloud property coefficients          !
#!       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
#!                                     bands, where nn = 01-16              !
#!                                                                          !
#!    the 'radlw_rrtm3_main.f' contains:                                    !
#!                                                                          !
#!       'module_radlw_main'        -- main lw radiation transfer           !
#!                                                                          !
#!    in the main module 'module_radlw_main' there are only two             !
#!    externally callable subroutines:                                      !
#!                                                                          !
#!                                                                          !
#!       'lwrad'     -- main lw radiation routine                           !
#!          inputs:                                                         !
#!           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
#!            clouds,icseed,aerosols,sfemis,sfgtmp,                         !
#!            dzlyr,delpin,de_lgth,                                         !
#!            npts, nlay, nlp1, lprnt,                                      !
#!          outputs:                                                        !
#!            hlwc,topflx,sfcflx,cldtau,                                    !
#!!         optional outputs:                                               !
#!            HLW0,HLWB,FLXPRF)                                             !
#!                                                                          !
#!       'rlwinit'   -- initialization routine                              !
#!          inputs:                                                         !
#!           ( me )                                                         !
#!          outputs:                                                        !
#!           (none)                                                         !
#!                                                                          !
#!    all the lw radiation subprograms become contained subprograms         !
#!    in module 'module_radlw_main' and many of them are not directly       !
#!    accessable from places outside the module.                            !
#!                                                                          !
#!    derived data type constructs used:                                    !
#!                                                                          !
#!     1. radiation flux at toa: (from module 'module_radlw_parameters')    !
#!          topflw_type   -  derived data type for toa rad fluxes           !
#!            upfxc              total sky upward flux at toa               !
#!            upfx0              clear sky upward flux at toa               !
#!                                                                          !
#!     2. radiation flux at sfc: (from module 'module_radlw_parameters')    !
#!          sfcflw_type   -  derived data type for sfc rad fluxes           !
#!            upfxc              total sky upward flux at sfc               !
#!            upfx0              clear sky upward flux at sfc               !
#!            dnfxc              total sky downward flux at sfc             !
#!            dnfx0              clear sky downward flux at sfc             !
#!                                                                          !
#!     3. radiation flux profiles(from module 'module_radlw_parameters')    !
#!          proflw_type    -  derived data type for rad vertical prof       !
#!            upfxc              level upward flux for total sky            !
#!            dnfxc              level downward flux for total sky          !
#!            upfx0              level upward flux for clear sky            !
#!            dnfx0              level downward flux for clear sky          !
#!                                                                          !
#!    external modules referenced:                                          !
#!                                                                          !
#!       'module physparam'                                                 !
#!       'module physcons'                                                  !
#!       'mersenne_twister'                                                 !
#!                                                                          !
#!    compilation sequence is:                                              !
#!                                                                          !
#!       'radlw_rrtm3_param.f'                                              !
#!       'radlw_rrtm3_datatb.f'                                             !
#!       'radlw_rrtm3_main.f'                                               !
#!                                                                          !
#!    and all should be put in front of routines that use lw modules        !
#!                                                                          !
#!==========================================================================!
#!                                                                          !
#!    the original aer's program declarations:                              !
#!                                                                          !
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!                                                                          |
#!  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
#!  This software may be used, copied, or redistributed as long as it is    |
#!  not sold and this copyright notice is reproduced on each copy made.     |
#!  This model is provided as is without any express or implied warranties. |
#!                       (http://www.rtweb.aer.com/)                        |
#!                                                                          |
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!                                                                          !
#! ************************************************************************ !
#!                                                                          !
#!                              rrtmg_lw                                    !
#!                                                                          !
#!                                                                          !
#!                   a rapid radiative transfer model                       !
#!                       for the longwave region                            ! 
#!             for application to general circulation models                !
#!                                                                          !
#!                                                                          !
#!            atmospheric and environmental research, inc.                  !
#!                        131 hartwell avenue                               !
#!                        lexington, ma 02421                               !
#!                                                                          !
#!                           eli j. mlawer                                  !
#!                        jennifer s. delamere                              !
#!                         michael j. iacono                                !
#!                         shepard a. clough                                !
#!                                                                          !
#!                                                                          !
#!                       email:  miacono@aer.com                            !
#!                       email:  emlawer@aer.com                            !
#!                       email:  jdelamer@aer.com                           !
#!                                                                          !
#!        the authors wish to acknowledge the contributions of the          !
#!        following people:  steven j. taubman, karen cady-pereira,         !
#!        patrick d. brown, ronald e. farren, luke chen, robert bergstrom.  !
#!                                                                          !
#! ************************************************************************ !
#!                                                                          !
#!    references:                                                           !
#!    (rrtm_lw/rrtmg_lw):                                                   !
#!      clough, s.A., m.w. shephard, e.j. mlawer, j.s. delamere,            !
#!      m.j. iacono, k. cady-pereira, s. boukabara, and p.d. brown:         !
#!      atmospheric radiative transfer modeling: a summary of the aer       !
#!      codes, j. quant. spectrosc. radiat. transfer, 91, 233-244, 2005.    !
#!                                                                          !
#!      mlawer, e.j., s.j. taubman, p.d. brown, m.j. iacono, and s.a.       !
#!      clough:  radiative transfer for inhomogeneous atmospheres: rrtm,    !
#!      a validated correlated-k model for the longwave.  j. geophys. res., !
#!      102, 16663-16682, 1997.                                             !
#!                                                                          !
#!    (mcica):                                                              !
#!      pincus, r., h. w. barker, and j.-j. morcrette: a fast, flexible,    !
#!      approximation technique for computing radiative transfer in         !
#!      inhomogeneous cloud fields, j. geophys. res., 108(d13), 4376,       !
#!      doi:10.1029/2002JD003322, 2003.                                     !
#!                                                                          !
#! ************************************************************************ !
#!                                                                          !
#!    aer's revision history:                                               !
#!     this version of rrtmg_lw has been modified from rrtm_lw to use a     !
#!     reduced set of g-points for application to gcms.                     !
#!                                                                          !
#! --  original version (derived from rrtm_lw), reduction of g-points,      !
#!     other revisions for use with gcms.                                   !
#!        1999: m. j. iacono, aer, inc.                                     !
#! --  adapted for use with ncar/cam3.                                      !
#!        may 2004: m. j. iacono, aer, inc.                                 !
#! --  revised to add mcica capability.                                     !
#!        nov 2005: m. j. iacono, aer, inc.                                 !
#! --  conversion to f90 formatting for consistency with rrtmg_sw.          !
#!        feb 2007: m. j. iacono, aer, inc.                                 !
#! --  modifications to formatting to use assumed-shape arrays.             !
#!        aug 2007: m. j. iacono, aer, inc.                                 !
#!                                                                          !
#! ************************************************************************ !
#!                                                                          !
#!    ncep modifications history log:                                       !
#!                                                                          !
#!       nov 1999,  ken campana       -- received the original code from    !
#!                    aer (1998 ncar ccm version), updated to link up with  !
#!                    ncep mrf model                                        !
#!       jun 2000,  ken campana       -- added option to switch random and  !
#!                    maximum/random cloud overlap                          !
#!           2001,  shrinivas moorthi -- further updates for mrf model      !
#!       may 2001,  yu-tai hou        -- updated on trace gases and cloud   !
#!                    property based on rrtm_v3.0 codes.                    !
#!       dec 2001,  yu-tai hou        -- rewritten code into fortran 90 std !
#!                    set ncep radiation structure standard that contains   !
#!                    three plug-in compatable fortran program files:       !
#!                    'radlw_param.f', 'radlw_datatb.f', 'radlw_main.f'     !
#!                    fixed bugs in subprograms taugb14, taugb2, etc. added !
#!                    out-of-bounds protections. (a detailed note of        !
#!                    up_to_date modifications/corrections by ncep was sent !
#!                    to aer in 2002)                                       !
#!       jun 2004,  yu-tai hou        -- added mike iacono's apr 2004       !
#!                    modification of variable diffusivity angles.          !
#!       apr 2005,  yu-tai hou        -- minor modifications on module      !
#!                    structures include rain/snow effect (this version of  !
#!                    code was given back to aer in jun 2006)               !
#!       mar 2007,  yu-tai hou        -- added aerosol effect for ncep      !
#!                    models using the generallized aerosol optical property!
#!                    scheme for gfs model.                                 !
#!       apr 2007,  yu-tai hou        -- added spectral band heating as an  !
#!                    optional output to support the 500 km gfs model's     !
#!                    upper stratospheric radiation calculations. and       !
#!                    restructure optional outputs for easy access by       !
#!                    different models.                                     !
#!       oct 2008,  yu-tai hou        -- modified to include new features   !
#!                    from aer's newer release v4.4-v4.7, including the     !
#!                    mcica sub-grid cloud option. add rain/snow optical    !
#!                    properties support to cloudy sky calculations.        !
#!                    correct errors in mcica cloud optical properties for  !
#!                    ebert & curry scheme (ilwcice=1) that needs band      !
#!                    index conversion. simplified and unified sw and lw    !
#!                    sub-column cloud subroutines into one module by using !
#!                    optional parameters.                                  !
#!       mar 2009,  yu-tai hou        -- replaced the original random number!
#!                    generator coming from the original code with ncep w3  !
#!                    library to simplify the program and moved sub-column  !
#!                    cloud subroutines inside the main module. added       !
#!                    option of user provided permutation seeds that could  !
#!                    be randomly generated from forecast time stamp.       !
#!       oct 2009,  yu-tai hou        -- modified subrtines "cldprop" and   !
#!                    "rlwinit" according updats from aer's rrtmg_lw v4.8.  !
#!       nov 2009,  yu-tai hou        -- modified subrtine "taumol" according
#!                    updats from aer's rrtmg_lw version 4.82. notice the   !
#!                    cloud ice/liquid are assumed as in-cloud quantities,  !
#!                    not as grid averaged quantities.                      !
#!       jun 2010,  yu-tai hou        -- optimized code to improve efficiency
#!       apr 2012,  b. ferrier and y. hou -- added conversion factor to fu's!
#!                    cloud-snow optical property scheme.                   !
#!       nov 2012,  yu-tai hou        -- modified control parameters thru   !
#!                     module 'physparam'.                                  !  
#!       FEB 2017    A.Cheng   - add odpth output, effective radius input   !
#!       jun 2018,  h-m lin/y-t hou   -- added new option of cloud overlap  !
#!                     method 'de-correlation-length' for mcica application !
#!                                                                          !
#!!!!!  ==============================================================  !!!!!
#!!!!!                         end descriptions                         !!!!!
#!!!!!  ==============================================================  !!!!!
#
#
#!> \defgroup module_radlw_main module_radlw_main
#!! \ingroup rad
#!! This module includes NCEP's modifications of the rrtmg-lw radiation
#!! code from AER.
#!!
#!! The RRTM-LW package includes three files:
#!! - radlw_param.f, which contains:
#!!  - module_radlw_parameters: band parameters set up
#!! - radlw_datatb.f, which contains modules:
#!!  - module_radlw_avplank: plank flux data
#!!  - module_radlw_ref: reference temperature and pressure
#!!  - module_radlw_cldprlw: cloud property coefficients
#!!  - module_radlw_kgbnn: absorption coeffients for 16 bands, where nn = 01-16
#!! - radlw_main.f, which contains:
#!!  - module_radlw_main, which is the main LW radiation transfer
#!!    program and contains two externally callable subroutines:
#!!   - lwrad(): the main LW radiation routine
#!!   - rlwinit(): the initialization routine
#!!
#!! All the LW radiation subprograms become contained subprograms in
#!! module 'module_radlw_main' and many of them are not directly
#!! accessable from places outside the module.
#!!
#!!\author   Eli J. Mlawer, emlawer@aer.com 
#!!\author   Jennifer S. Delamere, jdelamer@aer.com                    
#!!\author   Michael J. Iacono, miacono@aer.com  
#!!\author   Shepard A. Clough
#!!\version NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82
#!!                                                               
#!! The authors wish to acknowledge the contributions of the       
#!! following people:  Steven J. Taubman, Karen Cady-Pereira,
#!! Patrick D. Brown, Ronald E. Farren, Luke Chen, Robert Bergstrom.
#!!
#!!\copyright  2002-2007, Atmospheric & Environmental Research, Inc. (AER).
#!!  This software may be used, copied, or redistributed as long as it is
#!!  not sold and this copyright notice is reproduced on each copy made.
#!!  This model is provided as is without any express or implied warranties.
#!!  (http://www.rtweb.aer.com/)
#!! @{
#!========================================!
#      module module_radlw_main           !
#!........................................!
#!
import numpy as np
from radiation.python.phys_const import *
# use mersenne_twister, only : random_setseed, random_number,       &
#     &                             random_stat
from radiation.python.radlw.radlw_param import *

from radiation.python.radlw.radlw_avplank import totplnk
from radiation.python.radlw.radlw_ref import preflog, tref, chi_mls

VTAGLW = 'NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82'

# constant values
eps      = 1.0e-6
oneminus = 1.0-eps
cldmin   = 1.0e-80
bpade    = 1.0/0.278  # pade approx constant
stpfac   = 296.0/1013.0
wtdiff   = 0.5        # weight for radiance to flux conversion
tblint   = ntbl       # lookup table conversion factor
f_zero   = 0.0
f_one    = 1.0

#  ...  atomic weights for conversion from mass to volume mixing ratios
amdw    = con_amd/con_amw
amdo3   = con_amd/con_amo3

#  ...  band indices
nspa = np.array([1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9])
nspb = np.array([1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0])

#  ---  reset diffusivity angle for Bands 2-3 and 5-9 to vary (between 1.50
#       and 1.80) as a function of total column water vapor.  the function
#       has been defined to minimize flux and cooling rate errors in these bands
#       over a wide range of precipitable water values.
a0 = np.array([1.66,  1.55,  1.58,  1.66,  1.54, 1.454,  1.89,  1.33,  
  1.668,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66,  1.66])
a1 = np.array([0.00,  0.25,  0.22,  0.00,  0.13, 0.446, -0.10,  0.40,
 -0.006,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00])
a2 = np.array([0.00, -12.0, -11.7,  0.00, -0.72,-0.243,  0.19,-0.062,
  0.414,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00])

#! ---  logical flags for optional output fields

lhlwb  = False
lhlw0  = False
lflxprf= False

#  ---  those data will be set up only once by "rlwinit"

#  ...  fluxfac, heatfac are factors for fluxes (in w/m**2) and heating
#       rates (in k/day, or k/sec set by subroutine 'rlwinit')
#       semiss0 are default surface emissivity for each bands

semiss0 = np.ones(nbands)*nbands


tau_tbl = np.zeros(ntbl)  #clr-sky opt dep (for cldy transfer)
exp_tbl = np.zeros(ntbl)  #transmittance lookup table
tfn_tbl = np.zeros(ntbl)  #tau transition function; i.e. the
                          #transition of planck func from mean lyr
                          #temp to lyr boundary temp as a func of
                          #opt dep. "linear in tau" method is used.

#  ---  the following variables are used for sub-column cloud scheme

ipsdlw0 = ngptlw     # initial permutation seed



#> This subroutine is the main LW radiation routine.
#!!\param plyr           model layer mean pressure in mb
#!!\param plvl           model interface pressure in mb
#!!\param tlyr           model layer mean temperature in K
#!!\param tlvl           model interface temperature in K
#!!\param qlyr           layer specific humidity in gm/gm  
#!!\param olyr           layer ozone concentration in gm/gm 
#!!\param gasvmr         atmospheric gases amount:
#!!\n                    (:,:,1)  - co2 volume mixing ratio
#!!\n                    (:,:,2)  - n2o volume mixing ratio
#!!\n                    (:,:,3)  - ch4 volume mixing ratio
#!!\n                    (:,:,4)  - o2  volume mixing ratio
#!!\n                    (:,:,5)  - co  volume mixing ratio
#!!\n                    (:,:,6)  - cfc11 volume mixing ratio
#!!\n                    (:,:,7)  - cfc12 volume mixing ratio
#!!\n                    (:,:,8)  - cfc22 volume mixing ratio
#!!\n                    (:,:,9)  - ccl4  volume mixing ratio
#!!\param clouds         layer cloud profile
#!!\n   for  ilwcliq > 0  ---
#!!\n                    (:,:,1)  - layer total cloud fraction
#!!\n                    (:,:,2)  - layer in-cloud liq water path (\f$ g/m^2 \f$)
#!!\n                    (:,:,3)  - mean eff radius for liq cloud (micron)
#!!\n                    (:,:,4)  - layer in-cloud ice water path (\f$ g/m^2 \f$)
#!!\n                    (:,:,5)  - mean eff radius for ice cloud (micron)
#!!\n                    (:,:,6)  - layer rain drop water path    (\f$ g/m^2 \f$)
#!!\n                    (:,:,7)  - mean eff radius for rain drop (micron)
#!!\n                    (:,:,8)  - layer snow flake water path   (\f$ g/m^2 \f$)
#!!\n                    (:,:,9)  - mean eff radius for snow flake(micron)
#!!\n   for  ilwcliq = 0  ---
#!!\n                    (:,:,1)  - layer total cloud fraction
#!!\n                    (:,:,2)  - layer cloud optical depth
#!!\n                    (:,:,3)  - layer cloud single scattering albedo
#!!\n                    (:,:,4)  - layer cloud asymmetry factor
#!!\param icseed         auxiliary special cloud related array.
#!!\param aerosols       aerosol optical properties 
#!!\n                    (:,:,:,1) - optical depth
#!!\n                    (:,:,:,2) - single scattering albedo
#!!\n                    (:,:,:,3) - asymmetry parameter
#!!\param sfemis         surface emissivity
#!!\param sfgtmp         surface ground temperature in K
#!!\param dzlyr          layer thickness (km)
#!!\param delpin         layer pressure thickness (mb)
#!!\param de_lgth        cloud decorrelation length (km)
#!!\param npts           total number of horizontal points
#!!\param nlay, nlp1     total number of vertical layers, levels
#!!\param lprnt          cntl flag for diagnostic print out
#!!\param hlwc           total sky heating rate in k/day or k/sec
#!!\param topflx         radiation fluxes at top, components
#!!\n                    upfxc - total sky upward flux at top (\f$ w/m^2 \f$)
#!!\n                    upfx0 - clear sky upward flux at top (\f$ w/m^2 \f$)
#!!\param sfcflx         radiation fluxes at sfc, components
#!!\n                    upfxc - total sky upward flux at sfc (\f$ w/m^2 \f$)
#!!\n                    dnfxc - total sky downward flux at sfc (\f$ w/m^2 \f$)
#!!\n                    upfx0 - clear sky upward flux at sfc (\f$ w/m^2 \f$)
#!!\n                    dnfx0 - clear sky downward flux at sfc (\f$ w/m^2 \f$)
#!!\param cldtau         spectral band layer cloud optical depth (approx 10 mu)
#!!\param hlwb           spectral band total sky heating rates
#!!\param hlw0           clear sky heating rates (k/sec or k/day)
#!!\param flxprf         level radiation fluxes (\f$ w/m^2 \f$), components
#!!\n                    dnfxc - total sky downward flux
#!!\n                    upfxc - total sky upward flux
#!!\n                    dnfx0 - clear sky downward flux
#!!\n                    upfx0 - clear sky upward flux
#!> \section gen_lwrad General Algorithm
#!> @{
#! --------------------------------
def lwrad(plyr, plvl, tlyr, tlvl, qlyr, olyr, gasvmr, clouds, icseed,
          aerosols, sfemis, sfgtmp, dzlyr, delpin, de_lgth, npts,nlay, nlp1, lprnt,
          HLW0=None, HLWB=None, FLXPRF=None):

#!  ====================  defination of variables  ====================  !
#!                                                                       !
#!  input variables:                                                     !
#!     plyr (npts,nlay) : layer mean pressures (mb)                      !
#!     plvl (npts,nlp1) : interface pressures (mb)                       !
#!     tlyr (npts,nlay) : layer mean temperature (k)                     !
#!     tlvl (npts,nlp1) : interface temperatures (k)                     !
#!     qlyr (npts,nlay) : layer specific humidity (gm/gm)   *see inside  !
#!     olyr (npts,nlay) : layer ozone concentration (gm/gm) *see inside  !
#!     gasvmr(npts,nlay,:): atmospheric gases amount:                    !
#!                       (check module_radiation_gases for definition)   !
#!       gasvmr(:,:,1)  -   co2 volume mixing ratio                      !
#!       gasvmr(:,:,2)  -   n2o volume mixing ratio                      !
#!       gasvmr(:,:,3)  -   ch4 volume mixing ratio                      !
#!       gasvmr(:,:,4)  -   o2  volume mixing ratio                      !
#!       gasvmr(:,:,5)  -   co  volume mixing ratio                      !
#!       gasvmr(:,:,6)  -   cfc11 volume mixing ratio                    !
#!       gasvmr(:,:,7)  -   cfc12 volume mixing ratio                    !
#!       gasvmr(:,:,8)  -   cfc22 volume mixing ratio                    !
#!       gasvmr(:,:,9)  -   ccl4  volume mixing ratio                    !
#!     clouds(npts,nlay,:): layer cloud profiles:                        !
#!                       (check module_radiation_clouds for definition)  !
#!       clouds(:,:,1)  -   layer total cloud fraction                   !
#!       clouds(:,:,2)  -   layer in-cloud liq water path   (g/m**2)     !
#!       clouds(:,:,3)  -   mean eff radius for liq cloud   (micron)     !
#!       clouds(:,:,4)  -   layer in-cloud ice water path   (g/m**2)     !
#!       clouds(:,:,5)  -   mean eff radius for ice cloud   (micron)     !
#!       clouds(:,:,6)  -   layer rain drop water path      (g/m**2)     !
#!       clouds(:,:,7)  -   mean eff radius for rain drop   (micron)     !
#!       clouds(:,:,8)  -   layer snow flake water path     (g/m**2)     !
#!       clouds(:,:,9)  -   mean eff radius for snow flake  (micron)     !
#!     icseed(npts)   : auxiliary special cloud related array            !
#!                      when module variable isubclw=2, it provides      !
#!                      permutation seed for each column profile that    !
#!                      are used for generating random numbers.          !
#!                      when isubclw /=2, it will not be used.           !
#!     aerosols(npts,nlay,nbands,:) : aerosol optical properties         !
#!                       (check module_radiation_aerosols for definition)!
#!        (:,:,:,1)     - optical depth                                  !
#!        (:,:,:,2)     - single scattering albedo                       !
#!        (:,:,:,3)     - asymmetry parameter                            !
#!     sfemis (npts)  : surface emissivity                               !
#!     sfgtmp (npts)  : surface ground temperature (k)                   !
#!     dzlyr(npts,nlay) : layer thickness (km)                           !
#!     delpin(npts,nlay): layer pressure thickness (mb)                  !
#!     de_lgth(npts)    : cloud decorrelation length (km)                !
#!     npts           : total number of horizontal points                !
#!     nlay, nlp1     : total number of vertical layers, levels          !
#!     lprnt          : cntl flag for diagnostic print out               !
#!                                                                       !
#!  output variables:                                                    !
#!     hlwc  (npts,nlay): total sky heating rate (k/day or k/sec)        !
#!     topflx(npts)     : radiation fluxes at top, component:            !
#!                        (check module_radlw_paramters for definition)  !
#!        upfxc           - total sky upward flux at top (w/m2)          !
#!        upfx0           - clear sky upward flux at top (w/m2)          !
#!     sfcflx(npts)     : radiation fluxes at sfc, component:            !
#!                        (check module_radlw_paramters for definition)  !
#!        upfxc           - total sky upward flux at sfc (w/m2)          !
#!        upfx0           - clear sky upward flux at sfc (w/m2)          !
#!        dnfxc           - total sky downward flux at sfc (w/m2)        !
#!        dnfx0           - clear sky downward flux at sfc (w/m2)        !
#!     cldtau(npts,nlay): approx 10mu band layer cloud optical depth     !
#!                                                                       !
#!! optional output variables:                                           !
#!     hlwb(npts,nlay,nbands): spectral band total sky heating rates     !
#!     hlw0  (npts,nlay): clear sky heating rate (k/day or k/sec)        !
#!     flxprf(npts,nlp1): level radiative fluxes (w/m2), components:     !
#!                        (check module_radlw_paramters for definition)  !
#!        upfxc           - total sky upward flux                        !
#!        dnfxc           - total sky dnward flux                        !
#!        upfx0           - clear sky upward flux                        !
#!        dnfx0           - clear sky dnward flux                        !
#!                                                                       !
#!  external module variables:  (in physparam)                            !
#!   ilwrgas - control flag for rare gases (ch4,n2o,o2,cfcs, etc.)       !
#!           =0: do not include rare gases                               !
#!           >0: include all rare gases                                  !
#!   ilwcliq - control flag for liq-cloud optical properties             !
#!           =1: input cld liqp & reliq, hu & stamnes (1993)             !
#!           =2: not used                                                !
#!   ilwcice - control flag for ice-cloud optical properties             !
#!           =1: input cld icep & reice, ebert & curry (1997)            !
#!           =2: input cld icep & reice, streamer (1996)                 !
#!           =3: input cld icep & reice, fu (1998)                       !
#!   isubclw - sub-column cloud approximation control flag               !
#!           =0: no sub-col cld treatment, use grid-mean cld quantities  !
#!           =1: mcica sub-col, prescribed seeds to get random numbers   !
#!           =2: mcica sub-col, providing array icseed for random numbers!
#!   iovrlw  - cloud overlapping control flag                            !
#!           =0: random overlapping clouds                               !
#!           =1: maximum/random overlapping clouds                       !
#!           =2: maximum overlap cloud (used for isubclw>0 only)         !
#!           =3: decorrelation-length overlap (for isubclw>0 only)       !
#!   ivflip  - control flag for vertical index direction                 !
#!           =0: vertical index from toa to surface                      !
#!           =1: vertical index from surface to toa                      !
#!                                                                       !
#!  module parameters, control variables:                                !
#!     nbands           - number of longwave spectral bands              !
#!     maxgas           - maximum number of absorbing gaseous            !
#!     maxxsec          - maximum number of cross-sections               !
#!     ngptlw           - total number of g-point subintervals           !
#!     ng##             - number of g-points in band (##=1-16)           !
#!     ngb(ngptlw)      - band indices for each g-point                  !
#!     bpade            - pade approximation constant (1/0.278)          !
#!     nspa,nspb(nbands)- number of lower/upper ref atm's per band       !
#!     delwave(nbands)  - longwave band width (wavenumbers)              !
#!     ipsdlw0          - permutation seed for mcica sub-col clds        !
#!                                                                       !
#!  major local variables:                                               !
#!     pavel  (nlay)         - layer pressures (mb)                      !
#!     delp   (nlay)         - layer pressure thickness (mb)             !
#!     tavel  (nlay)         - layer temperatures (k)                    !
#!     tz     (0:nlay)       - level (interface) temperatures (k)        !
#!     semiss (nbands)       - surface emissivity for each band          !
#!     wx     (nlay,maxxsec) - cross-section molecules concentration     !
#!     coldry (nlay)         - dry air column amount                     !
#!                                   (1.e-20*molecules/cm**2)            !
#!     cldfrc (0:nlp1)       - layer cloud fraction                      !
#!     taucld (nbands,nlay)  - layer cloud optical depth for each band   !
#!     cldfmc (ngptlw,nlay)  - layer cloud fraction for each g-point     !
#!     tauaer (nbands,nlay)  - aerosol optical depths                    !
#!     fracs  (ngptlw,nlay)  - planck fractions                          !
#!     tautot (ngptlw,nlay)  - total optical depths (gaseous+aerosols)   !
#!     colamt (nlay,maxgas)  - column amounts of absorbing gases         !
#!                             1-maxgas are for watervapor, carbon       !
#!                             dioxide, ozone, nitrous oxide, methane,   !
#!                             oxigen, carbon monoxide, respectively     !
#!                             (molecules/cm**2)                         !
#!     pwvcm                 - column precipitable water vapor (cm)      !
#!     secdiff(nbands)       - variable diffusivity angle defined as     !
#!                             an exponential function of the column     !
#!                             water amount in bands 2-3 and 5-9.        !
#!                             this reduces the bias of several w/m2 in  !
#!                             downward surface flux in high water       !
#!                             profiles caused by using the constant     !
#!                             diffusivity angle of 1.66.         (mji)  !
#!     facij  (nlay)         - indicator of interpolation factors        !
#!                             =0/1: indicate lower/higher temp & height !
#!     selffac(nlay)         - scale factor for self-continuum, equals   !
#!                          (w.v. density)/(atm density at 296K,1013 mb) !
#!     selffrac(nlay)        - factor for temp interpolation of ref      !
#!                             self-continuum data                       !
#!     indself(nlay)         - index of the lower two appropriate ref    !
#!                             temp for the self-continuum interpolation !
#!     forfac (nlay)         - scale factor for w.v. foreign-continuum   !
#!     forfrac(nlay)         - factor for temp interpolation of ref      !
#!                             w.v. foreign-continuum data               !
#!     indfor (nlay)         - index of the lower two appropriate ref    !
#!                             temp for the foreign-continuum interp     !
#!     laytrop               - tropopause layer index at which switch is !
#!                             made from one conbination kew species to  !
#!                             another.                                  !
#!     jp(nlay),jt(nlay),jt1(nlay)                                       !
#!                           - lookup table indexes                      !
#!     totuflux(0:nlay)      - total-sky upward longwave flux (w/m2)     !
#!     totdflux(0:nlay)      - total-sky downward longwave flux (w/m2)   !
#!     htr(nlay)             - total-sky heating rate (k/day or k/sec)   !
#!     totuclfl(0:nlay)      - clear-sky upward longwave flux (w/m2)     !
#!     totdclfl(0:nlay)      - clear-sky downward longwave flux (w/m2)   !
#!     htrcl(nlay)           - clear-sky heating rate (k/day or k/sec)   !
#!     fnet    (0:nlay)      - net longwave flux (w/m2)                  !
#!     fnetc   (0:nlay)      - clear-sky net longwave flux (w/m2)        !
#!                                                                       !
#!                                                                       !
#!  ======================    end of definitions    ===================  !
    # init input variables
    gasvmr = np.zeros((npts, nlay, 9))
    clouds = np.zeros((npts, nlay, 9))
    sfemis = np.zeros(npts)
    sfgtmp = np.zeros(npts)
    de_lgth = np.zeros(npts)
    aerosols = np.zeros((npts, nlay, nbands, 3))

    # init output variables
    hlwc = np.zeros((npts, nlay))
    cldtau = np.zeros((npts, nlay))

    topflx = np.zeros(topflx)
    sfcflw = np.zeros(sfcflx)

    # init optional output variables
    hlwb = np.zeros((npts, nlay, nbands))
    hlw0 = np.zeros((npts, nlay, nbands))
    flxprf = np.zeros((npts, nlay, nbands))

    #  ---  locals:
    cldfrac = np.zeros(nlp1+1)

    totuflx = np.zeros(nlay+1)
    totdflx = np.zeros(nlay+1)
    totuclfl = np.zeros(nlay+1)
    totdclfl = np.zeros(nlay+1)
    tz = np.zeros(nlay+1)

    htr = np.zeros(nlay)
    htrcl = np.zeros(nlay)
    pavel = np.zeros(nlay)
    tavel = np.zeros(nlay)
    delp = np.zeros(nlay)
    clwp = np.zeros(nlay)
    ciwp = np.zeros(nlay)
    relw = np.zeros(nlay)
    reiw = np.zeros(nlay)
    cda1 = np.zeros(nlay)
    cda2 = np.zeros(nlay)
    cda3 = np.zeros(nlay)
    cda4 = np.zeros(nlay)
    coldry = np.zeros(nlay)
    colbrd = np.zeros(nlay)
    h2ovmr = np.zeros(nlay)
    o3vmr = np.zeros(nlay)
    fac00 = np.zeros(nlay)
    fac01 = np.zeros(nlay)
    fac10 = np.zeros(nlay)
    fac11 = np.zeros(nlay)
    selffac = np.zeros(nlay)
    selffrac = np.zeros(nlay)
    forfac = np.zeros(nlay)
    forfrac = np.zeros(nlay)
    minorfrac = np.zeros(nlay)
    scaleminor = np.zeros(nlay)
    scaleminorn2 = np.zeros(nlay)
    temcol = np.zeros(nlay)
    dz = np.zeros(nlay)

    pklev = np.zeros((nbands, nlay+1))
    pklay = np.zeros((nbands, nlay+1))

    htrb = np.zeros((nlay, nbands))

    taucld = np.zeros((nbands, nlay))
    tauaer = np.zeros((nbands, nlay))

    fracs = np.zeros((ngptlw, nlay))
    tautot = np.zeros((ngptlw, nlay))
    cldfmc = np.zeros((ngptlw, nlay))

    semiss = np.zeros(nbands)
    secdiff = np.zeros(nbands)

    #  ---  column amount of absorbing gases:
    #       (:,m) m = 1-h2o, 2-co2, 3-o3, 4-n2o, 5-ch4, 6-o2, 7-co
    colamt = np.zeros((nlay,maxgas))

    #  ---  column cfc cross-section amounts:
    #       (:,m) m = 1-ccl4, 2-cfc11, 3-cfc12, 4-cfc22
    wx = np.zeros((nlay,maxxsec))

    #  ---  reference ratios of binary species parameter in lower atmosphere:
    #       (:,m,:) m = 1-h2o/co2, 2-h2o/o3, 3-h2o/n2o, 4-h2o/ch4, 5-n2o/co2, 6-o3/co2
    rfrate = np.zeros((nlay,nrates,2))

    tem0 = 0
    tem1 = 0
    tem2 = 0
    pwvcm = 0
    summol = 0
    stemp = 0
    delgth = 0

    ipseed = np.zeros(npts)
    jp = np.zeros(nlay)
    jt = np.zeros(nlay)
    jt1 = np.zeros(nlay)
    indself = np.zeros(nlay)
    indfor = np.zeros(nlay)
    indminor = np.zeros(nlay)

    laytrop = 0
    iplon = 0
    i = 0
    j = 0
    k = 0
    k1 = 0
    
    # lcf1 (bool)

    # begin here

    #  --- ...  initialization
    if hlwb is not None:
        lhlwb = True
    if hlw0 is not None:
        lhlw0 = True
    if flxprf is not None:
        lflxprf = True
 

    #> -# Change random number seed value for each radiation invocation
    #!    (isubclw =1 or 2).

    if isubclw == 1:     # advance prescribed permutation seed
        for i in range(npts):
            ipseed[i] = ipsdlw0 + i
    elif isubclw == 2:   # use input array of permutaion seeds
        for i in range(npts):
            ipseed[i] = icseed[i]

    #  --- ...  loop over horizontal npts profiles
 
    for iplon in range(npts):
        #> -# Read surface emissivity.
        if sfemis[iplon] > eps and sfemis[iplon] <= 1.0:  # input surface emissivity
            for j in range(nbands):
                semiss[j] = sfemis[iplon]
        else:                                                      ! use default values
            for j in range(nbands):
                semiss[j] = semiss0[j]

        stemp = sfgtmp[iplon]          # surface ground temp
        if iovrlw == 3:
            delgth = de_lgth[iplon]    ! clouds decorr-length

        #> -# Prepare atmospheric profile for use in rrtm.
        #           the vertical index of internal array is from surface to top

        #!  --- ...  molecular amounts are input or converted to volume mixing ratio
        #!           and later then converted to molecular amount (molec/cm2) by the
        #!           dry air column coldry (in molec/cm2) which is calculated from the
        #!           layer pressure thickness (in mb), based on the hydrostatic equation
        #!  --- ...  and includes a correction to account for h2o in the layer.

        if ivflip == 0:       # input from toa to sfc
            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd
            tz[0] = tlvl[iplon, nlp1]

            for k in range(nlay):
                k1 = nlp1 - k
                pavel[k]= plyr[iplon, k1]
                delp[k] = delpin[iplon, k1]
                tavel[k]= tlyr[iplon, k1]
                tz[k]   = tlvl[iplon, k1]
                dz[k]   = dzlyr[iplon, k1]

                #> -# Set absorber amount for h2o, co2, and o3.
                h2ovmr[k] = np.maximum(f_zero, qlyr[iplon, k1]*amdw/(f_one-qlyr[iplon, k1])) # input specific humidity
                o3vmr[k] = np.maximum(f_zero, olyr[iplon, k1]*amdo3)  # input mass mixing ratio

                #  --- ...  tem0 is the molecular weight of moist air
                tem0 = (f_one - h2ovmr[k])*con_amd + h2ovmr[k]*con_amw
                coldry[k] = tem2*delp[k] / (tem1*tem0*(f_one+h2ovmr[k]))
                temcol[k] = 1.0e-12 * coldry[k]

                colamt[k, 0] = np.maximum(f_zero, coldry[k]*h2ovmr[k])          # h2o
                colamt[k, 1] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k1, 0)) # co2
                colamt[k, 2] = np.maximum(temcol[k], coldry[k]*o3vmr[k])           ! o3


            #> -# Set up column amount for rare gases n2o,ch4,o2,co,ccl4,cf11,cf12,
            #!    cf22, convert from volume mixing ratio to molec/cm2 based on
            #!    coldry (scaled to 1.0e-20).

            if ilwrgas > 0:
                for k in range(nlay):
                    k1 = nlp1 - k
                    colamt[k, 3] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k1, 1))  # n2o
                    colamt[k, 4] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k1, 2))  # ch4
                    colamt[k, 5] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 3))  # o2
                    colamt[k, 6] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 4))  # co

                    wx[k, 0] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 8])   # ccl4
                    wx[k, 1] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 5])   # cf11
                    wx[k, 2] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 6])   # cf12
                    wx[k, 3] = np.maximum(f_zero, coldry[k]*gasvmr[iplon, k1, 7])   # cf22
            else:
                for k in range(nlay):
                    colamt[k, 3] = f_zero     # n2o
                    colamt[k, 4] = f_zero     # ch4
                    colamt[k, 5] = f_zero     # o2
                    colamt[k, 6] = f_zero     # co

                    wx[k, 0] = f_zero
                    wx[k, 1] = f_zero
                    wx[k, 2] = f_zero
                    wx[k, 3] = f_zero


            #> -# Set aerosol optical properties.

            for in range(nlay):
                k1 = nlp1 - k
                for j in range(nbands):
                    tauaer[j, k] = aerosols[iplon, k1, j, 0] * (f_one - aerosols[iplon, k1, j, 1])

            #> -# Read cloud optical properties
            if ilwcliq > 0: # use prognostic cloud method
                for k in range(nlay):
                    k1 = nlp1 - k
                    cldfrc[k]= clouds[iplon, k1, 0]
                    clwp[k]  = clouds[iplon, k1, 1]
                    relw[k]  = clouds[iplon, k1, 2]
                    ciwp[k]  = clouds[iplon, k1, 3]
                    reiw[k]  = clouds[iplon, k1, 4]
                    cda1[k]  = clouds[iplon, k1, 5]
                    cda2[k]  = clouds[iplon, k1, 6]
                    cda3[k]  = clouds[iplon, k1, 7]
                    cda4[k]  = clouds[iplon, k1, 8]

            else:                       # use diagnostic cloud method
                for k in range(nlay):
                    k1 = nlp1 - k
                    cldfrc[k]= clouds[iplon, k1, 0]
                    cda1[k]  = clouds[iplon, k1, 1]

            cldfrc[0]    = f_one       # padding value only
            cldfrc[nlp1] = f_zero      # padding value only

            #> -# Compute precipitable water vapor for diffusivity angle adjustments.

            tem1 = f_zero
            tem2 = f_zero
            for k in range(nlay):
                tem1 = tem1 + coldry[k] + colamt[k, 0]
                tem2 = tem2 + colamt[k, 0]

            tem0 = 10.0 * tem2 / (amdw * tem1 * con_g)
            pwvcm = tem0 * plvl[iplon, nlp1]

        else:                       # input from sfc to toa
            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd
            tz[0] = tlvl[iplon, 0]

            for k in range(nlay):
                pavel[k]= plyr[iplon, k]
                delp[k] = delpin[iplon, k]
                tavel[k]= tlyr[iplon, k]
                tz[k]   = tlvl[iplon, k+1]
                dz[k]   = dzlyr[iplon, k]

                #  --- ...  set absorber amount
                h2ovmr[k] = np.maximum(f_zero, qlyr[iplon, k]*amdw/(f_one-qlyr[iplon, k]))           # input specific humidity
                o3vmr[k] = np.maximum(f_zero, olyr[iplon, k]*amdo3)  # input mass mixing ratio

                #  --- ...  tem0 is the molecular weight of moist air
                tem0 = (f_one - h2ovmr[k])*con_amd + h2ovmr[k]*con_amw
                coldry[k] = tem2*delp[k] / (tem1*tem0*(f_one+h2ovmr[k]))
                temcol[k] = 1.0e-12 * coldry[k]

                colamt[k, 0] = np.maximum(f_zero,    coldry[k]*h2ovmr[k])          # h2o
                colamt[k, 1] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k, 0])  # co2
                colamt[k, 2] = np.maximum(temcol[k], coldry[k]*o3vmr[k])           # o3

            #  --- ...  set up col amount for rare gases, convert from volume mixing ratio
            #           to molec/cm2 based on coldry (scaled to 1.0e-20)

            if ilwrgas > 0:
                for k in range(nlay):
                    colamt[k, 3] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k, 1])  # n2o
                    colamt[k, 4] = np.maximum(temcol[k], coldry[k]*gasvmr[iplon, k, 2])  # ch4
                    colamt[k, 5] = np.maximum(f_zero,    coldry[k]*gasvmr[iplon, k, 3])  # o2
                    colamt[k, 6] = np.maximum(f_zero,    coldry[k]*gasvmr[iplon, k, 4])  # co

                    wx[k, 0] = np.maximum( f_zero, coldry[k]*gasvmr[iplon, k, 8])   # ccl4
                    wx[k, 1] = np.maximum( f_zero, coldry[k]*gasvmr[iplon, k, 5])   # cf11
                    wx[k, 2] = np.maximum( f_zero, coldry[k]*gasvmr[iplon, k, 6])   # cf12
                    wx[k, 3] = np.maximum( f_zero, coldry[k]*gasvmr[iplon, k, 7])   # cf22

            else:
                for k in range(nlay):
                    colamt[k, 3] = f_zero     # n2o
                    colamt[k, 4] = f_zero     # ch4
                    colamt[k, 5] = f_zero     # o2
                    colamt[k, 6] = f_zero     # co

                    wx[k, 0] = f_zero
                    wx[k, 1] = f_zero
                    wx[k, 2] = f_zero
                    wx[k, 3] = f_zero

            #  --- ...  set aerosol optical properties

            for j in range(nbands):
                for k in range(nlay):
                    tauaer[j, k] = aerosols[iplon, k, j, 0] * (f_one - aerosols[iplon, k, j, 1])

            if ilwcliq > 0:  # use prognostic cloud method
                for k in range(nlay):
                    cldfrc[k]= clouds[iplon, k, 0]
                    clwp[k]  = clouds[iplon, k, 1]
                    relw[k]  = clouds[iplon, k, 2]
                    ciwp[k]  = clouds[iplon, k, 3]
                    reiw[k]  = clouds[iplon, k, 4]
                    cda1[k]  = clouds[iplon, k, 5]
                    cda2[k]  = clouds[iplon, k, 6]
                    cda3[k]  = clouds[iplon, k, 7]
                    cda4[k]  = clouds[iplon, k, 8]
            else:                       # use diagnostic cloud method
                for k in range(nlay):
                    cldfrc[k]= clouds[iplon, k, 0]
                    cda1[k]  = clouds[iplon, k, 1]

            cldfrc[0]    = f_one       # padding value only
            cldfrc[nlp1] = f_zero      # padding value only

            #  --- ...  compute precipitable water vapor for diffusivity angle adjustments

            tem1 = f_zero
            tem2 = f_zero
            for k in range(nlay):
                tem1 = tem1 + coldry[k] + colamt[k, 0]
                tem2 = tem2 + colamt[k, 0]

            tem0 = 10.0 * tem2 / (amdw * tem1 * con_g)
            pwvcm = tem0 * plvl[iplon, 0]



        #> -# Compute column amount for broadening gases.

        for k in range(nlay):
            summol = f_zero
            for i in range(1, maxgas):
                summol = summol + colamt[k, i]

            colbrd[k] = coldry[k] - summol    

        #> -# Compute diffusivity angle adjustments.

        tem1 = 1.80
        tem2 = 1.50
        for j in range(nbands):
            if j == 1 or j == 4 or j == 10:
                secdiff[j] = 1.66
            else:
                secdiff[j] = np.minimum(tem1, np.maximum(tem2, a0[j]+a1[j]*np.exp(a2[j]*pwvcm)))

        #> -# For cloudy atmosphere, call cldprop() to set cloud optical
        #!    properties.

        lcf1 = False
        for k in range(nlay):
            if cldfrc[k] > eps:
                lcf1 = True
                break

        if lcf1:
            cldfmc, taucld = cldprop(cldfrc,
                                     clwp,
                                     relw,
                                     ciwp,
                                     reiw,
                                     cda1,
                                     cda2,
                                     cda3,
                                     cda4,
                                     nlay,
                                     nlp1,
                                     ipseed[iplon],
                                     dz,
                                     delgth)



            #  --- ...  save computed layer cloud optical depth for output
            #           rrtm band-7 is apprx 10mu channel (or use spectral mean of bands 6-8)

            if ivflip == 0:       # input from toa to sfc
                for k in range(nlay):
                    k1 = nlp1 - k
                    cldtau[iplon, k1] = taucld[6, k]
            else:                        # input from sfc to toa
                for k in range(nlay):
                    cldtau[iplon, k] = taucld[6, k]

        else:
            ldfmc = f_zero
            taucld = f_zero

        #> -# Calling setcoef() to compute various coefficients needed in
        #!    radiative transfer calculations.
        (laytrop, pklay, pklev, jp, jt, jt1, rfrate, fac00, fac01, fac10,
        fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
        minorfrac, scaleminor, scaleminorn2, indminor) = setcoef(pavel,
                                                                tavel,
                                                                tz,
                                                                stemp,
                                                                h2ovmr,
                                                                colamt,
                                                                coldry,
                                                                colbrd,
                                                                nlay,
                                                                nlp1)

        #> -# Call taumol() to calculte the gaseous optical depths and Plank 
        #! fractions for each longwave spectral band.

        fracs, tautot = taumol(laytrop, pavel, coldry, colamt, colbrd, wx,
                               tauaer, rfrate, fac00, fac01, fac10, fac11,
                               jp, jt, jt1, selffac, selffrac, indself,
                               forfac, forfrac, indfor, minorfrac,
                               scaleminor, scaleminorn2, indminor, nlay)

        #> -# Call the radiative transfer routine based on cloud scheme
        #!    selection. Compute the upward/downward radiative fluxes, and
        #!    heating rates for both clear or cloudy atmosphere.
        #!\n  - call rtrn(): clouds are assumed as randomly overlaping in a
        #!                   vertical column
        #!\n  - call rtrnmr(): clouds are assumed as in maximum-randomly
        #!                     overlaping in a vertical column;
        #!\n  - call rtrnmc(): clouds are treated with the mcica stochastic
        #!                     approach.

        if isubclw <= 0:
            if iovrlw <= 0:
                (totuflux, totdflux, htr, totuclfl, totdclfl,
                 htrcl, htrb) = rtrn(semiss, delp, cldfrc, taucld, tautot,
                                     pklay, pklev, fracs, secdiff, nlay, nlp1)
            else:
                (totuflux, totdflux, htr, totuclfl, totdclfl, htrcl,
                 htrb) = rtrnmr(semiss, delp, cldfrc, taucld, tautot, pklay,
                                pklev, fracs, secdiff, nlay, nlp1)
        else:
            (totuflux,totdflux,htr, totuclfl,totdclfl,htrcl,
             htrb) = rtrnmc(semiss, delp, cldfmc, taucld, tautot, pklay,
                            pklev, fracs, secdiff, nlay, nlp1)

        #> -# Save outputs.

        topflx(iplon)%upfxc = totuflux(nlay)
        topflx(iplon)%upfx0 = totuclfl(nlay)

        sfcflx(iplon)%upfxc = totuflux(0)
        sfcflx(iplon)%upfx0 = totuclfl(0)
        sfcflx(iplon)%dnfxc = totdflux(0)
            sfcflx(iplon)%dnfx0 = totdclfl(0)

        if ivflip == 0:       # output from toa to sfc
            if lflxprf:
                for k in range(nlay):
                    k1 = nlp1 - k
                    flxprf(iplon,k1)%upfxc = totuflux(k)
                    flxprf(iplon,k1)%dnfxc = totdflux(k)
                    flxprf(iplon,k1)%upfx0 = totuclfl(k)
                    flxprf(iplon,k1)%dnfx0 = totdclfl(k)


            for k in range(nlay):
                k1 = nlp1 - k
                hlwc[iplon, k1] = htr[k]


            #! --- ...  optional clear sky heating rate
            if lhlw0:
                for k in range(nlay):
                    k1 = nlp1 - k
                    hlw0[iplon, k1] = htrcl[k]

            #! --- ...  optional spectral band heating rate
            if lhlwb:
                for j in range(nbands):
                    for k in range(nlay):
                        k1 = nlp1 - k
                        hlwb[iplon, k1, j] = htrb[k, j]

        else:                        # output from sfc to toa
            #! --- ...  optional fluxes
            if lflxprf:
                for k in range(nlay+1):
                    flxprf(iplon,k+1)%upfxc = totuflux(k)
                    flxprf(iplon,k+1)%dnfxc = totdflux(k)
                    flxprf(iplon,k+1)%upfx0 = totuclfl(k)
                    flxprf(iplon,k+1)%dnfx0 = totdclfl(k)


            for k in range(nlay):
                hlwc[iplon, k] = htr[k]

            #! --- ...  optional clear sky heating rate
            if lhlw0:
                for k in range(nlay):
                    hlw0[iplon, k] = htrcl[k]


            #! --- ...  optional spectral band heating rate
            if lhlwb:
                for j in range(nbands):
                    for k in range(nlay):
                        hlwb[iplon, k, j] = htrb[k, j]

    return hlwc, topflx, sfcflx, cldtau
