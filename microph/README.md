# GFDL Cloud Microphysics Module

## Description

GFDL cloud microphysics (MP) scheme is a six-category MP scheme to replace Zhao-Carr MP scheme, and moves the GFS from a total cloud water variable to five predicted hydrometeors (cloud water, cloud ice, rain, snow and graupel). This scheme utilizes the "bulk water" microphysical parameterization technique in Lin et al. (1983) [99] and has been significantly improved over years at GFDL (Lord et al. (1984) [106], Krueger et al. (1995) [95], Chen and Lin (2011) [27], Chen and Lin (2013) [28]). Physics processes of GFDL cloud MP are described in Figure 1 (also see warm_rain() and icloud()) and are feature with time-split between warm-rain (faster) and ice-phase (slower) processes (see 'conversion time scale' in gfdl_cloud_microphys.F90 for default values).


Figure 1: GFDL MP at a glance (Courtesy of S.J. Lin at GFDL)
Some unique attributes of GFDL cloud microphysics include:

## Precipitation and Cloud Effects on Dynamics

Figure 1: FV3 structure; Yellow represents external API routines, called once per physics time step; Green are called once per remapping time step; Blue are called once per acoustic time step.
The leftmost column of Figure 1 shows the external API calls used during a typical process-split model integration procedure. First, the solver is called, which advances the solver a full "physics" time step. This updated state is then passed to the physical parameterization package, which then computes the physics tendencies over the same time interval. Finally, the tendencies are then used to update the model state using a forward-in-time evaluation consistent with the dynamics. 
There are two levels of time-stepping inside FV3. The first is the "remapping" loop, the green column in Figure 1. This loop has three steps:

Perform the Lagrangian dynamics, the loop shown in the blue column of Figure 1.
Perform the subcycled tracer advection along Lagrangian surfaces, using accumulated mass fluxes from the Lagrangian dynamics. Subcycling is done independently within each layer to maintain local (within each layer) stability.
Remap the deformal Lagrangian surfaces on to the reference, or "Eulerian", coordinate levels.
This loop is typically performed once per call to the solver, although it is possible to improve the model's stability by executing the loop (and thereby the vertical remapping) multiple times per solver call.

At grid spacing of less than ~10 km, model dynamics should be able to "see" and "feel" the cloud and precipitation condensate; heat content, heat exchange with the environment, and momentum of condensate should be accounted for. The GFDL microphysics scheme is formulated to accomplish this through strict moist energy conservation during phase changes, and keeping heat and momentum budgets for all condensate. This results in thermodynamic consistency between the FV3 microphysics scheme and FV3 dyanmics.

In current fv3gfs, GFDL in-core fast saturation adjustment (phase-changes only) is called after the "Lagrangian-to-Eulerain" remapping. When GFDL In-Core Fast Saturation Adjustment Module is activated (do_sat_adj=.true. in fv_core_nml block), it adjusts cloud water evaporation (cloud water →water vapor), cloud water freezing (cloud water →cloud ice), and cloud ice deposition (water vapor →cloud ice). The process of condensation is an interesting and well known example. Say dynamics lifts a column of air above saturation, then an adjustment is made to temperature and moisture in order to reach saturation. The tendency of the dynamics has been included in this procedure in order to have the correct balance.

## Scale-awareness

Scale-awareness provided by assumed subgrid variability that is directly proportional to grid spacing. Horizontal sub-grid variability is a function of cell area:

Over land:
hvar=min{0.2,max[0.01, Dland (Ar/10^10)^0.25]}

Over Ocean:
hvar=min{0.2,max[0.01, Docean (Ar/10^10)^0.25]}

Where Ar is cell area, Dland and Docean are base values for sub-grid variability over land and ocean (larger sub-grid variability appears in larger area). Horizontal sub-grid variability is used in cloud fraction, relative humidity calculation, evaporation and condensation processes. Scale-awareness is achieved by this horizontal subgrid variability and a 2nd order FV-type vertical reconstruction (Lin et al. (1994) [100]).

[27] J.-H Chen and S.-J Lin. The remarkable predictability of inter-annual variability of atlantic hurricanes during the past decade. Geophysical Research Letters, 38(L11804):6, 2011.

[28] J-H. Chen and S-J. Lin. Seasonal predictions of tropical cyclones using a 25-km-resolution general circulation model. J. Climate, 26(2):380–398, 2013.

[95] S.K. Krueger, Q. Fu, K. N. Liou, and H-N. S. Chin. Improvement of an ice-phase microphysics parameterization for use in numerical simulations of tropical convection. Journal of Applied Meteorology, 34:281–287, January 1995.

[99] Y.-L. Lin, R. D. Farley, and H. D. Orville. Bulk parameterization of the snow field in a cloud model. J. Climate Appl. Meteor., 22:1065–1092, 1983.

[100] S-J. Lin, W. C. Chao, Y. C. Sud, and G. K. Walker. A class of the van leer-type transport schemes and its application to the moisture transport in a general circulation model. Monthly Weather Review, 122:1575–1593, 1994.

[106] S.J. Lord, H.E. Willoughby, and J.M. Piotrowicz. Role of a parameterized ice-phase microphysics in an axisymmetric, nonhydrostatic tropical cyclone model. J. Atmos. Sci., 41(19):2836–2848, October 1984.

## GFDL Cloud Driver General Algorithm

- Prevent excessive build-up of cloud ice from external sources.
- Convert moist mixing ratios to dry mixing ratios.
- Calculate cloud condensation nuclei (ccn), following klein eq. 15
- Calculate horizontal subgrid variability, which is used in cloud fraction, relative humidity calculation, evaporation and condensation processes. Horizontal sub-grid variability is a function of cell area and land/sea mask: 
- Over land: tland = dwland (Ar/10^10)^0.25
- Over ocean: tocean=dwocean (Ar/10^10)^0.25 where Ar is cell area. dwland=0.16 and dwocean=0.10 are base value for sub-grid variability over land and ocean. The total horizontal sub-grid variability is:
hvar=tland×frland+tocean×(1−frland)
hvar=min[0.2,max(0.01,hvar)]
- Calculate relative humidity increment.
- If requested, call neg_adj() and fix all negative water species.
- Do loop on cloud microphysics sub time step.
- Define air density based on hydrostatical property.
- Call warm_rain() - time-split warm rain processes: 1st pass.
- Sedimentation of cloud ice, snow, and graupel.
- Call fall_speed() to calculate the fall velocity of cloud ice, snow and graupel.
- Call terminal_fall() to calculate the terminal fall speed.
- Call sedi_heat() to calculate heat transportation during sedimentation.
- Call warm_rain() to - time-split warm rain processes: 2nd pass
- Call icloud(): ice-phase microphysics
- Calculate momentum transportation during sedimentation.
- Update moist air mass (actually hydrostatic pressure).
- Update cloud fraction tendency.

## GFDL Cloud Fast Physics General Algorithm (Saturation Adjustment)

- Define conversion scalar / factor.
- Define heat capacity of dry air and water vapor based on hydrostatical property.
- Define air density based on hydrostatical property.
- Define heat capacity and latend heat coefficient.
- Fix energy conservation.
- Fix negative cloud ice with snow.
- Melting of cloud ice to cloud water and rain.
- Update latend heat coefficient.
- Fix negative snow with graupel or graupel with available snow.
- Fix negative cloud water with rain or rain with available cloud water.
- Enforce complete freezing of cloud water to cloud ice below - 48 c.
- Update latend heat coefficient.
- Condensation/evaporation between water vapor and cloud water.
- Update latend heat coefficient.
- condensation/evaporation between water vapor and cloud water, last time step enforce upper (no super_sat) & lower (critical rh) bounds.
- Update latend heat coefficient.
- Homogeneous freezing of cloud water to cloud ice.
- Update latend heat coefficient.
- bigg mechanism (heterogeneous freezing of cloud water to cloud ice).
- Update latend heat coefficient.
- Freezing of rain to graupel.
- Update latend heat coefficient.
- Melting of snow to rain or cloud water.
- Autoconversion from cloud water to rain.
- Update latend heat coefficient.
- Sublimation/deposition between water vapor and cloud ice.
- Virtual temperature updated.
- Fix negative graupel with available cloud ice.
- Autoconversion from cloud ice to snow.
- Fix energy conservation.
- Update latend heat coefficient.
- Compute cloud fraction.
- If it is the last step, combine water species.
- Use the "liquid - frozen water temperature" (tin) to compute saturated specific humidity.
- higher than 10 m is considered "land" and will have higher subgrid variability
- "scale - aware" subgrid variability: 100 - km as the base
- calculate partial cloudiness by pdf; assuming subgrid linear distribution in horizontal; this is effectively a smoother for the binary cloud scheme; qa = 0.5 if qstar (i) == qpz
