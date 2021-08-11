# GFS NOAH Land Surface Model

## Description

Land-atmosphere interactions are a main driver of Earth's surface water and energy budgets. The importance of the land surface is rather intuitive, and has been demonstrated not only in terms of predictability on daily to seasonal timescale (Betts et al. (2017) [25]), but also in terms of influencing extremes such as drought and heatwaves (Paimazumder and Done (2016) [138]), PBL evolution and cloud formation (Milovac et al. (2016) [123]) and afternoon convection (Guillod et al. (2015) [72]), and tropical cyclone re-intensification (Andersen and Shepherd (2014) [6]). Other linkages, such as the role of soil moisture (SM) or vegetation heterogeneity in mesoscale circulation (Hsu et al. (2017) [88]) and planetary waves (Koster et al. (2014) [100]), and those driven by land use and land cover change or management (Hirsch et al. (2015) [82]; Findell et al. (2017) [52]) are topics of active research.

Figure 1 is a schematic of local land-atmosphere interactions in a quiescent synoptic regime, including the soil moisture-precipitation (SM-P) feedback pathways (Ek and Mahrt (1994) [46]; Ek and Holtslag (2004) [45] ). Solid arrows indicate a positive feedback pathway, and large dashed arrows represent a negative feedback, while red indicates radiative, black indicates surface layer and PBL, and brown indicates land surface processes. Thin red and grey dashed lines with arrows also represent positive feedbacks. The single horizontal gay-dotted line (no arrows) indicates the top of the PBL, and the seven small vertical dashed lines (no arrows) represent precipitation

Recently, the land surface updates in 2017 GFS operational physics includes:
- IGBP 20-type 1-km land classification
- STASGO 19-type 1-km soil classification
- MODIS-based snow free albedo
- MODIS-based maximum snow albedo
- Diurnal albedo treatment
- Unify snow cover, albedo between radiation and land surface model
- Increase ground heat flux under deep snow
- Upgrade surface layer parameterization scheme GFS Surface Layer Scheme to modify the roughness-length formulation and introduce a stability parameter constraint in the Monin-Obukhov similarity theory to prevent the land-atmosphere system from fully decoupling leading to excessive cooling of 2m temperature during sunset

## Subroutine sfc_flx

The land-surface model component was substantially upgraded from the Oregon State University (OSU) land surface model to EMC's new Noah Land Surface Model (Noah LSM) during the major implementation in the NCEP Global Forecast System (GFS) on May 31, 2005. Forecast System (GFS). The Noah LSM embodies about 10 years of upgrades (see [32], [99], [47]) to its ancestor, the OSU LSM. The Noah LSM upgrade includes:

- An increase from two (10, 190 cm thick) to four soil layers (10, 30, 60, 100 cm thick)
- Addition of frozen soil physics
- Add glacial ice treatment
- Two snowpack states (SWE, density)
- New formulations for infiltration and runoff account for sub-grid variability in precipitation and soil moisture
- Revised physics of the snowpack and its influence on surface heat fluxes and albedo
- Higher canopy resistance
- Spatially varying root depth
- Surface fluxes weighted by snow cover fraction
- Improved thermal conduction in soil/snow
- Improved seasonality of green vegetation cover.
- Improved evaporation treatment over bare soil and snowpack

## Algorithm

GFS Noah LSM General Algorithm

Set ice = -1 and green vegetation fraction (shdfac) = 0 for glacial-ice land.
Calculate soil layer depth below ground.
For ice, set green vegetation fraction (shdfac) = 0. and set sea-ice layers of equal thickness and sum to 3 meters
Otherwise, calculate depth (negative) below ground from top skin sfc to bottom of each soil layer.
Call redprm() to set the land-surface paramters, including soil-type and veg-type dependent parameters.
Calculate perturbated soil type "b" parameter. Following Gehne et al. (2019) [61] , a perturbation of LAI "leaf area index" (xlaip) and a perturbation of the empirical exponent parameter b in the soil hydraulic conductivity calculation (bexpp) are added to account for the uncertainties of LAI and b associated with different vegetation types and soil types using a linear scaling. The spatial pattern of xlaip is drawn from a normal distribution with a standard deviation of 0.25 while makes the LAI between 0 and 8. The spatial pattern of bexpp is drawn from a normal distribution with a standard deviation of 0.4, and is bounded between -1 and 1.
Calculate perturbated leaf area index.
Initialize precipitation logicals.
Over sea-ice or glacial-ice, if water-equivalent snow depth (sneqv) below threshold lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then set at lower bound and store the source increment in subsurface runoff/baseflow (runoff2).
For sea-ice and glacial-ice cases, set smc and sh2o values = 1.0 as a flag for non-soil medium.
If input snowpack (sneqv) is nonzero, then call csnow() to compute snow density (sndens) and snow thermal conductivity (sncond).
Determine if it's precipitating and what kind of precipitation it is. if it's precipitating and the air temperature is colder than 0oC, it's snowing! if it's precipitating and the air temperature is warmer than 0oC, but the ground temperature is colder than 0oC, freezing rain is presumed to be falling.
If either precipitation flag (snowng, frzgra) is set as true:
Since all precip is added to snowpack, no precip infiltrates into the soil so that prcp1 is set to zero.
Call snow_new() to update snow density based on new snowfall, using old and new snow.
Call csnow() to update snow thermal conductivity.
If precipitation is liquid (rain), hence save in the precip variable that later can wholely or partially infiltrate the soil (along with any canopy "drip" added to this later).
Determine snowcover fraction and albedo fraction over sea-ice, glacial-ice, and land.
Call snfrac() to calculate snow fraction cover.
Call alcalc() to calculate surface albedo modification due to snowdepth state.
Calculate thermal diffusivity (df1):
For sea-ice case and glacial-ice case, this is constant( df1=2.2).
For non-glacial land case, call tdfcnd() to calculate the thermal diffusivity of top soil layer ([142]).
Add subsurface heat flux reduction effect from the overlying green canopy, adapted from section 2.1.2 of [141].
Calculate subsurface heat flux, ssoil, from final thermal diffusivity of surface mediums,df1 above, and skin temperature and top mid-layer soil temperature.
For uncoupled mode, call snowz0() to calculate surface roughness (z0) over snowpack using snow condition from the previous timestep.
Calculate virtual temps and virtual potential temps needed by subroutines sfcdif and penman.
Calculate the total downward radiation (fdown) = net solar (swnet) + downward longwave (lwdn) as input of penman() and other surface energy budget calculations.
Call penman() to calculate potential evaporation (etp), and other partial products and sums for later calculations.
Call canres() to calculate the canopy resistance and convert it into pc if nonzero greenness fraction.
Now decide major pathway branch to take depending on whether snowpack exists or not:
For no snowpack is present, call nopac() to calculate soil moisture and heat flux values and update soil moisture contant and soil heat content values.
For a snowpack is present, call snopac().
Noah LSM post-processing:
Calculate sensible heat (h) for return to parent model.
Convert units and/or sign of total evap (eta), potential evap (etp), subsurface heat flux (s), and runoffs for what parent model expects.
Convert the sign of soil heat flux so that:
ssoil>0: warm the surface (night time)
ssoil<0: cool the surface (day time)
For the case of land (but not glacial-ice): convert runoff3 (internal layer runoff from supersat) from m to ms−1 and add to subsurface runoff/baseflow (runoff2). runoff2 is already a rate at this point.
For the case of sea-ice (ice=1) or glacial-ice (ice=-1), add any snowmelt directly to surface runoff (runoff1) since there is no soil medium, and thus no call to subroutine smflx (for soil moisture tendency).
Calculate total column soil moisture in meters (soilm) and root-zone soil moisture availability (fraction) relative to porosity/saturation.
