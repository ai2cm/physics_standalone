## Description

GFS Scale-Aware Mass-Flux Shallow Convection Scheme Module is an updated version of the previous mass-flux shallow convection scheme with scale and aerosol awareness and parameterizes the effect of shallow convection on the environment. GFS Scale-Aware Mass-Flux Shallow Convection Scheme Module is similar to GFS Scale-Aware Mass-Flux Deep Convection Scheme Module but with a few key differences. First, no quasi-equilibrium assumption is used for any grid size and the shallow cloud base mass flux is parameterized using a mean updraft velocity. Further, there are no convective downdrafts, the entrainment rate is greater than for deep convection, and the shallow convection is limited to not extend over the level where p=0.7psfc. The paramerization of scale and aerosol awareness follows that of the SAMF deep convection scheme, although it can be interpreted as only having the "static" and "feedback" control portions, since the "dynamic" control is not necessary to find the cloud base mass flux.

The previous version of the shallow convection scheme (shalcnv.f) is described in Han and Pan (2011) [68] and differences between the shallow and deep convection schemes are presented in Han and Pan (2011) [68] and Han et al. (2017) [70] . Details of scale- and aerosol-aware parameterizations are described in Han et al. (2017) [70] .

In further update for FY19 GFS implementation, interaction with turbulent kinetic energy (TKE), which is a prognostic variable used in a scale-aware TKE-based moist EDMF vertical turbulent mixing scheme, is included. Entrainment rates in updrafts are proportional to sub-cloud mean TKE. TKE is transported by cumulus convection. TKE contribution from cumulus convection is deduced from cumulus mass flux. On the other hand, tracers such as ozone and aerosol are also transported by cumulus convection.

To reduce too much convective cooling at the cloud top, the convection schemes have been modified for the rain conversion rate, entrainment and detrainment rates, overshooting layers, and maximum allowable cloudbase mass flux (as of June 2018).

[68] Jongil Han and Hua-Lu Pan. Revision of convection and vertical diffusion schemes in the ncep global forecast system. Weather and Forecasting, 26(4):520–533, 2016/03/25 2011.

[70] J. Han, W. Wang, Y. C. Kwon, S.-Y. Hong, V. Tallapragada, and F. Yang. Updates in the ncep gfs cumulus convective schemes with scale and aerosol awareness. Weather and Forecasting, 32:2005–2017, 2017.

## General Algorithm

This routine follows the GFS Scale-Aware Mass-Flux Deep Convection Scheme Module quite closely, although it can be interpreted as only having the "static" and "feedback" control portions, since the "dynamic" control is not necessary to find the cloud base mass flux. The algorithm is simplified from SAMF deep convection by excluding convective downdrafts and being confined to operate below p=0.7psfc. Also, entrainment is both simpler and stronger in magnitude compared to the deep scheme.

GFS samfshalcnv General Algorithm

Compute preliminary quantities needed for the static and feedback control portions of the algorithm.
Perform calculations related to the updraft of the entraining/detraining cloud model ("static control").
The cloud base mass flux is obtained using the cumulus updraft velocity averaged ove the whole cloud depth.
Calculate the tendencies of the state variables (per unit cloud base mass flux) and the cloud base mass flux.
For the "feedback control", calculate updated values of the state variables by multiplying the cloud base mass flux and the tendencies calculated per unit cloud base mass flux from the static control.
