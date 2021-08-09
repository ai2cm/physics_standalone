# GFS Scale-aware TKE-based Moist Eddy-Diffusion Mass-Flux (EDMF) PBL and Free Atmospheric Turbulence Scheme

## Description

The current operational GFS Hybrid Eddy-Diffusivity Mass-Flux PBL and Free Atmospheric Turbulence Scheme uses a hybrid EDMF parameterization for the convective PBL (Han et al. 2016 [69]; Han et al. 2017 [70]), where the EDMF scheme is applied only for the strongly unstable PBL, while the eddy-diffusivity counter-gradient(EDCG) scheme is used for the weakly unstable PBL. The new TKE-EDMF is an extended version of GFS Hybrid Eddy-Diffusivity Mass-Flux PBL and Free Atmospheric Turbulence Scheme with below enhancement:

Eddy diffusivity (K) is now a function of TKE which is prognostically predicted
EDMF approach is applied for all the unstable PBL
EDMF approach is also applied to the stratocumulus-top-driven turbulence mixing
It includes a moist-adiabatic process when updraft thermal becomes saturated
Scale-aware capability
It includes interaction between TKE and cumulus convection
The CCPP-compliant subroutine satmedmfvdif_run() computes subgrid vertical turbulence mixing using scale-aware TKE-based moist eddy-diffusion mass-flux paramterization (Han et al. 2019 [66])

For the convective boundary layer, the scheme adopts EDMF parameterization (Siebesma et al. (2007)[144]) to take into account nonlocal transport by large eddies(mfpblt.f)
A new mass-flux paramterization for stratocumulus-top-induced turbulence mixing has been introduced (mfscu.f; previously, it was an eddy diffusion form)
For local turbulence mixing, a TKE closure model is used.


[66] J. Han and C.S. Bretherton. Tke-based moist eddy-diffusivity mass-flux (edmf) parameterization for vertical turbulent mixing. Weather and Forecasting, accepted, 2019.

[69] Jongil Han, Marcin L. Witek, Joao Teixeira, Ruiyu Sun, Hua-Lu Pan, Jennifer K. Fletcher, and Christopher S. Bretherton. Implementation in the ncep gfs of a hybrid eddy-diffusivity mass-flux (edmf) boundary layer parameterization with dissipative heating and modified stable boundary layer mixing. Weather and Forecasting, 31(1):341–352, Feb 2016.

[70] J. Han, W. Wang, Y. C. Kwon, S.-Y. Hong, V. Tallapragada, and F. Yang. Updates in the ncep gfs cumulus convective schemes with scale and aerosol awareness. Weather and Forecasting, 32:2005–2017, 2017.

[144] A. Pier Siebesma, Pedro M. M. Soares, and João Teixeira. A combined eddy-diffusivity mass-flux approach for the convective boundary layer. Journal of the Atmospheric Sciences, 64(4):1230–1248, Apr 2007.

## GFS satmedmfvdif General Algorithm

satmedmfvdif_run() computes subgrid vertical turbulence mixing using the scale-aware TKE-based moist eddy-diffusion mass-flux (EDMF) parameterization of Han and Bretherton (2019) [66] .

- The local turbulent mixing is represented by an eddy-diffusivity scheme which is a function of a prognostic TKE.
- For the convective boundary layer, nonlocal transport by large eddies (mfpblt.f), is represented using a mass flux approach (Siebesma et al.(2007) [144] ).
- A mass-flux approach is also used to represent the stratocumulus-top-induced turbulence (mfscu.f).
