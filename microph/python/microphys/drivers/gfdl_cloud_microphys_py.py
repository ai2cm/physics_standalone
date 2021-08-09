from microphys.phys_const import *

from utility import *

import numpy as np
import math as mt
import time as tm


# Global variables for microphysics
c_air  = None
c_vap  = None
d0_vap = None   # The same as dc_vap, except that cp_vap can be cp_vap or cv_vap
lv00   = None   # The same as lv0, except that cp_vap can be cp_vap or cv_vap
fac_rc = None
cracs  = None
csacr  = None
cgacr  = None
cgacs  = None
acco   = None
csacw  = None
csaci  = None
cgacw  = None
cgaci  = None
cracw  = None
cssub  = None
crevp  = None
cgfr   = None
csmlt  = None
cgmlt  = None
ces0   = None
log_10 = None
tice0  = None
t_wfr  = None

do_sedi_w  = True   # Transport of vertical motion in sedimentation
do_setup   = True   # Setup constants and parameters
p_nonhydro = False  # Perform hydrosatic adjustment on air density
use_ccn    = True   # Must be true when prog_ccn is false


# Execute the full GFDL cloud microphysics
def gfdl_cloud_microphys_driver( input_data, hydrostatic, phys_hydrostatic, 
                                 kks, ktop, timings, n_iter, not_first_rep ):
    
    '''
    NOTE: -1 for the indices
    '''
    # Scalar input values
    iie     = input_data["iie"] - 1     # End of physics window along i-dimension
    kke     = input_data["kke"] - 1     # End of vertical dimension
    kbot    = input_data["kbot"] - 1    # Bottom of vertical compute domain
    seconds = input_data["seconds"]
    dt_in   = input_data["dt_in"]       # Physics time step
    lradar  = input_data["lradar"]
    reset   = input_data["reset"]
    
    # 2D input arrays
    area    = input_data["area"]        # Cell area
    land    = input_data["land"]        # Land fraction
    rain    = input_data["rain"]
    snow    = input_data["snow"]
    ice     = input_data["ice"]
    graupel = input_data["graupel"]
    
    # 3D input arrays
    dz        = input_data["dz"]
    delp      = input_data["delp"]
    uin       = input_data["uin"]
    vin       = input_data["vin"]
    qv        = input_data["qv"]
    ql        = input_data["ql"]
    qr        = input_data["qr"]
    qi        = input_data["qi"]
    qs        = input_data["qs"]
    qg        = input_data["qg"]
    qa        = input_data["qa"]
    qn        = input_data["qn"]
    p         = input_data["p"]
    pt        = input_data["pt"]
    qv_dt     = input_data["qv_dt"]
    ql_dt     = input_data["ql_dt"]
    qr_dt     = input_data["qr_dt"]
    qi_dt     = input_data["qi_dt"]
    qs_dt     = input_data["qs_dt"]
    qg_dt     = input_data["qg_dt"]
    qa_dt     = input_data["qa_dt"]
    pt_dt     = input_data["pt_dt"]
    udt       = input_data["udt"]
    vdt       = input_data["vdt"]
    w         = input_data["w"]
    refl_10cm = input_data["refl_10cm"]
    
    # Global variables
    global c_air
    global c_vap
    global d0_vap
    global lv00
    
    global do_sedi_w
    global p_nonhydro
    global use_ccn
    
    '''
    NOTE: -1 w.r.t. Fortran
    NOTE: Is this even needed?
    NOTE: What is the purpose of having iis? Why "is" not equal to "iis"?
    '''
    # Define start and end indices of the three dimensions
    k_s = 0
    k_e = kke - kks + 1
    
    ktop1 = ktop + 1
    kbot1 = kbot + 1
    
    # Define heat capacity of dry air and water vapor based on 
    # hydrostatical property
    if phys_hydrostatic or hydrostatic:
        c_air = cp_air
        c_vap = cp_vap
        p_nonhydro = False
    else:
        c_air = cv_air
        c_vap = cv_vap
        p_nonhydro = True
        
    d0_vap = c_vap - c_liq
    lv00 = hlv0 - d0_vap * t_ice
    
    if hydrostatic:
        do_sedi_w = False
    
    '''
    NOTE: Moved in phys_const.py to avoid having too many globals
    '''
    # ~ # Define latent heat coefficients used in wet bulb and bigg mechanism
    # ~ latv = hlv
    # ~ lati = hlf
    # ~ lats = latv + lati
    # ~ lat2 = lats * lats
    
    # ~ lcp = latv / cp_air
    # ~ icp = lati / cp_air
    # ~ tcp = (latv + lati) / cp_air
    
    # Define cloud microphysics sub time step
    mpdt   = np.minimum(dt_in, mp_time)
    rdt    = 1. / dt_in
    ntimes = int(round(dt_in / mpdt))
    
    # Small time step
    dts = dt_in / ntimes
    
    '''
    NOTE: Remember that in a gt4py stencil the fields will all have the 
          same dimension (i, j, k), with i=2034, j=1, k=79, thus in 
          cases where 2D arrays (2304, 1) are used every k-level of the 
          corresponding field must be equal to achieve coherence for 
          later accesses (look for notes labeled "2D coherence")
    NOTE: In this case the fields could just be initialized with zeros
    '''
    # Initialize precipitation
    graupel[...] = 0.
    rain[...]    = 0.
    snow[...]    = 0.
    ice[...]     = 0.
    cond         = np.zeros_like(ice)
    
    '''
    NOTE: j-loop pulled inside the function
    NOTE: This will be the main stencil (try keeping everything in one stencil)
    NOTE: Inline mpdrv
    '''
    ### Major cloud microphysics ###
    
    dt_rain = dts * 0.5
    
    qiz = qi[:, :, ktop:kbot1].copy()
    qsz = qs[:, :, ktop:kbot1].copy()
        
    # This is to prevent excessive build-up of cloud ice from 
    # external sources
    if de_ice:
        qio = qiz - dt_in * qi_dt[:, :, ktop:kbot1] # Orginal qi before phys
        qin = np.maximum(qio, qi0_max)              # Adjusted value
        
        '''
        NOTE: ijk replaces the if statement and acts as a "mask" on the arrays
        NOTE: This approach might be too tedious and not too similar to a gt4py implementation
        '''
        ijk = qiz > qin
        
        qsz[ijk] = qsz[ijk] + qiz[ijk] - qin[ijk]
        qiz[ijk] = qin[ijk]
        
        '''
        NOTE: dqi is could be a temporary variable in gt4py, thus should be declared outside of the if
        '''
        dqi        = (qin[ijk] - qio[ijk]) * rdt    # Modified qi tendency
        qs_dt[ijk] = qs_dt[ijk] + qi_dt[ijk] - dqi
        qi_dt[ijk] = dqi
        
        qi[ijk] = qiz[ijk]
        qs[ijk] = qsz[ijk]
    
    '''
    NOTE: t0 are read-only and pt, thus there's no need to copy pt
    '''
    t0  = pt[:, :, ktop:kbot1]
    tz  = t0.copy()
    dp1 = delp[:, :, ktop:kbot1].copy()
    dp0 = dp1.copy()    # Moist air mass * grav
    
    # Convert moist mixing ratios to dry mixing ratios
    qvz = qv[:, :, ktop:kbot1].copy()
    qlz = ql[:, :, ktop:kbot1].copy()
    qrz = qr[:, :, ktop:kbot1].copy()
    qgz = qg[:, :, ktop:kbot1].copy()
    
    dp1 = dp1 * (1. - qvz)
    omq = dp0 / dp1
    
    qvz = qvz * omq
    qlz = qlz * omq
    qrz = qrz * omq
    qiz = qiz * omq
    qsz = qsz * omq
    qgz = qgz * omq
    
    '''
    NOTE: qa0 and dz0 are read-only and qa and dz are never changed, thus there's no need to copy qa and dz
    '''
    qa0 = qa[:, :, ktop:kbot1]
    qaz = np.zeros_like(qa0)
    dz0 = dz[:, :, ktop:kbot1]
    
    den0 = -dp1 / (grav * dz0)  # Density of dry air
    p1   = den0 * rdgas * t0    # Dry air pressure
    
    # Save a copy of old values for computing tendencies
    qv0 = qvz.copy()
    ql0 = qlz.copy()
    qr0 = qrz.copy()
    qi0 = qiz.copy()
    qs0 = qsz.copy()
    qg0 = qgz.copy()
    
    '''
    NOTE: u0 and v0 are read-only and uin and vin are never changed, thus there's no need to copy uin and vin
    '''
    # For sedi_momentum
    u0 = uin[:, :, ktop:kbot1]
    v0 = vin[:, :, ktop:kbot1]
    u1 = u0.copy()
    v1 = v0.copy()
    m1 = np.zeros_like(u1)
    
    '''
    NOTE: This can be removed since w1 is copied back in w at the end of 
          the function, but w1 in Python is 3D anyway
    '''
    # ~ if do_sedi_w:
        # ~ w1 = w[:, :, ktop:kbot1].copy()
    
    # Calculate cloud condensation nuclei (ccn) based on klein eq. 15
    cpaut = c_paut * 0.104 * grav / 1.717e-5
    
    if prog_ccn:
        
        # Convert #/cc to #/m^3
        ccn     = qn[:, :, ktop:kbot1] * 1.e6
        c_praut = cpaut * (ccn * rhor)**(-1./3.)
        
        use_ccn = False
        
    else:
        
        '''
        NOTE: Remember that here land is 2D and not 1D, since it's not passed to mpdrv as slice
        NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
        NOTE: Try to do the computation in gt4py only for the first 
              k-level and then propagate, but keep in mind that this 
              implies an non-parallel region, so it might not be faster 
              anyway (one computation and multiple copies, but not in 
              parallel, compared to multiple parallel computations). 
              This is valid also in other cases where computation on 2D 
              arrays is done and should be equal on all k-levels.
        '''
        ccn0 = (ccn_l * land + ccn_o * (1. - land)) * 1.e6
        
        if use_ccn:
            
            '''
            NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
            '''
            # ccn is formulted as ccn = ccn_surface * (den / den_surface)
            ccn0 = ccn0 * rdgas * tz[:, :, kbot] / p1[:, :, kbot]
        
        '''
        NOTE: No need to copy ccn0, since it's only used inside this else statement
        NOTE: ccn and c_praut are 3D arrays with all k-levels equal, 
              thus in gt4py there's no need to duplicate ccn0 over k, 
              since ccn0 is already 3D and coherent due to gt4py 
              dimension requirements
        '''
        tmp     = cpaut * (ccn0 * rhor)**(-1./3.)
        ccn     = np.repeat(ccn0[:, :, np.newaxis], kbot-ktop+1, axis=2)
        c_praut = np.repeat(tmp[:, :, np.newaxis], kbot-ktop+1, axis=2)
        
    '''
    NOTE: Double sqrt is faster than ** and * is faster than / generally (?)
    NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
    '''
    # Calculate horizontal subgrid variability
    # Total water subgrid deviation in horizontal direction
    # Default area dependent form: use dx ~ 100 km as the base
    s_leng  = np.sqrt(np.sqrt(area * 1.e-10))
    t_land  = dw_land * s_leng
    t_ocean = dw_ocean * s_leng
    h_var   = t_land * land + t_ocean * (1. - land)
    h_var   = np.minimum(0.20, np.maximum(0.01, h_var))
    
    '''
    NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
    '''
    # Relative humidity increment
    rh_adj  = 1. - h_var - rh_inc
    rh_rain = np.maximum(0.35, rh_adj - rh_inr)
    
    # Fix all negative water species
    if fix_negative:
        neg_adj(ktop, kbot, tz, dp1, qvz, qlz, qrz, qiz, qsz, qgz)
    
    '''
    NOTE: Here m2_rain and m2_sol have dimension (i, j, k) and not (i, k) to be coherent with the other arrays
    '''
    m2_rain = np.zeros_like(pt)
    m2_sol  = np.zeros_like(pt)
    
    '''
    NOTE: Is it possible to have a for-loop inside a gt4py stencil?
    NOTE: vtrz is defined as empty outside the loop since in Fortran 
          this is done at the start of the function and not at each loop 
          iteration. Same for the other ones.
    '''
    r1      = np.empty_like(rain)
    g1      = np.empty_like(graupel)
    s1      = np.empty_like(snow)
    i1      = np.empty_like(ice)
    
    m1_rain = np.empty_like(pt)
    m1_sol  = np.empty_like(pt)
    
    vtrz    = np.empty_like(den0)
    vtsz    = np.empty_like(den0)
    vtiz    = np.empty_like(den0)
    vtgz    = np.empty_like(den0)
    
    for n in range(ntimes):
        
        # Start 1st warm rain timer
        if BENCHMARK and not_first_rep: t_warm_rain_1_start = tm.perf_counter()
        
        '''
        NOTE: No need to copy dz0 and den0 since both are never modified
        '''
        # Define air density based on hydrostatical property
        if p_nonhydro:
            
            dz1    = dz0
            den    = den0   # Dry air density remains the same
            denfac = np.sqrt(sfcrho / den)
            
        else:
            
            dz1    = dz0 * tz / t0  # Hydrostatic balance
            den    = den0 * dz0 / dz1
            denfac = np.sqrt(sfcrho / den)
        
        '''
        NOTE: r1 in Fortran is a scalar, but here it'll be a multi-dim array
        NOTE: w passed directly instead of passing w1
        '''
        # Time-split warm rain processes: 1st pass
        warm_rain( dt_rain, ktop, kbot, dp1, dz1, tz, qvz, qlz, qrz, 
                   qiz, qsz, qgz, den, denfac, ccn, c_praut, rh_rain, 
                   vtrz, r1, m1_rain, w, h_var )
        
        '''
        NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
        '''
        rain = rain + r1
        
        '''
        NOTE: Coherence along j-dimension of m2_rain (?)
        NOTE: m1_rain, m2_rain, m1_sol and m2_sol have to be explicitly 
              accessed from ktop to kbot since they were previously 
              allocated with the dimension of pt, that don't necessarily 
              correspond to ktop:kbot1 depending on the value of ktop 
              and kbot
        '''
        m2_rain[:, :, ktop:kbot1] = m2_rain[:, :, ktop:kbot1] + m1_rain[:, :, ktop:kbot1]
        
        m1 = m1 + m1_rain[:, :, ktop:kbot1]
        
        if BENCHMARK and not_first_rep:
            
            # Stop 1st warm rain timer
            t_warm_rain_1_end = tm.perf_counter()
            
            timings["warm_rain_1_run"][n_iter] += t_warm_rain_1_end - t_warm_rain_1_start
            
            # Start sedimentation timer
            t_sedimentation_start = tm.perf_counter()
        
        # Sedimentation of cloud ice, snow, and graupel
        fall_speed(ktop, kbot, den, qsz, qiz, qgz, qlz, tz, vtsz, vtiz, vtgz)
        
        '''
        NOTE: w passed directly instead of passing w1
        '''
        terminal_fall( dts, ktop, kbot, tz, qvz, qlz, qrz, qgz, qsz, 
                       qiz, dz1, dp1, den, vtgz, vtsz, vtiz, r1, g1, 
                       s1, i1, m1_sol, w )
        
        '''
        NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
        '''
        rain    = rain + r1     # From melted snow and ice that reached the ground
        snow    = snow + s1
        graupel = graupel + g1
        ice     = ice + i1
        
        # Heat transportation during sedimentation
        if do_sedi_heat:
            sedi_heat( ktop, kbot, dp1, m1_sol, dz1, tz, qvz, qlz, qrz, 
                       qiz, qsz, qgz, c_ice )
        
        if BENCHMARK and not_first_rep:
            
            # Stop sedimentation timer
            t_sedimentation_end = tm.perf_counter()
            
            timings["sedimentation_run"][n_iter] += t_sedimentation_end - t_sedimentation_start
            
            # Start 2nd warm rain timer
            t_warm_rain_2_start = tm.perf_counter()
        
        '''
        NOTE: r1 in Fortran is a scalar, but here it'll be a multi-dim array
        NOTE: w passed directly instead of passing w1
        '''               
        # Time-split warm rain processes: 2nd pass
        warm_rain( dt_rain, ktop, kbot, dp1, dz1, tz, qvz, qlz, qrz, 
                   qiz, qsz, qgz, den, denfac, ccn, c_praut, rh_rain, 
                   vtrz, r1, m1_rain, w, h_var )
        
        '''
        NOTE: 2D coherence (here it's enough to have the 2D array as 3D field from the beginning (?))
        '''
        rain = rain + r1
        
        m2_rain[:, :, ktop:kbot1] = m2_rain[:, :, ktop:kbot1] + m1_rain[:, :, ktop:kbot1]
        m2_sol[:, :, ktop:kbot1]  = m2_sol[:, :, ktop:kbot1] + m1_sol[:, :, ktop:kbot1]
        
        m1 = m1 + m1_rain[:, :, ktop:kbot1] + m1_sol[:, :, ktop:kbot1]
        
        if BENCHMARK and not_first_rep:
            
            # Stop 2nd warm rain timer
            t_warm_rain_2_end = tm.perf_counter()
            
            timings["warm_rain_2_run"][n_iter] += t_warm_rain_2_end - t_warm_rain_2_start
            
            # Start ice-phase microphysics timer
            t_icloud_start = tm.perf_counter()
        
        # Ice-phase microphysics
        icloud( ktop, kbot, tz, p1, qvz, qlz, qrz, qiz, qsz, qgz, dp1, 
                den, denfac, vtsz, vtgz, vtrz, qaz, rh_adj, rh_rain, 
                dts, h_var )
        
        if BENCHMARK and not_first_rep:
            
            # Stop ice-phase microphysics timer
            t_icloud_end = tm.perf_counter()
            
            timings["icloud_run"][n_iter] += t_icloud_end - t_icloud_start
                
    # Convert units from Pa*kg/kg to kg/m^2/s
    m2_rain[:, :, ktop:kbot1] = m2_rain[:, :, ktop:kbot1] * rdt * rgrav
    m2_sol[:, :, ktop:kbot1]  = m2_sol[:, :, ktop:kbot1] * rdt * rgrav
    
    # Momentum transportation during sedimentation (dp1 is dry mass; dp0 
    # is the old moist total mass)
    if sedi_transport:
        
        '''
        NOTE: In gt4py this needs to be done in a forward computation block
        '''
        for k in range(ktop1, kbot1):
           
            u1[:, :, k] = ( dp0[:, :, k  ] * u1[:, :, k  ] +   \
                            m1 [:, :, k-1] * u1[:, :, k-1] ) / \
                          ( dp0[:, :, k  ] + m1[:, :, k-1] )
            
            v1[:, :, k] = ( dp0[:, :, k  ] * v1[:, :, k  ] +   \
                            m1 [:, :, k-1] * v1[:, :, k-1] ) / \
                          ( dp0[:, :, k  ] + m1[:, :, k-1] )
        
        '''
        NOTE: But this can be done in parallel
        '''                  
        udt[:, :, ktop1:kbot1] = udt[:, :, ktop1:kbot1] + \
                                 ( u1[:, :, ktop1:kbot1] - u0[:, :, ktop1:kbot1] ) * rdt
                                  
        vdt[:, :, ktop1:kbot1] = vdt[:, :, ktop1:kbot1] + \
                                 ( v1[:, :, ktop1:kbot1] - v0[:, :, ktop1:kbot1] ) * rdt
    
    '''
    NOTE: This can be removed since w1 is copied back in w at the end of 
          the function, but w1 in Python is 3D anyway
    '''                              
    # ~ if do_sedi_w:
        # ~ w[:, :, ktop:kbot1] = w1
    
    # Update moist air mass (actually hydrostatic pressure) and convert 
    # to dry mixing ratios
    omq = dp1 / dp0
    qv_dt[:, :, ktop:kbot1] = qv_dt[:, :, ktop:kbot1] + rdt * (qvz - qv0) * omq
    ql_dt[:, :, ktop:kbot1] = ql_dt[:, :, ktop:kbot1] + rdt * (qlz - ql0) * omq
    qr_dt[:, :, ktop:kbot1] = qr_dt[:, :, ktop:kbot1] + rdt * (qrz - qr0) * omq
    qi_dt[:, :, ktop:kbot1] = qi_dt[:, :, ktop:kbot1] + rdt * (qiz - qi0) * omq
    qs_dt[:, :, ktop:kbot1] = qs_dt[:, :, ktop:kbot1] + rdt * (qsz - qs0) * omq
    qg_dt[:, :, ktop:kbot1] = qg_dt[:, :, ktop:kbot1] + rdt * (qgz - qg0) * omq
    
    cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
    
    pt_dt[:, :, ktop:kbot1] = pt_dt[:, :, ktop:kbot1] + rdt * (tz - t0) * cvm / cp_air
    
    # Update cloud fraction tendency
    if do_qa:
        qa_dt[:, :, ktop:kbot1] = 0.
    else:
        qa_dt[:, :, ktop:kbot1] = qa_dt[:, :, ktop:kbot1] + rdt * (qaz / ntimes - qa0)
    
    '''
    NOTE: do_qa removed since behavior is the same in if and else
    NOTE: In Fortran if ks < ktop the k-loop includes ktop too, is this intended?
    '''
    # No clouds allowed above ktop
    if k_s < ktop:
        qa_dt[:, :, k_s:ktop+1] = 0.

    # Convert to mm / day
    convt = 86400. * rdt * rgrav
    
    rain    = rain * convt
    snow    = snow * convt
    ice     = ice * convt
    graupel = graupel * convt
    
    prec_mp = rain + snow + ice + graupel
    
    '''
    NOTE: Radar part missing (never execute since lradar is false)
    '''
    
    return qi, qs, \
           qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, qa_dt, \
           pt_dt, w, udt, vdt, \
           rain, snow, ice, graupel, \
           refl_10cm


# Sedimentation of heat
def sedi_heat( ktop, kbot, dm, m1, dz, tz, qv, ql, qr, qi, qs, qg, cw, 
               condition=None ):
    
    # Input q fields are dry mixing ratios, and dm is dry air mass
    
    # Check if there is a condition, otherwise fill condition with True
    c = check_condition(condition, qv.shape[:2])
    
    dgz = np.empty_like(dz)
    cvn = np.empty_like(dm)
    
    dgz[c] = -0.5 * grav * dz[c]
    cvn[c] = dm[c] * (cv_air + qv[c] * cv_vap + (qr[c] + ql[c]) * c_liq + (qi[c] + qs[c] + qg[c]) * c_ice)
    
    # - Assumption: The ke in the falling condensates is negligible 
    #               compared to the potential energy that was 
    #               unaccounted for. Local thermal equilibrium is 
    #               assumed, and the loss in pe is transformed into 
    #               internal energy (to heat the whole grid box).
    # - Backward time-implicit upwind transport scheme:
    # - dm here is dry air mass
    '''
    NOTE: (tmp * tz(k) + m1(k) * dgz(k)) / tmp is equal to tz(k) + m1(k) * dgz(k) / tmp but you use one less multiplication
    '''
    tmp = cvn[:, :, ktop][c] + m1[:, :, ktop][c] * cw
    tz[:, :, ktop][c] = tz[:, :, ktop][c] + m1[:, :, ktop][c] * dgz[:, :, ktop][c] / tmp
    
    '''
    NOTE: This needs to be done in a forward computation block
    '''
    # Implicit algorithm: can't be vectorized
    for k in range(ktop+1, kbot+1):
        tz[:, :, k][c] = ( (cvn[:, :, k][c] + cw * (m1[:, :, k][c] - m1[:, :, k-1][c])) * \
                           tz[:, :, k][c] + m1[:, :, k-1][c] * cw * tz[:, :, k-1][c] + \
                           dgz[:, :, k][c] * (m1[:, :, k-1][c] + m1[:, :, k][c]) ) / \
                         ( cvn[:, :, k][c] + cw * m1[:, :, k][c] )


# Warm rain cloud microphysics
def warm_rain( dt, ktop, kbot, dp, dz, tz, qv, ql, qr, qi, qs, qg, den, 
               denfac, ccn, c_praut, rh_rain, vtr, r1, m1_rain, w1, 
               h_var ):
    
    ktop1 = ktop + 1
    kbot1 = kbot + 1
    
    so3 = 7./3.
    dt5 = 0.5 * dt
    
    # Terminal speed of rain
    m1_rain[...] = 0.
    
    '''
    NOTE: no_fall has been converted to a return value and it's 2D
    '''
    no_fall = check_column(ktop, kbot, qr)
    
    vtr[no_fall] = vf_min
    r1[no_fall]  = 0.
    
    '''
    NOTE: This represents the else condition
    '''
    nf_inv = ~no_fall
    
    # Fall speed of rain
    if const_vr:
        vtr[nf_inv] = vr_fac
    else:
        
        qden         = np.empty_like(qr)
        qden[nf_inv] = qr[nf_inv] * den[nf_inv]
        
        '''
        NOTE: Since nf_inv is 2D it has to be repeated along the k-axis.
              This could also be done masking the levels of ij-levels of qr.
        '''
        nf_inv3d = np.repeat(nf_inv[:, :, np.newaxis], kbot-ktop+1, axis=2)
        
        ijk = nf_inv3d & (qr < thr)
        
        vtr[ijk] = vr_min
        
        ijk_inv = nf_inv3d & (qr >= thr)
        
        '''
        NOTE: Computing the exponential and logarithm can be avoided and 
              this will also avoid python "divide by zero in log" 
              warnings
        NOTE: Actually computing exp and log is slightly faster than 
              using **, thus maybe change this and set python to ignore 
              the warnings
        '''
        # ~ vtr[ijk_inv] = vr_fac * vconr * \
                       # ~ np.sqrt(np.minimum(10., sfcrho / den[ijk_inv])) * \
                       # ~ np.exp(0.2 * np.log(qden[ijk_inv] / normr))
        vtr[ijk_inv] = vr_fac * vconr * \
                       np.sqrt(np.minimum(10., sfcrho / den[ijk_inv])) * \
                       np.exp(0.2 * np.log(qden[ijk_inv] / normr))
        
        vtr[ijk_inv] = np.minimum(vr_max, np.maximum(vr_min, vtr[ijk_inv]))
        
    '''
    NOTE: This has a different shape than the other fields, this 
          will cause problems in gt4py, thus here try to reduce the 
          k-dimension to ktop:kbot instead of ktop:kbot+1. If not 
          possible try having a whole field for the additional level
    NOTE: This needs to be done in a backward computation block
    NOTE: Remember ze[:, :, kbot+1] = zs
    NOTE: Suggestion on how to approach this (obviously in Python 
          this is not a problem, but it will be in gt4py, so I try 
          to structure the Python code in the same way)
    '''
    zs = 0.
    ze = np.empty_like(dz)
    
    ze_kbot1 = zs
    
    ze[:, :, kbot][nf_inv] = ze_kbot1 - dz[:, :, kbot][nf_inv]
    for k in range(kbot-1, ktop-1, -1):
        ze[:, :, k][nf_inv] = ze[:, :, k+1][nf_inv] - dz[:, :, k][nf_inv]   # dz < 0
    
    # Evaporation and accretion of rain for the first 1/2 time step
    revap_racc( ktop, kbot, dt5, tz, qv, ql, qr, qi, qs, qg, den, 
                denfac, rh_rain, h_var, nf_inv )
    
    if do_sedi_w:
        dm = np.empty_like(dp)
        
        dm[nf_inv] = dp[nf_inv] * \
                     ( 1. + qv[nf_inv] + ql[nf_inv] + qr[nf_inv] + \
                            qi[nf_inv] + qs[nf_inv] + qg[nf_inv] )
        
    # Mass flux induced by falling rain
    if use_ppm:
        
        zt = np.empty_like(dz)
        
        zt[:, :, ktop][nf_inv] = ze[:, :, ktop][nf_inv]
        
        zt[:, :, ktop1:kbot1][nf_inv] = ze[:, :, ktop1:kbot1][nf_inv] - \
                                        dt5 * ( vtr[:, :, ktop:kbot][nf_inv] + \
                                                vtr[:, :, ktop1:kbot1][nf_inv] )
        
        '''
        NOTE: This is done to preserve the same dimensions of fields
        NOTE: zt_kbot1 is 2D and corresponds to zt[:, :, kbot+1]
        '''
        zt_kbot1 = np.empty_like(zt[:, :, ktop])                                        
        
        zt_kbot1[nf_inv] = zs - dt * vtr[:, :, kbot][nf_inv]
        
        '''
        NOTE: This needs to be done in a forward computation block
        NOTE: The loop goes from ktop to kbot-1 and the last 
              iteration is done separately in order to update 
              zt_kbot1
        '''
        for k in range(ktop, kbot):
            
            ij = nf_inv & (zt[:, :, k+1] >= zt[:, :, k])
        
            zt[:, :, k+1][ij] = zt[:, :, k][ij] - dz_min
            
        ij = nf_inv & (zt_kbot1 >= zt[:, :, kbot])
        
        zt_kbot1[ij] = zt[:, :, kbot][ij] - dz_min
        
        '''
        NOTE: Here a 2D boolean array must be past to ensure that 
              only specific ij-levels are updated, since this 
              function call is inside an else-statement
        NOTE: Probably zt_kbot1 needs to be passed too
        '''
        lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qr, r1, 
                             m1_rain, mono_prof, nf_inv )
        
    else:
        
        '''
        NOTE: Here a 2D boolean array must be past to ensure that 
              only specific ij-levels are updated, since this 
              function call is inside an else-statement
        NOTE: Probably zs needs to be passed too
        '''
        implicit_fall( dt, ktop, kbot, ze, ze_kbot1, vtr, dp, qr, r1, m1_rain, 
                       nf_inv )
        
    # Vertical velocity transportation during sedimentation
    if do_sedi_w:
        
        w1[:, :, ktop][nf_inv] = ( dm[:, :, ktop][nf_inv] * w1[:, :, ktop][nf_inv] + \
                                   m1_rain[:, :, ktop][nf_inv] * vtr[:, :, ktop][nf_inv] ) / \
                                 ( dm[:, :, ktop][nf_inv] - m1_rain[:, :, ktop][nf_inv] )
        
        w1[:, :, ktop1:kbot1][nf_inv] = ( dm[:, :, ktop1:kbot1][nf_inv] * w1[:, :, ktop1:kbot1][nf_inv] - \
                                          m1_rain[:, :, ktop:kbot][nf_inv] * vtr[:, :, ktop:kbot][nf_inv] + \
                                          m1_rain[:, :, ktop1:kbot1][nf_inv] * vtr[:, :, ktop1:kbot1][nf_inv] ) / \
                                        ( dm[:, :, ktop1:kbot1][nf_inv] + \
                                          m1_rain[:, :, ktop:kbot][nf_inv] - \
                                          m1_rain[:, :, ktop1:kbot1][nf_inv] )
                                          
    # Heat transportation during sedimentation
    if do_sedi_heat:
        sedi_heat( ktop, kbot, dp, m1_rain, dz, tz, qv, ql, qr, qi, 
                   qs, qg, c_liq, nf_inv )
        
    # Evaporation and accretion of rain for the remaing 1/2 time step
    revap_racc( ktop, kbot, dt5, tz, qv, ql, qr, qi, qs, qg, den, 
                denfac, rh_rain, h_var )
                    
    # Auto-conversion assuming linear subgrid vertical distribution of 
    # cloud water following lin et al. 1994, mwr
    if irain_f != 0:
        
        # No subgrid variability
        qc0 = fac_rc * ccn
        
        ijk = tz > t_wfr
        
        if use_ccn:
            # ccn is formulted as ccn = ccn_surface * (den / den_surface)
            qc = qc0[ijk]
        else:
            qc = qc0[ijk] / den[ijk]
        
        dq = np.empty_like(qc0)
            
        dq[ijk] = ql[ijk] - qc
        
        ijk = ijk & (dq > 0.)
        
        '''
        NOTE: Computing the exponential and logarithm can be avoided and 
              this will also avoid python "divide by zero in log" 
              warnings
        '''
        sink    = np.minimum(dq[ijk], dt * c_praut[ijk] * den[ijk] * np.exp(so3 * np.log(ql[ijk])))
        ql[ijk] = ql[ijk] - sink
        qr[ijk] = qr[ijk] + sink
    
    else:
        
        dl = np.empty_like(ql)
        
        '''
        NOTE: ktop needs to be passed, to avoid having to pass a 
              reference to an array element (for ql and dl), but still 
              knowing the starting index (ktop). The implementation of 
              the function will change a bit!
        '''
        # With subgrid variability
        linear_prof(kbot-ktop+1, ktop, ql, dl, z_slope_liq, h_var)
                     
        qc0 = fac_rc * ccn
        
        ijk = tz > t_wfr + dt_fr
        
        dl[ijk] = np.minimum(np.maximum(1.e-6, dl[ijk]), 0.5 * ql[ijk])
        
        # As in klein's gfdl am2 stratiform scheme (with subgrid variations)
        if use_ccn:
            # ccn is formulted as ccn = ccn_surface * (den / den_surface)
            qc = qc0[ijk]
        else:
            qc = qc0[ijk] / den[ijk]
        
        dq = np.empty_like(qc0)
        
        dq[ijk] = 0.5 * (ql[ijk] + dl[ijk] - qc)
        
        # dq = dl if qc == q_minus = ql - dl
        # dq = 0 if qc == q_plus = ql + dl
        ijk = ijk & (dq > 0.)   # q_plus > qc
        
        # Revised continuous form: linearly decays (with subgrid dl) to 
        # zero at qc == ql + dl
        '''
        NOTE: Computing the exponential and logarithm can be avoided and 
              this will also avoid python "divide by zero in log" 
              warnings
        '''
        sink    = np.minimum(1., dq[ijk] / dl[ijk]) * dt * c_praut[ijk] * den[ijk] * np.exp(so3 * np.log(ql[ijk]))
        ql[ijk] = ql[ijk] - sink
        qr[ijk] = qr[ijk] + sink


# Evaporation of rain    
def revap_racc( ktop, kbot, dt, tz, qv, ql, qr, qi, qs, qg, den, denfac, 
                rh_rain, h_var, condition=None ):
    
    # Check if there is a condition, otherwise fill condition with True
    c = check_condition(condition, qv.shape[:2])
    
    c3d = np.repeat(c[:, :, np.newaxis], kbot-ktop+1, axis=2)

    ijk = c3d & (tz > t_wfr) & (qr > qrmin)
    
    # Define heat capacity and latent heat coefficient
    lhl     = np.empty_like(tz)
    q_liq   = np.empty_like(ql)
    q_sol   = np.empty_like(qi)
    cvm     = np.empty_like(qv)
    lcpk    = np.empty_like(qv)
    tin     = np.empty_like(tz)
    qpz     = np.empty_like(qv)
    qsat    = np.empty_like(den)
    dqsdt   = np.empty_like(qr)
    dqh     = np.empty_like(ql)
    dqv     = np.empty_like(qv)
    q_minus = np.empty_like(qv)
    q_plus  = np.empty_like(qv)
    
    '''
    NOTE: h_var needs to be copied along the k-dimension to ensure the 
          same dimensions during computation of dqh, since the fields 
          involved are 2D and 3D
    '''
    h_var3d = np.repeat(h_var[:, :, np.newaxis], kbot-ktop+1, axis=2)
    
    lhl[ijk]   = lv00 + d0_vap * tz[ijk]
    q_liq[ijk] = ql[ijk] + qr[ijk]
    q_sol[ijk] = qi[ijk] + qs[ijk] + qg[ijk]
    cvm[ijk]   = c_air + qv[ijk] * c_vap + q_liq[ijk] * c_liq + q_sol[ijk] * c_ice
    lcpk[ijk]  = lhl[ijk] / cvm[ijk]

    tin[ijk] = tz[ijk] - lcpk[ijk] * ql[ijk]    # Presence of clouds suppresses the rain evap
    qpz[ijk] = qv[ijk] + ql[ijk]
    
    qsat[ijk], dqsdt[ijk] = wqs2(tin[ijk], den[ijk])
    
    dqh[ijk]     = np.maximum(ql[ijk], h_var3d[ijk] * np.maximum(qpz[ijk], qcmin))
    dqh[ijk]     = np.minimum(dqh[ijk], 0.2 * qpz[ijk]) # New limiter
    dqv[ijk]     = qsat[ijk] - qv[ijk]  # Use this to prevent super-sat the gird box
    q_minus[ijk] = qpz[ijk] - dqh[ijk]
    q_plus[ijk]  = qpz[ijk] + dqh[ijk]

    # qsat must be > q_minus to activate evaporation
    # qsat must be < q_plus to activate accretion
    # Rain evaporation
    ijk_dqv = ijk & (dqv > qvmin) & (qsat > q_minus)
    
    ijk_qsat = ijk_dqv & (qsat > q_plus)
    
    dq = np.empty_like(qv)
    
    dq[ijk_qsat] = qsat[ijk_qsat] - qpz[ijk_qsat]
    
    ijk_qsat_inv = ijk_dqv & (qsat <= q_plus)
    
    # q_minus < qsat < q_plus
    # dq == dqh if qsat == q_minus
    dq[ijk_qsat_inv] = 0.25 * (q_minus[ijk_qsat_inv] - qsat[ijk_qsat_inv])**2 / dqh[ijk_qsat_inv]
    
    qden = np.empty_like(qr)
    
    qden[ijk_dqv] = qr[ijk_dqv] * den[ijk_dqv]
    t2            = tin[ijk_dqv] * tin[ijk_dqv]
    evap          = crevp[0] * t2 * dq[ijk_dqv] * \
                    ( crevp[1] * np.sqrt(qden[ijk_dqv]) + crevp[2] * \
                      np.exp(0.725 * np.log(qden[ijk_dqv])) ) / \
                    ( crevp[3] * t2 + crevp[4] * qsat[ijk_dqv] * den[ijk_dqv] )
    evap          = np.minimum(qr[ijk_dqv], np.minimum(dt * evap, dqv[ijk_dqv] / (1. + lcpk[ijk_dqv] * dqsdt[ijk_dqv])))

    # Alternative minimum evap in dry environmental air
    # sink = min (qr (k), dim (rh_rain * qsat, qv (k)) / (1. + lcpk (k) * dqsdt))
    # evap = max (evap, sink)
    qr[ijk_dqv]    = qr[ijk_dqv] - evap
    qv[ijk_dqv]    = qv[ijk_dqv] + evap
    q_liq[ijk_dqv] = q_liq[ijk_dqv] - evap
    cvm[ijk_dqv]   = c_air + qv[ijk_dqv] * c_vap + q_liq[ijk_dqv] * c_liq + q_sol[ijk_dqv] * c_ice
    tz[ijk_dqv]    = tz[ijk_dqv] - evap * lhl[ijk_dqv] / cvm[ijk_dqv]
    
    # Accretion: pracc
    ijk_qr = ijk & (qr > qrmin) & (ql > 1.e-6) & (qsat < q_minus)
    
    sink       = dt * denfac[ijk_qr] * cracw * np.exp(0.95 * np.log(qr[ijk_qr] * den[ijk_qr]))
    sink       = sink / (1. + sink) * ql[ijk_qr]
    ql[ijk_qr] = ql[ijk_qr] - sink
    qr[ijk_qr] = qr[ijk_qr] + sink


# Definition of vertical subgrid variability (used for cloud ice and 
# cloud water autoconversion)
def linear_prof(km, ktop, q, dm, z_var, h_var):
    
    if z_var:
        
        dq = np.empty_like(q)
        
        dq[:, :, 1:km] = 0.5 * (q[:, :, 1:km] - q[:, :, 0:km-1])
        
        dm[:, :, 0] = 0.
        
        # Use twice the strength of the positive definiteness limiter (lin et al 1994)
        dm[:, :, 1:km-1] = 0.5 * np.minimum(np.abs(dq[:, :, 1:km-1] + dq[:, :, 2:km]), 0.5 * q[:, :, 1:km-1])
        
        ijk = dq[:, :, 1:km-1] * dq[:, :, 2:km] <= 0.
        
        ijk_dq = ijk & (dq[:, :, 1:km-1] > 0.) # Local maximum
        
        dm[:, :, 1:km-1][ijk_dq] = np.minimum( dm[:, :, 1:km-1][ijk_dq], 
                                               np.minimum( dq[:, :, 1:km-1][ijk_dq], 
                                                           -dq[:, :, 2:km][ijk_dq] ) 
                                   )
        
        ijk_dq_inv = ijk & (dq[:, :, 1:km-1] <= 0.)
        
        dm[:, :, 1:km-1][ijk_dq_inv] = 0.
        
        dm[:, :, km-1] = 0.
        
        # Impose a presumed background horizontal variability that is 
        # proportional to the value itself
        h_var3d = np.repeat(h_var[:, :, np.newaxis], km, axis=2)
        
        dm[:, :, 0:km] = np.maximum(dm[:, :, 0:km], np.maximum(qvmin, h_var3d * q[:, :, 0:km]))
    
    else:
        
        h_var3d = np.repeat(h_var[:, :, np.newaxis], km, axis=2)
        
        dm[:, :, 0:km] = np.maximum(qvmin, h_var3d * q[:, :, 0:km])


# Ice cloud microphysics processes
def icloud( ktop, kbot, tzk, p1, qvk, qlk, qrk, qik, qsk, qgk, dp1, den, 
            denfac, vts, vtg, vtr, qak, rh_adj, rh_rain, dts, h_var ):
    
    dt5  = 0.5 * dts
    rdts = 1. / dts
    
    # Define conversion scalar / factor
    fac_i2s  = 1. - mt.exp(-dts / tau_i2s)
    fac_g2v  = 1. - mt.exp(-dts / tau_g2v)
    fac_v2g  = 1. - mt.exp(-dts / tau_v2g)
    fac_imlt = 1. - mt.exp(-dt5 / tau_imlt)
    
    # Define heat capacity and latend heat coefficient
    lhi   = li00 + dc_ice * tzk
    q_liq = qlk + qrk
    q_sol = qik + qsk + qgk
    cvm   = c_air + qvk * c_vap + q_liq * c_liq + q_sol * c_ice
    icpk  = lhi / cvm
    
    # - Sources of cloud ice: pihom, cold rain, and the sat_adj
    # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
    # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion
    ijk = (tzk > tice) & (qik > qcmin)
    '''
    NOTE: This represents an else if
    NOTE: ijk_alt has to be computed now since tzk changes inside the if
    '''
    ijk_alt = ~ijk & (tzk < t_wfr) & (qlk > qcmin)
    
    tmp = np.empty_like(qlk)
    
    # pimlt: instant melting of cloud ice
    melt       = np.minimum(qik[ijk], fac_imlt * (tzk[ijk] - tice) / icpk[ijk])
    '''
    NOTE: The where functionality is used to compute the positive difference (dim() in Fortran)
    '''
    tmp[ijk]   = np.minimum(melt, np.where(qlk[ijk] < ql_mlt, ql_mlt-qlk[ijk], 0)) # Maximum ql amount
    qlk[ijk]   = qlk[ijk] + tmp[ijk]
    qrk[ijk]   = qrk[ijk] + melt - tmp[ijk]
    qik[ijk]   = qik[ijk] - melt
    q_liq[ijk] = q_liq[ijk] + melt
    q_sol[ijk] = q_sol[ijk] - melt
    cvm[ijk]   = c_air + qvk[ijk] * c_vap + q_liq[ijk] * c_liq + q_sol[ijk] * c_ice
    tzk[ijk]   = tzk[ijk] - melt * lhi[ijk] / cvm[ijk]
    
    # - pihom: homogeneous freezing of cloud water into cloud ice
    # - This is the 1st occurance of liquid water freezing in the split mp process
    factor = np.empty_like(tzk)
    sink   = np.empty_like(tzk)
    
    dtmp            = t_wfr - tzk[ijk_alt]
    factor[ijk_alt] = np.minimum(1., dtmp / dt_fr)
    sink[ijk_alt]   = np.minimum(qlk[ijk_alt] * factor[ijk_alt], dtmp / icpk[ijk_alt])
    qi_crt          = qi_gen * np.minimum(qi_lim, 0.1 * (tice - tzk[ijk_alt])) / den[ijk_alt]
    '''
    NOTE: The where functionality is used to compute the positive difference (dim() in Fortran)
    '''
    tmp[ijk_alt]    = np.minimum(sink[ijk_alt], np.where(qik[ijk_alt] < qi_crt, qi_crt-qik[ijk_alt], 0))
    qlk[ijk_alt]    = qlk[ijk_alt] - sink[ijk_alt]
    qsk[ijk_alt]    = qsk[ijk_alt] + sink[ijk_alt] - tmp[ijk_alt]
    qik[ijk_alt]    = qik[ijk_alt] + tmp[ijk_alt]
    q_liq[ijk_alt]  = q_liq[ijk_alt] - sink[ijk_alt]
    q_sol[ijk_alt]  = q_sol[ijk_alt] + sink[ijk_alt]
    cvm[ijk_alt]    = c_air + qvk[ijk_alt] * c_vap + q_liq[ijk_alt] * c_liq + q_sol[ijk_alt] * c_ice
    tzk[ijk_alt]    = tzk[ijk_alt] + sink[ijk_alt] * lhi[ijk_alt] / cvm[ijk_alt]
    
    di = np.empty_like(qik)
    
    # Vertical subgrid variability
    linear_prof(kbot-ktop+1, ktop, qik, di, z_slope_ice, h_var)
    
    # Update capacity heat and latent heat coefficient
    lhl  = lv00 + d0_vap * tzk
    lhi  = li00 + dc_ice * tzk
    lcpk = lhl / cvm
    icpk = lhi / cvm
    tcpk = lcpk + icpk
    
    ijk = p1 >= p_min
    
    '''
    NOTE: This is not needed since * are copied back into *k at the end
    '''
    # ~ tz = tzk.copy()
    # ~ qv = qvk.copy()
    # ~ ql = qlk.copy()
    # ~ qi = qik.copy()
    # ~ qr = qrk.copy()
    # ~ qs = qsk.copy()
    # ~ qg = qgk.copy()
    
    pgacr = np.empty_like(qgk)
    pgacw = np.empty_like(qlk)
    
    pgacr[ijk] = 0.
    pgacw[ijk] = 0.
    
    tc = np.empty_like(tzk)
    
    tc[ijk] = tzk[ijk] - tice
    
    '''
    NOTE: ijk_tc_inv has to be computed now since tc changes inside the if
    '''
    ijk_tc     = ijk & (tc >= 0.)
    ijk_tc_inv = ijk & (tc < 0.)
    
    # Melting of snow
    dqs0 = np.empty_like(qvk)
    
    dqs0[ijk_tc] = ces0 / p1[ijk_tc] - qvk[ijk_tc]
    
    ijk_qs = ijk_tc & (qsk > qcmin)
    
    # psacw: accretion of cloud water by snow (only rate is used (for 
    # snow melt) since tc > 0.)
    ijk_ql = ijk_qs & (qlk > qrmin)
    
    psacw = np.empty_like(qlk)
    
    factor[ijk_ql] = denfac[ijk_ql] * csacw * np.exp(0.8125 * np.log(qsk[ijk_ql] * den[ijk_ql]))
    psacw[ijk_ql]  = factor[ijk_ql] / (1. + dts * factor[ijk_ql]) * qlk[ijk_ql] # Rate
        
    ijk_ql_inv = ijk_qs & (qlk <= qrmin)
    
    psacw[ijk_ql_inv] = 0.
    
    # psacr: accretion of rain by melted snow
    # pracs: accretion of snow by rain
    ijk_qr = ijk_qs & (qrk > qrmin)
    
    psacr = np.empty_like(vts)
    pracs = np.empty_like(vtr)
    
    psacr[ijk_qr] = np.minimum( 
                        acr3d( vts[ijk_qr], vtr[ijk_qr], qrk[ijk_qr], 
                               qsk[ijk_qr], csacr, acco, 0, 1, 
                               den[ijk_qr] ), 
                        qrk[ijk_qr] * rdts 
                    )
    pracs[ijk_qr] = acr3d( vtr[ijk_qr], vts[ijk_qr], qsk[ijk_qr], 
                           qrk[ijk_qr], cracs, acco, 0, 0, den[ijk_qr] )
    
    ijk_qr_inv = ijk_qs & (qrk <= qrmin)
    
    psacr[ijk_qr_inv] = 0.
    pracs[ijk_qr_inv] = 0.
    
    # Total snow sink
    # psmlt: snow melt (due to rain accretion)
    psmlt         = np.maximum( 0., smlt( tc[ijk_qs], dqs0[ijk_qs], 
                                          qsk[ijk_qs] * den[ijk_qs], 
                                          psacw[ijk_qs], psacr[ijk_qs], 
                                          csmlt, den[ijk_qs], 
                                          denfac[ijk_qs] ) )
    sink[ijk_qs]  = np.minimum(qsk[ijk_qs], np.minimum(dts * (psmlt + pracs[ijk_qs]), tc[ijk_qs] / icpk[ijk_qs]))
    qsk[ijk_qs]   = qsk[ijk_qs] - sink[ijk_qs]
    tmp[ijk_qs]   = np.minimum(sink[ijk_qs], np.where(qlk[ijk_qs] < qs_mlt, qs_mlt-qlk[ijk_qs], 0))    # Maximum ql due to snow melt
    qlk[ijk_qs]   = qlk[ijk_qs] + tmp[ijk_qs]
    qrk[ijk_qs]   = qrk[ijk_qs] + sink[ijk_qs] - tmp[ijk_qs]
    q_liq[ijk_qs] = q_liq[ijk_qs] + sink[ijk_qs]
    q_sol[ijk_qs] = q_sol[ijk_qs] - sink[ijk_qs]
    cvm[ijk_qs]   = c_air + qvk[ijk_qs] * c_vap + q_liq[ijk_qs] * c_liq + q_sol[ijk_qs] * c_ice
    tzk[ijk_qs]   = tzk[ijk_qs] - sink[ijk_qs] * lhi[ijk_qs] / cvm[ijk_qs]
    tc[ijk_qs]    = tzk[ijk_qs] - tice
    
    # Update capacity heat and latend heat coefficient
    lhi[ijk_tc]  = li00 + dc_ice * tzk[ijk_tc]
    icpk[ijk_tc] = lhi[ijk_tc] / cvm[ijk_tc]
    
    # Melting of graupel
    ijk_qg = ijk_tc & (qgk > qcmin) & (tc > 0.)
    
    # pgacr: accretion of rain by graupel
    ijk_qr = ijk_qg & (qrk > qrmin)
    
    pgacr[ijk_qr] = np.minimum(
                        acr3d( vtg[ijk_qr], vtr[ijk_qr], qrk[ijk_qr], 
                               qgk[ijk_qr], cgacr, acco, 0, 2, den[ijk_qr] ),
                        rdts * qrk[ijk_qr]
                    )
    
    # pgacw: accretion of cloud water by graupel
    qden = np.empty_like(den)
    
    qden[ijk_qg] = qgk[ijk_qg] * den[ijk_qg]
    
    ijk_ql = ijk_qg & (qlk > qrmin)
    
    factor[ijk_ql] = cgacw * qden[ijk_ql] / np.sqrt(den[ijk_ql] * np.sqrt(np.sqrt(qden[ijk_ql])))
    pgacw[ijk_ql]  = factor[ijk_ql] / (1. + dts * factor[ijk_ql]) * qlk[ijk_ql] # Rate
    
    # pgmlt: graupel melt
    pgmlt         = dts * gmlt( tc[ijk_qg], dqs0[ijk_qg], qden[ijk_qg], 
                                pgacw[ijk_qg], pgacr[ijk_qg], cgmlt, 
                                den[ijk_qg] )
    pgmlt         = np.minimum(np.maximum(0., pgmlt), np.minimum(qgk[ijk_qg], tc[ijk_qg] / icpk[ijk_qg]))
    qgk[ijk_qg]   = qgk[ijk_qg] - pgmlt
    qrk[ijk_qg]   = qrk[ijk_qg] + pgmlt
    q_liq[ijk_qg] = q_liq[ijk_qg] + pgmlt
    q_sol[ijk_qg] = q_sol[ijk_qg] - pgmlt
    cvm[ijk_qg]   = c_air + qvk[ijk_qg] * c_vap + q_liq[ijk_qg] * c_liq + q_sol[ijk_qg] * c_ice
    tzk[ijk_qg]   = tzk[ijk_qg] - pgmlt * lhi[ijk_qg] / cvm[ijk_qg]
    
    # Cloud ice proc
    # psaci: accretion of cloud ice by snow
    ijk_qi = ijk_tc_inv & (qik > 3.e-7)  # Cloud ice sink terms
    
    ijk_qs = ijk_qi & (qsk > 1.e-7)
    
    psaci = np.empty_like(qik)
    
    # sjl added (following lin eq. 23) the temperature dependency to 
    # reduce accretion, use esi = exp(0.05 * tc) as in hong et al 2004
    factor[ijk_qs] = dts * denfac[ijk_qs] * csaci * \
                     np.exp(0.05 * tc[ijk_qs] + 0.8125 * np.log(qsk[ijk_qs] * den[ijk_qs]))
    psaci[ijk_qs]  = factor[ijk_qs] / (1. + factor[ijk_qs]) * qik[ijk_qs]
    
    ijk_qs_inv = ijk_qi & (qsk <= 1.e-7)
    
    psaci[ijk_qs_inv] = 0.
    
    # pasut: autoconversion: cloud ice -- > snow
    # - Similar to lfo 1983: eq. 21 solved implicitly
    # - Threshold from wsm6 scheme, hong et al 2004, eq (13) : qi0_crt ~0.8e-4
    qim = np.empty_like(den)
    
    qim[ijk_qi] = qi0_crt / den[ijk_qi]
    
    # - Assuming linear subgrid vertical distribution of cloud ice
    # - The mismatch computation following lin et al. 1994, mwr
    
    if const_vi:
        tmp[ijk_qi] = fac_i2s
    else:
        tmp[ijk_qi] = fac_i2s * np.exp(0.025 * tc[ijk_qi])
    
    q_plus = np.empty_like(qik)
    
    di[ijk_qi]     = np.maximum(di[ijk_qi], qrmin)
    q_plus[ijk_qi] = qik[ijk_qi] + di[ijk_qi]
    
    ijk_qp = ijk_qi & (q_plus > (qim + qrmin))
    
    ijk_qim = ijk_qp & (qim > (qik - di))
    
    dq = np.empty_like(di)
    
    dq[ijk_qim] = (0.25 * (q_plus[ijk_qim] - qim[ijk_qim])**2) / di[ijk_qim]
    
    ijk_qim_inv = ijk_qp & (qim <= (qik - di))
    
    dq[ijk_qim_inv] = qik[ijk_qim_inv] - qim[ijk_qim_inv]
    
    psaut = np.empty_like(dq)
    
    psaut[ijk_qp] = tmp[ijk_qp] * dq[ijk_qp]
    
    ijk_qp_inv = ijk_qi & (q_plus <= (qim + qrmin))
    
    psaut[ijk_qp_inv] = 0.
    
    # sink is no greater than 75% of qi
    sink[ijk_qi] = np.minimum(0.75 * qik[ijk_qi], psaci[ijk_qi] + psaut[ijk_qi])
    qik[ijk_qi]  = qik[ijk_qi] - sink[ijk_qi]
    qsk[ijk_qi]  = qsk[ijk_qi] + sink[ijk_qi]
    
    # pgaci: accretion of cloud ice by graupel
    ijk_qg = ijk_qi & (qgk > 1.e-6)
    
    # - factor = dts * cgaci / sqrt (den (k)) * exp (0.05 * tc + 0.875 * log (qg * den (k)))
    # - Simplified form: remove temp dependency & set the exponent "0.875" -- > 1
    factor[ijk_qg] = dts * cgaci * np.sqrt(den[ijk_qg]) * qgk[ijk_qg]
    pgaci          = factor[ijk_qg] / (1. + factor[ijk_qg]) * qik[ijk_qg]
    qik[ijk_qg]    = qik[ijk_qg] - pgaci
    qgk[ijk_qg]    = qgk[ijk_qg] + pgaci
    
    # Cold-rain proc
    # Rain to ice, snow, graupel processes
    tc[ijk_tc_inv] = tzk[ijk_tc_inv] - tice
    
    ijk_qr = ijk_tc_inv & (qrk > 1e-7) & (tc < 0.)

    # - Sink terms to qr: psacr + pgfr
    # - Source terms to qs: psacr
    # - Source terms to qg: pgfr
    # psacr accretion of rain by snow
    ijk_qs = ijk_qr & (qsk > 1.e-7) # If snow exists
    
    psacr[ijk_qs] = dts * acr3d( vts[ijk_qs], vtr[ijk_qs], qrk[ijk_qs], 
                                 qsk[ijk_qs], csacr, acco, 0, 1, 
                                 den[ijk_qs] )
                                 
    ijk_qs_inv = ijk_qr & (qsk <= 1.e-7)
    
    psacr[ijk_qs_inv] = 0.
    
    # pgfr: rain freezing -- > graupel
    pgfr = dts * cgfr[0] / den[ijk_qr] * (np.exp(-cgfr[1] * tc[ijk_qr]) - 1.) * \
           np.exp(1.75 * np.log(qrk[ijk_qr] * den[ijk_qr]))

    # Total sink to qr
    sink[ijk_qr]   = psacr[ijk_qr] + pgfr
    factor[ijk_qr] = np.minimum( sink[ijk_qr], 
                                 np.minimum( qrk[ijk_qr], 
                                             -tc[ijk_qr] / icpk[ijk_qr] ) ) / \
                     np.maximum( sink[ijk_qr], qrmin )
    
    psacr[ijk_qr] = factor[ijk_qr] * psacr[ijk_qr]
    pgfr          = factor[ijk_qr] * pgfr
    
    sink[ijk_qr]  = psacr[ijk_qr] + pgfr
    qrk[ijk_qr]   = qrk[ijk_qr] - sink[ijk_qr]
    qsk[ijk_qr]   = qsk[ijk_qr] + psacr[ijk_qr]
    qgk[ijk_qr]   = qgk[ijk_qr] + pgfr
    q_liq[ijk_qr] = q_liq[ijk_qr] - sink[ijk_qr]
    q_sol[ijk_qr] = q_sol[ijk_qr] + sink[ijk_qr]
    cvm[ijk_qr]   = c_air + qvk[ijk_qr] * c_vap + q_liq[ijk_qr] * c_liq + q_sol[ijk_qr] * c_ice
    tzk[ijk_qr]   = tzk[ijk_qr] + sink[ijk_qr] * lhi[ijk_qr] / cvm[ijk_qr]
    
    # Update capacity heat and latend heat coefficient
    lhi[ijk_tc_inv]  = li00 + dc_ice * tzk[ijk_tc_inv]
    icpk[ijk_tc_inv] = lhi[ijk_tc_inv] / cvm[ijk_tc_inv]
    
    # Graupel production terms
    ijk_qs = ijk_tc_inv & (qsk > 1.e-7)
    
    # Accretion: snow -- > graupel
    ijk_qg = ijk_qs & (qgk > qrmin)
    
    sink[ijk_qg] = dts * acr3d( vtg[ijk_qg], vts[ijk_qg], qsk[ijk_qg], 
                                qgk[ijk_qg], cgacs, acco, 0, 3, 
                                den[ijk_qg] )
    
    ijk_qg_inv = ijk_qs & (qgk <= qrmin)
    
    sink[ijk_qg_inv] = 0.
    
    # Autoconversion snow -- > graupel
    qsm = np.empty_like(den)
    
    qsm[ijk_qs] = qs0_crt / den[ijk_qs]
    
    ijk_qsm = ijk_qs & (qsk > qsm)
    
    factor[ijk_qsm] = dts * 1.e-3 * np.exp(0.09 * (tzk[ijk_qsm] - tice))
    sink[ijk_qsm]   = sink[ijk_qsm] + factor[ijk_qsm] / (1. + factor[ijk_qsm]) * (qsk[ijk_qsm] - qsm[ijk_qsm])
    
    sink[ijk_qs] = np.minimum(qsk[ijk_qs], sink[ijk_qs])
    qsk[ijk_qs] = qsk[ijk_qs] - sink[ijk_qs]
    qgk[ijk_qs] = qgk[ijk_qs] + sink[ijk_qs]
    
    ijk_qg = ijk_tc_inv & (qgk > 1.e-7) & (tzk < tice0)
    
    # pgacw: accretion of cloud water by graupel
    ijk_ql = ijk_qg & (qlk > 1.e-6)
    
    qden[ijk_ql]   = qgk[ijk_ql] * den[ijk_ql]
    factor[ijk_ql] = dts * cgacw * qden[ijk_ql] / np.sqrt(den[ijk_ql] * np.sqrt(np.sqrt(qden[ijk_ql])))
    pgacw[ijk_ql]  = factor[ijk_ql] / (1. + factor[ijk_ql]) * qlk[ijk_ql]
    
    ijk_ql_inv = ijk_qg & (qlk <= 1.e-6)
    
    pgacw[ijk_ql_inv] = 0.
    
    # pgacr: accretion of rain by graupel
    ijk_qr = ijk_qg & (qrk > 1.e-6)
    
    pgacr[ijk_qr] = np.minimum( dts * acr3d( vtg[ijk_qr], vtr[ijk_qr], 
                                             qrk[ijk_qr], qgk[ijk_qr], 
                                             cgacr, acco, 0, 2, 
                                             den[ijk_qr] ), 
                                qrk[ijk_qr]
                    )
                                    
    ijk_qr_inv = ijk_qg & (qrk <= 1.e-6)
    
    pgacr[ijk_qr_inv] = 0.
    
    sink[ijk_qg]   = pgacr[ijk_qg] + pgacw[ijk_qg]
    factor[ijk_qg] = np.minimum(sink[ijk_qg], np.where(tzk[ijk_qg] < tice, tice-tzk[ijk_qg], 0) / icpk[ijk_qg]) / np.maximum(sink[ijk_qg], qrmin)
    pgacr[ijk_qg]  = factor[ijk_qg] * pgacr[ijk_qg]
    pgacw[ijk_qg]  = factor[ijk_qg] * pgacw[ijk_qg]
    
    sink[ijk_qg]  = pgacr[ijk_qg] + pgacw[ijk_qg]
    qgk[ijk_qg]   = qgk[ijk_qg] + sink[ijk_qg]
    qrk[ijk_qg]   = qrk[ijk_qg] - pgacr[ijk_qg]
    qlk[ijk_qg]   = qlk[ijk_qg] - pgacw[ijk_qg]
    q_liq[ijk_qg] = q_liq[ijk_qg] - sink[ijk_qg]
    q_sol[ijk_qg] = q_sol[ijk_qg] + sink[ijk_qg]
    cvm[ijk_qg]   = c_air + qvk[ijk_qg] * c_vap + q_liq[ijk_qg] * c_liq + q_sol[ijk_qg] * c_ice
    tzk[ijk_qg]   = tzk[ijk_qg] + sink[ijk_qg] * lhi[ijk_qg] / cvm[ijk_qg]
    
    # Subgrid cloud microphysics
    subgrid_z_proc( ktop, kbot, p1, den, denfac, dts, rh_adj, tzk, qvk, 
                    qlk, qrk, qik, qsk, qgk, qak, h_var, rh_rain )


# Temperature sentive high vertical resolution processes
def subgrid_z_proc( ktop, kbot, p1, den, denfac, dts, rh_adj, tz, qv, 
                    ql, qr, qi, qs, qg, qa, h_var, rh_rain ):
                        
    if fast_sat_adj:
        dt_evap = 0.5 * dts
    else:
        dt_evap = dts
    
    # Define conversion scalar / factor
    fac_v2l = 1. - mt.exp(-dt_evap / tau_v2l)
    fac_l2v = 1. - mt.exp(-dt_evap / tau_l2v)
    
    fac_g2v = 1. - mt.exp(-dts / tau_g2v)
    fac_v2g = 1. - mt.exp(-dts / tau_v2g)
    
    # Define heat capacity and latend heat coefficient
    lhl   = lv00 + d0_vap * tz
    lhi   = li00 + dc_ice * tz
    q_liq = ql + qr
    q_sol = qi + qs + qg
    cvm   = c_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
    lcpk  = lhl / cvm
    icpk  = lhi / cvm
    tcpk  = lcpk + icpk
    tcp3  = lcpk + icpk * np.minimum(1., np.where(tz < tice, tice-tz, 0) / (tice - t_wfr))
    
    ijk = p1 >= p_min
    
    # Instant deposit all water vapor to cloud ice when temperature is 
    # super low
    ijk_tz = ijk & (tz < t_min)
    '''
    NOTE: This represents the cycle, if the if is not entered then the 
          rest can be executed, while if the if is entered you cycle
    '''    
    ijk_tz_inv = ijk & (tz >= t_min)
    
    sink = np.empty_like(qv)
    
    sink[ijk_tz]  = np.where(1.e-7 < qv[ijk_tz], 1.e-7-qv[ijk_tz], 0)
    qv[ijk_tz]    = qv[ijk_tz] - sink[ijk_tz]
    qi[ijk_tz]    = qi[ijk_tz] + sink[ijk_tz]
    q_sol[ijk_tz] = q_sol[ijk_tz] + sink[ijk_tz]
    cvm[ijk_tz]   = c_air + qv[ijk_tz] * c_vap + q_liq[ijk_tz] * c_liq + q_sol[ijk_tz] * c_ice
    tz[ijk_tz]    = tz[ijk_tz] + sink[ijk_tz] * (lhl[ijk_tz] + lhi[ijk_tz]) / cvm[ijk_tz]
    
    if not do_qa:
        qa[ijk_tz] = qa[ijk_tz] + 1.    # Air fully saturated; 100% cloud cover
    
    lhl[ijk_tz_inv]  = lv00 + d0_vap * tz[ijk_tz_inv]
    lhi[ijk_tz_inv]  = li00 + dc_ice * tz[ijk_tz_inv]
    lcpk[ijk_tz_inv] = lhl[ijk_tz_inv] / cvm[ijk_tz_inv]
    icpk[ijk_tz_inv] = lhi[ijk_tz_inv] / cvm[ijk_tz_inv]
    tcpk[ijk_tz_inv] = lcpk[ijk_tz_inv] + icpk[ijk_tz_inv]
    tcp3[ijk_tz_inv] = lcpk[ijk_tz_inv] + icpk[ijk_tz_inv] * np.minimum(1., np.where(tz[ijk_tz_inv] < tice, tice-tz[ijk_tz_inv], 0) / (tice - t_wfr))
    
    # Instant evaporation / sublimation of all clouds if rh < rh_adj -- > cloud free
    qpz = np.empty_like(qv)
    tin = np.empty_like(tz)
    
    qpz[ijk_tz_inv] = qv[ijk_tz_inv] + ql[ijk_tz_inv] + qi[ijk_tz_inv]
    tin[ijk_tz_inv] = tz[ijk_tz_inv] - ( lhl[ijk_tz_inv] * \
                      ( ql[ijk_tz_inv] + qi[ijk_tz_inv] ) + \
                      lhi[ijk_tz_inv] * qi[ijk_tz_inv] ) / \
                      ( c_air + qpz[ijk_tz_inv] * c_vap + \
                      qr[ijk_tz_inv] * c_liq + \
                      ( qs[ijk_tz_inv] + qg[ijk_tz_inv] ) * c_ice )
    
    ijk_tin = ijk_tz_inv & (tin > t_sub + 6.)
    
    rh = np.empty_like(qv)
    
    rh[ijk_tin] = qpz[ijk_tin] / iqs1(tin[ijk_tin], den[ijk_tin])
    
    rh_adj3d = np.repeat(rh_adj[:, :, np.newaxis], kbot-ktop+1, axis=2)
    
    ijk_rh = ijk_tin & (rh < rh_adj3d)  # qpz / rh_adj < qs
    
    tz[ijk_rh] = tin[ijk_rh]
    qv[ijk_rh] = qpz[ijk_rh]
    ql[ijk_rh] = 0.
    qi[ijk_rh] = 0.
    
    '''
    NOTE: This represents the cycle inside a double if
    '''    
    ijk_rh_inv = ( ijk_tin & (rh >= rh_adj3d) ) | ( ijk_tz_inv & (tin <= t_sub + 6.) )
    
    # Cloud water < -- > vapor adjustment
    qsw   = np.empty_like(tz)
    dwsdt = np.empty_like(tz)
    dq0   = np.empty_like(tz)
    
    qsw[ijk_rh_inv], dwsdt[ijk_rh_inv] = wqs2(tz[ijk_rh_inv], den[ijk_rh_inv])
    
    dq0[ijk_rh_inv] = qsw[ijk_rh_inv] - qv[ijk_rh_inv]
    
    ijk_dq0 = ijk_rh_inv & (dq0 > 0.)
    
    # SJL 20170703 added ql factor to prevent the situation of high ql and low RH
    # factor = min (1., fac_l2v * sqrt (max (0., ql (k)) / 1.e-5) * 10. * dq0 / qsw)
    # factor = fac_l2v
    # factor = 1
    factor = np.empty_like(tz)
    evap   = np.empty_like(tz)
    
    factor[ijk_dq0] = np.minimum(1., fac_l2v * (10. * dq0[ijk_dq0] / qsw[ijk_dq0])) # The rh dependent factor = 1 at 90%
    evap[ijk_dq0]   = np.minimum( ql[ijk_dq0], 
                                  factor[ijk_dq0] * dq0[ijk_dq0] / (1. + tcp3[ijk_dq0] * dwsdt[ijk_dq0]) )
    
    ijk_dq0_inv = ijk_rh_inv & (dq0 <= 0.)  # Condensate all excess vapor into cloud water
    
    # evap = fac_v2l * dq0 / (1. + tcp3 (k) * dwsdt)
    # sjl, 20161108
    evap[ijk_dq0_inv] = dq0[ijk_dq0_inv] / (1. + tcp3[ijk_dq0_inv] * dwsdt[ijk_dq0_inv])
    
    qv[ijk_rh_inv]    = qv[ijk_rh_inv] + evap[ijk_rh_inv]
    ql[ijk_rh_inv]    = ql[ijk_rh_inv] - evap[ijk_rh_inv]
    q_liq[ijk_rh_inv] = q_liq[ijk_rh_inv] - evap[ijk_rh_inv]
    cvm[ijk_rh_inv]   = c_air + qv[ijk_rh_inv] * c_vap + \
                        q_liq[ijk_rh_inv] * c_liq + q_sol[ijk_rh_inv] * c_ice
    tz[ijk_rh_inv]    = tz[ijk_rh_inv] - evap[ijk_rh_inv] * lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    
    # Update heat capacity and latent heat coefficient
    lhi[ijk_rh_inv]  = li00 + dc_ice * tz[ijk_rh_inv]
    icpk[ijk_rh_inv] = lhi[ijk_rh_inv] / cvm[ijk_rh_inv]
    
    # Enforce complete freezing below -48 degrees Celsius
    dtmp = np.empty_like(tz)
    
    dtmp[ijk_rh_inv] = t_wfr - tz[ijk_rh_inv]   # [-40, -48]
    
    ijk_dtmp = ijk_rh_inv & (dtmp > 0.) & (ql > qcmin)
    
    sink[ijk_dtmp]  = np.minimum( ql[ijk_dtmp], 
                                  np.minimum( ql[ijk_dtmp] * dtmp[ijk_dtmp] * 0.125, 
                                              dtmp[ijk_dtmp] / icpk[ijk_dtmp] ) )
    ql[ijk_dtmp]    = ql[ijk_dtmp] - sink[ijk_dtmp]
    qi[ijk_dtmp]    = qi[ijk_dtmp] + sink[ijk_dtmp]
    q_liq[ijk_dtmp] = q_liq[ijk_dtmp] - sink[ijk_dtmp]
    q_sol[ijk_dtmp] = q_sol[ijk_dtmp] + sink[ijk_dtmp]
    cvm[ijk_dtmp]   = c_air + qv[ijk_dtmp] * c_vap + q_liq[ijk_dtmp] * c_liq + q_sol[ijk_dtmp] * c_ice
    tz[ijk_dtmp]    = tz[ijk_dtmp] + sink[ijk_dtmp] * lhi[ijk_dtmp] / cvm[ijk_dtmp]
    
    # Update heat capacity and latent heat coefficient
    lhi[ijk_rh_inv]  = li00 + dc_ice * tz[ijk_rh_inv]
    icpk[ijk_rh_inv] = lhi[ijk_rh_inv] / cvm[ijk_rh_inv]
    
    # Bigg mechanism
    if fast_sat_adj:
        dt_pisub = 0.5 * dts
    else:
        dt_pisub = dts
        
        tc = np.empty_like(tz)
        
        tc[ijk_rh_inv] = tice - tz[ijk_rh_inv]
        
        ijk_ql = ijk_rh_inv & (ql > qrmin) & (tc > 0.)
        
        sink[ijk_ql]  = 3.3333e-10 * dts * \
                        (np.exp(0.66 * tc[ijk_ql]) - 1.) * \
                        den[ijk_ql] * ql[ijk_ql] * ql[ijk_ql]
        sink[ijk_ql]  = np.minimum( ql[ijk_ql], 
                                    np.minimum(tc[ijk_ql] / icpk[ijk_ql], sink[ijk_ql]) )
        ql[ijk_ql]    = ql[ijk_ql] - sink[ijk_ql]
        qi[ijk_ql]    = qi[ijk_ql] + sink[ijk_ql]
        q_liq[ijk_ql] = q_liq[ijk_ql] - sink[ijk_ql]
        q_sol[ijk_ql] = q_sol[ijk_ql] + sink[ijk_ql]
        cvm[ijk_ql]   = c_air + qv[ijk_ql] * c_vap + q_liq[ijk_ql] * c_liq + q_sol[ijk_ql] * c_ice
        tz[ijk_ql]    = tz[ijk_ql] + sink[ijk_ql] * lhi[ijk_ql] / cvm[ijk_ql]
        
    # Update capacity heat and latent heat coefficient
    lhl[ijk_rh_inv]  = lv00 + d0_vap * tz[ijk_rh_inv]
    lhi[ijk_rh_inv]  = li00 + dc_ice * tz[ijk_rh_inv]
    lcpk[ijk_rh_inv] = lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    icpk[ijk_rh_inv] = lhi[ijk_rh_inv] / cvm[ijk_rh_inv]
    tcpk[ijk_rh_inv] = lcpk[ijk_rh_inv] + icpk[ijk_rh_inv]
    
    # Sublimation / deposition of ice
    ijk_tz = ijk_rh_inv & (tz < tice)
    
    qsi   = np.empty_like(tz)
    dqsdt = np.empty_like(tz)
    dq    = np.empty_like(tz)
    
    qsi[ijk_tz], dqsdt[ijk_tz] = iqs2(tz[ijk_tz], den[ijk_tz])
    
    dq[ijk_tz]   = qv[ijk_tz] - qsi[ijk_tz]
    sink[ijk_tz] = dq[ijk_tz] / (1. + tcpk[ijk_tz] * dqsdt[ijk_tz])
    
    ijk_qi = ijk_tz & (qi > qrmin)
    
    # - Eq 9, hong et al. 2004, mwr
    # - For a and b, see dudhia 1989: page 3103 eq (b7) and (b8)
    pidep = np.empty_like(tz)
    
    pidep[ijk_qi] = dt_pisub * dq[ijk_qi] * 349138.78 * \
                    np.exp(0.875 * np.log(qi[ijk_qi] * den[ijk_qi])) / \
                    ( qsi[ijk_qi] * den[ijk_qi] * lat2 / \
                      (0.0243 * rvgas * tz[ijk_qi]**2) + 4.42478e4 )
    
    ijk_qi_inv = ijk_tz & (qi <= qrmin)
    
    pidep[ijk_qi_inv] = 0.
    
    ijk_dq = ijk_tz & (dq > 0.) # Vapor -- > ice
    
    tmp = np.empty_like(tz)
    
    tmp[ijk_dq] = tice - tz[ijk_dq]
    
    # The following should produce more ice at higher altitude
    # qi_crt = 4.92e-11 * exp (1.33 * log (1.e3 * exp (0.1 * tmp))) / den (k)
    qi_crt         = qi_gen * np.minimum(qi_lim, 0.1 * tmp[ijk_dq]) / den[ijk_dq]
    sink[ijk_dq]   = np.minimum( sink[ijk_dq], 
                                 np.minimum( np.maximum(qi_crt - qi[ijk_dq], pidep[ijk_dq]), 
                                             tmp[ijk_dq] / tcpk[ijk_dq] ) )
    
    ijk_dq_inv = ijk_tz & (dq <= 0.)    # Ice -- > vapor
    
    pidep[ijk_dq_inv] = pidep[ijk_dq_inv] * np.minimum(1., np.where(t_sub < tz[ijk_dq_inv], tz[ijk_dq_inv]-t_sub, 0) * 0.2)
    sink[ijk_dq_inv]  = np.maximum( pidep[ijk_dq_inv], 
                                    np.maximum(sink[ijk_dq_inv], -qi[ijk_dq_inv]) )
    
    qv[ijk_tz]    = qv[ijk_tz] - sink[ijk_tz]
    qi[ijk_tz]    = qi[ijk_tz] + sink[ijk_tz]
    q_sol[ijk_tz] = q_sol[ijk_tz] + sink[ijk_tz]
    cvm[ijk_tz]   = c_air + qv[ijk_tz] * c_vap + q_liq[ijk_tz] * c_liq + q_sol[ijk_tz] * c_ice
    tz[ijk_tz]    = tz[ijk_tz] + sink[ijk_tz] * (lhl[ijk_tz] + lhi[ijk_tz]) / cvm[ijk_tz]
    
    # Update capacity heat and latend heat coefficient
    lhl[ijk_rh_inv]  = lv00 + d0_vap * tz[ijk_rh_inv]
    lhi[ijk_rh_inv]  = li00 + dc_ice * tz[ijk_rh_inv]
    lcpk[ijk_rh_inv] = lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    icpk[ijk_rh_inv] = lhi[ijk_rh_inv] / cvm[ijk_rh_inv]
    tcpk[ijk_rh_inv] = lcpk[ijk_rh_inv] + icpk[ijk_rh_inv]
    
    # - Sublimation / deposition of snow
    # - This process happens for the whole temperature range
    ijk_qs = ijk_rh_inv & (qs > qrmin)
    
    pssub = np.empty_like(den)
    
    qsi[ijk_qs], dqsdt[ijk_qs] = iqs2(tz[ijk_qs], den[ijk_qs])
    
    qden          = qs[ijk_qs] * den[ijk_qs]
    tmp[ijk_qs]   = np.exp(0.65625 * np.log(qden))
    tsq           = tz[ijk_qs] * tz[ijk_qs]
    dq[ijk_qs]    = (qsi[ijk_qs] - qv[ijk_qs]) / (1. + tcpk[ijk_qs] * dqsdt[ijk_qs])
    pssub[ijk_qs] = cssub[0] * tsq * \
                    ( cssub[1] * np.sqrt(qden) + cssub[2] * tmp[ijk_qs] * np.sqrt(denfac[ijk_qs]) ) / \
                    ( cssub[3] * tsq + cssub[4] * qsi[ijk_qs] * den[ijk_qs] )
    pssub[ijk_qs] = (qsi[ijk_qs] - qv[ijk_qs]) * dts * pssub[ijk_qs]
    
    ijk_ps     = ijk_qs & (pssub > 0.)  # qs -- > qv, sublimation
    ijk_ps_inv = ijk_qs & (pssub <= 0.)
    
    pssub[ijk_ps] = np.minimum( pssub[ijk_ps] * \
                                np.minimum( 1., np.where(t_sub < tz[ijk_ps], tz[ijk_ps]-t_sub, 0) * 0.2), 
                                qs[ijk_ps] )
    
    ijk_tz = ijk_ps_inv & (tz > tice)
    
    pssub[ijk_tz] = 0.  # No deposition
    
    ijk_tz_inv = ijk_ps_inv & (tz <= tice)
    
    pssub[ijk_tz_inv] = np.maximum( pssub[ijk_tz_inv], 
                                    np.maximum( dq[ijk_tz_inv], 
                                                (tz[ijk_tz_inv] - tice) / tcpk[ijk_tz_inv] ) 
                        )
    
    qs[ijk_qs]    = qs[ijk_qs] - pssub[ijk_qs]
    qv[ijk_qs]    = qv[ijk_qs] + pssub[ijk_qs]
    q_sol[ijk_qs] = q_sol[ijk_qs] - pssub[ijk_qs]
    cvm[ijk_qs]   = c_air + qv[ijk_qs] * c_vap + q_liq[ijk_qs] * c_liq + q_sol[ijk_qs] * c_ice
    tz[ijk_qs]    = tz[ijk_qs] - pssub[ijk_qs] * (lhl[ijk_qs] + lhi[ijk_qs]) / cvm[ijk_qs]
    
    # Update capacity heat and latent heat coefficient
    lhl[ijk_rh_inv]  = lv00 + d0_vap * tz[ijk_rh_inv]
    lhi[ijk_rh_inv]  = li00 + dc_ice * tz[ijk_rh_inv]
    lcpk[ijk_rh_inv] = lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    icpk[ijk_rh_inv] = lhi[ijk_rh_inv] / cvm[ijk_rh_inv]
    tcpk[ijk_rh_inv] = lcpk[ijk_rh_inv] + icpk[ijk_rh_inv]
    
    # Simplified 2-way grapuel sublimation-deposition mechanism
    ijk_qg = ijk_rh_inv & (qg > qrmin)
    
    pgsub = np.empty_like(qv)
    
    qsi[ijk_qg], dqsdt[ijk_qg] = iqs2(tz[ijk_qg], den[ijk_qg])
    
    dq[ijk_qg]    = (qv[ijk_qg] - qsi[ijk_qg]) / (1. + tcpk[ijk_qg] * dqsdt[ijk_qg])
    pgsub[ijk_qg] = (qv[ijk_qg] / qsi[ijk_qg] - 1.) * qg[ijk_qg]
    
    ijk_pg     = ijk_qg & (pgsub > 0.)  # Deposition
    ijk_pg_inv = ijk_qg & (pgsub <= 0.) # Sublimation
    
    ijk_tz = ijk_pg & (tz > tice)
    
    pgsub[ijk_tz] = 0.
    
    ijk_tz_inv = ijk_pg & (tz <= tice)
    
    pgsub[ijk_tz_inv] = np.minimum( 
                            np.minimum(fac_v2g * pgsub[ijk_tz_inv], 0.2 * dq[ijk_tz_inv]), 
                            np.minimum(ql[ijk_tz_inv] + qr[ijk_tz_inv], (tice - tz[ijk_tz_inv]) / tcpk[ijk_tz_inv])
                        )
    
    pgsub[ijk_pg_inv] = np.maximum(fac_g2v * pgsub[ijk_pg_inv], dq[ijk_pg_inv]) * \
                        np.minimum(1., np.where(t_sub < tz[ijk_pg_inv], tz[ijk_pg_inv]-t_sub, 0) * 0.1)
    
    qg[ijk_qg]    = qg[ijk_qg] + pgsub[ijk_qg]
    qv[ijk_qg]    = qv[ijk_qg] - pgsub[ijk_qg]
    q_sol[ijk_qg] = q_sol[ijk_qg] + pgsub[ijk_qg]
    cvm[ijk_qg]   = c_air + qv[ijk_qg] * c_vap + q_liq[ijk_qg] * c_liq + q_sol[ijk_qg] * c_ice
    tz[ijk_qg]    = tz[ijk_qg] + pgsub[ijk_qg] * (lhl[ijk_qg] + lhi[ijk_qg]) / cvm[ijk_qg]
    
    '''
    USE_MIN_EVAP
    '''
    # Update capacity heat and latend heat coefficient
    lhl[ijk_rh_inv]  = lv00 + d0_vap * tz[ijk_rh_inv]
    lcpk[ijk_rh_inv] = lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    
    # Minimum evap of rain in dry environmental air
    ijk_qr = ijk_rh_inv & (qr > qcmin)
    
    rh_rain3d = np.repeat(rh_rain[:, :, np.newaxis], kbot-ktop+1, axis=2)
    
    qsw[ijk_qr], dqsdt[ijk_qr] = wqs2(tz[ijk_qr], den[ijk_qr])
    
    sink[ijk_qr]  = np.minimum( qr[ijk_qr], 
                                np.where( qv[ijk_qr] < rh_rain3d[ijk_qr] * qsw[ijk_qr], 
                                          (rh_rain3d[ijk_qr] * qsw[ijk_qr]) - qv[ijk_qr], 
                                          0 ) / (1. + lcpk[ijk_qr] * dqsdt[ijk_qr]) )
    qv[ijk_qr]    = qv[ijk_qr] + sink[ijk_qr]
    qr[ijk_qr]    = qr[ijk_qr] - sink[ijk_qr]
    q_liq[ijk_qr] = q_liq[ijk_qr] - sink[ijk_qr]
    cvm[ijk_qr]   = c_air + qv[ijk_qr] * c_vap + q_liq[ijk_qr] * c_liq + q_sol[ijk_qr] * c_ice
    tz[ijk_qr]    = tz[ijk_qr] - sink[ijk_qr] * lhl[ijk_qr] / cvm[ijk_qr]
    '''
    END USE_MIN_EVAP
    '''
    
    # Update capacity heat and latent heat coefficient
    lhl[ijk_rh_inv]  = lv00 + d0_vap * tz[ijk_rh_inv]
    cvm[ijk_rh_inv]  = c_air + (qv[ijk_rh_inv] + q_liq[ijk_rh_inv] + q_sol[ijk_rh_inv]) * c_vap
    lcpk[ijk_rh_inv] = lhl[ijk_rh_inv] / cvm[ijk_rh_inv]
    
    # Compute cloud fraction        
    # Combine water species
    '''
    NOTE: Here we could simply return to end the function, but in gt4py 
          this would cause the stencil to be split
    '''
    ijk_do = ijk_rh_inv & (not do_qa)
    
    if rad_snow:
        q_sol[ijk_do] = qi[ijk_do] + qs[ijk_do]
    else:
        q_sol[ijk_do] = qi[ijk_do]
        
    if rad_rain:
        q_liq[ijk_do] = ql[ijk_do] + qr[ijk_do]
    else:
        q_liq[ijk_do] = ql[ijk_do]
    
    q_cond = np.empty_like(q_liq)
    
    q_cond[ijk_do] = q_liq[ijk_do] + q_sol[ijk_do]
    
    qpz[ijk_do] = qv[ijk_do] + q_cond[ijk_do]   # qpz is conserved
    
    # Use the "liquid - frozen water temperature" (tin) to compute saturated specific humidity
    tin[ijk_do] = tz[ijk_do] - ( lcpk[ijk_do] * q_cond[ijk_do] + \
                                 icpk[ijk_do] * q_sol[ijk_do] ) # Minimum temperature
    
    # Determine saturated specific humidity
    ijk_tin = ijk_do & (tin <= t_wfr)
    
    # Ice phase
    qstar = np.empty_like(den)
    
    qstar[ijk_tin] = iqs1(tin[ijk_tin], den[ijk_tin])
    
    ijk_tin_alt = ijk_do & (tin > t_wfr) & (tin >= tice)
    
    # Liquid phase
    qstar[ijk_tin_alt] = wqs1(tin[ijk_tin_alt], den[ijk_tin_alt])
    
    ijk_tin_inv = ijk_do & (tin > t_wfr) & (tin < tice)
    
    # Mixed phase
    qsi[ijk_tin_inv] = iqs1(tin[ijk_tin_inv], den[ijk_tin_inv])
    qsw[ijk_tin_inv] = wqs1(tin[ijk_tin_inv], den[ijk_tin_inv])
    
    ijk_q = ijk_tin_inv & (q_cond > 3.e-6)
    
    rqi = np.empty_like(q_sol)
    
    rqi[ijk_q] = q_sol[ijk_q] / q_cond[ijk_q]
    
    ijk_q_inv = ijk_tin_inv & (q_cond <= 3.e-6)
    
    # Mostly liquid water q_cond (k) at initial cloud development stage
    rqi[ijk_q_inv] = (tice - tin[ijk_q_inv]) / (tice - t_wfr)
    
    qstar[ijk_tin_inv] = rqi[ijk_tin_inv] * qsi[ijk_tin_inv] + \
                         (1. - rqi[ijk_tin_inv]) * qsw[ijk_tin_inv]
                         
    # Assuming subgrid linear distribution in horizontal; this is 
    # effectively a smoother for the binary cloud scheme
    ijk_qpz = ijk_do & (qpz > qrmin)
    
    # Partial cloudiness by pdf
    h_var3d = np.repeat(h_var[:, :, np.newaxis], kbot-ktop+1, axis=2)
    q_plus  = np.empty_like(qpz)
    q_minus = np.empty_like(qpz)
    
    dq[ijk_qpz]      = np.maximum(qcmin, h_var3d[ijk_qpz] * qpz[ijk_qpz])
    q_plus[ijk_qpz]  = qpz[ijk_qpz] + dq[ijk_qpz]   # Cloud free if qstar > q_plus
    q_minus[ijk_qpz] = qpz[ijk_qpz] - dq[ijk_qpz]
    
    ijk_qst = ijk_qpz & (qstar < q_minus)
    
    qa[ijk_qst] = qa[ijk_qst] + 1.  # Air fully saturated; 100% cloud cover
    
    ijk_qst_alt = ijk_qpz & (qstar >= q_minus) & (qstar < q_plus) & (q_cond > qc_crt)
    
    qa[ijk_qst_alt] = qa[ijk_qst_alt] + \
                      ( q_plus[ijk_qst_alt] - qstar[ijk_qst_alt] ) / \
                      ( dq[ijk_qst_alt] + dq[ijk_qst_alt] )   # Partial cloud cover


# Compute the terminal fall speed (it considers cloud ice, snow, and 
# graupel's melting during fall)
def terminal_fall( dtm, ktop, kbot, tz, qv, ql, qr, qg, qs, qi, dz, dp, 
                   den, vtg, vts, vti, r1, g1, s1, i1, m1_sol, w1 ):
    
    ktop1 = ktop + 1
    kbot1 = kbot + 1
    
    dt5      = 0.5 * dtm
    fac_imlt = 1. - np.exp(-dt5 / tau_imlt)
    
    # Define heat capacity and latent heat coefficient
    m1_sol[...] = 0.
    
    lhl   = lv00 + d0_vap * tz
    lhi   = li00 + dc_ice * tz
    q_liq = ql + qr
    q_sol = qi + qs + qg
    cvm   = c_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
    lcpk  = lhl / cvm
    icpk  = lhi / cvm
    
    '''
    NOTE: This needs to be done in a forward computation block
    NOTE: The first index k where stop_k[:, :, k] is equal to True is 
          equal to k0. This is needed if working with array programming.
    '''
    # Find significant melting level
    k0     = np.full(qi.shape[:2], kbot)
    stop_k = np.zeros_like(qi, dtype=bool)
    
    for k in range(ktop, kbot):
        ij = (~stop_k[:, :, k]) & (tz[:, :, k] > tice)
        
        k0[ij] = k
        
        stop_k[:, :, k:][ij] = True
    
    stop_k[:, :, kbot] = True
        
    # Melting of cloud ice (before fall)
    tc = np.empty_like(tz)
    
    tc[stop_k] = tz[stop_k] - tice
    
    ijk = stop_k & (qi > qcmin) & (tc > 0.)
    
    sink = np.empty_like(qi)
    tmp  = np.empty_like(qi)
    
    sink[ijk]  = np.minimum(qi[ijk], fac_imlt * tc[ijk] / icpk[ijk])
    '''
    NOTE: The where functionality is used to compute the positive difference (dim() in Fortran)
    '''
    tmp[ijk]   = np.minimum(sink[ijk], np.where(ql[ijk] < ql_mlt, ql_mlt-ql[ijk], 0))
    ql[ijk]    = ql[ijk] + tmp[ijk]
    qr[ijk]    = qr[ijk] + sink[ijk] - tmp[ijk]
    qi[ijk]    = qi[ijk] - sink[ijk]
    q_liq[ijk] = q_liq[ijk] + sink[ijk]
    q_sol[ijk] = q_sol[ijk] - sink[ijk]
    cvm[ijk]   = c_air + qv[ijk] * c_vap + q_liq[ijk] * c_liq + q_sol[ijk] * c_ice
    tz[ijk]    = tz[ijk] - sink[ijk] * lhi[ijk] / cvm[ijk]
    tc[ijk]    = tz[ijk] - tice
    
    # Turn off melting when cloud microphysics time step is small
    if dtm < 60.:
        k0 = kbot
        stop_k[:, :, ktop:kbot] = False
        
    # sjl, turn off melting of falling cloud ice, snow and graupel
    k0 = kbot
    stop_k[:, :, ktop:kbot] = False
    
    zs = 0.
    ze = np.empty_like(dz)
    
    ze_kbot1 = zs
    
    ze[:, :, kbot] = ze_kbot1 - dz[:, :, kbot]
    for k in range(kbot-1, ktop-1, -1):
        ze[:, :, k] = ze[:, :, k+1] - dz[:, :, k]   # dz < 0
    
    zt = np.empty_like(dz)
        
    zt[:, :, ktop] = ze[:, :, ktop]
    
    # Update capacity heat and latent heat coefficient
    lhi[stop_k]  = li00 + dc_ice * tz[stop_k]
    icpk[stop_k] = lhi[stop_k] / cvm[stop_k]
    
    # Melting of falling cloud ice into rain
    no_fall = check_column(ktop, kbot, qi)
    
    ij_vi = (vi_fac < 1.e-5) | no_fall
    
    i1[ij_vi] = 0.
    
    ij_vi_inv = ~ij_vi
    
    zt[:, :, ktop1:kbot1][ij_vi_inv] = ze[:, :, ktop1:kbot1][ij_vi_inv] - \
                                       dt5 * ( vti[:, :, ktop:kbot][ij_vi_inv] + \
                                               vti[:, :, ktop1:kbot1][ij_vi_inv] )
                                                
    zt_kbot1 = np.empty_like(zt[:, :, kbot])
    
    zt_kbot1[ij_vi_inv] = zs - dtm * vti[:, :, kbot][ij_vi_inv]
    
    for k in range(ktop, kbot):
        
        ij_zt = ij_vi_inv & (zt[:, :, k+1] >= zt[:, :, k])
        
        zt[:, :, k+1][ij_zt] = zt[:, :, k][ij_zt] - dz_min
    
    ij_zt = ij_vi_inv & (zt_kbot1 >= zt[:, :, kbot])
    
    zt_kbot1[ij_zt] = zt[:, :, kbot][ij_zt] - dz_min
    
    """
    WARNING: NEVER CALLED KEEP EMPTY FOR NOW
    if (k0 < kbot) then
        do k = kbot - 1, k0, - 1
            if (qi (k) > qrmin) then
                do m = k + 1, kbot
                    if (zt (k + 1) >= ze (m)) exit
                    if (zt (k) < ze (m + 1) .and. tz (m) > tice) then
                        dtime = min (1.0, (ze (m) - ze (m + 1)) / (max (vr_min, vti (k)) * tau_imlt))
                        sink = min (qi (k) * dp (k) / dp (m), dtime * (tz (m) - tice) / icpk (m))
                        tmp = min (sink, dim (ql_mlt, ql (m)))
                        ql (m) = ql (m) + tmp
                        qr (m) = qr (m) - tmp + sink
                        tz (m) = tz (m) - sink * icpk (m)
                        qi (k) = qi (k) - sink * dp (m) / dp (k)
                    endif
                enddo
            endif
        enddo
    endif
    """
    
    if do_sedi_w:
        
        dm = np.empty_like(dp)
        
        dm[ij_vi_inv] = dp[ij_vi_inv] * \
                        ( 1. + qv[ij_vi_inv] + ql[ij_vi_inv] + qr[ij_vi_inv] + \
                          qi[ij_vi_inv] + qs[ij_vi_inv] + qg[ij_vi_inv] )
                          
    if use_ppm:
        lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qi, 
                             i1, m1_sol, mono_prof, ij_vi_inv )
    else:
        implicit_fall( dtm, ktop, kbot, ze, ze_kbot1, vti, dp, qi, i1, 
                       m1_sol, ij_vi_inv )
                       
    if do_sedi_w:
        
        w1[:, :, ktop][ij_vi_inv] = ( dm[:, :, ktop][ij_vi_inv] * w1[:, :, ktop][ij_vi_inv] + \
                                      m1_sol[:, :, ktop][ij_vi_inv] * vti[:, :, ktop][ij_vi_inv] ) / \
                                    ( dm[:, :, ktop][ij_vi_inv] - m1_sol[:, :, ktop][ij_vi_inv] )
        
        w1[:, :, ktop1:kbot1][ij_vi_inv] = ( dm[:, :, ktop1:kbot1][ij_vi_inv] * \
                                             w1[:, :, ktop1:kbot1][ij_vi_inv] - \
                                             m1_sol[:, :, ktop:kbot][ij_vi_inv] * \
                                             vti[:, :, ktop:kbot][ij_vi_inv] + \
                                             m1_sol[:, :, ktop1:kbot1][ij_vi_inv] * \
                                             vti[:, :, ktop1:kbot1][ij_vi_inv] ) / \
                                           ( dm[:, :, ktop1:kbot1][ij_vi_inv] + \
                                             m1_sol[:, :, ktop:kbot][ij_vi_inv] - \
                                             m1_sol[:, :, ktop1:kbot1][ij_vi_inv] )
    
    # Melting of falling snow into rain
    r1[...] = 0.
    
    no_fall = check_column(ktop, kbot, qs)
    
    s1[no_fall] = 0.
    
    nf_inv = ~no_fall
    
    zt[:, :, ktop1:kbot1][nf_inv] = ze[:, :, ktop1:kbot1][nf_inv] - \
                                    dt5 * ( vts[:, :, ktop:kbot][nf_inv] + \
                                            vts[:, :, ktop1:kbot1][nf_inv] )
    
    zt_kbot1[nf_inv] = zs - dtm * vts[:, :, kbot][nf_inv]
    
    for k in range(ktop, kbot):
        
        ij_zt = nf_inv & (zt[:, :, k+1] >= zt[:, :, k])
        
        zt[:, :, k+1][ij_zt] = zt[:, :, k][ij_zt] - dz_min
    
    ij_zt = nf_inv & (zt_kbot1 >= zt[:, :, kbot])
    
    zt_kbot1[ij_zt] = zt[:, :, kbot][ij_zt] - dz_min
    
    """
    WARNING: NEVER CALLED KEEP EMPTY FOR NOW
    if (k0 < kbot) then
        do k = kbot - 1, k0, - 1
            if (qs (k) > qrmin) then
                do m = k + 1, kbot
                    if (zt (k + 1) >= ze (m)) exit
                    dtime = min (dtm, (ze (m) - ze (m + 1)) / (vr_min + vts (k)))
                    if (zt (k) < ze (m + 1) .and. tz (m) > tice) then
                        dtime = min (1.0, dtime / tau_smlt)
                        sink = min (qs (k) * dp (k) / dp (m), dtime * (tz (m) - tice) / icpk (m))
                        tz (m) = tz (m) - sink * icpk (m)
                        qs (k) = qs (k) - sink * dp (m) / dp (k)
                        if (zt (k) < zs) then
                            r1 = r1 + sink * dp (m) ! precip as rain
                        else
                            ! qr source here will fall next time step (therefore, can evap)
                            qr (m) = qr (m) + sink
                        endif
                    endif
                    if (qs (k) < qrmin) exit
                enddo
            endif
        enddo
    endif
    """
    
    if do_sedi_w:
        
        dm[nf_inv] = dp[nf_inv] * \
                     ( 1. + qv[nf_inv] + ql[nf_inv] + qr[nf_inv] + \
                       qi[nf_inv] + qs[nf_inv] + qg[nf_inv] )
    
    m1 = np.empty_like(m1_sol)
                          
    if use_ppm:
        lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qs, 
                             s1, m1, mono_prof, nf_inv )
    else:
        implicit_fall( dtm, ktop, kbot, ze, ze_kbot1, vts, dp, qs, s1, 
                       m1, nf_inv )
                       
    m1_sol[nf_inv] = m1_sol[nf_inv] + m1[nf_inv]
    
    if do_sedi_w:
        
        w1[:, :, ktop][nf_inv] = ( dm[:, :, ktop][nf_inv] * w1[:, :, ktop][nf_inv] + \
                                   m1[:, :, ktop][nf_inv] * vts[:, :, ktop][nf_inv] ) / \
                                 ( dm[:, :, ktop][nf_inv] - m1[:, :, ktop][nf_inv] )
        
        w1[:, :, ktop1:kbot1][nf_inv] = ( dm[:, :, ktop1:kbot1][nf_inv] * \
                                          w1[:, :, ktop1:kbot1][nf_inv] - \
                                          m1[:, :, ktop:kbot][nf_inv] * \
                                          vts[:, :, ktop:kbot][nf_inv] + \
                                          m1[:, :, ktop1:kbot1][nf_inv] * \
                                          vts[:, :, ktop1:kbot1][nf_inv] ) / \
                                        ( dm[:, :, ktop1:kbot1][nf_inv] + \
                                          m1[:, :, ktop:kbot][nf_inv] - \
                                          m1[:, :, ktop1:kbot1][nf_inv] )
    
    # Melting of falling graupel into rain
    no_fall = check_column(ktop, kbot, qg)
    
    g1[no_fall] = 0.
    
    nf_inv = ~no_fall
    
    zt[:, :, ktop1:kbot1][nf_inv] = ze[:, :, ktop1:kbot1][nf_inv] - \
                                    dt5 * ( vtg[:, :, ktop:kbot][nf_inv] + \
                                            vtg[:, :, ktop1:kbot1][nf_inv] )
    
    zt_kbot1[nf_inv] = zs - dtm * vtg[:, :, kbot][nf_inv]
    
    for k in range(ktop, kbot):
        
        ij_zt = nf_inv & (zt[:, :, k+1] >= zt[:, :, k])
        
        zt[:, :, k+1][ij_zt] = zt[:, :, k][ij_zt] - dz_min
    
    ij_zt = nf_inv & (zt_kbot1 >= zt[:, :, kbot])
    
    zt_kbot1[ij_zt] = zt[:, :, kbot][ij_zt] - dz_min
    
    """
    WARNING: NEVER CALLED KEEP EMPTY FOR NOW
    if (k0 < kbot) then
        do k = kbot - 1, k0, - 1
            if (qg (k) > qrmin) then
                do m = k + 1, kbot
                    if (zt (k + 1) >= ze (m)) exit
                    dtime = min (dtm, (ze (m) - ze (m + 1)) / vtg (k))
                    if (zt (k) < ze (m + 1) .and. tz (m) > tice) then
                        dtime = min (1., dtime / tau_g2r)
                        sink = min (qg (k) * dp (k) / dp (m), dtime * (tz (m) - tice) / icpk (m))
                        tz (m) = tz (m) - sink * icpk (m)
                        qg (k) = qg (k) - sink * dp (m) / dp (k)
                        if (zt (k) < zs) then
                            r1 = r1 + sink * dp (m)
                        else
                            qr (m) = qr (m) + sink
                        endif
                    endif
                    if (qg (k) < qrmin) exit
                enddo
            endif
        enddo
    endif
    """
    
    if do_sedi_w:
        
        dm[nf_inv] = dp[nf_inv] * \
                     ( 1. + qv[nf_inv] + ql[nf_inv] + qr[nf_inv] + \
                       qi[nf_inv] + qs[nf_inv] + qg[nf_inv] )
                          
    if use_ppm:
        lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qg, 
                             g1, m1, mono_prof, nf_inv )
    else:
        implicit_fall( dtm, ktop, kbot, ze, ze_kbot1, vtg, dp, qg, g1, 
                       m1, nf_inv )
                       
    m1_sol[nf_inv] = m1_sol[nf_inv] + m1[nf_inv]
    
    if do_sedi_w:
        
        w1[:, :, ktop][nf_inv] = ( dm[:, :, ktop][nf_inv] * w1[:, :, ktop][nf_inv] + \
                                   m1[:, :, ktop][nf_inv] * vtg[:, :, ktop][nf_inv] ) / \
                                 ( dm[:, :, ktop][nf_inv] - m1[:, :, ktop][nf_inv] )
        
        w1[:, :, ktop1:kbot1][nf_inv] = ( dm[:, :, ktop1:kbot1][nf_inv] * \
                                          w1[:, :, ktop1:kbot1][nf_inv] - \
                                          m1[:, :, ktop:kbot][nf_inv] * \
                                          vtg[:, :, ktop:kbot][nf_inv] + \
                                          m1[:, :, ktop1:kbot1][nf_inv] * \
                                          vtg[:, :, ktop1:kbot1][nf_inv] ) / \
                                        ( dm[:, :, ktop1:kbot1][nf_inv] + \
                                          m1[:, :, ktop:kbot][nf_inv] - \
                                          m1[:, :, ktop1:kbot1][nf_inv] )


# Checks if the water species are large enough to fall
def check_column(ktop, kbot, q):
    
    return ~np.any(q > qrmin, axis=2)


# Compute the time-implicit monotonic scheme
def implicit_fall( dt, ktop, kbot, ze, ze_kbot1, vt, dp, q, precip, m1, 
                   condition=None ):
    
    # Check if there is a condition, otherwise fill condition with True
    c = check_condition(condition, q.shape[:2])
    
    ktop1 = ktop + 1
    kbot1 = kbot + 1
    
    dz = np.empty_like(q)
    dd = np.empty_like(vt)
    
    dz[:, :, ktop:kbot][c] = ze[:, :, ktop:kbot][c] - ze[:, :, ktop1:kbot1][c]
    dz[:, :, kbot][c]      = ze[:, :, kbot][c] - ze_kbot1
    dd[c]                  = dt * vt[c]
    q[c]                   = q[c] * dp[c]
    
    # Sedimentation: non-vectorizable loop
    qm = np.empty_like(q)
    
    qm[:, :, ktop][c] = q[:, :, ktop][c] / (dz[:, :, ktop][c] + dd[:, :, ktop][c])
    
    for k in range(ktop1, kbot1):
        qm[:, :, k][c] = (q[:, :, k][c] + dd[:, :, k-1][c] * qm[:, :, k-1][c]) / \
                         (dz[:, :, k][c] + dd[:, :, k][c])
        
    # qm is density at this stage
    qm[c] = qm[c] * dz[c]
    
    # Output mass fluxes: non-vectorizable loop
    m1[:, :, ktop][c] = q[:, :, ktop][c] - qm[:, :, ktop][c]
    for k in range(ktop1, kbot1):
        m1[:, :, k][c] = m1[:, :, k-1][c] + q[:, :, k][c] - qm[:, :, k][c]
    
    precip[c] = m1[:, :, kbot][c]
    
    # Update
    q[c] = qm[c] / dp[c]


# Lagrangian scheme
def lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, q, 
                         precip, m1, mono, condition=None ):
    """
    WARNING: NEVER CALLED KEEP EMPTY FOR NOW
    # Check if there is a condition, otherwise fill condition with True
    c = check_condition(condition, q.shape[:2])
    
    ktop1 = ktop + 1
    kbot1 = kbot + 1
                             
    # Density
    dz = np.empty_like(q)
    a1 = np.empty_like(q)
    a2 = np.empty_like(q)
    a3 = np.empty_like(q)
    a4 = np.empty_like(q)
    qm = np.empty_like(q)
    
    dz[:, :, ktop:kbot][c] = zt[:, :, ktop:kbot][c] - zt[:, :, ktop1:kbot1][c]  # dz is positive
    dz[:, :, kbot][c]      = zt[:, :, kbot][c] - zt_kbot1[c]
    q[c]                   = q[c] * dp[c]
    '''
    NOTE: Here we split the array a4 into its 4 components, that are now 
          3D, due to gt4py requirements
    '''
    a1[c]                  = q[c] / dz[c]
    qm[c]                  = 0.
    
    # Construct vertical profile with zt as coordinate
    # ~ cs_profile (a4 (1, ktop), dz (ktop), kbot - ktop + 1, mono)
    
    '''
    NOTE: This is a real mess, even for gt4py imo
    '''
    # ~ k0 = ktop
    #for k in range(ktop, kbot1):
    #    for n in range(k0, kbot1):
    # ~ for k in range(ktop, kbot):
        # ~ for n in range(k0, kbot):
            # ~ ij = c & (ze[:, :, k] <= zt[:, :, n]) & (ze[:, :, k] >= zt[:, :, n+1])
            
            # ~ pl = np.empty_like(dz[:, :, ktop])
            
            # ~ pl[ij] = (zt[:, :, n][ij] - ze[:, :, k][ij]) / dz[:, :, n][ij]
            
            # ~ ij_zt = ij & (zt[:, :, n+1] <= ze[:, :, k+1])
            
            # ~ # Entire new grid is within the original grid
            # ~ pr = (zt[:, :, n][ij_zt] - ze[:, :, k+1][ij_zt]) / dz[:, :, n][ij_zt]
            
            # ~ qm[:, :, k][ij_zt] = a2[:, :, n][ij_zt] + 0.5 * \
                                 # ~ ( a4[:, :, n][ij_zt] + a3[:, :, n][ij_zt] - \
                                   # ~ a2[:, :, n][ij_zt] ) * \
                                 # ~ ( pr + pl[ij_zt] ) - 
                                 # ~ a4[:, :, n][ij_zt] * r3 * \
                                 # ~ ( pr * (pr + pl[ij_zt]) + pl[ij_zt]**2 )
            # ~ qm[:, :, k][ij_zt] = qm[:, :, k][ij_zt] * (ze[:, :, k][ij_zt] - ze[:, :, k+1][ij_zt])
            
            # ~ k0 = n
            
            
    
    m1[:, :, ktop][c]        = q[:, :, ktop][c] - qm[:, :, ktop][c]
    
    for k in range(ktop1, kbot1):
        m1[:, :, k][c] = m1[:, :, k-1][c] + q[:, :, k][c] - qm[:, :, k][c]
        
    precip[c] = m1[:, :, kbot][c]
    
    # - Convert back to * dry * mixing ratio
    # - dp must be dry air_mass (because moist air mass will be changed 
    #   due to terminal fall).
    q[c] = qm[c] / dp[c]
    """
    return


# Calculate the vertical fall speed
def fall_speed(ktop, kbot, den, qs, qi, qg, ql, tk, vts, vti, vtg):
    
    # Marshall-Palmer formula
    
    # Try the local air density -- for global model; the true value 
    # could be much smaller than sfcrho over high mountains
    rhof = np.sqrt(np.minimum(10., sfcrho / den))
    
    # Ice
    if const_vi:
        vti = vi_fac
    else:
        
        # Use deng and mace (2008, grl), which gives smaller fall speed 
        # than hd90 formula
        vi0 = 0.01 * vi_fac
        
        ijk = qi < thi
        
        vti[ijk] = vf_min
        
        ijk_inv = ~ijk
        
        tc           = tk[ijk_inv] - tice
        vti[ijk_inv] = ( 3. + np.log10(qi[ijk_inv] * den[ijk_inv]) ) * \
                       ( tc * (aa * tc + bb) + cc ) + dd * tc + ee
        vti[ijk_inv] = vi0 * np.exp(log_10 * vti[ijk_inv]) * 0.8 
        vti[ijk_inv] = np.minimum(vi_max, np.maximum(vf_min, vti[ijk_inv]))
    
    # Snow
    if const_vs:
        vts = vs_fac
    else:
        
        ijk = qs < ths
        
        vts[ijk] = vf_min
        
        ijk_inv = ~ijk
        
        '''
        NOTE: Computing the exponential and logarithm can be avoided and 
              this will also avoid python "divide by zero in log" 
              warnings
        '''
        vts[ijk_inv] = vs_fac * vcons * rhof[ijk_inv] * np.exp(0.0625 * np.log(qs[ijk_inv] * den[ijk_inv] / norms))
        vts[ijk_inv] = np.minimum(vs_max, np.maximum(vf_min, vts[ijk_inv]))
    
    # Graupel
    if const_vg:
        vtg = vg_fac
    else:
        
        ijk = qg < thg
        
        vtg[ijk] = vf_min
        
        ijk_inv = ~ijk
        
        vtg[ijk_inv] = vg_fac * vcong * rhof[ijk_inv] * \
                       np.sqrt(np.sqrt(np.sqrt(qg[ijk_inv] * den[ijk_inv] / normg)))
        vtg[ijk_inv] = np.minimum(vg_max, np.maximum(vf_min, vtg[ijk_inv]))


# Set up gfdl cloud microphysics parameters
def setupm():
    
    # Global variables
    global fac_rc
    global cracs
    global csacr
    global cgacr
    global cgacs
    global acco
    global csacw
    global csaci
    global cgacw
    global cgaci
    global cracw
    global cssub
    global crevp
    global cgfr
    global csmlt
    global cgmlt
    global ces0
    
    gam263 = 1.456943
    gam275 = 1.608355
    gam290 = 1.827363
    gam325 = 2.54925
    gam350 = 3.323363
    gam380 = 4.694155
    
    # Intercept parameters
    rnzs = 3.0e6
    rnzr = 8.0e6
    rnzg = 4.0e6
    
    # Density parameters
    acc = np.array([5., 2., 0.5])
    
    pie = 4. * mt.atan(1.)
    
    # S. Klein's formular (eq 16) from am2
    fac_rc = (4./3.) * pie * rhor * rthresh**3
    
    vdifu = 2.11e-5
    tcond = 2.36e-2
    
    visk = 1.259e-5
    hlts = 2.8336e6
    hltc = 2.5e6
    hltf = 3.336e5
    
    ch2o = 4.1855e3
    
    pisq = pie * pie
    scm3 = (visk / vdifu)**(1./3.)
    
    cracs = pisq * rnzr * rnzs * rhos
    csacr = pisq * rnzr * rnzs * rhor
    cgacr = pisq * rnzr * rnzg * rhor
    cgacs = pisq * rnzg * rnzs * rhos
    cgacs = cgacs * c_pgacs
    
    act    = np.empty(8)
    act[0] = pie * rnzs * rhos
    act[1] = pie * rnzr * rhor
    act[5] = pie * rnzg * rhog
    act[2] = act[1]
    act[3] = act[0]
    act[4] = act[1]
    act[6] = act[0]
    act[7] = act[5]
    
    acco = np.empty((3,4))
    for i in range(3):
        for k in range(4):
            acco[i, k] = acc[i] / (act[2*k]**((6 - i) * 0.25) * act[2*k + 1]**((i+1) * 0.25))
            
    gcon = 40.74 * mt.sqrt(sfcrho)
    
    # Decreasing csacw to reduce cloud water --- > snow
    csacw = pie * rnzs * clin * gam325 / (4. * act[0]**0.8125)
    
    craci = pie * rnzr * alin * gam380 / (4. * act[1]**0.95)
    csaci = csacw * c_psaci
    
    cgacw = pie * rnzg * gam350 * gcon / (4. * act[5]**0.875)
    
    cgaci = cgacw * 0.05
    
    cracw = craci
    cracw = c_cracw * cracw
    
    # Subl and revap: five constants for three separate processes
    cssub    = np.empty(5)
    cssub[0] = 2. * pie * vdifu * tcond * rvgas * rnzs
    cssub[1] = 0.78 / mt.sqrt(act[0])
    cssub[2] = 0.31 * scm3 * gam263 * mt.sqrt(clin / visk) / act[0]**0.65625
    cssub[3] = tcond * rvgas
    cssub[4] = (hlts**2) * vdifu
    
    cgsub    = np.empty(5)
    cgsub[0] = 2. * pie * vdifu * tcond * rvgas * rnzg
    cgsub[1] = 0.78 / mt.sqrt(act[5])
    cgsub[2] = 0.31 * scm3 * gam275 * mt.sqrt(gcon / visk) / act[5]**0.6875
    cgsub[3] = cssub[3]
    cgsub[4] = cssub[4]
    
    crevp    = np.empty(5)
    crevp[0] = 2. * pie * vdifu * tcond * rvgas * rnzr
    crevp[1] = 0.78 / mt.sqrt(act[1])
    crevp[2] = 0.31 * scm3 * gam290 * mt.sqrt(alin / visk) / act[1]**0.725
    crevp[3] = cssub[3]
    crevp[4] = hltc**2 * vdifu
    
    cgfr     = np.empty(2)
    cgfr[0]  = 20.e2 * pisq * rnzr * rhor / act[1]**1.75
    cgfr[1]  = 0.66
    
    # smlt: five constants (lin et al. 1983)
    csmlt    = np.empty(5)
    csmlt[0] = 2. * pie * tcond * rnzs / hltf
    csmlt[1] = 2. * pie * vdifu * rnzs * hltc / hltf
    csmlt[2] = cssub[1]
    csmlt[3] = cssub[2]
    csmlt[4] = ch2o / hltf
    
    # gmlt: five constants
    cgmlt    = np.empty(5)
    cgmlt[0] = 2. * pie * tcond * rnzg / hltf
    cgmlt[1] = 2. * pie * vdifu * rnzg * hltc / hltf
    cgmlt[2] = cgsub[1]
    cgmlt[3] = cgsub[2]
    cgmlt[4] = ch2o / hltf
    
    es0  = 6.107799961e2    # ~6.1 mb
    ces0 = eps * es0


# Initialization of gfdl cloud microphysics
def gfdl_cloud_microphys_init():
    
    # Global variables
    global log_10
    global tice0
    global t_wfr
    
    global do_setup
    
    if do_setup:
        setupm()
        do_setup = False
    
    log_10 = mt.log(10.)
    tice0  = tice - 0.01
    t_wfr  = tice - 40.0    # Supercooled water can exist down to -48 degrees Celsius, which is the "absolute"


# Accretion function    
def acr3d(v1, v2, q1, q2, c, cac, i, k, rho):
    
    t1 = np.sqrt(q1 * rho)
    s1 = np.sqrt(q2 * rho)
    s2 = np.sqrt(s1)
    
    return c * np.abs(v1 - v2) * q1 * s2 * \
           ( cac[i, k] * t1 + cac[i+1, k] * np.sqrt(t1) * s2 + \
             cac[i+2, k] * s1 )


# Melting of snow function (psacw and psacr must be calc before smlt is 
# called)
def smlt(tc, dqs, qsrho, psacw, psacr, c, rho, rhofac):
    
    return (c[0] * tc / rho - c[1] * dqs) * \
           (c[2] * np.sqrt(qsrho) + c[3] * qsrho**0.65625 * np.sqrt(rhofac)) + \
           c[4] * tc * (psacw + psacr)
           
           
# Melting of graupel function (pgacw and pgacr must be calc before gmlt 
# is called)
def gmlt(tc, dqs, qgrho, pgacw, pgacr, c, rho):
    
    return (c[0] * tc / rho - c[1] * dqs) * \
           (c[2] * np.sqrt(qgrho) + c[3] * qgrho**0.6875 / rho ** 0.25) + \
           c[4] * tc * (pgacw + pgacr)
    

# Compute the saturated specific humidity
def wqs1(ta, den):
    
    return ( e00 * \
             np.exp( ( dc_vap * np.log(ta / t_ice) + \
                       lv0 * (ta - t_ice) / (ta * t_ice) ) / rvgas ) ) / \
           ( rvgas * ta * den )


# Compute saturated specific humidity and its gradient
def wqs2(ta, den):
    
    tmp = wqs1(ta, den)
    
    return tmp, tmp * (dc_vap + lv0 / ta) / (rvgas * ta)
    
    
# Compute the saturated specific humidity  
def iqs1(ta, den):
    
    res = np.empty_like(ta)
    
    # Over ice between -160 degrees Celsius and 0 degrees Celsius        
    ijk = ta < t_ice
    
    ijk_ta = ijk & (ta >= t_ice - 160.)
    
    res[ijk_ta] = ( e00 * \
                    np.exp( ( d2ice * np.log(ta[ijk_ta] / t_ice) + \
                              li2 * (ta[ijk_ta] - t_ice) / (ta[ijk_ta] * t_ice) ) / rvgas ) ) / \
                  ( rvgas * ta[ijk_ta] * den[ijk_ta] )
    
    ijk_ta_inv = ijk & (ta < t_ice - 160.)
    
    res[ijk_ta_inv] = ( e00 * \
                        np.exp( ( d2ice * np.log(1. - 160. / t_ice) - \
                                  li2 * 160. / ((t_ice - 160.) * t_ice) ) / rvgas ) ) / \
                      ( rvgas * (t_ice - 160.) * den[ijk_ta_inv] )
    
    # Over water between 0 degrees Celsius and 102 degrees Celsius
    ijk = ~ijk
    
    ijk_ta = ijk & (ta <= t_ice + 102.)
    
    res[ijk_ta] = wqs1(ta[ijk_ta], den[ijk_ta])
    
    ijk_ta_inv = ijk & (ta > t_ice + 102.)
    
    res[ijk_ta_inv] = wqs1(t_ice + 102., den[ijk_ta_inv])
    
    return res
    

# Compute the gradient of saturated specific humidity
def iqs2(ta, den):
    
    tmp = iqs1(ta, den)
    
    dtmp = np.empty_like(ta)
    
    # Over ice between -160 degrees Celsius and 0 degrees Celsius        
    ijk = ta < t_ice
    
    ijk_ta = ijk & (ta >= t_ice - 160.)
    
    dtmp[ijk_ta] = tmp[ijk_ta] * (d2ice + li2 / ta[ijk_ta]) / (rvgas * ta[ijk_ta])
    
    ijk_ta_inv = ijk & (ta < t_ice - 160.)
    
    dtmp[ijk_ta_inv] = tmp[ijk_ta_inv] * (d2ice + li2 / (t_ice - 160.)) / (rvgas * (t_ice - 160.))
    
    # Over water between 0 degrees Celsius and 102 degrees Celsius
    ijk = ~ijk
    
    ijk_ta = ijk & (ta <= t_ice + 102.)
    
    dtmp[ijk_ta] = tmp[ijk_ta] * (dc_vap + lv0 / ta[ijk_ta]) / (rvgas * ta[ijk_ta])
    
    ijk_ta_inv = ijk & (ta > t_ice + 102.)
    
    dtmp[ijk_ta_inv] = tmp[ijk_ta_inv] * (dc_vap + lv0 / (t_ice + 102.)) / (rvgas * (t_ice + 102.))
    
    return tmp, dtmp
    

# Fix negative water species (designed for 6-class micro-physics schemes)
def neg_adj(ktop, kbot, pt, dp, qv, ql, qr, qi, qs, qg):
    
    '''
    NOTE: Better to just pass these as a arguments
    '''
    global c_air
    global c_vap
    global d0_vap
    global lv00
    
    # Define heat capacity and latent heat coefficient
    cvm  = c_air + qv  * c_vap + (qr + ql) * c_liq + (qi + qs + qg) * c_ice
    lcpk = (lv00 + d0_vap * pt) / cvm
    icpk = (li00 + dc_ice * pt) / cvm
    
    # Ice phase
        
    # If cloud ice < 0, borrow from snow
    ijk = qi < 0.
    
    qs[ijk] = qs[ijk] + qi[ijk]
    qi[ijk] = 0.
    
    # If snow < 0, borrow from graupel
    ijk = qs < 0.
    
    qg[ijk] = qg[ijk] + qs[ijk]
    qs[ijk] = 0.
    
    # If graupel < 0, borrow from rain
    ijk = qg < 0.
    
    qr[ijk] = qr[ijk] + qg[ijk]
    pt[ijk] = pt[ijk] - qg[ijk] * icpk[ijk] # Heating
    qg[ijk] = 0.
        
    # Liquid phase
    
    # If rain < 0, borrow from cloud water
    ijk = qr < 0.
    
    ql[ijk] = ql[ijk] + qr[ijk]
    qr[ijk] = 0.
    
    # If cloud water < 0, borrow from water vapor
    ijk = ql < 0.
    
    qv[ijk] = qv[ijk] + ql[ijk]
    pt[ijk] = pt[ijk] - ql[ijk] * lcpk[ijk] # Heating
    ql[ijk] = 0.
    
    '''
    NOTE: In this case a forward computation block will be needed since 
          the computation has dependencies on values from previous 
          k-levels
    NOTE: ij is a 2D mask, since the condition has to be applied to a specific k-level in this case (non-parallel computation)
    '''
    # Fix water vapor; borrow from below
    for k in range(ktop, kbot):
        
        ij = qv[:, :, k] < 0.
        
        qv[:, :, k+1][ij] = qv[:, :, k+1][ij] + qv[:, :, k][ij] * dp[:, :, k][ij] / dp[:, :, k+1][ij]
        qv[:, :, k][ij]   = 0.
    
    # Bottom layer; borrow from above
    ij = (qv[:, :, kbot] < 0.) & (qv[:, :, kbot-1] > 0.)
    
    '''
    NOTE: dq needs to be allocated first in order to have the right 
          dimensions, otherwise when ij is false no element is put into 
          the list (not really, since it's just use in this if 
          statement)
    NOTE: Would it be better/faster to use np.empty instead of np.empty_like
    '''
    dq = np.minimum(-qv[:, :, kbot][ij] * dp[:, :, kbot][ij], qv[:, :, kbot-1][ij] * dp[:, :, kbot-1][ij])
    
    qv[:, :, kbot-1][ij] = qv[:, :, kbot-1][ij] - dq / dp[:, :, kbot-1][ij]
    qv[:, :, kbot][ij]   = qv[:, :, kbot][ij] + dq / dp[:, :, kbot][ij]


# If there is no condition return an array full of True values
def check_condition(condition, shape):
    
    if condition is None:
        return np.ones(shape, dtype=bool)
    else:
        return condition
