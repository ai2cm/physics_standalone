from microphys.phys_const import *
from microphys.microphys_funcs import *

from utility import *
from utility.ufuncs_gt4py import *

import gt4py as gt
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, BACKWARD, FORWARD, computation, interval


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, debug_mode=DEBUG_MODE)
def fields_init( land   : FIELD_FLT, 
                 area   : FIELD_FLT, 
                 h_var  : FIELD_FLT, 
                 rh_adj : FIELD_FLT, 
                 rh_rain: FIELD_FLT, 
                 graupel: FIELD_FLT, 
                 ice    : FIELD_FLT, 
                 rain   : FIELD_FLT, 
                 snow   : FIELD_FLT, 
                 qa     : FIELD_FLT, 
                 qg     : FIELD_FLT, 
                 qi     : FIELD_FLT, 
                 ql     : FIELD_FLT, 
                 qn     : FIELD_FLT, 
                 qr     : FIELD_FLT, 
                 qs     : FIELD_FLT, 
                 qv     : FIELD_FLT, 
                 pt     : FIELD_FLT, 
                 delp   : FIELD_FLT, 
                 dz     : FIELD_FLT, 
                 qgz    : FIELD_FLT, 
                 qiz    : FIELD_FLT, 
                 qlz    : FIELD_FLT, 
                 qrz    : FIELD_FLT, 
                 qsz    : FIELD_FLT, 
                 qvz    : FIELD_FLT, 
                 tz     : FIELD_FLT, 
                 qi_dt  : FIELD_FLT, 
                 qs_dt  : FIELD_FLT, 
                 uin    : FIELD_FLT, 
                 vin    : FIELD_FLT, 
                 qa0    : FIELD_FLT, 
                 qg0    : FIELD_FLT, 
                 qi0    : FIELD_FLT, 
                 ql0    : FIELD_FLT, 
                 qr0    : FIELD_FLT, 
                 qs0    : FIELD_FLT, 
                 qv0    : FIELD_FLT, 
                 t0     : FIELD_FLT, 
                 dp0    : FIELD_FLT, 
                 den0   : FIELD_FLT, 
                 dz0    : FIELD_FLT, 
                 u0     : FIELD_FLT, 
                 v0     : FIELD_FLT, 
                 dp1    : FIELD_FLT, 
                 p1     : FIELD_FLT, 
                 u1     : FIELD_FLT, 
                 v1     : FIELD_FLT, 
                 ccn    : FIELD_FLT, 
                 c_praut: FIELD_FLT, 
                 use_ccn: DTYPE_INT, 
                 c_air  : DTYPE_FLT, 
                 c_vap  : DTYPE_FLT, 
                 d0_vap : DTYPE_FLT, 
                 lv00   : DTYPE_FLT, 
                 dt_in  : DTYPE_FLT, 
                 rdt    : DTYPE_FLT, 
                 cpaut  : DTYPE_FLT ):
    
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            # Initialize precipitation
            graupel = 0.
            rain    = 0.
            snow    = 0.
            ice     = 0.
            
            # This is to prevent excessive build-up of cloud ice from 
            # external sources
            if de_ice == 1:
                
                qio = qi - dt_in * qi_dt    # Orginal qi before phys
                qin = max(qio, qi0_max)     # Adjusted value
                
                if qi > qin:
                    
                    qs = qs + qi - qin
                    qi = qin
                    
                    dqi   = (qin - qio) * rdt   # Modified qi tendency
                    qs_dt = qs_dt + qi_dt - dqi
                    qi_dt = dqi
            
            qiz = qi
            qsz = qs
            
            t0  = pt
            tz  = t0
            dp1 = delp
            dp0 = dp1   # Moist air mass * grav
            
            # Convert moist mixing ratios to dry mixing ratios
            qvz = qv
            qlz = ql
            qrz = qr
            qgz = qg
            
            dp1 = dp1 * (1. - qvz)
            omq = dp0 / dp1
            
            qvz = qvz * omq
            qlz = qlz * omq
            qrz = qrz * omq
            qiz = qiz * omq
            qsz = qsz * omq
            qgz = qgz * omq
            
            qa0 = qa
            dz0 = dz
            
            den0 = -dp1 / (grav * dz0)  # Density of dry air
            p1   = den0 * rdgas * t0    # Dry air pressure
            
            # Save a copy of old values for computing tendencies
            qv0 = qvz
            ql0 = qlz
            qr0 = qrz
            qi0 = qiz
            qs0 = qsz
            qg0 = qgz
            
            # For sedi_momentum
            u0 = uin
            v0 = vin
            u1 = u0
            v1 = v0
            
            if prog_ccn == 1:
                
                # Convert #/cc to #/m^3
                ccn     = qn * 1.e6
                c_praut = cpaut * (ccn * rhor)**(-1./3.)
                
            else:
                
                ccn = (ccn_l * land + ccn_o * (1. - land)) * 1.e6
                
                if use_ccn == 1:
                    
                    # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                    ccn = ccn * rdgas * tz / p1
                
                c_praut = cpaut * (ccn * rhor)**(-1./3.)
            
            # Calculate horizontal subgrid variability
            # Total water subgrid deviation in horizontal direction
            # Default area dependent form: use dx ~ 100 km as the base
            s_leng_1  = sqrt(sqrt(area * 1.e-10))
            t_land_1  = dw_land * s_leng_1
            t_ocean_1 = dw_ocean * s_leng_1
            h_var     = t_land_1 * land + t_ocean_1 * (1. - land)
            h_var     = min(0.2, max(0.01, h_var))
            
            # Relative humidity increment
            rh_adj  = 1. - h_var - rh_inc
            rh_rain = max(0.35, rh_adj - rh_inr)
            
            # Fix all negative water species
            if fix_negative == 1:
                
                # Define heat capacity and latent heat coefficient
                cvm_1  = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
                lcpk_1 = (lv00 + d0_vap * tz) / cvm_1
                icpk_1 = (li00 + dc_ice * tz) / cvm_1
                
                # Ice phase
                
                # If cloud ice < 0, borrow from snow
                if qiz < 0.:
                    
                    qsz = qsz + qiz
                    qiz = 0.
                
                # If snow < 0, borrow from graupel
                if qsz < 0.:
                    
                    qgz = qgz + qsz
                    qsz = 0.
                
                # If graupel < 0, borrow from rain
                if qgz < 0.:
                    
                    qrz = qrz + qgz
                    tz  = tz - qgz * icpk_1   # Heating
                    qgz = 0.
                    
                # Liquid phase
                
                # If rain < 0, borrow from cloud water
                if qrz < 0.:
                    
                    qlz = qlz + qrz
                    qrz = 0.
                
                # If cloud water < 0, borrow from water vapor
                if qlz < 0.:
                    
                    qvz = qvz + qlz
                    tz  = tz - qlz * lcpk_1   # Heating
                    qlz = 0.
        
        with interval(0, -1):
            
            # Initialize precipitation
            graupel = 0.
            rain    = 0.
            snow    = 0.
            ice     = 0.
            
            # This is to prevent excessive build-up of cloud ice from 
            # external sources
            if de_ice == 1:
                
                qio = qi - dt_in * qi_dt    # Orginal qi before phys
                qin = max(qio, qi0_max)     # Adjusted value
                
                if qi > qin:
                    
                    qs = qs + qi - qin
                    qi = qin
                    
                    dqi   = (qin - qio) * rdt   # Modified qi tendency
                    qs_dt = qs_dt + qi_dt - dqi
                    qi_dt = dqi
            
            qiz = qi
            qsz = qs
            
            t0  = pt
            tz  = t0
            dp1 = delp
            dp0 = dp1   # Moist air mass * grav
            
            # Convert moist mixing ratios to dry mixing ratios
            qvz = qv
            qlz = ql
            qrz = qr
            qgz = qg
            
            dp1 = dp1 * (1. - qvz)
            omq = dp0 / dp1
            
            qvz = qvz * omq
            qlz = qlz * omq
            qrz = qrz * omq
            qiz = qiz * omq
            qsz = qsz * omq
            qgz = qgz * omq
            
            qa0 = qa
            dz0 = dz
            
            den0 = -dp1 / (grav * dz0)  # Density of dry air
            p1   = den0 * rdgas * t0    # Dry air pressure
            
            # Save a copy of old values for computing tendencies
            qv0 = qvz
            ql0 = qlz
            qr0 = qrz
            qi0 = qiz
            qs0 = qsz
            qg0 = qgz
            
            # For sedi_momentum
            u0 = uin
            v0 = vin
            u1 = u0
            v1 = v0
            
            if prog_ccn == 1:
                
                # Convert #/cc to #/m^3
                ccn     = qn * 1.e6
                c_praut = cpaut * (ccn * rhor)**(-1./3.)
                
            else:
                
                ccn = (ccn_l * land + ccn_o * (1. - land)) * 1.e6
                
                if use_ccn == 1:
                
                    # Propagate downwards previously computed values of ccn
                    ccn = ccn[0, 0, +1]
                
                c_praut = cpaut * (ccn * rhor)**(-1./3.)
            
            # Calculate horizontal subgrid variability
            # Total water subgrid deviation in horizontal direction
            # Default area dependent form: use dx ~ 100 km as the base
            s_leng_2  = sqrt(sqrt(area * 1.e-10))
            t_land_2  = dw_land * s_leng_2
            t_ocean_2 = dw_ocean * s_leng_2
            h_var     = t_land_2 * land + t_ocean_2 * (1. - land)
            h_var     = min(0.2, max(0.01, h_var))
            
            # Relative humidity increment
            rh_adj  = 1. - h_var - rh_inc
            rh_rain = max(0.35, rh_adj - rh_inr)
            
            # Fix all negative water species
            if fix_negative == 1:
                
                # Define heat capacity and latent heat coefficient
                cvm_2  = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
                lcpk_2 = (lv00 + d0_vap * tz) / cvm_2
                icpk_2 = (li00 + dc_ice * tz) / cvm_2
                
                # Ice phase
                
                # If cloud ice < 0, borrow from snow
                if qiz < 0.:
                    
                    qsz = qsz + qiz
                    qiz = 0.
                
                # If snow < 0, borrow from graupel
                if qsz < 0.:
                    
                    qgz = qgz + qsz
                    qsz = 0.
                
                # If graupel < 0, borrow from rain
                if qgz < 0.:
                    
                    qrz = qrz + qgz
                    tz  = tz - qgz * icpk_2   # Heating
                    qgz = 0.
                    
                # Liquid phase
                
                # If rain < 0, borrow from cloud water
                if qrz < 0.:
                    
                    qlz = qlz + qrz
                    qrz = 0.
                
                # If cloud water < 0, borrow from water vapor
                if qlz < 0.:
                    
                    qvz = qvz + qlz
                    tz  = tz - qlz * lcpk_2   # Heating
                    qlz = 0.
                
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if fix_negative == 1:
                
                qvz_copy = qvz
                
                if qvz_copy < 0.: qvz = 0.
        
        with interval(1, -2):
            
            if fix_negative == 1:
                
                qvz_copy = qvz
                
                # Fix water vapor; borrow from below
                if qvz_copy[0, 0, -1] < 0.:
                    qvz_copy = qvz_copy[0, 0, 0] + qvz_copy[0, 0, -1] * dp1[0, 0, -1] / dp1[0, 0, 0]
                
                if qvz_copy < 0.:
                    qvz = 0.
                else:
                    qvz = qvz_copy
        
        with interval(-2, -1):
            
            if fix_negative == 1:
                
                qvz_copy = qvz
                
                # Fix water vapor; borrow from below
                if qvz_copy[0, 0, -1] < 0.:
                    qvz_copy = qvz_copy[0, 0, 0] + qvz_copy[0, 0, -1] * dp1[0, 0, -1] / dp1[0, 0, 0]
                
                if qvz_copy < 0.:
                    qvz = 0.
                else:
                    qvz = qvz_copy
                
                # Bottom layer; borrow from above
                flag = 0
                
                if qvz_copy < 0.: qvz1 = qvz[0, 0, +1] + qvz_copy[0, 0, 0] * dp1[0, 0, 0] / dp1[0, 0, +1]
                else:             qvz1 = qvz[0, 0, +1]
                
                if (qvz1 < 0.) and (qvz > 0.):
                    
                    dq   = min(-qvz1 * dp1[0, 0, +1], qvz[0, 0, 0] * dp1[0, 0, 0])
                    qvz  = qvz - dq / dp1
                    flag = 1
        
        with interval(-1, None):
            
            if fix_negative == 1:
                
                # Fix water vapor; borrow from below
                if qvz_copy[0, 0, -1] < 0.:
                    qvz[0, 0, 0] = qvz[0, 0, 0] + qvz_copy[0, 0, -1] * dp1[0, 0, -1] / dp1[0, 0, 0]
                
                # Bottom layer; borrow from above
                if flag[0, 0, -1] == 1:
                    qvz = qvz - dq[0, 0, -1] / dp1


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, debug_mode=DEBUG_MODE)
def warm_rain( h_var     : FIELD_FLT, 
               rain      : FIELD_FLT, 
               qgz       : FIELD_FLT, 
               qiz       : FIELD_FLT, 
               qlz       : FIELD_FLT, 
               qrz       : FIELD_FLT, 
               qsz       : FIELD_FLT, 
               qvz       : FIELD_FLT, 
               tz        : FIELD_FLT, 
               den       : FIELD_FLT, 
               denfac    : FIELD_FLT, 
               w         : FIELD_FLT, 
               t0        : FIELD_FLT, 
               den0      : FIELD_FLT, 
               dz0       : FIELD_FLT, 
               dz1       : FIELD_FLT, 
               dp1       : FIELD_FLT, 
               m1        : FIELD_FLT, 
               vtrz      : FIELD_FLT, 
               ccn       : FIELD_FLT, 
               c_praut   : FIELD_FLT, 
               m1_sol    : FIELD_FLT, 
               m2_rain   : FIELD_FLT, 
               m2_sol    : FIELD_FLT, 
               is_first  : DTYPE_INT, 
               do_sedi_w : DTYPE_INT, 
               p_nonhydro: DTYPE_INT, 
               use_ccn   : DTYPE_INT, 
               c_air     : DTYPE_FLT, 
               c_vap     : DTYPE_FLT, 
               d0_vap    : DTYPE_FLT, 
               lv00      : DTYPE_FLT, 
               fac_rc    : DTYPE_FLT, 
               cracw     : DTYPE_FLT, 
               crevp_0   : DTYPE_FLT, 
               crevp_1   : DTYPE_FLT, 
               crevp_2   : DTYPE_FLT, 
               crevp_3   : DTYPE_FLT, 
               crevp_4   : DTYPE_FLT, 
               t_wfr     : DTYPE_FLT, 
               so3       : DTYPE_FLT, 
               dt_rain   : DTYPE_FLT, 
               zs        : DTYPE_FLT ):
    
    with computation(FORWARD):
        
        with interval(0, 1):
        
            if is_first == 1:
                
                # Define air density based on hydrostatical property
                if p_nonhydro == 1:
                    
                    dz1    = dz0
                    den    = den0   # Dry air density remains the same
                    denfac = sqrt(sfcrho / den)
                    
                else:
                    
                    dz1    = dz0 * tz / t0  # Hydrostatic balance
                    den    = den0 * dz0 / dz1
                    denfac = sqrt(sfcrho / den)

            # Time-split warm rain processes: 1st pass
            
            # Terminal speed of rain
            m1_rain = 0.
            
            if qrz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, None):
            
            if is_first == 1:
                
                # Define air density based on hydrostatical property
                if p_nonhydro == 1:
                    
                    dz1    = dz0
                    den    = den0   # Dry air density remains the same
                    denfac = sqrt(sfcrho / den)
                    
                else:
                    
                    dz1    = dz0 * tz / t0  # Hydrostatic balance
                    den    = den0 * dz0 / dz1
                    denfac = sqrt(sfcrho / den)

            # Time-split warm rain processes: 1st pass
            
            # Terminal speed of rain
            m1_rain = 0.
            
            if no_fall[0, 0, -1] == 1:
                
                if (qrz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
                
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            vtrz, r1 = compute_rain_fspeed(no_fall, qrz, den)
            
            if no_fall == 0:
                
                ze = zs - dz1
                
                # Evaporation and accretion of rain for the first 1/2 time step
                if (tz > t_wfr) and (qrz > qrmin):
                    
                    # Define heat capacity and latent heat coefficient
                    lhl_rr11   = lv00 + d0_vap * tz
                    q_liq_rr11 = qlz + qrz
                    q_sol_rr11 = qiz + qsz + qgz
                    cvm_rr11   = c_air + qvz * c_vap + q_liq_rr11 * c_liq + q_sol_rr11 * c_ice
                    lcpk_rr11  = lhl_rr11 / cvm_rr11

                    tin_rr11 = tz - lcpk_rr11 * qlz    # Presence of clouds suppresses the rain evap
                    qpz_rr11 = qvz + qlz
                    
                    qsat_rr11  = ( e00 * exp((dc_vap * log(tin_rr11 / t_ice) + lv0 * (tin_rr11 - t_ice) / (tin_rr11 * t_ice)) / rvgas) ) / \
                                 ( rvgas * tin_rr11 * den )
                    dqsdt_rr11 = qsat_rr11 * (dc_vap + lv0 / tin_rr11) / (rvgas * tin_rr11)
                    
                    dqh_rr11     = max(qlz, h_var * max(qpz_rr11, qcmin))
                    dqh_rr11     = min(dqh_rr11, 0.2 * qpz_rr11)   # New limiter
                    dqv_rr11     = qsat_rr11 - qvz                 # Use this to prevent super-sat the gird box
                    q_minus_rr11 = qpz_rr11 - dqh_rr11
                    q_plus_rr11  = qpz_rr11 + dqh_rr11
                    
                    dt5_rr11 = 0.5 * dt_rain
                    
                    # qsat must be > q_minus to activate evaporation
                    # qsat must be < q_plus to activate accretion
                    # Rain evaporation
                    if (dqv_rr11 > qvmin) and (qsat_rr11 > q_minus_rr11):
                    
                        if qsat_rr11 > q_plus_rr11:
                            
                            dq_rr11 = qsat_rr11 - qpz_rr11
                            
                        else:
                            
                            # q_minus < qsat < q_plus
                            # dq == dqh if qsat == q_minus
                            dq_rr11 = 0.25 * (q_minus_rr11 - qsat_rr11)**2 / dqh_rr11
                        
                        qden_rr11 = qrz * den
                        t2_rr11   = tin_rr11 * tin_rr11
                        evap_rr11 = crevp_0 * t2_rr11 * dq_rr11 * \
                                    ( crevp_1 * sqrt(qden_rr11) + crevp_2 * \
                                      exp(0.725 * log(qden_rr11)) ) / \
                                    ( crevp_3 * t2_rr11 + crevp_4 * qsat_rr11 * den )
                        evap_rr11 = min(qrz, min(dt5_rr11 * evap_rr11, dqv_rr11 / (1. + lcpk_rr11 * dqsdt_rr11)))

                        # Alternative minimum evap in dry environmental air
                        qrz        = qrz - evap_rr11
                        qvz        = qvz + evap_rr11
                        q_liq_rr11 = q_liq_rr11 - evap_rr11
                        cvm_rr11   = c_air + qvz * c_vap + q_liq_rr11 * c_liq + q_sol_rr11 * c_ice
                        tz         = tz - evap_rr11 * lhl_rr11 / cvm_rr11
                        
                    # Accretion: pracc
                    if (qrz > qrmin) and (qlz > 1.e-6) and (qsat_rr11 < q_minus_rr11):
                        
                        sink_rr11 = dt5_rr11 * denfac * cracw * exp(0.95 * log(qrz * den))
                        sink_rr11 = sink_rr11 / (1. + sink_rr11) * qlz
                        qlz       = qlz - sink_rr11
                        qrz       = qrz + sink_rr11
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
        
        with interval(0, -1):
        
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            vtrz, r1 = compute_rain_fspeed(no_fall, qrz, den)
            
            if no_fall == 0:
                
                ze = ze[0, 0, +1] - dz1    # dz < 0
                
                # Evaporation and accretion of rain for the first 1/2 time step
                if (tz > t_wfr) and (qrz > qrmin):
                    
                    # Define heat capacity and latent heat coefficient
                    lhl_rr12   = lv00 + d0_vap * tz
                    q_liq_rr12 = qlz + qrz
                    q_sol_rr12 = qiz + qsz + qgz
                    cvm_rr12   = c_air + qvz * c_vap + q_liq_rr12 * c_liq + q_sol_rr12 * c_ice
                    lcpk_rr12  = lhl_rr12 / cvm_rr12

                    tin_rr12 = tz - lcpk_rr12 * qlz    # Presence of clouds suppresses the rain evap
                    qpz_rr12 = qvz + qlz
                    
                    qsat_rr12  = ( e00 * exp((dc_vap * log(tin_rr12 / t_ice) + lv0 * (tin_rr12 - t_ice) / (tin_rr12 * t_ice)) / rvgas) ) / \
                                 ( rvgas * tin_rr12 * den )
                    dqsdt_rr12 = qsat_rr12 * (dc_vap + lv0 / tin_rr12) / (rvgas * tin_rr12)
                    
                    dqh_rr12     = max(qlz, h_var * max(qpz_rr12, qcmin))
                    dqh_rr12     = min(dqh_rr12, 0.2 * qpz_rr12)   # New limiter
                    dqv_rr12     = qsat_rr12 - qvz                 # Use this to prevent super-sat the gird box
                    q_minus_rr12 = qpz_rr12 - dqh_rr12
                    q_plus_rr12  = qpz_rr12 + dqh_rr12
                    
                    dt5_rr12 = 0.5 * dt_rain

                    # qsat must be > q_minus to activate evaporation
                    # qsat must be < q_plus to activate accretion
                    # Rain evaporation
                    if (dqv_rr12 > qvmin) and (qsat_rr12 > q_minus_rr12):
                    
                        if qsat_rr12 > q_plus_rr12:
                            
                            dq_rr12 = qsat_rr12 - qpz_rr12
                            
                        else:
                            
                            # q_minus < qsat < q_plus
                            # dq == dqh if qsat == q_minus
                            dq_rr12 = 0.25 * (q_minus_rr12 - qsat_rr12)**2 / dqh_rr12
                        
                        qden_rr12 = qrz * den
                        t2_rr12   = tin_rr12 * tin_rr12
                        evap_rr12 = crevp_0 * t2_rr12 * dq_rr12 * \
                                    ( crevp_1 * sqrt(qden_rr12) + crevp_2 * \
                                      exp(0.725 * log(qden_rr12)) ) / \
                                    ( crevp_3 * t2_rr12 + crevp_4 * qsat_rr12 * den )
                        evap_rr12 = min(qrz, min(dt5_rr12 * evap_rr12, dqv_rr12 / (1. + lcpk_rr12 * dqsdt_rr12)))

                        # Alternative minimum evap in dry environmental air
                        qrz        = qrz - evap_rr12
                        qvz        = qvz + evap_rr12
                        q_liq_rr12 = q_liq_rr12 - evap_rr12
                        cvm_rr12   = c_air + qvz * c_vap + q_liq_rr12 * c_liq + q_sol_rr12 * c_ice
                        tz         = tz - evap_rr12 * lhl_rr12 / cvm_rr12
                        
                    # Accretion: pracc
                    if (qrz > qrmin) and (qlz > 1.e-6) and (qsat_rr12 < q_minus_rr12):
                        
                        sink_rr12 = dt5_rr12 * denfac * cracw * exp(0.95 * log(qrz * den))
                        sink_rr12 = sink_rr12 / (1. + sink_rr12) * qlz
                        qlz       = qlz - sink_rr12
                        qrz       = qrz + sink_rr12
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
        
    # Mass flux induced by falling rain
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 1) and (no_fall == 0):
                zt = ze
        
        with interval(1, -1):
            
            if use_ppm == 1:
                
                if no_fall == 0:
                    zt = ze - 0.5 * dt_rain * (vtrz[0, 0, -1] + vtrz)
                
                if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                    zt = zt[0, 0, -1] - dz_min
        
        with interval(-1, None):
            
            if use_ppm == 1:
                
                if no_fall == 0:
                
                    zt = ze - 0.5 * dt_rain * (vtrz[0, 0, -1] + vtrz)
                    
                    zt_kbot1 = zs - dt_rain * vtrz
                    
                    if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                        zt = zt[0, 0, -1] - dz_min
                        
                    if zt_kbot1 >= zt:
                        zt_kbot1 = zt - dz_min
    
    with computation(BACKWARD), interval(0, -1):
        
        if (use_ppm == 1) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, +1]
    
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                
                dz = ze - ze[0, 0, +1]
                
                dd  = dt_rain * vtrz
                qrz = qrz * dp1
                
                qm = qrz / (dz + dd)
                
                m1_rain = qrz - qm * dz
            
        with interval(1, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                
                dz = ze - ze[0, 0, +1]
                
                dd  = dt_rain * vtrz
                qrz = qrz * dp1
                
                qm = (qrz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
                m1_rain = m1_rain[0, 0, -1] + qrz[0, 0, 0] - qm * dz
                
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                
                dz = ze - zs
                
                dd  = dt_rain * vtrz
                qrz = qrz * dp1
                
                qm = (qrz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
                m1_rain = m1_rain[0, 0, -1] + qrz[0, 0, 0] - qm * dz
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    r1 = m1_rain
                    
                    # Update
                    qrz = (qm * dz) / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                    \
                                   m1_rain[0, 0, -1] * vtrz[0, 0, -1] + \
                                   m1_rain * vtrz ) /                   \
                                 ( dm + m1_rain[0, 0, -1] - m1_rain )
        
        with interval(1, -1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    r1 = r1[0, 0, +1]
                    
                    # Update
                    qrz = (qm * dz) / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                    \
                                   m1_rain[0, 0, -1] * vtrz[0, 0, -1] + \
                                   m1_rain * vtrz ) /                   \
                                 ( dm + m1_rain[0, 0, -1] - m1_rain )
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    r1 = r1[0, 0, +1]
                    
                    # Update
                    qrz = (qm * dz) / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_rain * vtrz) / (dm - m1_rain)
                
                if do_sedi_heat == 1:
                
                    # Input q fields are dry mixing ratios, and dm is dry air mass
                    cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
                    
                    # - Assumption: The ke in the falling condensates is negligible 
                    #               compared to the potential energy that was 
                    #               unaccounted for. Local thermal equilibrium is 
                    #               assumed, and the loss in pe is transformed into 
                    #               internal energy (to heat the whole grid box).
                    # - Backward time-implicit upwind transport scheme:
                    # - dm here is dry air mass
                    tz = tz - 0.5 * m1_rain * grav * dz1 / (cvn + m1_rain * c_liq)
                          
    with computation(FORWARD):
        
        with interval(0, 1):
                
            if no_fall == 0:
                
                tz_copy = tz
                
                # Evaporation and accretion of rain for the remaining 1/2 time step
                if (tz > t_wfr) and (qrz > qrmin):
                    
                    # Define heat capacity and latent heat coefficient
                    lhl_rr21   = lv00 + d0_vap * tz
                    q_liq_rr21 = qlz + qrz
                    q_sol_rr21 = qiz + qsz + qgz
                    cvm_rr21   = c_air + qvz * c_vap + q_liq_rr21 * c_liq + q_sol_rr21 * c_ice
                    lcpk_rr21  = lhl_rr21 / cvm_rr21

                    tin_rr21 = tz - lcpk_rr21 * qlz    # Presence of clouds suppresses the rain evap
                    qpz_rr21 = qvz + qlz
                    
                    qsat_rr21  = ( e00 * exp((dc_vap * log(tin_rr21 / t_ice) + lv0 * (tin_rr21 - t_ice) / (tin_rr21 * t_ice)) / rvgas) ) / \
                                 ( rvgas * tin_rr21 * den )
                    dqsdt_rr21 = qsat_rr21 * (dc_vap + lv0 / tin_rr21) / (rvgas * tin_rr21)
                    
                    dqh_rr21     = max(qlz, h_var * max(qpz_rr21, qcmin))
                    dqh_rr21     = min(dqh_rr21, 0.2 * qpz_rr21)   # New limiter
                    dqv_rr21     = qsat_rr21 - qvz                 # Use this to prevent super-sat the gird box
                    q_minus_rr21 = qpz_rr21 - dqh_rr21
                    q_plus_rr21  = qpz_rr21 + dqh_rr21
                    
                    dt5_rr21 = 0.5 * dt_rain
                    
                    # qsat must be > q_minus to activate evaporation
                    # qsat must be < q_plus to activate accretion
                    # Rain evaporation
                    if (dqv_rr21 > qvmin) and (qsat_rr21 > q_minus_rr21):
                    
                        if qsat_rr21 > q_plus_rr21:
                            
                            dq_rr21 = qsat_rr21 - qpz_rr21
                            
                        else:
                            
                            # q_minus < qsat < q_plus
                            # dq == dqh if qsat == q_minus
                            dq_rr21 = 0.25 * (q_minus_rr21 - qsat_rr21)**2 / dqh_rr21
                        
                        qden_rr21 = qrz * den
                        t2_rr21   = tin_rr21 * tin_rr21
                        evap_rr21 = crevp_0 * t2_rr21 * dq_rr21 * \
                                    ( crevp_1 * sqrt(qden_rr21) + crevp_2 * \
                                      exp(0.725 * log(qden_rr21)) ) / \
                                    ( crevp_3 * t2_rr21 + crevp_4 * qsat_rr21 * den )
                        evap_rr21 = min(qrz, min(dt5_rr21 * evap_rr21, dqv_rr21 / (1. + lcpk_rr21 * dqsdt_rr21)))

                        # Alternative minimum evap in dry environmental air
                        qrz        = qrz - evap_rr21
                        qvz        = qvz + evap_rr21
                        q_liq_rr21 = q_liq_rr21 - evap_rr21
                        cvm_rr21   = c_air + qvz * c_vap + q_liq_rr21 * c_liq + q_sol_rr21 * c_ice
                        tz         = tz - evap_rr21 * lhl_rr21 / cvm_rr21
                        
                    # Accretion: pracc
                    if (qrz > qrmin) and (qlz > 1.e-6) and (qsat_rr21 < q_minus_rr21):
                        
                        sink_rr21 = dt5_rr21 * denfac * cracw * exp(0.95 * log(qrz * den))
                        sink_rr21 = sink_rr21 / (1. + sink_rr21) * qlz
                        qlz       = qlz - sink_rr21
                        qrz       = qrz + sink_rr21
            
            # Auto-conversion assuming linear subgrid vertical distribution of 
            # cloud water following lin et al. 1994, mwr
            if irain_f != 0:
                
                # No subgrid variability
                qc0_nsv1 = fac_rc * ccn
                
                if tz > t_wfr:
                
                    if use_ccn == 1:
                        
                        # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                        qc_nsv1 = qc0_nsv1
                        
                    else:
                        
                        qc_nsv1 = qc0_nsv1 / den
                        
                    dq_nsv1 = qlz - qc_nsv1
                    
                    if dq_nsv1 > 0.:
                    
                        sink_nsv1 = min(dq_nsv1, dt_rain * c_praut * den * exp(so3 * log(qlz)))
                        qlz       = qlz - sink_nsv1
                        qrz       = qrz + sink_nsv1
        
        with interval(1, None):
                
            if no_fall == 0:
                
                if do_sedi_heat == 1:
                    
                    cvn_ia = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
                    
                    tz[0, 0, 0] = ( (cvn_ia + c_liq * (m1_rain - m1_rain[0, 0, -1])) *               \
                                    tz[0, 0, 0] + m1_rain[0, 0, -1] * c_liq * tz_copy[0, 0, -1] - \
                                    0.5 * grav * dz1 * (m1_rain[0, 0, -1] + m1_rain) ) /                       \
                                  ( cvn_ia + c_liq * m1_rain )
                    
                    tz_copy = tz
                
                # Evaporation and accretion of rain for the remaining 1/2 time step
                if (tz > t_wfr) and (qrz > qrmin):
                    
                    # Define heat capacity and latent heat coefficient
                    lhl_rr22   = lv00 + d0_vap * tz
                    q_liq_rr22 = qlz + qrz
                    q_sol_rr22 = qiz + qsz + qgz
                    cvm_rr22   = c_air + qvz * c_vap + q_liq_rr22 * c_liq + q_sol_rr22 * c_ice
                    lcpk_rr22  = lhl_rr22 / cvm_rr22

                    tin_rr22 = tz - lcpk_rr22 * qlz    # Presence of clouds suppresses the rain evap
                    qpz_rr22 = qvz + qlz
                    
                    qsat_rr22  = ( e00 * exp((dc_vap * log(tin_rr22 / t_ice) + lv0 * (tin_rr22 - t_ice) / (tin_rr22 * t_ice)) / rvgas) ) / \
                                 ( rvgas * tin_rr22 * den )
                    dqsdt_rr22 = qsat_rr22 * (dc_vap + lv0 / tin_rr22) / (rvgas * tin_rr22)
                    
                    dqh_rr22     = max(qlz, h_var * max(qpz_rr22, qcmin))
                    dqh_rr22     = min(dqh_rr22, 0.2 * qpz_rr22)   # New limiter
                    dqv_rr22     = qsat_rr22 - qvz                 # Use this to prevent super-sat the gird box
                    q_minus_rr22 = qpz_rr22 - dqh_rr22
                    q_plus_rr22  = qpz_rr22 + dqh_rr22
                    
                    dt5_rr22 = 0.5 * dt_rain
                    
                    # qsat must be > q_minus to activate evaporation
                    # qsat must be < q_plus to activate accretion
                    # Rain evaporation
                    if (dqv_rr22 > qvmin) and (qsat_rr22 > q_minus_rr22):
                    
                        if qsat_rr22 > q_plus_rr22:
                            
                            dq_rr22 = qsat_rr22 - qpz_rr22
                            
                        else:
                            
                            # q_minus < qsat < q_plus
                            # dq == dqh if qsat == q_minus
                            dq_rr22 = 0.25 * (q_minus_rr22 - qsat_rr22)**2 / dqh_rr22
                        
                        qden_rr22 = qrz * den
                        t2_rr22   = tin_rr22 * tin_rr22
                        evap_rr22 = crevp_0 * t2_rr22 * dq_rr22 * \
                                    ( crevp_1 * sqrt(qden_rr22) + crevp_2 * \
                                      exp(0.725 * log(qden_rr22)) ) / \
                                    ( crevp_3 * t2_rr22 + crevp_4 * qsat_rr22 * den )
                        evap_rr22 = min(qrz, min(dt5_rr22 * evap_rr22, dqv_rr22 / (1. + lcpk_rr22 * dqsdt_rr22)))

                        # Alternative minimum evap in dry environmental air
                        qrz        = qrz - evap_rr22
                        qvz        = qvz + evap_rr22
                        q_liq_rr22 = q_liq_rr22 - evap_rr22
                        cvm_rr22   = c_air + qvz * c_vap + q_liq_rr22 * c_liq + q_sol_rr22 * c_ice
                        tz         = tz - evap_rr22 * lhl_rr22 / cvm_rr22
                        
                    # Accretion: pracc
                    if (qrz > qrmin) and (qlz > 1.e-6) and (qsat_rr22 < q_minus_rr22):
                        
                        sink_rr22 = dt5_rr22 * denfac * cracw * exp(0.95 * log(qrz * den))
                        sink_rr22 = sink_rr22 / (1. + sink_rr22) * qlz
                        qlz       = qlz - sink_rr22
                        qrz       = qrz + sink_rr22
            
            # Auto-conversion assuming linear subgrid vertical distribution of 
            # cloud water following lin et al. 1994, mwr
            if irain_f != 0:
                
                # No subgrid variability
                qc0_nsv2 = fac_rc * ccn
                
                if tz > t_wfr:
                
                    if use_ccn == 1:
                        
                        # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                        qc_nsv2 = qc0_nsv2
                        
                    else:
                        
                        qc_nsv2 = qc0_nsv2 / den
                        
                    dq_nsv2 = qlz - qc_nsv2
                    
                    if dq_nsv2 > 0.:
                    
                        sink_nsv2 = min(dq_nsv2, dt_rain * c_praut * den * exp(so3 * log(qlz)))
                        qlz       = qlz - sink_nsv2
                        qrz       = qrz + sink_nsv2
            
            if (irain_f == 0) and (z_slope_liq == 1):
                dq = 0.5 * (qlz[0, 0, 0] - qlz[0, 0, -1])
                
    with computation(PARALLEL): 
        
        with interval(0, 1):
            
            if irain_f == 0:
                
                if z_slope_liq == 1:
                    
                    dl = 0.
                    
                    # Impose a presumed background horizontal variability that is 
                    # proportional to the value itself
                    dl = max(dl, max(qvmin, h_var * qlz))
            
                else:
                    
                    dl = max(qvmin, h_var * qlz)
                
                qlz, qrz = autoconv_subgrid_var( use_ccn, 
                                                 fac_rc, t_wfr, so3, dt_rain, 
                                                 qlz, qrz, tz, den, 
                                                 ccn, c_praut, dl )
            
            rain    = rain + r1
            m2_rain = m2_rain + m1_rain
            
            if is_first == 1:
                
                m1 = m1 + m1_rain
                
            else:
                
                m2_sol = m2_sol + m1_sol
                m1     = m1 + m1_rain + m1_sol
            
        with interval(1, -1):
            
            if irain_f == 0:
                
                if z_slope_liq == 1:
                    
                    # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                    dl = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qlz[0, 0, 0])
                    
                    if dq * dq[0, 0, +1] <= 0.:
                    
                        if dq > 0.: # Local maximum
                    
                            dl = min(dl, min(dq, -dq[0, 0, +1]))
                    
                        else:
                            
                            dl = 0.
                    
                    # Impose a presumed background horizontal variability that is 
                    # proportional to the value itself
                    dl = max(dl, max(qvmin, h_var * qlz))
            
                else:
                    
                    dl = max(qvmin, h_var * qlz)
                
                qlz, qrz = autoconv_subgrid_var( use_ccn, 
                                                 fac_rc, t_wfr, so3, dt_rain, 
                                                 qlz, qrz, tz, den, 
                                                 ccn, c_praut, dl )
            
            rain    = rain + r1
            m2_rain = m2_rain + m1_rain
            
            if is_first == 1:
                
                m1 = m1 + m1_rain
                
            else:
                
                m2_sol = m2_sol + m1_sol
                m1     = m1 + m1_rain + m1_sol
        
        with interval(-1, None):
            
            if irain_f == 0:
                
                if z_slope_liq == 1:
                    
                    dl = 0.
                    
                    # Impose a presumed background horizontal variability that is 
                    # proportional to the value itself
                    dl = max(dl, max(qvmin, h_var * qlz))
            
                else:
                    
                    dl = max(qvmin, h_var * qlz)
                
                qlz, qrz = autoconv_subgrid_var( use_ccn, 
                                                 fac_rc, t_wfr, so3, dt_rain, 
                                                 qlz, qrz, tz, den, 
                                                 ccn, c_praut, dl )
            
            rain    = rain + r1
            m2_rain = m2_rain + m1_rain
            
            if is_first == 1:
                
                m1 = m1 + m1_rain
                
            else:
                
                m2_sol = m2_sol + m1_sol
                m1     = m1 + m1_rain + m1_sol


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, debug_mode=DEBUG_MODE)
def sedimentation( graupel  : FIELD_FLT, 
                   ice      : FIELD_FLT, 
                   rain     : FIELD_FLT, 
                   snow     : FIELD_FLT, 
                   qgz      : FIELD_FLT, 
                   qiz      : FIELD_FLT, 
                   qlz      : FIELD_FLT, 
                   qrz      : FIELD_FLT, 
                   qsz      : FIELD_FLT, 
                   qvz      : FIELD_FLT, 
                   tz       : FIELD_FLT, 
                   den      : FIELD_FLT, 
                   w        : FIELD_FLT, 
                   dz1      : FIELD_FLT, 
                   dp1      : FIELD_FLT, 
                   vtgz     : FIELD_FLT, 
                   vtsz     : FIELD_FLT, 
                   m1_sol   : FIELD_FLT, 
                   do_sedi_w: DTYPE_INT, 
                   c_air    : DTYPE_FLT, 
                   c_vap    : DTYPE_FLT, 
                   d0_vap   : DTYPE_FLT, 
                   lv00     : DTYPE_FLT, 
                   log_10   : DTYPE_FLT, 
                   zs       : DTYPE_FLT, 
                   dts      : DTYPE_FLT, 
                   fac_imlt : DTYPE_FLT ):
    
    with computation(FORWARD):
        
        with interval(0, 1):
            
            # Sedimentation of cloud ice, snow, and graupel
            # Marshall-Palmer formula
            
            # Try the local air density -- for global model; the true value 
            # could be much smaller than sfcrho over high mountains
            rhof_fs1 = sqrt(min(10., sfcrho / den))
            
            # Ice
            if const_vi == 1:
                
                vtiz = vi_fac
                
            else:
                
                # Use deng and mace (2008, grl), which gives smaller fall speed 
                # than hd90 formula
                vi0_fs1 = 0.01 * vi_fac
                
                if qiz < thi:
                
                    vtiz = vf_min
                
                else:
                
                    tc_fs1 = tz - tice
                    vtiz   = (3. + log(qiz * den) / log_10) * (tc_fs1 * (aa * tc_fs1 + bb) + cc) + dd_fs * tc_fs1 + ee
                    vtiz   = vi0_fs1 * exp(log_10 * vtiz) * 0.8 
                    vtiz   = min(vi_max, max(vf_min, vtiz))
            
            # Snow
            if const_vs == 1:
                
                vtsz = vs_fac
                
            else:
                
                if qsz < ths:
                
                    vtsz = vf_min
                
                else:
                    
                    vtsz = vs_fac * vcons * rhof_fs1 * exp(0.0625 * log(qsz * den / norms))
                    vtsz = min(vs_max, max(vf_min, vtsz))
            
            # Graupel
            if const_vg == 1:
                
                vtgz = vg_fac
                
            else:
                
                if qgz < thg:
                
                    vtgz = vf_min
                
                else:
                
                    vtgz = vg_fac * vcong * rhof_fs1 * sqrt(sqrt(sqrt(qgz * den / normg)))
                    vtgz = min(vg_max, max(vf_min, vtgz))
            
            # Define heat capacity and latent heat coefficient
            m1_sol = 0.
            
            lhi_1   = li00 + dc_ice * tz
            q_liq_1 = qlz + qrz
            q_sol_1 = qiz + qsz + qgz
            cvm_1   = c_air + qvz * c_vap + q_liq_1 * c_liq + q_sol_1 * c_ice
            icpk_1  = lhi_1 / cvm_1

            # Find significant melting level
            if tz > tice: stop_k = 1
            else:         stop_k = 0
            
            if stop_k == 1:
            
                # Melting of cloud ice (before fall)
                tc_1 = tz - tice
                
                if (qiz > qcmin) and (tc_1 > 0.):
                
                    sink_1  = min(qiz, fac_imlt * tc_1 / icpk_1)
                    tmp_1   = min(sink_1, dim(ql_mlt, qlz))
                    qlz     = qlz + tmp_1
                    qrz     = qrz + sink_1 - tmp_1
                    qiz     = qiz - sink_1
                    q_liq_1 = q_liq_1 + sink_1
                    q_sol_1 = q_sol_1 - sink_1
                    cvm_1   = c_air + qvz * c_vap + q_liq_1 * c_liq + q_sol_1 * c_ice
                    tz      = tz - sink_1 * lhi_1 / cvm_1
                    tc_1    = tz - tice
                
            # Turn off melting when cloud microphysics time step is small
            if dts < 60.:
                stop_k = 0
                
            # sjl, turn off melting of falling cloud ice, snow and graupel
            stop_k = 0
            
            if stop_k == 1:
        
                # Update capacity heat and latent heat coefficient
                lhi_1  = li00 + dc_ice * tz
                icpk_1 = lhi_1 / cvm_1
            
            if qiz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, -1):
            
            # Sedimentation of cloud ice, snow, and graupel
            # Marshall-Palmer formula
            
            # Try the local air density -- for global model; the true value 
            # could be much smaller than sfcrho over high mountains
            rhof_fs2 = sqrt(min(10., sfcrho / den))
            
            # Ice
            if const_vi == 1:
                
                vtiz = vi_fac
                
            else:
                
                # Use deng and mace (2008, grl), which gives smaller fall speed 
                # than hd90 formula
                vi0_fs2 = 0.01 * vi_fac
                
                if qiz < thi:
                
                    vtiz = vf_min
                
                else:
                
                    tc_fs2 = tz - tice
                    vtiz   = (3. + log(qiz * den) / log_10) * (tc_fs2 * (aa * tc_fs2 + bb) + cc) + dd_fs * tc_fs2 + ee
                    vtiz   = vi0_fs2 * exp(log_10 * vtiz) * 0.8 
                    vtiz   = min(vi_max, max(vf_min, vtiz))
            
            # Snow
            if const_vs == 1:
                
                vtsz = vs_fac
                
            else:
                
                if qsz < ths:
                
                    vtsz = vf_min
                
                else:
                    
                    vtsz = vs_fac * vcons * rhof_fs2 * exp(0.0625 * log(qsz * den / norms))
                    vtsz = min(vs_max, max(vf_min, vtsz))
            
            # Graupel
            if const_vg == 1:
                
                vtgz = vg_fac
                
            else:
                
                if qgz < thg:
                
                    vtgz = vf_min
                
                else:
                
                    vtgz = vg_fac * vcong * rhof_fs2 * sqrt(sqrt(sqrt(qgz * den / normg)))
                    vtgz = min(vg_max, max(vf_min, vtgz))
            
            # Define heat capacity and latent heat coefficient
            m1_sol = 0.
            
            # Find significant melting level
            if stop_k[0, 0, -1] == 0:
                
                if tz > tice: stop_k = 1
                else:         stop_k = 0
                
            else:
                
                stop_k = 1
            
            lhi_2   = li00 + dc_ice * tz
            q_liq_2 = qlz + qrz
            q_sol_2 = qiz + qsz + qgz
            cvm_2   = c_air + qvz * c_vap + q_liq_2 * c_liq + q_sol_2 * c_ice
            icpk_2  = lhi_2 / cvm_2
            
            if stop_k == 1:
            
                # Melting of cloud ice (before fall)
                tc_2 = tz - tice
                
                if (qiz > qcmin) and (tc_2 > 0.):
                
                    sink_2  = min(qiz, fac_imlt * tc_2 / icpk_2)
                    tmp_2   = min(sink_2, dim(ql_mlt, qlz))
                    qlz     = qlz + tmp_2
                    qrz     = qrz + sink_2 - tmp_2
                    qiz     = qiz - sink_2
                    q_liq_2 = q_liq_2 + sink_2
                    q_sol_2 = q_sol_2 - sink_2
                    cvm_2   = c_air + qvz * c_vap + q_liq_2 * c_liq + q_sol_2 * c_ice
                    tz      = tz - sink_2 * lhi_2 / cvm_2
                    tc_2    = tz - tice
            
            # Turn off melting when cloud microphysics time step is small
            if dts < 60.:
                stop_k = 0
                
            # sjl, turn off melting of falling cloud ice, snow and graupel
            stop_k = 0
            
            if stop_k == 1:
        
                # Update capacity heat and latent heat coefficient
                lhi_2  = li00 + dc_ice * tz
                icpk_2 = lhi_2 / cvm_2
            
            if no_fall[0, 0, -1] == 1:
                
                if (qiz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
        
        with interval(-1, None):
            
            # Sedimentation of cloud ice, snow, and graupel
            # Marshall-Palmer formula
            
            # Try the local air density -- for global model; the true value 
            # could be much smaller than sfcrho over high mountains
            rhof_fs3 = sqrt(min(10., sfcrho / den))
            
            # Ice
            if const_vi == 1:
                
                vtiz = vi_fac
                
            else:
                
                # Use deng and mace (2008, grl), which gives smaller fall speed 
                # than hd90 formula
                vi0_fs3 = 0.01 * vi_fac
                
                if qiz < thi:
                
                    vtiz = vf_min
                
                else:
                
                    tc_fs3 = tz - tice
                    vtiz   = (3. + log(qiz * den) / log_10) * (tc_fs3 * (aa * tc_fs3 + bb) + cc) + dd_fs * tc_fs3 + ee
                    vtiz   = vi0_fs3 * exp(log_10 * vtiz) * 0.8 
                    vtiz   = min(vi_max, max(vf_min, vtiz))
            
            # Snow
            if const_vs == 1:
                
                vtsz = vs_fac
                
            else:
                
                if qsz < ths:
                
                    vtsz = vf_min
                
                else:
                    
                    vtsz = vs_fac * vcons * rhof_fs3 * exp(0.0625 * log(qsz * den / norms))
                    vtsz = min(vs_max, max(vf_min, vtsz))
            
            # Graupel
            if const_vg == 1:
                
                vtgz = vg_fac
                
            else:
                
                if qgz < thg:
                
                    vtgz = vf_min
                
                else:
                
                    vtgz = vg_fac * vcong * rhof_fs3 * sqrt(sqrt(sqrt(qgz * den / normg)))
                    vtgz = min(vg_max, max(vf_min, vtgz))
            
            # Define heat capacity and latent heat coefficient
            m1_sol = 0.
            
            lhi_3   = li00 + dc_ice * tz
            q_liq_3 = qlz + qrz
            q_sol_3 = qiz + qsz + qgz
            cvm_3   = c_air + qvz * c_vap + q_liq_3 * c_liq + q_sol_3 * c_ice
            icpk_3  = lhi_3 / cvm_3
            
            # Find significant melting level
            stop_k = 1
            
            if stop_k == 1:
            
                # Melting of cloud ice (before fall)
                tc_3 = tz - tice
                
                if (qiz > qcmin) and (tc_3 > 0.):
                
                    sink_3  = min(qiz, fac_imlt * tc_3 / icpk_3)
                    tmp_3   = min(sink_3, dim(ql_mlt, qlz))
                    qlz     = qlz + tmp_3
                    qrz     = qrz + sink_3 - tmp_3
                    qiz     = qiz - sink_3
                    q_liq_3 = q_liq_3 + sink_3
                    q_sol_3 = q_sol_3 - sink_3
                    cvm_3   = c_air + qvz * c_vap + q_liq_3 * c_liq + q_sol_3 * c_ice
                    tz      = tz - sink_3 * lhi_3 / cvm_3
                    tc_3    = tz - tice
                
                # Update capacity heat and latent heat coefficient
                lhi_3  = li00 + dc_ice * tz
                icpk_3 = lhi_3 / cvm_3
            
            if no_fall[0, 0, -1] == 1:
                
                if (qiz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
        
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            ze = zs - dz1
            
            if (vi_fac < 1.e-5) or (no_fall == 1):
                
                i1 = 0.
            
            else:
                
                zt       = ze - 0.5 * dts * (vtiz[0, 0, -1] + vtiz)
                zt_kbot1 = zs - dts * vtiz
                
        with interval(1, -1):
            
            ze = ze[0, 0, +1] - dz1    # dz < 0
            
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            if (vi_fac < 1.e-5) or (no_fall == 1): 
                
                i1 = 0.
            
            else:
                
                zt = ze - 0.5 * dts * (vtiz[0, 0, -1] + vtiz)
            
        with interval(0, 1):
            
            ze = ze[0, 0, +1] - dz1    # dz < 0
            zt = ze
            
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            if (vi_fac < 1.e-5) or (no_fall == 1): i1 = 0.
                
    with computation(FORWARD):
        
        with interval(1, -1):
        
            if (vi_fac >= 1.e-5) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
                
        with interval(-1, None):
        
            if (vi_fac >= 1.e-5) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
            
            if (vi_fac >= 1.e-5) and (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min
                
    with computation(BACKWARD), interval(0, -1):
        
        if (vi_fac >= 1.e-5) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min
    
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtiz
                    qiz = qiz * dp1
                    
                    qm = qiz / (dz + dd)
                    
                    m1_sol = qiz - qm * dz
                    
        with interval(1, -1):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtiz
                    qiz = qiz * dp1
                    
                    qm = (qiz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_sol = m1_sol[0, 0, -1] + qiz[0, 0, 0] - qm * dz
            
        with interval(-1, None):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - zs
                    
                    dd  = dts * vtiz
                    qiz = qiz * dp1
                    
                    qm = (qiz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_sol = m1_sol[0, 0, -1] + qiz[0, 0, 0] - qm * dz
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if use_ppm == 0:
                    
                    i1 = m1_sol
                    
                    # Update
                    qiz = (qm * dz) / dp1
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                   \
                                   m1_sol[0, 0, -1] * vtiz[0, 0, -1] + \
                                   m1_sol * vtiz ) /                   \
                                 ( dm + m1_sol[0, 0, -1] - m1_sol )
        
        with interval(1, -1):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if use_ppm == 0:
                    
                    i1 = i1[0, 0, +1]
                    
                    # Update
                    qiz = (qm * dz) / dp1
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                   \
                                   m1_sol[0, 0, -1] * vtiz[0, 0, -1] + \
                                   m1_sol * vtiz ) /                   \
                                 ( dm + m1_sol[0, 0, -1] - m1_sol )
        
        with interval(0, 1):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if use_ppm == 0:
                    
                    i1 = i1[0, 0, +1]
                    
                    # Update
                    qiz = (qm * dz) / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_sol * vtiz) / (dm - m1_sol)

    # Melting of falling snow into rain
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if qsz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, None):
        
            if no_fall[0, 0, -1] == 1:
                
                if (qsz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
                
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            r1 = 0.
        
            if no_fall == 1:
                
                s1 = 0.
                
            else:
                
                zt       = ze - 0.5 * dts * (vtsz[0, 0, -1] + vtsz)
                zt_kbot1 = zs - dts * vtsz
        
        with interval(1, -1):
        
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            r1 = 0.
        
            if no_fall == 1:
                
                s1 = 0.
                
            else:
                
                zt = ze - 0.5 * dts * (vtsz[0, 0, -1] + vtsz)
        
        with interval(0, 1):
            
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            r1 = 0.
        
            if no_fall == 1: s1 = 0.
                
    with computation(FORWARD):
        
        with interval(1, -1):
        
            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
                
        with interval(-1, None):
        
            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
            
            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min
    
    with computation(FORWARD):
        
        with interval(0, 1):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtsz
                    qsz = qsz * dp1
                    
                    qm = qsz / (dz + dd)
                    
                    m1_tf = qsz - qm * dz
        
        with interval(1, -1):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtsz
                    qsz = qsz * dp1
                    
                    qm = (qsz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_tf = m1_tf[0, 0, -1] + qsz[0, 0, 0] - qm * dz
        
        with interval(-1, None):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - zs
                    
                    dd  = dts * vtsz
                    qsz = qsz * dp1
                    
                    qm = (qsz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_tf = m1_tf[0, 0, -1] + qsz[0, 0, 0] - qm * dz
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    s1 = m1_tf
                    
                    # Update
                    qsz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtsz[0, 0, -1] + \
                                   m1_tf * vtsz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )
        
        with interval(1, -1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    s1 = s1[0, 0, +1]
                    
                    # Update
                    qsz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtsz[0, 0, -1] + \
                                   m1_tf * vtsz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    s1 = s1[0, 0, +1]
                    
                    # Update
                    qsz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtsz) / (dm - m1_tf)
    
    # Melting of falling graupel into rain
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if qgz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, None):
        
            if no_fall[0, 0, -1] == 1:
                
                if (qgz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
                
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if no_fall == 1:
                
                g1 = 0.
            
            else:
                
                zt       = ze - 0.5 * dts * (vtgz[0, 0, -1] + vtgz)
                zt_kbot1 = zs - dts * vtgz
        
        with interval(1, -1):
        
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            if no_fall == 1:
                
                g1 = 0.
            
            else:
                
                zt = ze - 0.5 * dts * (vtgz[0, 0, -1] + vtgz)
        
        with interval(0, 1):
            
            if no_fall[0, 0, +1] == 0:
                no_fall = no_fall[0, 0, +1]
            
            if no_fall == 1: g1 = 0.
                
    with computation(FORWARD):
        
        with interval(1, -1):
        
            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
                
        with interval(-1, None):
        
            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
            
            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min
    
    with computation(FORWARD):
        
        with interval(0, 1):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtgz
                    qgz = qgz * dp1
                    
                    qm = qgz / (dz + dd)
                    
                    m1_tf = qgz - qm * dz
        
        with interval(1, -1):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - ze[0, 0, +1]
                    
                    dd  = dts * vtgz
                    qgz = qgz * dp1
                    
                    qm = (qgz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_tf = m1_tf[0, 0, -1] + qgz[0, 0, 0] - qm * dz
        
        with interval(-1, None):
        
            if no_fall == 0:
                
                if do_sedi_w == 1:
                    dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
                
                if use_ppm == 0:
                    
                    dz = ze - zs
                    
                    dd  = dts * vtgz
                    qgz = qgz * dp1
                    
                    qm = (qgz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                    
                    m1_tf = m1_tf[0, 0, -1] + qgz[0, 0, 0] - qm * dz
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    g1 = m1_tf
                    
                    # Update
                    qgz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtgz[0, 0, -1] + \
                                   m1_tf * vtgz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )
            
            rain    = rain + r1     # From melted snow and ice that reached the ground
            snow    = snow + s1
            graupel = graupel + g1
            ice     = ice + i1
        
        with interval(1, -1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    g1 = g1[0, 0, +1]
                    
                    # Update
                    qgz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtgz[0, 0, -1] + \
                                   m1_tf * vtgz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )
            
            rain    = rain + r1     # From melted snow and ice that reached the ground
            snow    = snow + s1
            graupel = graupel + g1
            ice     = ice + i1
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    g1 = g1[0, 0, +1]
                    
                    # Update
                    qgz = (qm * dz) / dp1
                    
                m1_sol = m1_sol + m1_tf
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtgz) / (dm - m1_tf)
            
            rain    = rain + r1     # From melted snow and ice that reached the ground
            snow    = snow + s1
            graupel = graupel + g1
            ice     = ice + i1
            
            if do_sedi_heat == 1:
                
                # Input q fields are dry mixing ratios, and dm is dry air mass
                cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
                
                # - Assumption: The ke in the falling condensates is negligible 
                #               compared to the potential energy that was 
                #               unaccounted for. Local thermal equilibrium is 
                #               assumed, and the loss in pe is transformed into 
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tz = tz - 0.5 * m1_sol * grav * dz1 / (cvn + m1_sol * c_ice)
    
    # Implicit algorithm            
    with computation(FORWARD), interval(1, None):
        
        if do_sedi_heat == 1:
            
            # Input q fields are dry mixing ratios, and dm is dry air mass
            cvn_ia = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
            
            tz[0, 0, 0] = ( (cvn_ia + c_ice * (m1_sol - m1_sol[0, 0, -1])) *        \
                            tz[0, 0, 0] + m1_sol[0, 0, -1] * c_ice * tz[0, 0, -1] - \
                            0.5 * grav * dz1 * (m1_sol[0, 0, -1] + m1_sol) ) /                   \
                          ( cvn_ia + c_ice * m1_sol )


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, debug_mode=DEBUG_MODE)
def icloud( h_var   : FIELD_FLT, 
            rh_adj  : FIELD_FLT, 
            rh_rain : FIELD_FLT, 
            qaz     : FIELD_FLT, 
            qgz     : FIELD_FLT, 
            qiz     : FIELD_FLT, 
            qlz     : FIELD_FLT, 
            qrz     : FIELD_FLT, 
            qsz     : FIELD_FLT, 
            qvz     : FIELD_FLT, 
            tz      : FIELD_FLT, 
            den     : FIELD_FLT, 
            denfac  : FIELD_FLT, 
            p1      : FIELD_FLT, 
            vtgz    : FIELD_FLT, 
            vtrz    : FIELD_FLT, 
            vtsz    : FIELD_FLT, 
            c_air   : DTYPE_FLT, 
            c_vap   : DTYPE_FLT, 
            d0_vap  : DTYPE_FLT, 
            lv00    : DTYPE_FLT, 
            cracs   : DTYPE_FLT, 
            csacr   : DTYPE_FLT, 
            cgacr   : DTYPE_FLT, 
            cgacs   : DTYPE_FLT, 
            acco_00 : DTYPE_FLT, 
            acco_01 : DTYPE_FLT, 
            acco_02 : DTYPE_FLT, 
            acco_03 : DTYPE_FLT, 
            acco_10 : DTYPE_FLT, 
            acco_11 : DTYPE_FLT, 
            acco_12 : DTYPE_FLT, 
            acco_13 : DTYPE_FLT, 
            acco_20 : DTYPE_FLT, 
            acco_21 : DTYPE_FLT, 
            acco_22 : DTYPE_FLT, 
            acco_23 : DTYPE_FLT, 
            csacw   : DTYPE_FLT, 
            csaci   : DTYPE_FLT, 
            cgacw   : DTYPE_FLT, 
            cgaci   : DTYPE_FLT, 
            cracw   : DTYPE_FLT, 
            cssub_0 : DTYPE_FLT, 
            cssub_1 : DTYPE_FLT, 
            cssub_2 : DTYPE_FLT, 
            cssub_3 : DTYPE_FLT, 
            cssub_4 : DTYPE_FLT, 
            cgfr_0  : DTYPE_FLT, 
            cgfr_1  : DTYPE_FLT, 
            csmlt_0 : DTYPE_FLT, 
            csmlt_1 : DTYPE_FLT, 
            csmlt_2 : DTYPE_FLT, 
            csmlt_3 : DTYPE_FLT, 
            csmlt_4 : DTYPE_FLT, 
            cgmlt_0 : DTYPE_FLT, 
            cgmlt_1 : DTYPE_FLT, 
            cgmlt_2 : DTYPE_FLT, 
            cgmlt_3 : DTYPE_FLT, 
            cgmlt_4 : DTYPE_FLT, 
            ces0    : DTYPE_FLT, 
            tice0   : DTYPE_FLT, 
            t_wfr   : DTYPE_FLT, 
            dts     : DTYPE_FLT, 
            rdts    : DTYPE_FLT, 
            fac_i2s : DTYPE_FLT, 
            fac_g2v : DTYPE_FLT, 
            fac_v2g : DTYPE_FLT, 
            fac_imlt: DTYPE_FLT, 
            fac_l2v : DTYPE_FLT ):
    
    with computation(FORWARD):
        
        with interval(0, 1):

            # Ice-phase microphysics
            
            # Define heat capacity and latent heat coefficient
            lhi   = li00 + dc_ice * tz
            q_liq = qlz + qrz
            q_sol = qiz + qsz + qgz
            cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            icpk  = lhi / cvm
            
            # - Sources of cloud ice: pihom, cold rain, and the sat_adj
            # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
            # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion
            t_wfr_tmp = t_wfr
            if (tz > tice) and (qiz > qcmin):
            
                # pimlt: instant melting of cloud ice
                melt  = min(qiz, fac_imlt * (tz - tice) / icpk)
                tmp   = min(melt, dim(ql_mlt, qlz))    # Maximum ql amount
                qlz   = qlz + tmp
                qrz   = qrz + melt - tmp
                qiz   = qiz - melt
                q_liq = q_liq + melt
                q_sol = q_sol - melt
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz    = tz - melt * lhi / cvm
            
            elif (tz < t_wfr) and (qlz > qcmin):
                
                # - pihom: homogeneous freezing of cloud water into cloud ice
                # - This is the 1st occurance of liquid water freezing in the split mp process
                dtmp   = t_wfr_tmp - tz
                factor = min(1., dtmp / dt_fr)
                sink   = min(qlz * factor, dtmp / icpk)
                qi_crt = qi_gen * min(qi_lim, 0.1 * (tice - tz)) / den
                tmp    = min(sink, dim(qi_crt, qiz))
                qlz    = qlz - sink
                qsz    = qsz + sink - tmp
                qiz    = qiz + tmp
                q_liq  = q_liq - sink
                q_sol  = q_sol + sink
                cvm    = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz     = tz + sink * lhi / cvm
        
        with interval(1, None):

            # Ice-phase microphysics
            
            # Define heat capacity and latent heat coefficient
            lhi   = li00 + dc_ice * tz
            q_liq = qlz + qrz
            q_sol = qiz + qsz + qgz
            cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            icpk  = lhi / cvm
            
            # - Sources of cloud ice: pihom, cold rain, and the sat_adj
            # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
            # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion
            t_wfr_tmp = t_wfr
            if (tz > tice) and (qiz > qcmin):
            
                # pimlt: instant melting of cloud ice
                melt  = min(qiz, fac_imlt * (tz - tice) / icpk)
                tmp   = min(melt, dim(ql_mlt, qlz))    # Maximum ql amount
                qlz   = qlz + tmp
                qrz   = qrz + melt - tmp
                qiz   = qiz - melt
                q_liq = q_liq + melt
                q_sol = q_sol - melt
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz    = tz - melt * lhi / cvm
            
            elif (tz < t_wfr) and (qlz > qcmin):
                
                # - pihom: homogeneous freezing of cloud water into cloud ice
                # - This is the 1st occurance of liquid water freezing in the split mp process
                dtmp   = t_wfr_tmp - tz
                factor = min(1., dtmp / dt_fr)
                sink   = min(qlz * factor, dtmp / icpk)
                qi_crt = qi_gen * min(qi_lim, 0.1 * (tice - tz)) / den
                tmp    = min(sink, dim(qi_crt, qiz))
                qlz    = qlz - sink
                qsz    = qsz + sink - tmp
                qiz    = qiz + tmp
                q_liq  = q_liq - sink
                q_sol  = q_sol + sink
                cvm    = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz     = tz + sink * lhi / cvm
            
            if z_slope_ice == 1:
                dq = 0.5 * (qiz[0, 0, 0] - qiz[0, 0, -1])
    
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if z_slope_ice == 1:
                
                di = 0.
                
                # Impose a presumed background horizontal variability that is 
                # proportional to the value itself
                di = max(di, max(qvmin, h_var * qiz))
        
            else:
                
                di = max(qvmin, h_var * qiz)
            
            qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = \
            icloud_main( c_air, c_vap, d0_vap, lv00, 
                         cracs, csacr, cgacr, cgacs, 
                         acco_00, acco_01, acco_02, acco_03, 
                         acco_10, acco_11, acco_12, acco_13, 
                         acco_20, acco_21, acco_22, acco_23, 
                         csacw, csaci, cgacw, cgaci, 
                         cssub_0, cssub_1, cssub_2, cssub_3, cssub_4, 
                         cgfr_0, cgfr_1, 
                         csmlt_0, csmlt_1, csmlt_2, csmlt_3, csmlt_4, 
                         cgmlt_0, cgmlt_1, cgmlt_2, cgmlt_3, cgmlt_4, 
                         ces0, tice0, t_wfr, 
                         dts, rdts, 
                         fac_i2s, fac_g2v, fac_v2g, fac_l2v, 
                         h_var, rh_adj, rh_rain, 
                         qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz, den, denfac, 
                         vtgz, vtrz, vtsz, 
                         p1, 
                         di, 
                         q_liq, q_sol, cvm )
        
        with interval(1, -1):
            
            if z_slope_ice == 1:
                
                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                di = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qiz[0, 0, 0])
                
                if dq * dq[0, 0, +1] <= 0.:
                
                    if dq > 0.: # Local maximum
                
                        di = min(di, min(dq, -dq[0, 0, +1]))
                
                    else:
                        
                        di = 0.
                
                # Impose a presumed background horizontal variability that is 
                # proportional to the value itself
                di = max(di, max(qvmin, h_var * qiz))
        
            else:
                
                di = max(qvmin, h_var * qiz)
            
            qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = \
            icloud_main( c_air, c_vap, d0_vap, lv00, 
                         cracs, csacr, cgacr, cgacs, 
                         acco_00, acco_01, acco_02, acco_03, 
                         acco_10, acco_11, acco_12, acco_13, 
                         acco_20, acco_21, acco_22, acco_23, 
                         csacw, csaci, cgacw, cgaci, 
                         cssub_0, cssub_1, cssub_2, cssub_3, cssub_4, 
                         cgfr_0, cgfr_1, 
                         csmlt_0, csmlt_1, csmlt_2, csmlt_3, csmlt_4, 
                         cgmlt_0, cgmlt_1, cgmlt_2, cgmlt_3, cgmlt_4, 
                         ces0, tice0, t_wfr, 
                         dts, rdts, 
                         fac_i2s, fac_g2v, fac_v2g, fac_l2v, 
                         h_var, rh_adj, rh_rain, 
                         qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz, den, denfac, 
                         vtgz, vtrz, vtsz, 
                         p1, 
                         di, 
                         q_liq, q_sol, cvm )
        
        with interval(-1, None):
            
            if z_slope_ice == 1:
                
                di = 0.
                
                # Impose a presumed background horizontal variability that is 
                # proportional to the value itself
                di = max(di, max(qvmin, h_var * qiz))
        
            else:
                
                di = max(qvmin, h_var * qiz)
            
            qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = \
            icloud_main( c_air, c_vap, d0_vap, lv00, 
                         cracs, csacr, cgacr, cgacs, 
                         acco_00, acco_01, acco_02, acco_03, 
                         acco_10, acco_11, acco_12, acco_13, 
                         acco_20, acco_21, acco_22, acco_23, 
                         csacw, csaci, cgacw, cgaci, 
                         cssub_0, cssub_1, cssub_2, cssub_3, cssub_4, 
                         cgfr_0, cgfr_1, 
                         csmlt_0, csmlt_1, csmlt_2, csmlt_3, csmlt_4, 
                         cgmlt_0, cgmlt_1, cgmlt_2, cgmlt_3, cgmlt_4, 
                         ces0, tice0, t_wfr, 
                         dts, rdts, 
                         fac_i2s, fac_g2v, fac_v2g, fac_l2v, 
                         h_var, rh_adj, rh_rain, 
                         qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz, den, denfac, 
                         vtgz, vtrz, vtsz, 
                         p1, 
                         di, 
                         q_liq, q_sol, cvm )


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD, debug_mode=DEBUG_MODE)
def fields_update( graupel: FIELD_FLT, 
                   ice    : FIELD_FLT, 
                   rain   : FIELD_FLT, 
                   snow   : FIELD_FLT, 
                   qaz    : FIELD_FLT, 
                   qgz    : FIELD_FLT, 
                   qiz    : FIELD_FLT, 
                   qlz    : FIELD_FLT, 
                   qrz    : FIELD_FLT, 
                   qsz    : FIELD_FLT, 
                   qvz    : FIELD_FLT, 
                   tz     : FIELD_FLT, 
                   udt    : FIELD_FLT, 
                   vdt    : FIELD_FLT, 
                   qa_dt  : FIELD_FLT, 
                   qg_dt  : FIELD_FLT, 
                   qi_dt  : FIELD_FLT, 
                   ql_dt  : FIELD_FLT, 
                   qr_dt  : FIELD_FLT, 
                   qs_dt  : FIELD_FLT, 
                   qv_dt  : FIELD_FLT, 
                   pt_dt  : FIELD_FLT, 
                   qa0    : FIELD_FLT, 
                   qg0    : FIELD_FLT, 
                   qi0    : FIELD_FLT, 
                   ql0    : FIELD_FLT, 
                   qr0    : FIELD_FLT, 
                   qs0    : FIELD_FLT, 
                   qv0    : FIELD_FLT, 
                   t0     : FIELD_FLT, 
                   dp0    : FIELD_FLT, 
                   u0     : FIELD_FLT, 
                   v0     : FIELD_FLT, 
                   dp1    : FIELD_FLT, 
                   u1     : FIELD_FLT, 
                   v1     : FIELD_FLT, 
                   m1     : FIELD_FLT, 
                   m2_rain: FIELD_FLT, 
                   m2_sol : FIELD_FLT, 
                   ntimes : DTYPE_INT, 
                   c_air  : DTYPE_FLT, 
                   c_vap  : DTYPE_FLT, 
                   rdt    : DTYPE_FLT ):
    
    with computation(FORWARD):
        
        with interval(0, 1):
        
            # Convert units from Pa*kg/kg to kg/m^2/s
            m2_rain = m2_rain * rdt * rgrav
            m2_sol  = m2_sol * rdt * rgrav
            
            # Update moist air mass (actually hydrostatic pressure) and convert 
            # to dry mixing ratios
            omq = dp1 / dp0
            qv_dt = qv_dt + rdt * (qvz - qv0) * omq
            ql_dt = ql_dt + rdt * (qlz - ql0) * omq
            qr_dt = qr_dt + rdt * (qrz - qr0) * omq
            qi_dt = qi_dt + rdt * (qiz - qi0) * omq
            qs_dt = qs_dt + rdt * (qsz - qs0) * omq
            qg_dt = qg_dt + rdt * (qgz - qg0) * omq
            
            cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            
            pt_dt = pt_dt + rdt * (tz - t0) * cvm / cp_air
            
            # Update cloud fraction tendency
            if do_qa == 1:
                
                qa_dt = 0.
                
            else:
                
                qa_dt = qa_dt + rdt * (qaz / ntimes - qa0)
            
            '''
            LEFT OUT FOR NOW
            # No clouds allowed above ktop
            if k_s < ktop:
                qa_dt[:, :, k_s:ktop+1] = 0.
            '''
            
            # Convert to mm / day
            convt = 86400. * rdt * rgrav
            
            rain    = rain * convt
            snow    = snow * convt
            ice     = ice * convt
            graupel = graupel * convt
        
        with interval(1, None):
        
            # Convert units from Pa*kg/kg to kg/m^2/s
            m2_rain = m2_rain * rdt * rgrav
            m2_sol  = m2_sol * rdt * rgrav
            
            if sedi_transport == 1:
                
                u1[0, 0, 0] = ( dp0[0, 0, 0] * u1[0, 0, 0] + m1[0, 0, -1] * u1[0, 0, -1] ) / \
                              ( dp0[0, 0, 0] + m1[0, 0, -1] )   
                v1[0, 0, 0] = ( dp0[0, 0, 0] * v1[0, 0, 0] + m1[0, 0, -1] * v1[0, 0, -1] ) / \
                              ( dp0[0, 0, 0] + m1[0, 0, -1] )
                
                udt = udt + (u1 - u0) * rdt
                vdt = vdt + (v1 - v0) * rdt
            
            # Update moist air mass (actually hydrostatic pressure) and convert 
            # to dry mixing ratios
            omq = dp1 / dp0
            qv_dt = qv_dt + rdt * (qvz - qv0) * omq
            ql_dt = ql_dt + rdt * (qlz - ql0) * omq
            qr_dt = qr_dt + rdt * (qrz - qr0) * omq
            qi_dt = qi_dt + rdt * (qiz - qi0) * omq
            qs_dt = qs_dt + rdt * (qsz - qs0) * omq
            qg_dt = qg_dt + rdt * (qgz - qg0) * omq
            
            cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            
            pt_dt = pt_dt + rdt * (tz - t0) * cvm / cp_air
            
            # Update cloud fraction tendency
            if do_qa == 1:
                
                qa_dt = 0.
                
            else:
                
                qa_dt = qa_dt + rdt * (qaz / ntimes - qa0)
            
            # Convert to mm / day
            convt = 86400. * rdt * rgrav
            
            rain    = rain * convt
            snow    = snow * convt
            ice     = ice * convt
            graupel = graupel * convt
