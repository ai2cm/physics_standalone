from microphys.phys_const import *

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
    
    with computation(PARALLEL), interval(...):
        
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
    
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (prog_ccn == 0) and (use_ccn == 1):
                
                # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                ccn = ccn * rdgas * tz / p1
        
        with interval(0, -1):
            
            if (prog_ccn == 0) and (use_ccn == 1):
                
                # Propagate downwards previously computed values of ccn
                ccn = ccn[0, 0, +1]
    
    with computation(PARALLEL), interval(...):
        
        if prog_ccn == 0:
            c_praut = cpaut * (ccn * rhor)**(-1./3.)
        
        # Calculate horizontal subgrid variability
        # Total water subgrid deviation in horizontal direction
        # Default area dependent form: use dx ~ 100 km as the base
        s_leng  = sqrt(sqrt(area * 1.e-10))
        t_land  = dw_land * s_leng
        t_ocean = dw_ocean * s_leng
        h_var   = t_land * land + t_ocean * (1. - land)
        h_var   = min(0.2, max(0.01, h_var))
        
        # Relative humidity increment
        rh_adj  = 1. - h_var - rh_inc
        rh_rain = max(0.35, rh_adj - rh_inr)
        
        # Fix all negative water species
        if fix_negative == 1:
            
            # Define heat capacity and latent heat coefficient
            cvm  = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            lcpk = (lv00 + d0_vap * tz) / cvm
            icpk = (li00 + dc_ice * tz) / cvm
            
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
                tz  = tz - qgz * icpk   # Heating
                qgz = 0.
                
            # Liquid phase
            
            # If rain < 0, borrow from cloud water
            if qrz < 0.:
                
                qlz = qlz + qrz
                qrz = 0.
            
            # If cloud water < 0, borrow from water vapor
            if qlz < 0.:
                
                qvz = qvz + qlz
                tz  = tz - qlz * lcpk   # Heating
                qlz = 0.
    
    with computation(FORWARD), interval(1, None):
        
        # Fix water vapor; borrow from below
        if (fix_negative == 1) and (qvz[0, 0, -1] < 0.):
            qvz[0, 0, 0] = qvz[0, 0, 0] + qvz[0, 0, -1] * dp1[0, 0, -1] / dp1[0, 0, 0]
    
    with computation(PARALLEL), interval(0, -1):
        
        if (fix_negative == 1) and (qvz < 0.): qvz = 0.
    
    # Bottom layer; borrow from above
    with computation(PARALLEL):
        
        with interval(-2, -1):
            
            flag = 0
            
            if (fix_negative == 1) and (qvz[0, 0, +1] < 0.) and (qvz > 0.):
                
                dq   = min(-qvz[0, 0, +1] * dp1[0, 0, +1], qvz[0, 0, 0] * dp1[0, 0, 0])
                flag = 1
        
        with interval(-1, None):
            
            flag = 0
            
            if (fix_negative == 1) and (qvz < 0.) and (qvz[0, 0, -1] > 0.):
                
                dq   = min(-qvz[0, 0, 0] * dp1[0, 0, 0], qvz[0, 0, -1] * dp1[0, 0, -1])
                flag = 1
    
    with computation(PARALLEL):
        
        with interval(-2, -1):
            
            if flag == 1:
                
                qvz = qvz - dq / dp1
        
        with interval(-1, None):
            
            if flag == 1:
                
                qvz = qvz + dq / dp1


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
    
    with computation(PARALLEL), interval(...):
        
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

        '''
        ################################################################ WARM_RAIN
        '''

        # Time-split warm rain processes: 1st pass
        dt5 = 0.5 * dt_rain
        
        # Terminal speed of rain
        m1_rain = 0.
    
    '''
    ####################################################### CHECK_COLUMN
    '''
        
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if qrz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, None):
        
            if no_fall[0, 0, -1] == 1:
                
                if (qrz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    '''
    ################################################### END CHECK_COLUMN
    '''
    
    with computation(PARALLEL), interval(...):
    
        if no_fall == 1:
            
            vtrz = vf_min
            r1   = 0.
        
        else:
            
            # Fall speed of rain
            if const_vr == 1:
                
                vtrz = vr_fac
                
            else:
                
                qden = qrz * den
                
                if qrz < thr:
                    
                    vtrz = vr_min
                
                else:
                
                    vtrz = vr_fac * vconr * sqrt(min(10., sfcrho / den)) * exp(0.2 * log(qden / normr))
                    vtrz = min(vr_max, max(vr_min, vtrz))
    
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if no_fall == 0:
                ze = zs - dz1
                
        with interval(0, -1):
            
            if no_fall == 0:
                ze = ze[0, 0, +1] - dz1    # dz < 0
        
    with computation(PARALLEL), interval(...):
        
        if no_fall == 0:

            '''
            ################################################# REVAP_RACC
            '''
    
            # Evaporation and accretion of rain for the first 1/2 time step
            if (tz > t_wfr) and (qrz > qrmin):
                
                # Define heat capacity and latent heat coefficient
                lhl   = lv00 + d0_vap * tz
                q_liq = qlz + qrz
                q_sol = qiz + qsz + qgz
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                lcpk  = lhl / cvm

                tin = tz - lcpk * qlz    # Presence of clouds suppresses the rain evap
                qpz = qvz + qlz
                
                qsat  = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                        ( rvgas * tin * den )
                dqsdt = qsat * (dc_vap + lv0 / tin) / (rvgas * tin)
                
                dqh     = max(qlz, h_var * max(qpz, qcmin))
                dqh     = min(dqh, 0.2 * qpz)   # New limiter
                dqv     = qsat - qvz            # Use this to prevent super-sat the gird box
                q_minus = qpz - dqh
                q_plus  = qpz + dqh

                # qsat must be > q_minus to activate evaporation
                # qsat must be < q_plus to activate accretion
                # Rain evaporation
                if (dqv > qvmin) and (qsat > q_minus):
                
                    if qsat > q_plus:
                        
                        dq = qsat - qpz
                        
                    else:
                        
                        # q_minus < qsat < q_plus
                        # dq == dqh if qsat == q_minus
                        dq = 0.25 * (q_minus - qsat)**2 / dqh
                    
                    qden = qrz * den
                    t2   = tin * tin
                    evap = crevp_0 * t2 * dq * \
                           ( crevp_1 * sqrt(qden) + crevp_2 * \
                             exp(0.725 * log(qden)) ) / \
                           ( crevp_3 * t2 + crevp_4 * qsat * den )
                    evap = min(qrz, min(dt5 * evap, dqv / (1. + lcpk * dqsdt)))

                    # Alternative minimum evap in dry environmental air
                    qrz   = qrz - evap
                    qvz   = qvz + evap
                    q_liq = q_liq - evap
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz - evap * lhl / cvm
                    
                # Accretion: pracc
                if (qrz > qrmin) and (qlz > 1.e-6) and (qsat < q_minus):
                    
                    sink = dt5 * denfac * cracw * exp(0.95 * log(qrz * den))
                    sink = sink / (1. + sink) * qlz
                    qlz  = qlz - sink
                    qrz  = qrz + sink

            '''
            ############################################# END REVAP_RACC
            '''
            
            if do_sedi_w == 1:
                dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
        
    # Mass flux induced by falling rain
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if (use_ppm == 1) and (no_fall == 0):
                zt = ze
        
        with interval(1, -1):
            
            if (use_ppm == 1) and (no_fall == 0):
                zt = ze - dt5 * (vtrz[0, 0, -1] + vtrz)
        
        with interval(-1, None):
            
            if (use_ppm == 1) and (no_fall == 0):
                
                zt = ze - dt5 * (vtrz[0, 0, -1] + vtrz)
                
                zt_kbot1 = zs - dt_rain * vtrz
            
    with computation(FORWARD):
        
        with interval(1, -1):
            
            if (use_ppm == 1) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min
        
        with interval(-1, None):
            
            if use_ppm == 1:
                
                if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                    zt = zt[0, 0, -1] - dz_min
                    
                if (no_fall == 0) and (zt_kbot1 >= zt):
                    zt_kbot1 = zt - dz_min
                    
    with computation(BACKWARD), interval(0, -1):
        
        if (use_ppm == 1) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, +1]
    
    '''        
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 1) and (no_fall == 0):
            lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qr, r1, 
                                 m1_rain, mono_prof, nf_inv )
    '''

    '''
    ###################################################### IMPLICIT_FALL
    '''

    with computation(PARALLEL):
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]
                
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
                
    with computation(PARALLEL), interval(...):
           
        if (use_ppm == 0) and (no_fall == 0):
            
            dd  = dt_rain * vtrz
            qrz = qrz * dp1
    
    # Sedimentation
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = qrz / (dz + dd)
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = (qrz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 0) and (no_fall == 0):
            
            # qm is density at this stage
            qm = qm * dz
    
    # Output mass fluxes
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = qrz - qm
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = m1_rain[0, 0, -1] + qrz[0, 0, 0] - qm
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                r1 = m1_rain
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                r1 = r1[0, 0, +1]
                
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qrz = qm / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_rain * vtrz) / (dm - m1_rain)
        
        with interval(1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qrz = qm / dp1
                
                '''
                ###################################### END IMPLICIT_FALL
                '''
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                    \
                                   m1_rain[0, 0, -1] * vtrz[0, 0, -1] + \
                                   m1_rain * vtrz ) /                   \
                                 ( dm + m1_rain[0, 0, -1] - m1_rain )
    
    '''
    ########################################################## SEDI_HEAT
    '''

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if (do_sedi_heat == 1) and (no_fall == 0):
                
                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
                
                # - Assumption: The ke in the falling condensates is negligible 
                #               compared to the potential energy that was 
                #               unaccounted for. Local thermal equilibrium is 
                #               assumed, and the loss in pe is transformed into 
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_rain * c_liq
                tz  = tz + m1_rain * dgz / tmp
                
        with interval(1, None):
            
            if (do_sedi_heat == 1) and (no_fall == 0):
                
                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
    
    # Implicit algorithm            
    with computation(FORWARD), interval(1, None):
        
        if (do_sedi_heat == 1) and (no_fall == 0):
            
            tz[0, 0, 0] = ( (cvn + c_liq * (m1_rain - m1_rain[0, 0, -1])) *          \
                            tz[0, 0, 0] + m1_rain[0, 0, -1] * c_liq * tz[0, 0, -1] + \
                            dgz * (m1_rain[0, 0, -1] + m1_rain) ) /                  \
                          ( cvn + c_liq * m1_rain )

    '''
    ###################################################### END SEDI_HEAT
    '''
                          
    with computation(PARALLEL), interval(...):
                
        if no_fall == 0:

            '''
            ################################################# REVAP_RACC
            '''
            
            # Evaporation and accretion of rain for the remaining 1/2 time step
            if (tz > t_wfr) and (qrz > qrmin):
                
                # Define heat capacity and latent heat coefficient
                lhl   = lv00 + d0_vap * tz
                q_liq = qlz + qrz
                q_sol = qiz + qsz + qgz
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                lcpk  = lhl / cvm

                tin = tz - lcpk * qlz    # Presence of clouds suppresses the rain evap
                qpz = qvz + qlz
                
                qsat  = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                        ( rvgas * tin * den )
                dqsdt = qsat * (dc_vap + lv0 / tin) / (rvgas * tin)
                
                dqh     = max(qlz, h_var * max(qpz, qcmin))
                dqh     = min(dqh, 0.2 * qpz)   # New limiter
                dqv     = qsat - qvz            # Use this to prevent super-sat the gird box
                q_minus = qpz - dqh
                q_plus  = qpz + dqh

                # qsat must be > q_minus to activate evaporation
                # qsat must be < q_plus to activate accretion
                # Rain evaporation
                if (dqv > qvmin) and (qsat > q_minus):
                
                    if qsat > q_plus:
                        
                        dq = qsat - qpz
                        
                    else:
                        
                        # q_minus < qsat < q_plus
                        # dq == dqh if qsat == q_minus
                        dq = 0.25 * (q_minus - qsat)**2 / dqh
                    
                    qden = qrz * den
                    t2   = tin * tin
                    evap = crevp_0 * t2 * dq * \
                           ( crevp_1 * sqrt(qden) + crevp_2 * \
                             exp(0.725 * log(qden)) ) / \
                           ( crevp_3 * t2 + crevp_4 * qsat * den )
                    evap = min(qrz, min(dt5 * evap, dqv / (1. + lcpk * dqsdt)))

                    # Alternative minimum evap in dry environmental air
                    qrz   = qrz - evap
                    qvz   = qvz + evap
                    q_liq = q_liq - evap
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz - evap * lhl / cvm
                    
                # Accretion: pracc
                if (qrz > qrmin) and (qlz > 1.e-6) and (qsat < q_minus):
                    
                    sink = dt5 * denfac * cracw * exp(0.95 * log(qrz * den))
                    sink = sink / (1. + sink) * qlz
                    qlz  = qlz - sink
                    qrz  = qrz + sink
                        
            '''
            ############################################# END REVAP_RACC
            '''
        
        # Auto-conversion assuming linear subgrid vertical distribution of 
        # cloud water following lin et al. 1994, mwr
        if irain_f != 0:
    
            # No subgrid variability
            qc0 = fac_rc * ccn
            
            if tz > t_wfr:
            
                if use_ccn == 1:
                    
                    # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                    qc = qc0
                    
                else:
                    
                    qc = qc0 / den
                    
                dq = qlz - qc
                
                if dq > 0.:
                
                    sink = min(dq, dt_rain * c_praut * den * exp(so3 * log(qlz)))
                    qlz  = qlz - sink
                    qrz  = qrz + sink

    '''
    ######################################################## LINEAR_PROF
    '''
    
    # With subgrid variability    
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.
    
        with interval(1, None):
            
            if (irain_f == 0) and (z_slope_liq == 1):
                dq = 0.5 * (qlz[0, 0, 0] - qlz[0, 0, -1])
                
    with computation(PARALLEL): 
        
        with interval(1, -1):
            
            if (irain_f == 0) and (z_slope_liq == 1):
                
                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                dl = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qlz[0, 0, 0])
                
                if dq * dq[0, 0, +1] <= 0.:
                
                    if dq > 0.: # Local maximum
                
                        dl = min(dl, min(dq, -dq[0, 0, +1]))
                
                    else:
                        
                        dl = 0.
        
        with interval(-1, None):
            
            if (irain_f == 0) and (z_slope_liq == 1): dl = 0.
            
    with computation(PARALLEL), interval(...):
        
        if irain_f == 0:
            
            if z_slope_liq == 1:
                
                # Impose a presumed background horizontal variability that is 
                # proportional to the value itself
                dl = max(dl, max(qvmin, h_var * qlz))
        
            else:
                
                dl = max(qvmin, h_var * qlz)

            '''
            ############################################ END LINEAR_PROF
            '''
            
            qc0 = fac_rc * ccn
            
            if tz > t_wfr + dt_fr:
                
                dl = min(max(1.e-6, dl), 0.5 * qlz)
                
                # As in klein's gfdl am2 stratiform scheme (with subgrid variations)
                if use_ccn == 1:
                    
                    # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                    qc = qc0
                    
                else:
                    
                    qc = qc0 / den
                
                dq = 0.5 * (qlz + dl - qc)
                
                # dq = dl if qc == q_minus = ql - dl
                # dq = 0 if qc == q_plus = ql + dl
                if dq > 0.:     # q_plus > qc
                    
                    # Revised continuous form: linearly decays (with subgrid dl) to zero at qc == ql + dl
                    sink = min(1., dq / dl) * dt_rain * c_praut * den * exp(so3 * log(qlz))
                    qlz  = qlz - sink
                    qrz  = qrz + sink

        '''
        ################################################################ END WARM_RAIN
        '''
        
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
    
    with computation(PARALLEL), interval(...):
        
        '''
        ################################################################ FALL_SPEED
        '''
        
        # Sedimentation of cloud ice, snow, and graupel
        # Marshall-Palmer formula
        
        # Try the local air density -- for global model; the true value 
        # could be much smaller than sfcrho over high mountains
        rhof = sqrt(min(10., sfcrho / den))
        
        # Ice
        if const_vi == 1:
            
            vtiz = vi_fac
            
        else:
            
            # Use deng and mace (2008, grl), which gives smaller fall speed 
            # than hd90 formula
            vi0 = 0.01 * vi_fac
            
            if qiz < thi:
            
                vtiz = vf_min
            
            else:
            
                tc   = tz - tice
                vtiz = (3. + log(qiz * den) / log_10) * (tc * (aa * tc + bb) + cc) + dd_fs * tc + ee
                vtiz = vi0 * exp(log_10 * vtiz) * 0.8 
                vtiz = min(vi_max, max(vf_min, vtiz))
        
        # Snow
        if const_vs == 1:
            
            vtsz = vs_fac
            
        else:
            
            if qsz < ths:
            
                vtsz = vf_min
            
            else:
                
                vtsz = vs_fac * vcons * rhof * exp(0.0625 * log(qsz * den / norms))
                vtsz = min(vs_max, max(vf_min, vtsz))
        
        # Graupel
        if const_vg == 1:
            
            vtgz = vg_fac
            
        else:
            
            if qgz < thg:
            
                vtgz = vf_min
            
            else:
            
                vtgz = vg_fac * vcong * rhof * sqrt(sqrt(sqrt(qgz * den / normg)))
                vtgz = min(vg_max, max(vf_min, vtgz))
        
        '''
        ################################################################ END FALL_SPEED
        '''

        '''
        ################################################################ TERMINAL_FALL
        '''
        
        dt5 = 0.5 * dts
        
        # Define heat capacity and latent heat coefficient
        m1_sol = 0.
        
        lhi   = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk  = lhi / cvm
    
    # Find significant melting level
    '''
    k0 removed to avoid having to introduce a k_idx field
    '''    
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if tz > tice: stop_k = 1
            else:         stop_k = 0
            
        with interval(1, -1):
            
            if stop_k[0, 0, -1] == 0:
                
                if tz > tice: stop_k = 1
                else:         stop_k = 0
                
            else:
                
                stop_k = 1
                
        with interval(-1, None):
            
            stop_k = 1
    
    with computation(PARALLEL), interval(...):
        
        if stop_k == 1:
            
            # Melting of cloud ice (before fall)
            tc = tz - tice
            
            if (qiz > qcmin) and (tc > 0.):
            
                sink  = min(qiz, fac_imlt * tc / icpk)
                tmp   = min(sink, dim(ql_mlt, qlz))
                qlz   = qlz + tmp
                qrz   = qrz + sink - tmp
                qiz   = qiz - sink
                q_liq = q_liq + sink
                q_sol = q_sol - sink
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz    = tz - sink * lhi / cvm
                tc    = tz - tice
                
    with computation(PARALLEL), interval(0, -1):
        
        # Turn off melting when cloud microphysics time step is small
        if dts < 60.:
            stop_k = 0
            
        # sjl, turn off melting of falling cloud ice, snow and graupel
        stop_k = 0
        
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            ze = zs - dz1
                
        with interval(1, -1):
            
            ze = ze[0, 0, +1] - dz1    # dz < 0
            
        with interval(0, 1):
            
            ze = ze[0, 0, +1] - dz1    # dz < 0
            zt = ze
            
    with computation(PARALLEL), interval(...):
        
        if stop_k == 1:
        
            # Update capacity heat and latent heat coefficient
            lhi  = li00 + dc_ice * tz
            icpk = lhi / cvm

    '''
    ####################################################### CHECK_COLUMN
    '''

    # Melting of falling cloud ice into rain
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if qiz > qrmin: no_fall = 0
            else:           no_fall = 1
        
        with interval(1, None):
        
            if no_fall[0, 0, -1] == 1:
                
                if (qiz > qrmin): no_fall = 0
                else:             no_fall = 1
                
            else:
                
                no_fall = 0
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    '''
    ################################################### END CHECK_COLUMN
    '''
            
    with computation(PARALLEL), interval(...):
        
        if (vi_fac < 1.e-5) or (no_fall == 1): i1 = 0.
        
    with computation(PARALLEL):
        
        with interval(1, -1):
        
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                zt = ze - dt5 * (vtiz[0, 0, -1] + vtiz)
                
        with interval(-1, None):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                zt       = ze - dt5 * (vtiz[0, 0, -1] + vtiz)
                zt_kbot1 = zs - dts * vtiz
                
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
        
    with computation(PARALLEL), interval(...):
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
        
        if (vi_fac >= 1.e-5) and (no_fall == 0):
            
            if do_sedi_w == 1:
                dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
            
            '''
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qi, 
                                     i1, m1_sol, mono_prof, ij_vi_inv )
            '''

    '''
    ###################################################### IMPLICIT_FALL
    '''
                       
    with computation(PARALLEL):
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]
                
        with interval(-1, None):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                dz = ze - zs
                
    with computation(PARALLEL), interval(...):
           
        if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
            
            dd  = dts * vtiz
            qiz = qiz * dp1
    
    # Sedimentation
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                qm = qiz / (dz + dd)
            
        with interval(1, None):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                qm = (qiz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
            
            # qm is density at this stage
            qm = qm * dz
    
    # Output mass fluxes
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                m1_sol = qiz - qm
            
        with interval(1, None):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                m1_sol = m1_sol[0, 0, -1] + qiz[0, 0, 0] - qm
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                i1 = m1_sol
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (vi_fac >= 1.e-5) and (no_fall == 0):
                i1 = i1[0, 0, +1]
                
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if use_ppm == 0:
                    
                    # Update
                    qiz = qm / dp1
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_sol * vtiz) / (dm - m1_sol)
        
        with interval(1, None):
            
            if (vi_fac >= 1.e-5) and (no_fall == 0):
                
                if use_ppm == 0:
                    
                    # Update
                    qiz = qm / dp1
                
                '''
                ###################################### END IMPLICIT_FALL
                '''
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                   \
                                   m1_sol[0, 0, -1] * vtiz[0, 0, -1] + \
                                   m1_sol * vtiz ) /                   \
                                 ( dm + m1_sol[0, 0, -1] - m1_sol )

    '''
    ####################################################### CHECK_COLUMN
    '''

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
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    '''
    ################################################### END CHECK_COLUMN
    '''
            
    with computation(PARALLEL), interval(...):
        
        r1 = 0.
        
        if no_fall == 1: s1 = 0.
        
    with computation(PARALLEL):
        
        with interval(1, -1):
        
            if no_fall == 0:
                zt = ze - dt5 * (vtsz[0, 0, -1] + vtsz)
                
        with interval(-1, None):
            
            if no_fall == 0:
                
                zt       = ze - dt5 * (vtsz[0, 0, -1] + vtsz)
                zt_kbot1 = zs - dts * vtsz
                
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
    
    with computation(PARALLEL), interval(...):
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
        
        if no_fall == 0:
            
            if do_sedi_w == 1:
                dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
            
            '''
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qs, 
                                     s1, m1_tf, mono_prof, nf_inv )
            '''

    '''
    ###################################################### IMPLICIT_FALL
    '''

    with computation(PARALLEL):
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]
                
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
                
    with computation(PARALLEL), interval(...):
           
        if (use_ppm == 0) and (no_fall == 0):
            
            dd  = dts * vtsz
            qsz = qsz * dp1
    
    # Sedimentation
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = qsz / (dz + dd)
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = (qsz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 0) and (no_fall == 0):
            
            # qm is density at this stage
            qm = qm * dz
            
    # Output mass fluxes
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qsz - qm
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, -1] + qsz[0, 0, 0] - qm
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                s1 = m1_tf
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                s1 = s1[0, 0, +1]
                
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qsz = qm / dp1
                    
                m1_sol = m1_sol + m1_tf
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtsz) / (dm - m1_tf)
        
        with interval(1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qsz = qm / dp1
                
                '''
                ###################################### END IMPLICIT_FALL
                '''
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtsz[0, 0, -1] + \
                                   m1_tf * vtsz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )

    '''
    ####################################################### CHECK_COLUMN
    '''
    
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
                
    with computation(BACKWARD), interval(0, -1):
        
        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    '''
    ################################################### END CHECK_COLUMN
    '''
            
    with computation(PARALLEL), interval(...):
        
        if no_fall == 1: g1 = 0.
        
    with computation(PARALLEL):
        
        with interval(1, -1):
        
            if no_fall == 0:
                zt = ze - dt5 * (vtgz[0, 0, -1] + vtgz)
                
        with interval(-1, None):
            
            if no_fall == 0:
                
                zt       = ze - dt5 * (vtgz[0, 0, -1] + vtgz)
                zt_kbot1 = zs - dts * vtgz
                
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
    
    with computation(PARALLEL), interval(...):
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
        
        if no_fall == 0:
            
            if do_sedi_w == 1:
                dm = dp1 * (1. + qvz + qlz + qrz + qiz + qsz + qgz)
            
            '''
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qg, 
                                     g1, m1_tf, mono_prof, nf_inv )
            '''

    '''
    ###################################################### IMPLICIT_FALL
    '''

    with computation(PARALLEL):
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]
                
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
                
    with computation(PARALLEL), interval(...):
           
        if (use_ppm == 0) and (no_fall == 0):
            
            dd  = dts * vtgz
            qgz = qgz * dp1
    
    # Sedimentation
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = qgz / (dz + dd)
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                qm = (qgz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)
                
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 0) and (no_fall == 0):
            
            # qm is density at this stage
            qm = qm * dz
            
    # Output mass fluxes
    with computation(FORWARD):
        
        with interval(0, 1):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qgz - qm
            
        with interval(1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, -1] + qgz[0, 0, 0] - qm
            
    with computation(BACKWARD):
        
        with interval(-1, None):
            
            if (use_ppm == 0) and (no_fall == 0):
                g1 = m1_tf
        
        with interval(0, -1):
            
            if (use_ppm == 0) and (no_fall == 0):
                g1 = g1[0, 0, +1]
    
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qgz = qm / dp1
                    
                m1_sol = m1_sol + m1_tf
                
                # Vertical velocity transportation during sedimentation                    
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtgz) / (dm - m1_tf)
        
        with interval(1, None):
            
            if no_fall == 0:
                
                if use_ppm == 0:
                    
                    # Update
                    qgz = qm / dp1
                
                '''
                ###################################### END IMPLICIT_FALL
                '''
                    
                m1_sol = m1_sol + m1_tf
                    
                if do_sedi_w == 1:
                    
                    w[0, 0, 0] = ( dm * w[0, 0, 0] -                  \
                                   m1_tf[0, 0, -1] * vtgz[0, 0, -1] + \
                                   m1_tf * vtgz ) /                   \
                                 ( dm + m1_tf[0, 0, -1] - m1_tf )

    '''
    #################################################################### END TERMINAL_FALL
    '''
                                   
    with computation(PARALLEL), interval(...):
        
        rain    = rain + r1     # From melted snow and ice that reached the ground
        snow    = snow + s1
        graupel = graupel + g1
        ice     = ice + i1

    '''
    #################################################################### SEDI_HEAT
    '''

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if do_sedi_heat == 1:
                
                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
                
                # - Assumption: The ke in the falling condensates is negligible 
                #               compared to the potential energy that was 
                #               unaccounted for. Local thermal equilibrium is 
                #               assumed, and the loss in pe is transformed into 
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_sol * c_ice
                tz  = tz + m1_sol * dgz / tmp
                
        with interval(1, None):
            
            if do_sedi_heat == 1:
                
                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (cv_air + qvz * cv_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice)
    
    # Implicit algorithm            
    with computation(FORWARD), interval(1, None):
        
        if do_sedi_heat == 1:
            
            tz[0, 0, 0] = ( (cvn + c_ice * (m1_sol - m1_sol[0, 0, -1])) *           \
                            tz[0, 0, 0] + m1_sol[0, 0, -1] * c_ice * tz[0, 0, -1] + \
                            dgz * (m1_sol[0, 0, -1] + m1_sol) ) /                   \
                          ( cvn + c_ice * m1_sol )

    '''
    #################################################################### END SEDI_HEAT
    '''


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
    
    with computation(PARALLEL), interval(...):

        '''
        ################################################################ ICLOUD
        '''

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
        '''
        THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
        '''
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
            '''
            THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
            dtmp   = t_wfr - tz
            '''
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

    '''
    ######################################################## LINEAR_PROF
    '''
        
    # Vertical subgrid variability
    with computation(PARALLEL):
        
        with interval(0, 1):
            
            if z_slope_ice == 1:
                di = 0.
    
        with interval(1, None):
            
            if z_slope_ice == 1:
                dq = 0.5 * (qiz[0, 0, 0] - qiz[0, 0, -1])
                
    with computation(PARALLEL): 
        
        with interval(1, -1):
            
            if z_slope_ice == 1:
                
                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                di = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qiz[0, 0, 0])
                
                if dq * dq[0, 0, +1] <= 0.:
                
                    if dq > 0.: # Local maximum
                
                        di = min(di, min(dq, -dq[0, 0, +1]))
                
                    else:
                        
                        di = 0.
        
        with interval(-1, None):
            
            if z_slope_ice == 1: di = 0.
            
    with computation(PARALLEL), interval(...):
            
        if z_slope_ice == 1:
            
            # Impose a presumed background horizontal variability that is 
            # proportional to the value itself
            di = max(di, max(qvmin, h_var * qiz))
    
        else:
            
            di = max(qvmin, h_var * qiz)

        '''
        ################################################ END LINEAR_PROF
        '''
        
        # Update capacity heat and latent heat coefficient
        lhi  = li00 + dc_ice * tz
        icpk = lhi / cvm
        
        if p1 >= p_min:
            
            pgacr = 0.
            pgacw = 0.
            
            tc = tz - tice
            
            if tc >= 0.:
            
                # Melting of snow
                dqs0 = ces0 / p1 - qvz
                
                if qsz > qcmin:
                
                    # psacw: accretion of cloud water by snow (only rate is used (for 
                    # snow melt) since tc > 0.)
                    if qlz > qrmin:
                        
                        factor = denfac * csacw * exp(0.8125 * log(qsz * den))
                        psacw  = factor / (1. + dts * factor) * qlz # Rate
                        
                    else:
                    
                        psacw = 0.
                    
                    # psacr: accretion of rain by melted snow
                    # pracs: accretion of snow by rain
                    if qrz > qrmin:
                        
                        t1 = sqrt(qrz * den)
                        t2 = sqrt(t1)
                        s1 = sqrt(qsz * den)
                        s2 = sqrt(s1)
                               
                        psacr = min( csacr * abs(vtsz - vtrz) * qrz * s2 * ( acco_01 * t1 + acco_11 * t2 * s2 + acco_21 * s1 ), 
                                     qrz * rdts )
                        pracs = cracs * abs(vtrz - vtsz) * qsz * t2 * ( acco_00 * s1 + acco_10 * s2 * t2 + acco_20 * t1 )
                    
                    else:
                                          
                        psacr = 0.
                        pracs = 0.
                    
                    # Total snow sink
                    # psmlt: snow melt (due to rain accretion)
                    psmlt = max( 0., ( csmlt_0 * tc / den - csmlt_1 * dqs0 ) *                                       \
                                     ( csmlt_2 * sqrt(qsz * den) + csmlt_3 * (qsz * den)**0.65625 * sqrt(denfac) ) + \
                                     csmlt_4 * tc * (psacw + psacr) )
                    sink  = min(qsz, min(dts * (psmlt + pracs), tc / icpk))
                    qsz   = qsz - sink
                    tmp   = min(sink, dim(qs_mlt, qlz)) # Maximum ql due to snow melt
                    qlz   = qlz + tmp
                    qrz   = qrz + sink - tmp
                    q_liq = q_liq + sink
                    q_sol = q_sol - sink
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz - sink * lhi / cvm
                    tc    = tz - tice
                
                # Update capacity heat and latent heat coefficient
                lhi  = li00 + dc_ice * tz
                icpk = lhi / cvm
                
                # Melting of graupel
                if (qgz > qcmin) and (tc > 0.):
                
                    # pgacr: accretion of rain by graupel
                    if qrz > qrmin:
                        
                        t1 = sqrt(qrz * den)
                        s1 = sqrt(qgz * den)
                        s2 = sqrt(s1)
                               
                        pgacr = min( cgacr * abs(vtgz - vtrz) * qrz * s2 * ( acco_02 * t1 + acco_12 * sqrt(t1) * s2 + acco_22 * s1 ), 
                                     rdts * qrz )
                
                    # pgacw: accretion of cloud water by graupel                
                    qden = qgz * den
                    
                    if qlz > qrmin:
                
                        factor = cgacw * qden / sqrt(den * sqrt(sqrt(qden)))
                        pgacw  = factor / (1. + dts * factor) * qlz # Rate
                
                    # pgmlt: graupel melt
                    pgmlt = dts * ( ( cgmlt_0 * tc / den - cgmlt_1 * dqs0 ) *                       \
                                    ( cgmlt_2 * sqrt(qden) + cgmlt_3 * qden**0.6875 / den**0.25 ) + \
                                    cgmlt_4 * tc * (pgacw + pgacr) )
                    pgmlt = min(max(0., pgmlt), min(qgz, tc / icpk))
                    qgz   = qgz - pgmlt
                    qrz   = qrz + pgmlt
                    q_liq = q_liq + pgmlt
                    q_sol = q_sol - pgmlt
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz - pgmlt * lhi / cvm
                
            else:
                
                # Cloud ice proc
                # psaci: accretion of cloud ice by snow
                if qiz > 3.e-7: # Cloud ice sink terms
                
                    if qsz > 1.e-7:
                    
                        # sjl added (following lin eq. 23) the temperature dependency to 
                        # reduce accretion, use esi = exp(0.05 * tc) as in hong et al 2004
                        factor = dts * denfac * csaci * exp(0.05 * tc + 0.8125 * log(qsz * den))
                        psaci  = factor / (1. + factor) * qiz
                    
                    else:
                    
                        psaci = 0.
                    
                    # pasut: autoconversion: cloud ice -- > snow
                    # - Similar to lfo 1983: eq. 21 solved implicitly
                    # - Threshold from wsm6 scheme, hong et al 2004, eq (13) : qi0_crt ~0.8e-4
                    qim = qi0_crt / den
                    
                    # - Assuming linear subgrid vertical distribution of cloud ice
                    # - The mismatch computation following lin et al. 1994, mwr
                    if const_vi == 1:
                        
                        tmp = fac_i2s
                        
                    else:
                        
                        tmp = fac_i2s * exp(0.025 * tc)
                    
                    di     = max(di, qrmin)
                    q_plus = qiz + di
                    
                    if q_plus > (qim + qrmin):
                    
                        if qim > (qiz - di):
                    
                            dq = (0.25 * (q_plus - qim)**2) / di
                    
                        else:
                            
                            dq = qiz - qim
                    
                        psaut = tmp * dq
                    
                    else:
                    
                        psaut = 0.
                    
                    # sink is no greater than 75% of qi
                    sink = min(0.75 * qiz, psaci + psaut)
                    qiz  = qiz - sink
                    qsz  = qsz + sink
                    
                    # pgaci: accretion of cloud ice by graupel
                    if qgz > 1.e-6:
                    
                        # - factor = dts * cgaci / sqrt (den (k)) * exp (0.05 * tc + 0.875 * log (qg * den (k)))
                        # - Simplified form: remove temp dependency & set the exponent "0.875" -- > 1
                        factor = dts * cgaci * sqrt(den) * qgz
                        pgaci  = factor / (1. + factor) * qiz
                        qiz    = qiz - pgaci
                        qgz    = qgz + pgaci
                
                # Cold-rain proc
                # Rain to ice, snow, graupel processes
                tc = tz - tice
                
                if (qrz > 1e-7) and (tc < 0.):

                    # - Sink terms to qr: psacr + pgfr
                    # - Source terms to qs: psacr
                    # - Source terms to qg: pgfr
                    # psacr accretion of rain by snow
                    if (qsz > 1.e-7):   # If snow exists
                        
                        t1 = sqrt(qrz * den)
                        s1 = sqrt(qsz * den)
                        s2 = sqrt(s1)
                        
                        psacr = dts * csacr * abs(vtsz - vtrz) * qrz * s2 * ( acco_01 * t1 + acco_11 * sqrt(t1) * s2 + acco_21 * s1 )
                                                 
                    else:
                    
                        psacr = 0.
                    
                    # pgfr: rain freezing -- > graupel
                    pgfr = dts * cgfr_0 / den * (exp(-cgfr_1 * tc) - 1.) * exp(1.75 * log(qrz * den))

                    # Total sink to qr
                    sink   = psacr + pgfr
                    factor = min(sink, min(qrz, -tc / icpk)) / max(sink, qrmin)
                    
                    psacr = factor * psacr
                    pgfr  = factor * pgfr
                    
                    sink  = psacr + pgfr
                    qrz   = qrz - sink
                    qsz   = qsz + psacr
                    qgz   = qgz + pgfr
                    q_liq = q_liq - sink
                    q_sol = q_sol + sink
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz + sink * lhi / cvm
                
                # Update capacity heat and latent heat coefficient
                lhi  = li00 + dc_ice * tz
                icpk = lhi / cvm
                
                # Graupel production terms
                if qsz > 1.e-7:
                
                    # Accretion: snow -- > graupel
                    if qgz > qrmin:
                        
                        t1 = sqrt(qsz * den)
                        s1 = sqrt(qgz * den)
                        s2 = sqrt(s1)
                        
                        sink = dts * cgacs * abs(vtgz - vtsz) * qsz * s2 * ( acco_03 * t1 + acco_13 * sqrt(t1) * s2 + acco_23 * s1 )
                    
                    else:
                    
                        sink = 0.
                    
                    # Autoconversion snow -- > graupel
                    qsm = qs0_crt / den
                    
                    if qsz > qsm:
                    
                        factor = dts * 1.e-3 * exp(0.09 * (tz - tice))
                        sink   = sink + factor / (1. + factor) * (qsz - qsm)
                    
                    sink = min(qsz, sink)
                    qsz  = qsz - sink
                    qgz  = qgz + sink
                
                if (qgz > 1.e-7) and (tz < tice0):
                
                    # pgacw: accretion of cloud water by graupel
                    if qlz > 1.e-6:
                    
                        qden   = qgz * den
                        factor = dts * cgacw * qden / sqrt(den * sqrt(sqrt(qden)))
                        pgacw  = factor / (1. + factor) * qlz
                    
                    else:
                    
                        pgacw = 0.
                    
                    # pgacr: accretion of rain by graupel
                    if qrz > 1.e-6:
                        
                        t1 = sqrt(qrz * den)
                        s1 = sqrt(qgz * den)
                        s2 = sqrt(s1)
                        
                        pgacr = min( dts * cgacr * abs(vtgz - vtrz) * qrz * s2 * ( acco_02 * t1 + acco_12 * sqrt(t1) * s2 + acco_22 * s1 ), 
                                     qrz )
                                                    
                    else:
                    
                        pgacr = 0.
                    
                    sink   = pgacr + pgacw
                    factor = min(sink, dim(tice, tz) / icpk) / max(sink, qrmin)
                    pgacr  = factor * pgacr
                    pgacw  = factor * pgacw
                    
                    sink  = pgacr + pgacw
                    qgz   = qgz + sink
                    qrz   = qrz - pgacr
                    qlz   = qlz - pgacw
                    q_liq = q_liq - sink
                    q_sol = q_sol + sink
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz + sink * lhi / cvm
        
        # Subgrid cloud microphysics
        # Define heat capacity and latent heat coefficient
        lhl   = lv00 + d0_vap * tz
        lhi   = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        lcpk  = lhl / cvm
        icpk  = lhi / cvm
        tcpk  = lcpk + icpk
        tcp3  = lcpk + icpk * min(1., dim(tice, tz) / (tice - t_wfr))
        
        if p1 >= p_min:
        
            # Instant deposit all water vapor to cloud ice when temperature is super low
            if tz < t_min:
                
                sink  = dim(1.e-7, qvz)
                qvz   = qvz - sink
                qiz   = qiz + sink
                q_sol = q_sol + sink
                cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz    = tz + sink * (lhl + lhi) / cvm
                
                if do_qa == 0:
                    qaz = qaz + 1.  # Air fully saturated; 100% cloud cover
            
            else:
                
                # Update heat capacity and latent heat coefficient
                lhl  = lv00 + d0_vap * tz
                lhi  = li00 + dc_ice * tz
                lcpk = lhl / cvm
                icpk = lhi / cvm
                tcpk = lcpk + icpk
                tcp3 = lcpk + icpk * min(1., dim(tice, tz) / (tice - t_wfr))
                
                # Instant evaporation / sublimation of all clouds if rh < rh_adj -- > cloud free                
                qpz = qvz + qlz + qiz
                tin = tz - ( lhl * (qlz + qiz) + lhi * qiz ) / \
                           ( c_air + qpz * c_vap + qrz * c_liq + (qsz + qgz) * c_ice )
                
                if tin > t_sub + 6.:
                    
                    if tin < t_ice:
        
                        # Over ice between -160 degrees Celsius and 0 degrees Celsius
                        if tin >= t_ice - 160.:
                        
                            iqs1 = ( e00 * exp((d2ice * log(tin / t_ice) + li2 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                   ( rvgas * tin * den )
                        
                        else:
                        
                            iqs1 = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                   ( rvgas * (t_ice - 160.) * den )
                    else:
                        
                        # Over water between 0 degrees Celsius and 102 degrees Celsius
                        if tin <= t_ice + 102.:
                        
                            iqs1 = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                   ( rvgas * tin * den )
                        
                        else:
                            
                            iqs1 = ( e00 * exp((dc_vap * log(1. + 102. / t_ice) + lv0 * 102. / ((t_ice + 102.) * t_ice)) / rvgas) ) / \
                                   ( rvgas * (t_ice + 102.) * den )
                    
                    rh = qpz / iqs1
                
                    if rh < rh_adj: # qpz / rh_adj < qs
                        
                        tz  = tin
                        qvz = qpz
                        qlz = 0.
                        qiz = 0.
                        
                if ((tin > t_sub + 6.) and (rh >= rh_adj)) or (tin <= t_sub + 6.):
                
                    # Cloud water < -- > vapor adjustment
                    qsw   = ( e00 * exp((dc_vap * log(tz / t_ice) + lv0 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                            ( rvgas * tz * den )
                    dwsdt = qsw * (dc_vap + lv0 / tz) / (rvgas * tz)
                    
                    dq0 = qsw - qvz
                    
                    if dq0 > 0.:
                    
                        # Added ql factor to prevent the situation of high ql and low RH
                        factor = min(1., fac_l2v * (10. * dq0 / qsw))   # The rh dependent factor = 1 at 90%
                        evap   = min(qlz, factor * dq0 / (1. + tcp3 * dwsdt))
                    
                    else:   # Condensate all excess vapor into cloud water
                        
                        evap = dq0 / (1. + tcp3 * dwsdt)
                    
                    qvz   = qvz + evap
                    qlz   = qlz - evap
                    q_liq = q_liq - evap
                    cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz    = tz - evap * lhl / cvm
                    
                    # Update heat capacity and latent heat coefficient
                    lhi  = li00 + dc_ice * tz
                    icpk = lhi / cvm
                    
                    # Enforce complete freezing below -48 degrees Celsius
                    dtmp = t_wfr - tz   # [-40, -48]
                    
                    if (dtmp > 0.) and (qlz > qcmin):
                    
                        sink  = min(qlz, min(qlz * dtmp * 0.125, dtmp / icpk))
                        qlz   = qlz - sink
                        qiz   = qiz + sink
                        q_liq = q_liq - sink
                        q_sol = q_sol + sink
                        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz    = tz + sink * lhi / cvm
                    
                    # Update heat capacity and latent heat coefficient
                    lhi  = li00 + dc_ice * tz
                    icpk = lhi / cvm
                    
                    # Bigg mechanism
                    if fast_sat_adj == 1:
                        
                        dt_pisub = 0.5 * dts
                        
                    else:
                        
                        dt_pisub = dts
                        
                        tc = tice - tz
                        
                        if (qlz > qrmin) and (tc > 0.):
                        
                            sink  = 3.3333e-10 * dts * (exp(0.66 * tc) - 1.) * den * qlz * qlz
                            sink  = min(qlz, min(tc / icpk, sink))
                            qlz   = qlz - sink
                            qiz   = qiz + sink
                            q_liq = q_liq - sink
                            q_sol = q_sol + sink
                            cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                            tz    = tz + sink * lhi / cvm
                        
                    # Update capacity heat and latent heat coefficient
                    lhl  = lv00 + d0_vap * tz
                    lhi  = li00 + dc_ice * tz
                    lcpk = lhl / cvm
                    icpk = lhi / cvm
                    tcpk = lcpk + icpk
                    
                    # Sublimation / deposition of ice
                    if tz < tice:
                        
                        # Over ice between -160 degrees Celsius and 0 degrees Celsius 
                        if tz >= t_ice - 160.:
                            
                            qsi   = ( e00 * exp((d2ice * log(tz / t_ice) + li2 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                    ( rvgas * tz * den )
                            dqsdt = qsi * (d2ice + li2 / tz) / (rvgas * tz)
                        
                        else:
                            
                            qsi   = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                    ( rvgas * (t_ice - 160.) * den )
                            dqsdt = qsi * (d2ice + li2 / (t_ice - 160.)) / (rvgas * (t_ice - 160.))
                        
                        dq   = qvz - qsi
                        sink = dq / (1. + tcpk * dqsdt)
                    
                        if qiz > qrmin:
                    
                            # - Eq 9, hong et al. 2004, mwr
                            # - For a and b, see dudhia 1989: page 3103 eq (b7) and (b8)
                            pidep = dt_pisub * dq * 349138.78 * exp(0.875 * log(qiz * den)) / \
                                    ( qsi * den * lat2 / (0.0243 * rvgas * tz**2) + 4.42478e4 )
                    
                        else:
                            
                            pidep = 0.
                    
                        if dq > 0.: # Vapor -- > ice
                    
                            tmp = tice - tz
                            
                            # The following should produce more ice at higher altitude
                            qi_crt = qi_gen * min(qi_lim, 0.1 * tmp) / den
                            sink   = min(sink, min(max(qi_crt - qiz, pidep), tmp / tcpk))
                    
                        else:   # Ice -- > vapor
                    
                            pidep = pidep * min(1., dim(tz, t_sub) * 0.2)
                            sink  = max(pidep, max(sink, -qiz))
                    
                        qvz   = qvz - sink
                        qiz   = qiz + sink
                        q_sol = q_sol + sink
                        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz    = tz + sink * (lhl + lhi) / cvm
                    
                    # Update capacity heat and latent heat coefficient
                    lhl  = lv00 + d0_vap * tz
                    lhi  = li00 + dc_ice * tz
                    lcpk = lhl / cvm
                    icpk = lhi / cvm
                    tcpk = lcpk + icpk
                    
                    # - Sublimation / deposition of snow
                    # - This process happens for the whole temperature range
                    if qsz > qrmin:
                        
                        if tz < t_ice:
                            
                            # Over ice between -160 degrees Celsius and 0 degrees Celsius 
                            if tz >= t_ice - 160.:
                                
                                qsi   = ( e00 * exp((d2ice * log(tz / t_ice) + li2 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                        ( rvgas * tz * den )
                                dqsdt = qsi * (d2ice + li2 / tz) / (rvgas * tz)
                            
                            else:
                                
                                qsi   = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                        ( rvgas * (t_ice - 160.) * den )
                                dqsdt = qsi * (d2ice + li2 / (t_ice - 160.)) / (rvgas * (t_ice - 160.))
                        
                        else:
                            
                            # Over water between 0 degrees Celsius and 102 degrees Celsius
                            if tz <= t_ice + 102.:
                                
                                qsi   = ( e00 * exp((dc_vap * log(tz / t_ice) + lv0 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                        ( rvgas * tz * den )
                                dqsdt = qsi * (dc_vap + lv0 / tz) / (rvgas * tz)
                            
                            else:
                                
                                qsi   = ( e00 * exp((dc_vap * log(1. + 102. / t_ice) + lv0 * 102. / ((t_ice + 102.) * t_ice)) / rvgas) ) / \
                                        ( rvgas * (t_ice + 102.) * den )
                                dqsdt = qsi * (dc_vap + lv0 / (t_ice + 102.)) / (rvgas * (t_ice + 102.))
                        
                        qden  = qsz * den
                        tmp   = exp(0.65625 * log(qden))
                        tsq   = tz * tz
                        dq    = (qsi - qvz) / (1. + tcpk * dqsdt)
                        pssub = cssub_0 * tsq *                                           \
                                ( cssub_1 * sqrt(qden) + cssub_2 * tmp * sqrt(denfac) ) / \
                                ( cssub_3 * tsq + cssub_4 * qsi * den )
                        pssub = (qsi - qvz) * dts * pssub
                    
                        if pssub > 0.:  # qs -- > qv, sublimation
                    
                            pssub = min(pssub * min(1., dim(tz, t_sub) * 0.2), qsz)
                            
                        else:
                            
                            if tz > tice:
                                
                                pssub = 0.  # No deposition
                                
                            else:
                                
                                pssub = max(pssub, max(dq, (tz - tice) / tcpk))
                        
                        qsz   = qsz - pssub
                        qvz   = qvz + pssub
                        q_sol = q_sol - pssub
                        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz    = tz - pssub * (lhl + lhi) / cvm
                    
                    # Update capacity heat and latent heat coefficient
                    lhl  = lv00 + d0_vap * tz
                    lhi  = li00 + dc_ice * tz
                    lcpk = lhl / cvm
                    icpk = lhi / cvm
                    tcpk = lcpk + icpk
                    
                    # Simplified 2-way grapuel sublimation-deposition mechanism
                    if qgz > qrmin:
                        
                        if tz < t_ice:
                            
                            # Over ice between -160 degrees Celsius and 0 degrees Celsius 
                            if tz >= t_ice - 160.:
                                
                                qsi   = ( e00 * exp((d2ice * log(tz / t_ice) + li2 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                        ( rvgas * tz * den )
                                dqsdt = qsi * (d2ice + li2 / tz) / (rvgas * tz)
                            
                            else:
                                
                                qsi   = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                        ( rvgas * (t_ice - 160.) * den )
                                dqsdt = qsi * (d2ice + li2 / (t_ice - 160.)) / (rvgas * (t_ice - 160.))
                        
                        else:
                            
                            # Over water between 0 degrees Celsius and 102 degrees Celsius
                            if tz <= t_ice + 102.:
                                
                                qsi   = ( e00 * exp((dc_vap * log(tz / t_ice) + lv0 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                        ( rvgas * tz * den )
                                dqsdt = qsi * (dc_vap + lv0 / tz) / (rvgas * tz)
                            
                            else:
                                
                                qsi   = ( e00 * exp((dc_vap * log(1. + 102. / t_ice) + lv0 * 102. / ((t_ice + 102.) * t_ice)) / rvgas) ) / \
                                        ( rvgas * (t_ice + 102.) * den )
                                dqsdt = qsi * (dc_vap + lv0 / (t_ice + 102.)) / (rvgas * (t_ice + 102.))
                        
                        dq    = (qvz - qsi) / (1. + tcpk * dqsdt)
                        pgsub = (qvz / qsi - 1.) * qgz
                        
                        if pgsub > 0.:  # Deposition
                        
                            if tz > tice:
                            
                                pgsub = 0.
                            
                            else:
                                
                                pgsub = min(min(fac_v2g * pgsub, 0.2 * dq), min(qlz + qrz, (tice - tz) / tcpk))
                                
                        else:   # Sublimation
                            
                            pgsub = max(fac_g2v * pgsub, dq) * min(1., dim(tz, t_sub) * 0.1)
                        
                        qgz   = qgz + pgsub
                        qvz   = qvz - pgsub
                        q_sol = q_sol + pgsub
                        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz    = tz + pgsub * (lhl + lhi) / cvm
                    
                    '''
                    USE_MIN_EVAP
                    '''
                    # Update capacity heat and latent heat coefficient
                    lhl  = lv00 + d0_vap * tz
                    lcpk = lhl / cvm
                    
                    # Minimum evap of rain in dry environmental air
                    if qrz > qcmin:
                        
                        qsw   = ( e00 * exp((dc_vap * log(tz / t_ice) + lv0 * (tz - t_ice) / (tz * t_ice)) / rvgas) ) / \
                                ( rvgas * tz * den )
                        dqsdt = qsw * (dc_vap + lv0 / tz) / (rvgas * tz)
                        
                        sink  = min(qrz, dim(rh_rain * qsw, qvz) / (1. + lcpk * dqsdt))
                        qvz   = qvz + sink
                        qrz   = qrz - sink
                        q_liq = q_liq - sink
                        cvm   = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz    = tz - sink * lhl / cvm
                    '''
                    END USE_MIN_EVAP
                    '''
                    
                    # Update capacity heat and latent heat coefficient
                    lhl  = lv00 + d0_vap * tz
                    cvm  = c_air + (qvz + q_liq + q_sol) * c_vap
                    lcpk = lhl / cvm
                    
                    # Compute cloud fraction        
                    # Combine water species
                    if do_qa == 0:
                    
                        if rad_snow == 1: q_sol = qiz + qsz
                        else:             q_sol = qiz
                            
                        if rad_rain == 1: q_liq = qlz + qrz
                        else:             q_liq = qlz
                        
                        q_cond = q_liq + q_sol
                        
                        qpz = qvz + q_cond  # qpz is conserved
                        
                        # Use the "liquid-frozen water temperature" (tin) to compute saturated specific humidity
                        tin = tz - (lcpk * q_cond + icpk * q_sol)   # Minimum temperature
                        
                        # Determine saturated specific humidity
                        '''
                        THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
                        '''
                        t_wfr_tmp = t_wfr
                        if tin <= t_wfr:
                        
                            # Ice phase
                            if tin < t_ice:
                                
                                # Over ice between -160 degrees Celsius and 0 degrees Celsius
                                if tin >= t_ice - 160.:
                                
                                    qstar = ( e00 * exp((d2ice * log(tin / t_ice) + li2 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                            ( rvgas * tin * den )
                                
                                else:
                                
                                    qstar = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                            ( rvgas * (t_ice - 160.) * den )
                            else:
                                
                                # Over water between 0 degrees Celsius and 102 degrees Celsius
                                if tin <= t_ice + 102.:
                                
                                    qstar = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                            ( rvgas * tin * den )
                                
                                else:
                                    
                                    qstar = ( e00 * exp((dc_vap * log(1. + 102. / t_ice) + lv0 * 102. / ((t_ice + 102.) * t_ice)) / rvgas) ) / \
                                            ( rvgas * (t_ice + 102.) * den )
                        
                        elif tin >= tice:
                        
                            # Liquid phase
                            qstar = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                    ( rvgas * tin * den )
                            
                        else:
                            
                            # Mixed phase
                            if tin < t_ice:
                
                                # Over ice between -160 degrees Celsius and 0 degrees Celsius
                                if tin >= t_ice - 160.:
                                
                                    qsi = ( e00 * exp((d2ice * log(tin / t_ice) + li2 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                          ( rvgas * tin * den )
                                
                                else:
                                
                                    qsi = ( e00 * exp((d2ice * log(1. - 160. / t_ice) - li2 * 160. / ((t_ice - 160.) * t_ice)) / rvgas) ) / \
                                          ( rvgas * (t_ice - 160.) * den )
                            else:
                                
                                # Over water between 0 degrees Celsius and 102 degrees Celsius
                                if tin <= t_ice + 102.:
                                
                                    qsi = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                          ( rvgas * tin * den )
                                
                                else:
                                    
                                    qsi = ( e00 * exp((dc_vap * log(1. + 102. / t_ice) + lv0 * 102. / ((t_ice + 102.) * t_ice)) / rvgas) ) / \
                                          ( rvgas * (t_ice + 102.) * den )
                            
                            qsw = ( e00 * exp((dc_vap * log(tin / t_ice) + lv0 * (tin - t_ice) / (tin * t_ice)) / rvgas) ) / \
                                  ( rvgas * tin * den )
                        
                            if q_cond > 3.e-6:
                                
                                rqi = q_sol / q_cond
                        
                            else:
                                
                                # Mostly liquid water q_cond (k) at initial cloud development stage
                                '''
                                THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
                                rqi = (tice - tin) / (tice - t_wfr)
                                '''
                                rqi = (tice - tin) / (tice - t_wfr_tmp)
                        
                            qstar = rqi * qsi + (1. - rqi) * qsw
                                             
                        # Assuming subgrid linear distribution in horizontal; this is 
                        # effectively a smoother for the binary cloud scheme
                        if qpz > qrmin:
                        
                            # Partial cloudiness by pdf
                            dq      = max(qcmin, h_var * qpz)
                            q_plus  = qpz + dq  # Cloud free if qstar > q_plus
                            q_minus = qpz - dq
                            
                            if qstar < q_minus:
                            
                                qaz = qaz + 1.  # Air fully saturated; 100% cloud cover
                                
                            elif (qstar < q_plus) and (q_cond > qc_crt):
                                
                                qaz = qaz + (q_plus - qstar) / (dq + dq)    # Partial cloud cover
    
    '''
    #################################################################### END ICLOUD
    '''


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
    
    with computation(PARALLEL), interval(...):
        
        # Convert units from Pa*kg/kg to kg/m^2/s
        m2_rain = m2_rain * rdt * rgrav
        m2_sol  = m2_sol * rdt * rgrav
    
    # Momentum transportation during sedimentation (dp1 is dry mass; dp0 
    # is the old moist total mass)
    with computation(FORWARD), interval(1, None):
        
        if sedi_transport == 1:
            
            u1[0, 0, 0] = ( dp0[0, 0, 0] * u1[0, 0, 0] + m1[0, 0, -1] * u1[0, 0, -1] ) / \
                          ( dp0[0, 0, 0] + m1[0, 0, -1] )   
            v1[0, 0, 0] = ( dp0[0, 0, 0] * v1[0, 0, 0] + m1[0, 0, -1] * v1[0, 0, -1] ) / \
                          ( dp0[0, 0, 0] + m1[0, 0, -1] )
    
    with computation(PARALLEL), interval(1, None):
        
        if sedi_transport == 1:
            
            udt = udt + (u1 - u0) * rdt
            vdt = vdt + (v1 - v0) * rdt
    
    with computation(PARALLEL), interval(...):
        
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
