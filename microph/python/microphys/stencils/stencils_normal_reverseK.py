from microphys.phys_const import *
from microphys.microphys_funcs import *

from config import *
from utility.ufuncs_gt4py import *

import gt4py as gt
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, BACKWARD, FORWARD, computation, interval


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def fields_init(
    land: FIELD_FLT,
    area: FIELD_FLT,
    h_var: FIELD_FLT,
    rh_adj: FIELD_FLT,
    rh_rain: FIELD_FLT,
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qa: FIELD_FLT,
    qg: FIELD_FLT,
    qi: FIELD_FLT,
    ql: FIELD_FLT,
    qn: FIELD_FLT,
    qr: FIELD_FLT,
    qs: FIELD_FLT,
    qv: FIELD_FLT,
    pt: FIELD_FLT,
    delp: FIELD_FLT,
    dz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    qi_dt: FIELD_FLT,
    qs_dt: FIELD_FLT,
    uin: FIELD_FLT,
    vin: FIELD_FLT,
    qa0: FIELD_FLT,
    qg0: FIELD_FLT,
    qi0: FIELD_FLT,
    ql0: FIELD_FLT,
    qr0: FIELD_FLT,
    qs0: FIELD_FLT,
    qv0: FIELD_FLT,
    t0: FIELD_FLT,
    dp0: FIELD_FLT,
    den0: FIELD_FLT,
    dz0: FIELD_FLT,
    u0: FIELD_FLT,
    v0: FIELD_FLT,
    dp1: FIELD_FLT,
    p1: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    ccn: FIELD_FLT,
    c_praut: FIELD_FLT,
    use_ccn: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    dt_in: DTYPE_FLT,
    rdt: DTYPE_FLT,
    cpaut: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Initialize precipitation
        graupel = 0.0
        rain = 0.0
        snow = 0.0
        ice = 0.0

        # This is to prevent excessive build-up of cloud ice from
        # external sources
        if de_ice == 1:

            qio = qi - dt_in * qi_dt  # Orginal qi before phys
            qin = max(qio, qi0_max)  # Adjusted value

            if qi > qin:

                qs = qs + qi - qin
                qi = qin

                dqi = (qin - qio) * rdt  # Modified qi tendency
                qs_dt = qs_dt + qi_dt - dqi
                qi_dt = dqi

        qiz = qi
        qsz = qs

        t0 = pt
        tz = t0
        dp1 = delp
        dp0 = dp1  # Moist air mass * grav

        # Convert moist mixing ratios to dry mixing ratios
        qvz = qv
        qlz = ql
        qrz = qr
        qgz = qg

        dp1 = dp1 * (1.0 - qvz)
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
        p1 = den0 * rdgas * t0  # Dry air pressure

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
            ccn = qn * 1.0e6
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        else:

            ccn = (ccn_l * land + ccn_o * (1.0 - land)) * 1.0e6
    # with computation(BACKWARD):

    #     with interval(-1, None):
    with computation(FORWARD):

        with interval(0, 1):

            if (prog_ccn == 0) and (use_ccn == 1):

                # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                ccn = ccn * rdgas * tz / p1
        # with interval(0, -1):
        with interval(1, None):

            if (prog_ccn == 0) and (use_ccn == 1):

                # Propagate downwards in the atmosphere previously computed values of ccn
                ccn = ccn[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if prog_ccn == 0:
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        # Calculate horizontal subgrid variability
        # Total water subgrid deviation in horizontal direction
        # Default area dependent form: use dx ~ 100 km as the base
        s_leng = sqrt(sqrt(area * 1.0e-10))
        t_land = dw_land * s_leng
        t_ocean = dw_ocean * s_leng
        h_var = t_land * land + t_ocean * (1.0 - land)
        h_var = min(0.2, max(0.01, h_var))

        # Relative humidity increment
        rh_adj = 1.0 - h_var - rh_inc
        rh_rain = max(0.35, rh_adj - rh_inr)

        # Fix all negative water species
        if fix_negative == 1:

            # Define heat capacity and latent heat coefficient
            cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            lcpk = (lv00 + d0_vap * tz) / cvm
            icpk = (li00 + dc_ice * tz) / cvm

            # Ice phase

            # If cloud ice < 0, borrow from snow
            if qiz < 0.0:

                qsz = qsz + qiz
                qiz = 0.0

            # If snow < 0, borrow from graupel
            if qsz < 0.0:

                qgz = qgz + qsz
                qsz = 0.0

            # If graupel < 0, borrow from rain
            if qgz < 0.0:

                qrz = qrz + qgz
                tz = tz - qgz * icpk  # Heating
                qgz = 0.0

            # Liquid phase

            # If rain < 0, borrow from cloud water
            if qrz < 0.0:

                qlz = qlz + qrz
                qrz = 0.0

            # If cloud water < 0, borrow from water vapor
            if qlz < 0.0:

                qvz = qvz + qlz
                tz = tz - qlz * lcpk  # Heating
                qlz = 0.0

    # with computation(FORWARD), interval(1, None):
    with computation(BACKWARD), interval(0, -1):

        # Fix water vapor; borrow from below
        if (fix_negative == 1) and (qvz[0, 0, 1] < 0.0):
            qvz[0, 0, 0] = qvz[0, 0, 0] + qvz[0, 0, 1] * dp1[0, 0, 1] / dp1[0, 0, 0]

    # with computation(PARALLEL), interval(0, -1):
    with computation(PARALLEL), interval(1, None):

        if (fix_negative == 1) and (qvz < 0.0):
            qvz = 0.0

    # Bottom layer; borrow from above
    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            flag = 0

            if (fix_negative == 1) and (qvz < 0.0) and (qvz[0, 0, 1] > 0.0):

                dq = min(-qvz[0, 0, 0] * dp1[0, 0, 0], qvz[0, 0, 1] * dp1[0, 0, 1])
                flag = 1
        # with interval(-2, -1):
        with interval(1, 2):

            flag = 0

            if (fix_negative == 1) and (qvz[0, 0, -1] < 0.0) and (qvz > 0.0):

                dq = min(-qvz[0, 0, -1] * dp1[0, 0, -1], qvz[0, 0, 0] * dp1[0, 0, 0])
                flag = 1

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if flag == 1:

                qvz = qvz + dq / dp1
        # with interval(-2, -1):
        with interval(1, 2):

            if flag == 1:

                qvz = qvz - dq / dp1


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def warm_rain(
    h_var: FIELD_FLT,
    rain: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    denfac: FIELD_FLT,
    w: FIELD_FLT,
    t0: FIELD_FLT,
    den0: FIELD_FLT,
    dz0: FIELD_FLT,
    dz1: FIELD_FLT,
    dp1: FIELD_FLT,
    m1: FIELD_FLT,
    vtrz: FIELD_FLT,
    ccn: FIELD_FLT,
    c_praut: FIELD_FLT,
    m1_sol: FIELD_FLT,
    m2_rain: FIELD_FLT,
    m2_sol: FIELD_FLT,
    is_first: DTYPE_INT,
    do_sedi_w: DTYPE_INT,
    p_nonhydro: DTYPE_INT,
    use_ccn: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    fac_rc: DTYPE_FLT,
    cracw: DTYPE_FLT,
    crevp_0: DTYPE_FLT,
    crevp_1: DTYPE_FLT,
    crevp_2: DTYPE_FLT,
    crevp_3: DTYPE_FLT,
    crevp_4: DTYPE_FLT,
    t_wfr: DTYPE_FLT,
    so3: DTYPE_FLT,
    dt_rain: DTYPE_FLT,
    zs: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        if is_first == 1:

            # Define air density based on hydrostatical property
            if p_nonhydro == 1:

                dz1 = dz0
                den = den0  # Dry air density remains the same
                denfac = sqrt(sfcrho / den)

            else:

                dz1 = dz0 * tz / t0  # Hydrostatic balance
                den = den0 * dz0 / dz1
                denfac = sqrt(sfcrho / den)

        """
        ################################################################ WARM_RAIN
        """

        # Time-split warm rain processes: 1st pass
        dt5 = 0.5 * dt_rain

        # Terminal speed of rain
        m1_rain = 0.0

    """
    ####################################################### CHECK_COLUMN
    """

    # with computation(FORWARD):
    with computation(BACKWARD):
        # with interval(0, 1):
        with interval(-1, None):

            if qrz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        # with interval(1, None):
        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qrz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    """
    ################################################### END CHECK_COLUMN
    """

    with computation(PARALLEL), interval(...):

        vtrz, r1 = compute_rain_fspeed(no_fall, qrz, den)

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):

            if no_fall == 0:
                ze = zs - dz1

        # with interval(0, -1):
        with interval(1, None):

            if no_fall == 0:
                ze = ze[0, 0, -1] - dz1  # dz < 0

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            """
            ################################################# REVAP_RACC
            """

            # Evaporation and accretion of rain for the first 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

            """
            ############################################# END REVAP_RACC
            """

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    # Mass flux induced by falling rain
    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 1) and (no_fall == 0):

                zt = ze - dt5 * (vtrz[0, 0, 1] + vtrz)

                zt_kbot1 = zs - dt_rain * vtrz
        with interval(1, -1):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze - dt5 * (vtrz[0, 0, 1] + vtrz)
        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze

    # with computation(FORWARD):
    with computation(BACKWARD):

        with interval(1, -1):

            if (use_ppm == 1) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        # with interval(-1, None):
        with interval(0, 1):

            if use_ppm == 1:

                if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                    zt = zt[0, 0, 1] - dz_min

                if (no_fall == 0) and (zt_kbot1 >= zt):
                    zt_kbot1 = zt - dz_min

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if (use_ppm == 1) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, -1]

    """        
    with computation(PARALLEL), interval(...):
        
        if (use_ppm == 1) and (no_fall == 0):
            lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qr, r1, 
                                 m1_rain, mono_prof, nf_inv )
    """

    """
    ###################################################### IMPLICIT_FALL
    """

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dt_rain * vtrz
            qrz = qrz * dp1

    # Sedimentation
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qrz / (dz + dd)

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qrz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = qrz - qm

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = m1_rain[0, 0, 1] + qrz[0, 0, 0] - qm

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):
            if (use_ppm == 0) and (no_fall == 0):
                r1 = m1_rain

        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                r1 = r1[0, 0, -1]

    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                """
                ###################################### END IMPLICIT_FALL
                """

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_rain[0, 0, 1] * vtrz[0, 0, 1]
                        + m1_rain * vtrz
                    ) / (dm + m1_rain[0, 0, 1] - m1_rain)
        # with interval(0, 1):
        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_rain * vtrz) / (dm - m1_rain)

    """
    ########################################################## SEDI_HEAT
    """

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )
        # with interval(0, 1):
        with interval(-1, None):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_rain * c_liq
                tz = tz + m1_rain * dgz / tmp

    # Implicit algorithm
    # with computation(FORWARD), interval(1, None):
    with computation(BACKWARD), interval(0, -1):

        if (do_sedi_heat == 1) and (no_fall == 0):

            tz[0, 0, 0] = (
                (cvn + c_liq * (m1_rain - m1_rain[0, 0, 1])) * tz[0, 0, 0]
                + m1_rain[0, 0, 1] * c_liq * tz[0, 0, 1]
                + dgz * (m1_rain[0, 0, 1] + m1_rain)
            ) / (cvn + c_liq * m1_rain)

    """
    ###################################################### END SEDI_HEAT
    """

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            """
            ################################################# REVAP_RACC
            """

            # Evaporation and accretion of rain for the remaining 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

            """
            ############################################# END REVAP_RACC
            """

        # Auto-conversion assuming linear subgrid vertical distribution of
        # cloud water following lin et al. 1994, mwr
        if irain_f != 0:

            qlz, qrz = autoconv_no_subgrid_var(
                use_ccn, fac_rc, t_wfr, so3, dt_rain, qlz, qrz, tz, den, ccn, c_praut
            )

    """
    ######################################################## LINEAR_PROF
    """

    # With subgrid variability
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

        # with interval(1, None):
        with interval(0, -1):

            if (irain_f == 0) and (z_slope_liq == 1):
                dq = 0.5 * (qlz[0, 0, 0] - qlz[0, 0, 1])

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

        with interval(1, -1):

            if (irain_f == 0) and (z_slope_liq == 1):

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                dl = 0.5 * min(abs(dq + dq[0, 0, -1]), 0.5 * qlz[0, 0, 0])

                if dq * dq[0, 0, -1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        dl = min(dl, min(dq, -dq[0, 0, -1]))

                    else:

                        dl = 0.0

    with computation(PARALLEL), interval(...):

        if irain_f == 0:

            if z_slope_liq == 1:

                # Impose a presumed background horizontal variability that is
                # proportional to the value itself
                dl = max(dl, max(qvmin, h_var * qlz))

            else:

                dl = max(qvmin, h_var * qlz)

            """
            ############################################ END LINEAR_PROF
            """

            qlz, qrz = autoconv_subgrid_var(
                use_ccn,
                fac_rc,
                t_wfr,
                so3,
                dt_rain,
                qlz,
                qrz,
                tz,
                den,
                ccn,
                c_praut,
                dl,
            )

        """
        ################################################################ END WARM_RAIN
        """

        rain = rain + r1
        m2_rain = m2_rain + m1_rain

        if is_first == 1:

            m1 = m1 + m1_rain

        else:

            m2_sol = m2_sol + m1_sol
            m1 = m1 + m1_rain + m1_sol


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def sedimentation(
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    w: FIELD_FLT,
    dz1: FIELD_FLT,
    dp1: FIELD_FLT,
    vtgz: FIELD_FLT,
    vtsz: FIELD_FLT,
    m1_sol: FIELD_FLT,
    do_sedi_w: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    log_10: DTYPE_FLT,
    zs: DTYPE_FLT,
    dts: DTYPE_FLT,
    fac_imlt: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        """
        ################################################################ FALL_SPEED
        """

        # Sedimentation of cloud ice, snow, and graupel
        vtgz, vtiz, vtsz = fall_speed(log_10, qgz, qiz, qlz, qsz, tz, den)

        """
        ################################################################ END FALL_SPEED
        """

        """
        ################################################################ TERMINAL_FALL
        """

        dt5 = 0.5 * dts

        # Define heat capacity and latent heat coefficient
        m1_sol = 0.0

        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

    # Find significant melting level
    """
    k0 removed to avoid having to introduce a k_idx field
    """
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if tz > tice:
                stop_k = 1
            else:
                stop_k = 0

        with interval(1, -1):

            if stop_k[0, 0, 1] == 0:

                if tz > tice:
                    stop_k = 1
                else:
                    stop_k = 0

            else:

                stop_k = 1

        # with interval(-1, None):
        with interval(0, 1):

            stop_k = 1

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Melting of cloud ice (before fall)
            tc = tz - tice

            if (qiz > qcmin) and (tc > 0.0):

                sink = min(qiz, fac_imlt * tc / icpk)
                tmp = min(sink, dim(ql_mlt, qlz))
                qlz = qlz + tmp
                qrz = qrz + sink - tmp
                qiz = qiz - sink
                q_liq = q_liq + sink
                q_sol = q_sol - sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - sink * lhi / cvm
                tc = tz - tice

    # with computation(PARALLEL), interval(0, -1):
    with computation(PARALLEL), interval(1, None):

        # Turn off melting when cloud microphysics time step is small
        if dts < 60.0:
            stop_k = 0

        # sjl, turn off melting of falling cloud ice, snow and graupel
        stop_k = 0

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):

            ze = zs - dz1

        with interval(1, -1):

            ze = ze[0, 0, -1] - dz1  # dz < 0

        # with interval(0, 1):
        with interval(-1, None):

            ze = ze[0, 0, -1] - dz1  # dz < 0
            zt = ze

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Update capacity heat and latent heat coefficient
            lhi = li00 + dc_ice * tz
            icpk = lhi / cvm

    """
    ####################################################### CHECK_COLUMN
    """

    # Melting of falling cloud ice into rain
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if qiz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        # with interval(1, None):
        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qiz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    """
    ################################################### END CHECK_COLUMN
    """

    with computation(PARALLEL), interval(...):

        if (vi_fac < 1.0e-5) or (no_fall == 1):
            i1 = 0.0

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                zt = ze - dt5 * (vtiz[0, 0, 1] + vtiz)
                zt_kbot1 = zs - dts * vtiz
        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):
                zt = ze - dt5 * (vtiz[0, 0, 1] + vtiz)

    # with computation(FORWARD):
    with computation(BACKWARD):

        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        # with interval(-1, None):
        with interval(0, 1):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (vi_fac >= 1.0e-5) and (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if (vi_fac >= 1.0e-5) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

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

        if (vi_fac >= 1.0e-5) and (no_fall == 0):

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

            """
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qi, 
                                     i1, m1_sol, mono_prof, ij_vi_inv )
            """

    """
    ###################################################### IMPLICIT_FALL
    """

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - zs
        # with interval(0, -1):
        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            dd = dts * vtiz
            qiz = qiz * dp1

    # Sedimentation
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = qiz / (dz + dd)

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = (qiz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = qiz - qm

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = m1_sol[0, 0, 1] + qiz[0, 0, 0] - qm

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = m1_sol

        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = i1[0, 0, -1]

    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                """
                ###################################### END IMPLICIT_FALL
                """

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_sol[0, 0, 1] * vtiz[0, 0, 1]
                        + m1_sol * vtiz
                    ) / (dm + m1_sol[0, 0, 1] - m1_sol)
        # with interval(0, 1):
        with interval(-1, None):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_sol * vtiz) / (dm - m1_sol)

    """
    ####################################################### CHECK_COLUMN
    """

    # Melting of falling snow into rain
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if qsz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        # with interval(1, None):
        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qsz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    """
    ################################################### END CHECK_COLUMN
    """

    with computation(PARALLEL), interval(...):

        r1 = 0.0

        if no_fall == 1:
            s1 = 0.0

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if no_fall == 0:

                zt = ze - dt5 * (vtsz[0, 0, 1] + vtsz)
                zt_kbot1 = zs - dts * vtsz

        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtsz[0, 0, 1] + vtsz)

    # with computation(FORWARD):
    with computation(BACKWARD):

        with interval(1, -1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        # with interval(-1, None):
        with interval(0, 1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

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
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

            """
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qs, 
                                     s1, m1_tf, mono_prof, nf_inv )
            """

    """
    ###################################################### IMPLICIT_FALL
    """

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtsz
            qsz = qsz * dp1

    # Sedimentation
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qsz / (dz + dd)

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qsz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qsz - qm

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, 1] + qsz[0, 0, 0] - qm

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = m1_tf

        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = s1[0, 0, -1]

    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                """
                ###################################### END IMPLICIT_FALL
                """

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0] - m1_tf[0, 0, 1] * vtsz[0, 0, 1] + m1_tf * vtsz
                    ) / (dm + m1_tf[0, 0, 1] - m1_tf)

        # with interval(0, 1):
        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtsz) / (dm - m1_tf)

    """
    ####################################################### CHECK_COLUMN
    """

    # Melting of falling graupel into rain
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if qgz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        # with interval(1, None):
        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qgz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    """
    ################################################### END CHECK_COLUMN
    """

    with computation(PARALLEL), interval(...):

        if no_fall == 1:
            g1 = 0.0

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if no_fall == 0:

                zt = ze - dt5 * (vtgz[0, 0, 1] + vtgz)
                zt_kbot1 = zs - dts * vtgz
        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtgz[0, 0, 1] + vtgz)

    # with computation(FORWARD):
    with computation(BACKWARD):

        with interval(1, -1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        # with interval(-1, None):
        with interval(0, 1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    # with computation(BACKWARD), interval(0, -1):
    with computation(FORWARD), interval(1, None):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

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
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

            """
            if use_ppm == 1:
                lagrangian_fall_ppm( ktop, kbot, zs, ze, zt, zt_kbot1, dp, qg, 
                                     g1, m1_tf, mono_prof, nf_inv )
            """

    """
    ###################################################### IMPLICIT_FALL
    """

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtgz
            qgz = qgz * dp1

    # Sedimentation
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qgz / (dz + dd)

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qgz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qgz - qm

        # with interval(1, None):
        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, 1] + qgz[0, 0, 0] - qm

    # with computation(BACKWARD):
    with computation(FORWARD):

        # with interval(-1, None):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = m1_tf

        # with interval(0, -1):
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = g1[0, 0, -1]

    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                """
                ###################################### END IMPLICIT_FALL
                """

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0] - m1_tf[0, 0, 1] * vtgz[0, 0, 1] + m1_tf * vtgz
                    ) / (dm + m1_tf[0, 0, 1] - m1_tf)
        # with interval(0, 1):
        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtgz) / (dm - m1_tf)

    """
    #################################################################### END TERMINAL_FALL
    """

    with computation(PARALLEL), interval(...):

        rain = rain + r1  # From melted snow and ice that reached the ground
        snow = snow + s1
        graupel = graupel + g1
        ice = ice + i1

    """
    #################################################################### SEDI_HEAT
    """

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        # with interval(1, None):
        with interval(0, -1):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )
        # with interval(0, 1):
        with interval(-1, None):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_sol * c_ice
                tz = tz + m1_sol * dgz / tmp

    # Implicit algorithm
    # with computation(FORWARD), interval(1, None):
    with computation(BACKWARD), interval(0, -1):

        if do_sedi_heat == 1:

            tz[0, 0, 0] = (
                (cvn + c_ice * (m1_sol - m1_sol[0, 0, 1])) * tz[0, 0, 0]
                + m1_sol[0, 0, 1] * c_ice * tz[0, 0, 1]
                + dgz * (m1_sol[0, 0, 1] + m1_sol)
            ) / (cvn + c_ice * m1_sol)

    """
    #################################################################### END SEDI_HEAT
    """


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def icloud(
    h_var: FIELD_FLT,
    rh_adj: FIELD_FLT,
    rh_rain: FIELD_FLT,
    qaz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    denfac: FIELD_FLT,
    p1: FIELD_FLT,
    vtgz: FIELD_FLT,
    vtrz: FIELD_FLT,
    vtsz: FIELD_FLT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    cracs: DTYPE_FLT,
    csacr: DTYPE_FLT,
    cgacr: DTYPE_FLT,
    cgacs: DTYPE_FLT,
    acco_00: DTYPE_FLT,
    acco_01: DTYPE_FLT,
    acco_02: DTYPE_FLT,
    acco_03: DTYPE_FLT,
    acco_10: DTYPE_FLT,
    acco_11: DTYPE_FLT,
    acco_12: DTYPE_FLT,
    acco_13: DTYPE_FLT,
    acco_20: DTYPE_FLT,
    acco_21: DTYPE_FLT,
    acco_22: DTYPE_FLT,
    acco_23: DTYPE_FLT,
    csacw: DTYPE_FLT,
    csaci: DTYPE_FLT,
    cgacw: DTYPE_FLT,
    cgaci: DTYPE_FLT,
    cracw: DTYPE_FLT,
    cssub_0: DTYPE_FLT,
    cssub_1: DTYPE_FLT,
    cssub_2: DTYPE_FLT,
    cssub_3: DTYPE_FLT,
    cssub_4: DTYPE_FLT,
    cgfr_0: DTYPE_FLT,
    cgfr_1: DTYPE_FLT,
    csmlt_0: DTYPE_FLT,
    csmlt_1: DTYPE_FLT,
    csmlt_2: DTYPE_FLT,
    csmlt_3: DTYPE_FLT,
    csmlt_4: DTYPE_FLT,
    cgmlt_0: DTYPE_FLT,
    cgmlt_1: DTYPE_FLT,
    cgmlt_2: DTYPE_FLT,
    cgmlt_3: DTYPE_FLT,
    cgmlt_4: DTYPE_FLT,
    ces0: DTYPE_FLT,
    tice0: DTYPE_FLT,
    t_wfr: DTYPE_FLT,
    dts: DTYPE_FLT,
    rdts: DTYPE_FLT,
    fac_i2s: DTYPE_FLT,
    fac_g2v: DTYPE_FLT,
    fac_v2g: DTYPE_FLT,
    fac_imlt: DTYPE_FLT,
    fac_l2v: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        """
        ################################################################ ICLOUD
        """

        # Ice-phase microphysics

        # Define heat capacity and latent heat coefficient
        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

        # - Sources of cloud ice: pihom, cold rain, and the sat_adj
        # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
        # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion
        """
        THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
        """
        t_wfr_tmp = t_wfr
        if (tz > tice) and (qiz > qcmin):

            # pimlt: instant melting of cloud ice
            melt = min(qiz, fac_imlt * (tz - tice) / icpk)
            tmp = min(melt, dim(ql_mlt, qlz))  # Maximum ql amount
            qlz = qlz + tmp
            qrz = qrz + melt - tmp
            qiz = qiz - melt
            q_liq = q_liq + melt
            q_sol = q_sol - melt
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz - melt * lhi / cvm

        elif (tz < t_wfr) and (qlz > qcmin):

            # - pihom: homogeneous freezing of cloud water into cloud ice
            # - This is the 1st occurance of liquid water freezing in the split mp process
            """
            THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
            dtmp   = t_wfr - tz
            """
            dtmp = t_wfr_tmp - tz
            factor = min(1.0, dtmp / dt_fr)
            sink = min(qlz * factor, dtmp / icpk)
            qi_crt = qi_gen * min(qi_lim, 0.1 * (tice - tz)) / den
            tmp = min(sink, dim(qi_crt, qiz))
            qlz = qlz - sink
            qsz = qsz + sink - tmp
            qiz = qiz + tmp
            q_liq = q_liq - sink
            q_sol = q_sol + sink
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz + sink * lhi / cvm

    """
    ######################################################## LINEAR_PROF
    """

    # Vertical subgrid variability
    # with computation(FORWARD):
    with computation(BACKWARD):

        # with interval(0, 1):
        with interval(-1, None):

            if z_slope_ice == 1:
                di = 0.0

        # with interval(1, None):
        with interval(0, -1):

            if z_slope_ice == 1:
                dq = 0.5 * (qiz[0, 0, 0] - qiz[0, 0, 1])

    with computation(PARALLEL):
        # with interval(-1, None):
        with interval(0, 1):

            if z_slope_ice == 1:
                di = 0.0

        with interval(1, -1):

            if z_slope_ice == 1:

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                di = 0.5 * min(abs(dq + dq[0, 0, -1]), 0.5 * qiz[0, 0, 0])

                if dq * dq[0, 0, -1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        di = min(di, min(dq, -dq[0, 0, -1]))

                    else:

                        di = 0.0

    with computation(PARALLEL), interval(...):

        if z_slope_ice == 1:

            # Impose a presumed background horizontal variability that is
            # proportional to the value itself
            di = max(di, max(qvmin, h_var * qiz))

        else:

            di = max(qvmin, h_var * qiz)

        """
        ################################################ END LINEAR_PROF
        """

        qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = icloud_main(
            c_air,
            c_vap,
            d0_vap,
            lv00,
            cracs,
            csacr,
            cgacr,
            cgacs,
            acco_00,
            acco_01,
            acco_02,
            acco_03,
            acco_10,
            acco_11,
            acco_12,
            acco_13,
            acco_20,
            acco_21,
            acco_22,
            acco_23,
            csacw,
            csaci,
            cgacw,
            cgaci,
            cssub_0,
            cssub_1,
            cssub_2,
            cssub_3,
            cssub_4,
            cgfr_0,
            cgfr_1,
            csmlt_0,
            csmlt_1,
            csmlt_2,
            csmlt_3,
            csmlt_4,
            cgmlt_0,
            cgmlt_1,
            cgmlt_2,
            cgmlt_3,
            cgmlt_4,
            ces0,
            tice0,
            t_wfr,
            dts,
            rdts,
            fac_i2s,
            fac_g2v,
            fac_v2g,
            fac_l2v,
            h_var,
            rh_adj,
            rh_rain,
            qaz,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            vtgz,
            vtrz,
            vtsz,
            p1,
            di,
            q_liq,
            q_sol,
            cvm,
        )

    """
    #################################################################### END ICLOUD
    """


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def fields_update(
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qaz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    udt: FIELD_FLT,
    vdt: FIELD_FLT,
    qa_dt: FIELD_FLT,
    qg_dt: FIELD_FLT,
    qi_dt: FIELD_FLT,
    ql_dt: FIELD_FLT,
    qr_dt: FIELD_FLT,
    qs_dt: FIELD_FLT,
    qv_dt: FIELD_FLT,
    pt_dt: FIELD_FLT,
    qa0: FIELD_FLT,
    qg0: FIELD_FLT,
    qi0: FIELD_FLT,
    ql0: FIELD_FLT,
    qr0: FIELD_FLT,
    qs0: FIELD_FLT,
    qv0: FIELD_FLT,
    t0: FIELD_FLT,
    dp0: FIELD_FLT,
    u0: FIELD_FLT,
    v0: FIELD_FLT,
    dp1: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    m1: FIELD_FLT,
    m2_rain: FIELD_FLT,
    m2_sol: FIELD_FLT,
    ntimes: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    rdt: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Convert units from Pa*kg/kg to kg/m^2/s
        m2_rain = m2_rain * rdt * rgrav
        m2_sol = m2_sol * rdt * rgrav

    # Momentum transportation during sedimentation (dp1 is dry mass; dp0
    # is the old moist total mass)
    # with computation(FORWARD), interval(1, None):
    with computation(BACKWARD), interval(0, -1):

        if sedi_transport == 1:

            u1[0, 0, 0] = (dp0[0, 0, 0] * u1[0, 0, 0] + m1[0, 0, 1] * u1[0, 0, 1]) / (
                dp0[0, 0, 0] + m1[0, 0, 1]
            )
            v1[0, 0, 0] = (dp0[0, 0, 0] * v1[0, 0, 0] + m1[0, 0, 1] * v1[0, 0, 1]) / (
                dp0[0, 0, 0] + m1[0, 0, 1]
            )

    # with computation(PARALLEL), interval(1, None):
    with computation(PARALLEL), interval(0, -1):

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

            qa_dt = 0.0

        else:

            qa_dt = qa_dt + rdt * (qaz / ntimes - qa0)

        """
        LEFT OUT FOR NOW
        # No clouds allowed above ktop
        if k_s < ktop:
            qa_dt[:, :, k_s:ktop+1] = 0.
        """

        # Convert to mm / day
        convt = 86400.0 * rdt * rgrav

        rain = rain * convt
        snow = snow * convt
        ice = ice * convt
        graupel = graupel * convt
