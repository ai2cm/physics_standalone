from physics.constants import *

from physics.stencils.microphysics import *
import numpy as np
import gt4py as gt
import math as mt
from physics.config import *
from copy import deepcopy

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
    acc = np.array([5.0, 2.0, 0.5])

    pie = 4.0 * mt.atan(1.0)

    # S. Klein's formular (eq 16) from am2
    fac_rc = (4.0 / 3.0) * pie * rhor * rthresh ** 3

    vdifu = 2.11e-5
    tcond = 2.36e-2

    visk = 1.259e-5
    hlts = 2.8336e6
    hltc = 2.5e6
    hltf = 3.336e5

    ch2o = 4.1855e3

    pisq = pie * pie
    scm3 = (visk / vdifu) ** (1.0 / 3.0)

    cracs = pisq * rnzr * rnzs * rhos
    csacr = pisq * rnzr * rnzs * rhor
    cgacr = pisq * rnzr * rnzg * rhor
    cgacs = pisq * rnzg * rnzs * rhos
    cgacs = cgacs * c_pgacs

    act = np.empty(8)
    act[0] = pie * rnzs * rhos
    act[1] = pie * rnzr * rhor
    act[5] = pie * rnzg * rhog
    act[2] = act[1]
    act[3] = act[0]
    act[4] = act[1]
    act[6] = act[0]
    act[7] = act[5]

    acco = np.empty((3, 4))
    for i in range(3):
        for k in range(4):
            acco[i, k] = acc[i] / (
                act[2 * k] ** ((6 - i) * 0.25) * act[2 * k + 1] ** ((i + 1) * 0.25)
            )

    gcon = 40.74 * mt.sqrt(sfcrho)

    # Decreasing csacw to reduce cloud water --- > snow
    csacw = pie * rnzs * clin * gam325 / (4.0 * act[0] ** 0.8125)

    craci = pie * rnzr * alin * gam380 / (4.0 * act[1] ** 0.95)
    csaci = csacw * c_psaci

    cgacw = pie * rnzg * gam350 * gcon / (4.0 * act[5] ** 0.875)

    cgaci = cgacw * 0.05

    cracw = craci
    cracw = c_cracw * cracw

    # Subl and revap: five constants for three separate processes
    cssub = np.empty(5)
    cssub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzs
    cssub[1] = 0.78 / mt.sqrt(act[0])
    cssub[2] = 0.31 * scm3 * gam263 * mt.sqrt(clin / visk) / act[0] ** 0.65625
    cssub[3] = tcond * rvgas
    cssub[4] = (hlts ** 2) * vdifu

    cgsub = np.empty(5)
    cgsub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzg
    cgsub[1] = 0.78 / mt.sqrt(act[5])
    cgsub[2] = 0.31 * scm3 * gam275 * mt.sqrt(gcon / visk) / act[5] ** 0.6875
    cgsub[3] = cssub[3]
    cgsub[4] = cssub[4]

    crevp = np.empty(5)
    crevp[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzr
    crevp[1] = 0.78 / mt.sqrt(act[1])
    crevp[2] = 0.31 * scm3 * gam290 * mt.sqrt(alin / visk) / act[1] ** 0.725
    crevp[3] = cssub[3]
    crevp[4] = hltc ** 2 * vdifu

    cgfr = np.empty(2)
    cgfr[0] = 20.0e2 * pisq * rnzr * rhor / act[1] ** 1.75
    cgfr[1] = 0.66

    # smlt: five constants (lin et al. 1983)
    csmlt = np.empty(5)
    csmlt[0] = 2.0 * pie * tcond * rnzs / hltf
    csmlt[1] = 2.0 * pie * vdifu * rnzs * hltc / hltf
    csmlt[2] = cssub[1]
    csmlt[3] = cssub[2]
    csmlt[4] = ch2o / hltf

    # gmlt: five constants
    cgmlt = np.empty(5)
    cgmlt[0] = 2.0 * pie * tcond * rnzg / hltf
    cgmlt[1] = 2.0 * pie * vdifu * rnzg * hltc / hltf
    cgmlt[2] = cgsub[1]
    cgmlt[3] = cgsub[2]
    cgmlt[4] = ch2o / hltf

    es0 = 6.107799961e2  # ~6.1 mb
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

    log_10 = mt.log(10.0)
    tice0 = tice - 0.01
    t_wfr = (
        tice - 40.0
    )  # Supercooled water can exist down to -48 degrees Celsius, which is the "absolute"


# Scale the input dataset for benchmarks
def scale_dataset(data, factor):
    divider = factor[0]
    multiplier = factor[1]

    do_divide = divider < 1.0

    scaled_data = {}

    for var in data:

        data_var = data[var]
        if isinstance(data_var, np.ndarray):
            ndim = data_var.ndim
        else:
            ndim = 0

        if ndim == 3:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider), :, :]

            scaled_data[var] = np.tile(data_var, (multiplier, 1, 1))

        elif ndim == 2:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider), :]

            scaled_data[var] = np.tile(data_var, (multiplier, 1))

        elif ndim == 1:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider)]

            scaled_data[var] = np.tile(data_var, multiplier)

        elif ndim == 0:

            if var == "iie":
                scaled_data[var] = DTYPE_INT(data[var] * multiplier * divider)
            else:
                if var in INT_VARS:
                    scaled_data[var] = DTYPE_INT(data[var])
                elif var in FLT_VARS:
                    scaled_data[var] = DTYPE_FLT(data[var])
                else:
                    scaled_data[var] = data[var]

    return scaled_data


# Transform a dictionary of numpy arrays into a dictionary of gt4py
# storages of shape (iie-iis+1, jje-jjs+1, kke-kks+1)
def numpy_dict_to_gt4py_dict(np_dict):

    shape = np_dict["qi"].shape
    gt4py_dict = {}

    for var in np_dict:

        data = np_dict[var]
        if isinstance(data, np.ndarray):
            ndim = data.ndim
        else:
            ndim = 0

        if (ndim > 0) and (ndim <= 3) and (data.size >= 2):

            reshaped_data = np.empty(shape)

            if ndim == 1:  # 1D array (i-dimension)
                reshaped_data[...] = data[:, np.newaxis, np.newaxis]
            elif ndim == 2:  # 2D array (i-dimension, j-dimension)
                reshaped_data[...] = data[:, :, np.newaxis]
            elif ndim == 3:  # 3D array (i-dimension, j-dimension, k-dimension)
                reshaped_data[...] = data[...]

            dtype = DTYPE_INT if var in INT_VARS else DTYPE_FLT
            gt4py_dict[var] = gt.storage.from_array(
                reshaped_data, BACKEND, DEFAULT_ORIGIN, dtype=dtype
            )

        else:  # Scalars

            gt4py_dict[var] = deepcopy(data)

    return gt4py_dict


# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        if BACKEND == "gtcuda":
            data.synchronize()

        np_dict[var] = data.view(np.ndarray)

    return np_dict


# Global variables for microphysics
c_air = None
c_vap = None
d0_vap = None  # The same as dc_vap, except that cp_vap can be cp_vap or cv_vap
lv00 = None  # The same as lv0, except that cp_vap can be cp_vap or cv_vap
fac_rc = None
cracs = None
csacr = None
cgacr = None
cgacs = None
acco = None
csacw = None
csaci = None
cgacw = None
cgaci = None
cracw = None
cssub = None
crevp = None
cgfr = None
csmlt = None
cgmlt = None
ces0 = None
log_10 = None
tice0 = None
t_wfr = None

do_sedi_w = 1  # Transport of vertical motion in sedimentation
do_setup = True  # Setup constants and parameters
p_nonhydro = 0  # Perform hydrosatic adjustment on air density
use_ccn = 1  # Must be true when prog_ccn is false


def run(input_data: dict, kke: int, kbot: int, dt_in: float):
    """
    input_data: dictionary of required input data
    kke: vertical axis dimension
    kbot: top of the atmosphere (0 or kke)
    dt_in: physics time step in seconds
    """
    gfdl_cloud_microphys_init()
    input_data = scale_dataset(input_data, (1.0, 1))
    input_data = numpy_dict_to_gt4py_dict(input_data)
    hydrostatic = False
    phys_hydrostatic = True
    kks = 0

    # 2D input arrays
    area = input_data["area"]  # Cell area
    land = input_data["land"]  # Land fraction
    rain = input_data["rain"]
    snow = input_data["snow"]
    ice = input_data["ice"]
    graupel = input_data["graupel"]

    # 3D input arrays
    dz = input_data["dz"]
    delp = input_data["delp"]
    uin = input_data["uin"]
    vin = input_data["vin"]
    qv = input_data["qv"]
    ql = input_data["ql"]
    qr = input_data["qr"]
    qi = input_data["qi"]
    qs = input_data["qs"]
    qg = input_data["qg"]
    qa = input_data["qa"]
    qn = input_data["qn"]
    p = input_data["p"]
    pt = input_data["pt"]
    qv_dt = input_data["qv_dt"]
    ql_dt = input_data["ql_dt"]
    qr_dt = input_data["qr_dt"]
    qi_dt = input_data["qi_dt"]
    qs_dt = input_data["qs_dt"]
    qg_dt = input_data["qg_dt"]
    qa_dt = input_data["qa_dt"]
    pt_dt = input_data["pt_dt"]
    udt = input_data["udt"]
    vdt = input_data["vdt"]
    w = input_data["w"]
    refl_10cm = input_data["refl_10cm"]

    # Common 3D shape of all gt4py storages
    shape = qi.shape

    # 2D local arrays
    h_var = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    rh_adj = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    rh_rain = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)

    # 3D local arrays
    qaz = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qgz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qiz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qlz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qrz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qsz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qvz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    den = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    denfac = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    tz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qa0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qg0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qi0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    ql0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qr0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qs0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qv0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    t0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dp0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    den0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dz0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    u0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    v0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dz1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dp1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    p1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    u1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    v1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m1 = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtgz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtrz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtsz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    ccn = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    c_praut = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m1_sol = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m2_rain = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m2_sol = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)

    # Global variables
    global c_air
    global c_vap
    global d0_vap
    global lv00

    global do_sedi_w
    global p_nonhydro
    global use_ccn

    # Define start and end indices of the vertical dimensions
    k_s = kks
    k_e = kke - kks + 1

    # Define heat capacity of dry air and water vapor based on
    # hydrostatical property
    if phys_hydrostatic or hydrostatic:

        c_air = cp_air
        c_vap = cp_vap
        p_nonhydro = 0

    else:

        c_air = cv_air
        c_vap = cv_vap
        p_nonhydro = 1

    d0_vap = c_vap - c_liq
    lv00 = hlv0 - d0_vap * t_ice

    if hydrostatic:
        do_sedi_w = 0

    # Define cloud microphysics sub time step
    mpdt = np.minimum(dt_in, mp_time)
    rdt = 1.0 / dt_in
    ntimes = DTYPE_INT(round(dt_in / mpdt))

    # Small time step
    dts = dt_in / ntimes

    dt_rain = dts * 0.5

    # Calculate cloud condensation nuclei (ccn) based on klein eq. 15
    cpaut = c_paut * 0.104 * grav / 1.717e-5

    # Set use_ccn to false if prog_ccn is true
    if prog_ccn == 1:
        use_ccn = 0
    exec_info = {}
    ### Major cloud microphysics ###
    fields_init(
        land,
        area,
        h_var,
        rh_adj,
        rh_rain,
        graupel,
        ice,
        rain,
        snow,
        qa,
        qg,
        qi,
        ql,
        qn,
        qr,
        qs,
        qv,
        pt,
        delp,
        dz,
        qgz,
        qiz,
        qlz,
        qrz,
        qsz,
        qvz,
        tz,
        qi_dt,
        qs_dt,
        uin,
        vin,
        qa0,
        qg0,
        qi0,
        ql0,
        qr0,
        qs0,
        qv0,
        t0,
        dp0,
        den0,
        dz0,
        u0,
        v0,
        dp1,
        p1,
        u1,
        v1,
        ccn,
        c_praut,
        DTYPE_INT(use_ccn),
        c_air,
        c_vap,
        d0_vap,
        lv00,
        dt_in,
        rdt,
        cpaut,
        exec_info=exec_info,
    )

    so3 = 7.0 / 3.0

    zs = 0.0

    rdts = 1.0 / dts

    if fast_sat_adj:
        dt_evap = 0.5 * dts
    else:
        dt_evap = dts

    # Define conversion scalar / factor
    fac_i2s = 1.0 - mt.exp(-dts / tau_i2s)
    fac_g2v = 1.0 - mt.exp(-dts / tau_g2v)
    fac_v2g = 1.0 - mt.exp(-dts / tau_v2g)
    fac_imlt = 1.0 - mt.exp(-0.5 * dts / tau_imlt)
    fac_l2v = 1.0 - mt.exp(-dt_evap / tau_l2v)

    for n in range(ntimes):

        exec_info = {}

        # Time-split warm rain processes: 1st pass
        warm_rain(
            h_var,
            rain,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            w,
            t0,
            den0,
            dz0,
            dz1,
            dp1,
            m1,
            vtrz,
            ccn,
            c_praut,
            m1_sol,
            m2_rain,
            m2_sol,
            DTYPE_INT(1),
            DTYPE_INT(do_sedi_w),
            DTYPE_INT(p_nonhydro),
            DTYPE_INT(use_ccn),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            fac_rc,
            cracw,
            crevp[0],
            crevp[1],
            crevp[2],
            crevp[3],
            crevp[4],
            t_wfr,
            so3,
            dt_rain,
            zs,
            exec_info=exec_info,
        )

        exec_info = {}

        # Sedimentation of cloud ice, snow, and graupel
        sedimentation(
            graupel,
            ice,
            rain,
            snow,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            w,
            dz1,
            dp1,
            vtgz,
            vtsz,
            m1_sol,
            DTYPE_INT(do_sedi_w),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            log_10,
            zs,
            dts,
            fac_imlt,
            exec_info=exec_info,
        )

        exec_info = {}

        # Time-split warm rain processes: 2nd pass
        warm_rain(
            h_var,
            rain,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            w,
            t0,
            den0,
            dz0,
            dz1,
            dp1,
            m1,
            vtrz,
            ccn,
            c_praut,
            m1_sol,
            m2_rain,
            m2_sol,
            DTYPE_INT(0),
            DTYPE_INT(do_sedi_w),
            DTYPE_INT(p_nonhydro),
            DTYPE_INT(use_ccn),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            fac_rc,
            cracw,
            crevp[0],
            crevp[1],
            crevp[2],
            crevp[3],
            crevp[4],
            t_wfr,
            so3,
            dt_rain,
            zs,
            exec_info=exec_info,
        )

        exec_info = {}

        # Ice-phase microphysics
        icloud(
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
            p1,
            vtgz,
            vtrz,
            vtsz,
            c_air,
            c_vap,
            d0_vap,
            lv00,
            cracs,
            csacr,
            cgacr,
            cgacs,
            acco[0, 0],
            acco[0, 1],
            acco[0, 2],
            acco[0, 3],
            acco[1, 0],
            acco[1, 1],
            acco[1, 2],
            acco[1, 3],
            acco[2, 0],
            acco[2, 1],
            acco[2, 2],
            acco[2, 3],
            csacw,
            csaci,
            cgacw,
            cgaci,
            cracw,
            cssub[0],
            cssub[1],
            cssub[2],
            cssub[3],
            cssub[4],
            cgfr[0],
            cgfr[1],
            csmlt[0],
            csmlt[1],
            csmlt[2],
            csmlt[3],
            csmlt[4],
            cgmlt[0],
            cgmlt[1],
            cgmlt[2],
            cgmlt[3],
            cgmlt[4],
            ces0,
            tice0,
            t_wfr,
            dts,
            rdts,
            fac_i2s,
            fac_g2v,
            fac_v2g,
            fac_imlt,
            fac_l2v,
            exec_info=exec_info,
        )
    exec_info = {}
    fields_update(
        graupel,
        ice,
        rain,
        snow,
        qaz,
        qgz,
        qiz,
        qlz,
        qrz,
        qsz,
        qvz,
        tz,
        udt,
        vdt,
        qa_dt,
        qg_dt,
        qi_dt,
        ql_dt,
        qr_dt,
        qs_dt,
        qv_dt,
        pt_dt,
        qa0,
        qg0,
        qi0,
        ql0,
        qr0,
        qs0,
        qv0,
        t0,
        dp0,
        u0,
        v0,
        dp1,
        u1,
        v1,
        m1,
        m2_rain,
        m2_sol,
        ntimes,
        c_air,
        c_vap,
        rdt,
        exec_info=exec_info,
    )

    """
    NOTE: Radar part missing (never executed since lradar is false)
    """

    output = view_gt4py_storage(
        {
            "qi": qi[:, :, :],
            "qs": qs[:, :, :],
            "qv_dt": qv_dt[:, :, :],
            "ql_dt": ql_dt[:, :, :],
            "qr_dt": qr_dt[:, :, :],
            "qi_dt": qi_dt[:, :, :],
            "qs_dt": qs_dt[:, :, :],
            "qg_dt": qg_dt[:, :, :],
            "qa_dt": qa_dt[:, :, :],
            "pt_dt": pt_dt[:, :, :],
            "w": w[:, :, :],
            "udt": udt[:, :, :],
            "vdt": vdt[:, :, :],
            "rain": rain[:, :, 0],
            "snow": snow[:, :, 0],
            "ice": ice[:, :, 0],
            "graupel": graupel[:, :, 0],
            "refl_10cm": refl_10cm[:, :, :],
        }
    )
    return output
