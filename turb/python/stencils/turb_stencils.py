import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from config import *
from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    PARALLEL,
    FORWARD,
    computation,
    interval,
    exp,
    floor,
    sqrt,
)

STENCIL_OPTS = {"backend": BACKEND}
if BACKEND != "numpy":
    STENCIL_OPTS["skip_passes"] = ["graph_merge_horizontal_executions"]

STENCIL_OPTS_2 = {"backend": BACKEND}
if BACKEND != "numpy":
    STENCIL_OPTS_2["skip_passes"] = ["graph_merge_horizontal_executions", "GreedyMerging"]

backend = BACKEND


@gtscript.function
def fpvsx(t):
    con_ttp = 2.7316e2
    con_cvap = 1.8460e3
    con_cliq = 4.1855e3
    con_hvap = 2.5000e6
    con_rv = 4.6150e2
    con_csol = 2.1060e3
    con_hfus = 3.3358e5
    con_psat = 6.1078e2

    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)

    tr = con_ttp / t

    fpvsx = 0.0
    if t > tliq:
        fpvsx = con_psat * tr ** xponal * exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = con_psat * tr ** xponai * exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = con_psat * (tr ** xponal) * exp(xponbl * (1.0 - tr))
        pvi = con_psat * (tr ** xponai) * exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx


@gtscript.function
def fpvs(t):
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = min(max(c1xpvs + c2xpvs * t[0, 0, 0], 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    jx = floor(jx)

    # Convert jx to "x"
    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

    return fpvs


@gtscript.stencil(backend=backend)
def mask_init(mask: FIELD_INT):
    with computation(FORWARD), interval(1, None):
        mask = mask[0, 0, -1] + 1


@gtscript.stencil(backend=backend)
def init(
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    phii: FIELD_FLT,
    phil: FIELD_FLT,
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    garea: FIELD_FLT_IJ,
    gdx: FIELD_FLT_IJ,
    tke: FIELD_FLT,
    q1: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    prn: FIELD_FLT,
    kx1: FIELD_INT,
    prsi: FIELD_FLT,
    mask: FIELD_INT,
    kinver: FIELD_INT_IJ,
    tx1: FIELD_FLT_IJ,
    tx2: FIELD_FLT_IJ,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    kpblx: FIELD_INT_IJ,
    hpblx: FIELD_FLT_IJ,
    pblflg: FIELD_BOOL_IJ,
    sfcflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    scuflg: FIELD_BOOL_IJ,
    zorl: FIELD_FLT_IJ,
    dusfc: FIELD_FLT_IJ,
    dvsfc: FIELD_FLT_IJ,
    dtsfc: FIELD_FLT_IJ,
    dqsfc: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    hpbl: FIELD_FLT_IJ,
    rbsoil: FIELD_FLT_IJ,
    radmin: FIELD_FLT_IJ,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    kcld: FIELD_INT_IJ,
    theta: FIELD_FLT,
    prslk: FIELD_FLT,
    psk: FIELD_FLT_IJ,
    t1: FIELD_FLT,
    pix: FIELD_FLT,
    qlx: FIELD_FLT,
    slx: FIELD_FLT,
    thvx: FIELD_FLT,
    qtx: FIELD_FLT,
    thlx: FIELD_FLT,
    thlvx: FIELD_FLT,
    svx: FIELD_FLT,
    thetae: FIELD_FLT,
    gotvx: FIELD_FLT,
    prsl: FIELD_FLT,
    plyr: FIELD_FLT,
    rhly: FIELD_FLT,
    qstl: FIELD_FLT,
    bf: FIELD_FLT,
    cfly: FIELD_FLT,
    crb: FIELD_FLT_IJ,
    dtdz1: FIELD_FLT,
    evap: FIELD_FLT_IJ,
    heat: FIELD_FLT_IJ,
    hlw: FIELD_FLT,
    radx: FIELD_FLT,
    rbup: FIELD_FLT_IJ,
    sflux: FIELD_FLT_IJ,
    shr2: FIELD_FLT,
    stress: FIELD_FLT_IJ,
    swh: FIELD_FLT,
    thermal: FIELD_FLT_IJ,
    tsea: FIELD_FLT_IJ,
    u10m: FIELD_FLT_IJ,
    ustar: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    v10m: FIELD_FLT_IJ,
    xmu: FIELD_FLT_IJ,
    gravi: float,
    dt2: float,
    el2orc: float,
    tkmin: float,
    xkzm_h: float,
    xkzm_m: float,
    xkzm_s: float,
    km1: int,
    ntiw: int,
    fv: float,
    elocp: float,
    g: float,
    eps: float,
    ntke: int,
    ntcw: int,
    hvap: float,
    hfus: float,
    rbcr: float,
    f0: float,
    crbmin: float,
    crbmax: float,
    qmin: float,
    qlmin: float,
    cql: float,
    dw2min: float,
    xkgdx: float,
    xkzinv: float,
    ck1: float,
    ch1: float,
    cp: float,
):

    with computation(FORWARD), interval(0, 1):
        pcnvflg = 0
        scuflg = 1
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl = 1
        hpbl = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = 1
        lcld = km1 - 1
        kcld = km1 - 1
        mrad = km1
        krad = 0
        radmin = 0.0
        sfcflg = 1
        if rbsoil[0, 0] > 0.0:
            sfcflg = 0
        gdx = sqrt(garea[0, 0])

    with computation(PARALLEL), interval(...):
        zi = phii[0, 0, 0] * gravi
        zl = phil[0, 0, 0] * gravi
        tke = max(q1[0, 0, 0][ntke], tkmin)
    with computation(PARALLEL), interval(0, -1):
        ckz = ck1
        chz = ch1
        prn = 1.0
        kx1 = 0.0
        zm = zi[0, 0, 1]
        rdzt = 1.0 / (zl[0, 0, 1] - zl[0, 0, 0])

        if gdx[0, 0] >= xkgdx:
            xkzm_hx = xkzm_h
            xkzm_mx = xkzm_m
        else:
            xkzm_hx = 0.01 + ((xkzm_h - 0.01) * (1.0 / (xkgdx - 5.0))) * (
                gdx[0, 0] - 5.0
            )
            xkzm_mx = 0.01 + ((xkzm_m - 0.01) * (1.0 / (xkgdx - 5.0))) * (
                gdx[0, 0] - 5.0
            )

        if mask[0, 0, 0] < kinver[0, 0]:
            ptem = prsi[0, 0, 1] * tx1[0, 0]
            xkzo = xkzm_hx * min(1.0, exp(-((1.0 - ptem) * (1.0 - ptem) * 10.0)))

            if ptem >= xkzm_s:
                xkzmo = xkzm_mx
                kx1 = mask[0, 0, 0] + 1
            else:
                tem1 = min(
                    1.0,
                    exp(
                        -(
                            (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * 5.0
                        )
                    ),
                )
                xkzmo = xkzm_mx * tem1

        pix = psk[0, 0] / prslk[0, 0, 0]
        theta = t1[0, 0, 0] * pix[0, 0, 0]
        if (ntiw + 1) > 0:
            tem = max(q1[0, 0, 0][ntcw], qlmin)
            tem1 = max(q1[0, 0, 0][ntiw], qlmin)
            ptem = hvap * tem + (hvap + hfus) * tem1
            qlx = tem + tem1
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - ptem
        else:
            qlx = max(q1[0, 0, 0][ntcw], qlmin)
            slx = cp * t1[0, 0, 0] + phil[0, 0, 0] - hvap * qlx[0, 0, 0]

        tem = 1.0 + fv * max(q1[0, 0, 0][0], qmin) - qlx[0, 0, 0]
        thvx = theta[0, 0, 0] * tem
        qtx = max(q1[0, 0, 0][0], qmin) + qlx[0, 0, 0]
        thlx = theta[0, 0, 0] - pix[0, 0, 0] * elocp * qlx[0, 0, 0]
        thlvx = thlx[0, 0, 0] * (1.0 + fv * qtx[0, 0, 0])
        svx = cp * t1[0, 0, 0] * tem
        thetae = theta[0, 0, 0] + elocp * pix[0, 0, 0] * max(q1[0, 0, 0][0], qmin)
        gotvx = g / (t1[0, 0, 0] * tem)

        tem = (t1[0, 0, 1] - t1[0, 0, 0]) * tem * rdzt[0, 0, 0]
        if tem > 1.0e-5:
            xkzo = min(xkzo[0, 0, 0], xkzinv)
            xkzmo = min(xkzmo[0, 0, 0], xkzinv)

        plyr = 0.01 * prsl[0, 0, 0]
        es = 0.01 * fpvs(t1)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + (eps - 1) * es))
        rhly = max(0.0, min(1.0, max(qmin, q1[0, 0, 0][0]) / qs))
        qstl = qs

    with computation(FORWARD), interval(...):
        cfly = 0.0
        clwt = 1.0e-6 * (plyr[0, 0, 0] * 0.001)
        if qlx[0, 0, 0] > clwt:
            onemrh = max(1.0e-10, 1.0 - rhly[0, 0, 0])
            tem1 = cql / min(max((onemrh * qstl[0, 0, 0]) ** 0.49, 0.0001), 1.0)
            val = max(min(tem1 * qlx[0, 0, 0], 50.0), 0.0)
            cfly = min(max(sqrt(sqrt(rhly[0, 0, 0])) * (1.0 - exp(-val)), 0.0), 1.0)

    with computation(PARALLEL), interval(0, -2):
        tem1 = 0.5 * (t1[0, 0, 0] + t1[0, 0, 1])
        cfh = min(cfly[0, 0, 1], 0.5 * (cfly[0, 0, 0] + cfly[0, 0, 1]))
        alp = g / (0.5 * (svx[0, 0, 0] + svx[0, 0, 1]))
        gamma = el2orc * (0.5 * (qstl[0, 0, 0] + qstl[0, 0, 1])) / (tem1 ** 2)
        epsi = tem1 / elocp
        beta = (1.0 + gamma * epsi * (1.0 + fv)) / (1.0 + gamma)
        chx = cfh * alp * beta + (1.0 - cfh) * alp
        cqx = cfh * alp * hvap * (beta - epsi)
        cqx = cqx + (1.0 - cfh) * fv * g
        bf = chx * ((slx[0, 0, 1] - slx[0, 0, 0]) * rdzt[0, 0, 0]) + cqx * (
            (qtx[0, 0, 1] - qtx[0, 0, 0]) * rdzt[0, 0, 0]
        )
        radx = (zi[0, 0, 1] - zi[0, 0, 0]) * (swh[0, 0, 0] * xmu[0, 0] + hlw[0, 0, 0])

    with computation(FORWARD):
        with interval(0, 1):
            sflux = heat[0, 0] + evap[0, 0] * fv * theta[0, 0, 0]

            if sfcflg[0, 0] == 0 or sflux[0, 0] <= 0.0:
                pblflg = 0

            if pblflg[0, 0]:
                thermal = thlvx[0, 0, 0]
                crb = rbcr
            else:
                tem1 = 1e-7 * (
                    max(sqrt(u10m[0, 0] ** 2 + v10m[0, 0] ** 2), 1.0)
                    / (f0 * 0.01 * zorl[0, 0])
                )
                thermal = tsea[0, 0] * (1.0 + fv * max(q1[0, 0, 0][0], qmin))
                crb = max(min(0.16 * (tem1 ** (-0.18)), crbmax), crbmin)

            dtdz1 = dt2 / (zi[0, 0, 1] - zi[0, 0, 0])
            ustar = sqrt(stress[0, 0])

    with computation(PARALLEL):
        with interval(0, -2):
            dw2 = (u1[0, 0, 0] - u1[0, 0, 1]) ** 2 + (v1[0, 0, 0] - v1[0, 0, 1]) ** 2
            shr2 = max(dw2, dw2min) * rdzt[0, 0, 0] * rdzt[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            rbup = rbsoil[0, 0]


# Possible stencil name : mrf_pbl_scheme_part1
@gtscript.stencil(backend=backend)
def mrf_pbl_scheme_part1(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    g: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]

            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(1, None):
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]


# Possible stencil name : mrf_pbl_2_thermal_1
@gtscript.stencil(**STENCIL_OPTS)
def mrf_pbl_2_thermal_1(
    crb: FIELD_FLT_IJ,
    evap: FIELD_FLT_IJ,
    fh: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    fm: FIELD_FLT_IJ,
    gotvx: FIELD_FLT,
    heat: FIELD_FLT_IJ,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    phih: FIELD_FLT_IJ,
    phim: FIELD_FLT_IJ,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    rbsoil: FIELD_FLT_IJ,
    sfcflg: FIELD_BOOL_IJ,
    sflux: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    theta: FIELD_FLT,
    ustar: FIELD_FLT_IJ,
    vpert: FIELD_FLT_IJ,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT_IJ,
    fv: float,
    wfac: float,
    cfac: float,
    gamcrt: float,
    sfcfrac: float,
    vk: float,
    rimin: float,
    zolcru: float,
    zfmin: float,
    aphi5: float,
    aphi16: float,
    h1: float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == kpblx[0, 0]:
            if kpblx[0, 0] > 0:
                if rbdn[0, 0] >= crb[0, 0]:
                    rbint = 0.0
                elif rbup[0, 0] <= crb[0, 0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])
                hpblx = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

                if hpblx[0, 0] < zi[0, 0, 0]:
                    kpblx = kpblx[0, 0] - 1
            else:
                hpblx = zl[0, 0, 0]
                kpblx = 0

            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]

            if kpbl[0, 0] <= 0:
                pblflg = 0

    with computation(FORWARD), interval(0, 1):
        zol = max(rbsoil[0, 0] * fm[0, 0] * fm[0, 0] / fh[0, 0], rimin)
        if sfcflg[0, 0]:
            zol = min(zol[0, 0], -zfmin)
        else:
            zol = max(zol[0, 0], zfmin)

        zol1 = zol[0, 0] * sfcfrac * hpbl[0, 0] / zl[0, 0, 0]

        if sfcflg[0, 0]:
            phih = sqrt(1.0 / (1.0 - aphi16 * zol1))
            phim = sqrt(phih[0, 0])
        else:
            phim = 1.0 + aphi5 * zol1
            phih = phim[0, 0]

        pcnvflg = pblflg[0, 0] and (zol[0, 0] < zolcru)

        wst3 = gotvx[0, 0, 0] * sflux[0, 0] * hpbl[0, 0]
        ust3 = ustar[0, 0] ** 3.0

        if pblflg[0, 0]:
            wscale = max((ust3 + wfac * vk * wst3 * sfcfrac) ** h1, ustar[0, 0] / aphi5)

        flg = 1

        if pcnvflg[0, 0]:
            hgamt = heat[0, 0] / wscale
            hgamq = evap[0, 0] / wscale
            vpert = max(hgamt + hgamq * fv * theta[0, 0, 0], 0.0)
            thermal = thermal[0, 0] + min(cfac * vpert[0, 0], gamcrt)
            flg = 0
            rbup = rbsoil[0, 0]


# Possible stencil name : thermal_2
@gtscript.stencil(backend=backend)
def thermal_2(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thermal: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlvx_0: FIELD_FLT_IJ,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    zl: FIELD_FLT,
    g: float,
):

    with computation(FORWARD):
        with interval(1, 2):
            thlvx_0 = thlvx[0, 0, -1]
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(2, None):
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]


# Possible stencil name : pbl_height_enhance
@gtscript.stencil(backend=backend)
def pbl_height_enhance(
    crb: FIELD_FLT_IJ,
    flg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    mask: FIELD_INT,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    scuflg: FIELD_BOOL_IJ,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zstblmax: float,
):

    with computation(FORWARD), interval(...):
        if pcnvflg[0, 0] and kpbl[0, 0] == mask[0, 0, 0]:
            if rbdn[0, 0] >= crb[0, 0]:
                rbint = 0.0
            elif rbup[0, 0] <= crb[0, 0]:
                rbint = 1.0
            else:
                rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])

            hpbl[0, 0] = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

            if hpbl[0, 0] < zi[0, 0, 0]:
                kpbl[0, 0] = kpbl[0, 0] - 1

            if kpbl[0, 0] <= 0:
                pblflg[0, 0] = 0
                pcnvflg[0, 0] = 0

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]
            if flg[0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0
        with interval(1, -1):
            if flg[0, 0] and (zl[0, 0, 0] >= zstblmax):
                lcld = mask[0, 0, 0]
                flg = 0


# Possible stencil name : stratocumulus
@gtscript.stencil(backend=backend)
def stratocumulus(
    flg: FIELD_BOOL_IJ,
    kcld: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    lcld: FIELD_INT_IJ,
    mask: FIELD_INT,
    radmin: FIELD_FLT_IJ,
    radx: FIELD_FLT,
    qlx: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    km1: int,
    qlcr: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and (mask[0, 0, 0] <= lcld[0, 0]) and (qlx[0, 0, 0] >= qlcr):
                kcld = mask[0, 0, 0]
                flg = 0

        with interval(0, -1):
            if flg[0, 0] and (mask[0, 0, 0] <= lcld[0, 0]) and (qlx[0, 0, 0] >= qlcr):
                kcld = mask[0, 0, 0]
                flg = 0

    with computation(FORWARD):
        with interval(0, 1):
            if scuflg[0, 0] and (kcld[0, 0] == (km1 - 1)):
                scuflg = 0
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

        with interval(0, -1):
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= qlcr:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if scuflg[0, 0] and krad[0, 0] <= 0:
            scuflg = 0
        if scuflg[0, 0] and radmin[0, 0] >= 0.0:
            scuflg = 0


@gtscript.stencil(backend=backend)
def mass_flux_comp(
    pcnvflg: FIELD_BOOL_IJ,
    q1: FIELD_FLT_8,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
    t1: FIELD_FLT,
    tcdo: FIELD_FLT,
    tcko: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
):
    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            tcko = t1[0, 0, 0]
            ucko = u1[0, 0, 0]
            vcko = v1[0, 0, 0]
            for ii in range(8):
                qcko[0, 0, 0][ii] = q1[0, 0, 0][ii]
        if scuflg[0, 0]:
            tcdo = t1[0, 0, 0]
            ucdo = u1[0, 0, 0]
            vcdo = v1[0, 0, 0]
            for i2 in range(8):
                qcdo[0, 0, 0][i2] = q1[0, 0, 0][i2]


# Possible stencil name : prandtl_comp_exchg_coeff
@gtscript.stencil(**STENCIL_OPTS)
def prandtl_comp_exchg_coeff(
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    pcnvflg: FIELD_BOOL_IJ,
    phih: FIELD_FLT_IJ,
    phim: FIELD_FLT_IJ,
    prn: FIELD_FLT,
    zi: FIELD_FLT,
    sfcfrac: float,
    prmin: float,
    prmax: float,
    ck0: float,
    ck1: float,
    ch0: float,
    ch1: float,
):

    with computation(PARALLEL), interval(...):
        tem1 = max(zi[0, 0, 1] - sfcfrac * hpbl[0, 0], 0.0)
        ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0] ** 2.0)
        if mask[0, 0, 0] < kpbl[0, 0]:
            if pcnvflg[0, 0]:
                prn = 1.0 + ((phih[0, 0] / phim[0, 0]) - 1.0) * exp(ptem)
            else:
                prn = phih[0, 0] / phim[0, 0]

        if mask[0, 0, 0] < kpbl[0, 0]:
            prn = max(min(prn[0, 0, 0], prmax), prmin)
            ckz = max(min(ck1 + (ck0 - ck1) * exp(ptem), ck0), ck1)
            chz = max(min(ch1 + (ch0 - ch1) * exp(ptem), ch0), ch1)


# Possible stencil name : compute_eddy_buoy_shear
@gtscript.stencil(backend=backend)
def compute_eddy_buoy_shear(
    bf: FIELD_FLT,
    buod: FIELD_FLT,
    buou: FIELD_FLT,
    chz: FIELD_FLT,
    ckz: FIELD_FLT,
    dku: FIELD_FLT,
    dkt: FIELD_FLT,
    dkq: FIELD_FLT,
    ele: FIELD_FLT,
    elm: FIELD_FLT,
    gdx: FIELD_FLT_IJ,
    gotvx: FIELD_FLT,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    pblflg: FIELD_BOOL_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    phim: FIELD_FLT_IJ,
    prn: FIELD_FLT,
    prod: FIELD_FLT,
    radj: FIELD_FLT_IJ,
    rdzt: FIELD_FLT,
    rlam: FIELD_FLT,
    rle: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    sflux: FIELD_FLT_IJ,
    shr2: FIELD_FLT,
    stress: FIELD_FLT_IJ,
    tke: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    ustar: FIELD_FLT_IJ,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xkzo: FIELD_FLT,
    xkzmo: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    zi: FIELD_FLT,
    zl: FIELD_FLT,
    zol: FIELD_FLT_IJ,
    vk: float,
    rimin: float,
    tdzmin: float,
    prmax: float,
    prtke: float,
    prscu: float,
    dkmax: float,
    ck1: float,
    ch1: float,
    ce0: float,
    rchck: float,
):

    with computation(FORWARD):
        with interval(0, -1):
            if zol[0, 0] < 0.0:
                zk = vk * zl[0, 0, 0] * (1.0 - 100.0 * zol[0, 0]) ** 0.2
            elif zol[0, 0] >= 1.0:
                zk = vk * zl[0, 0, 0] / 3.7
            else:
                zk = vk * zl[0, 0, 0] / (1.0 + 2.7 * zol[0, 0])

            elm = zk * rlam[0, 0, 0] / (rlam[0, 0, 0] + zk)
            dz = zi[0, 0, 1] - zi[0, 0, 0]
            tem = max(gdx[0, 0], dz)
            elm = min(elm[0, 0, 0], tem)
            ele = min(ele[0, 0, 0], tem)

        with interval(-1, None):
            elm = elm[0, 0, -1]
            ele = ele[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        tem = (
            0.5
            * (elm[0, 0, 0] + elm[0, 0, 1])
            * sqrt(0.5 * (tke[0, 0, 0] + tke[0, 0, 1]))
        )
        ri = max(bf[0, 0, 0] / shr2[0, 0, 0], rimin)

        if mask[0, 0, 0] < kpbl[0, 0]:
            if pblflg[0, 0]:
                dku = ckz[0, 0, 0] * tem
                dkt = dku[0, 0, 0] / prn[0, 0, 0]
            else:
                dkt = chz[0, 0, 0] * tem
                dku = dkt[0, 0, 0] * prn[0, 0, 0]
        else:
            if ri < 0.0:
                dku = ck1 * tem
                dkt = rchck * dku[0, 0, 0]
            else:
                dkt = ch1 * tem
                dku = dkt[0, 0, 0] * min(1.0 + 2.1 * ri, prmax)

        tem = ckz[0, 0, 0] * tem
        dku_tmp = max(dku[0, 0, 0], tem)
        dkt_tmp = max(dkt[0, 0, 0], tem / prscu)

        if scuflg[0, 0]:
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                dku = dku_tmp
                dkt = dkt_tmp

        dkq = prtke * dkt[0, 0, 0]

        dkt = max(min(dkt[0, 0, 0], dkmax), xkzo[0, 0, 0])

        dkq = max(min(dkq[0, 0, 0], dkmax), xkzo[0, 0, 0])

        dku = max(min(dku[0, 0, 0], dkmax), xkzmo[0, 0, 0])

    with computation(PARALLEL), interval(...):
        if mask[0, 0, 0] == krad[0, 0]:
            if scuflg[0, 0]:
                tem1 = bf[0, 0, 0] / gotvx[0, 0, 0]
                if tem1 < tdzmin:
                    tem1 = tdzmin
                ptem = radj[0, 0] / tem1
                dkt = dkt[0, 0, 0] + ptem
                dku = dku[0, 0, 0] + ptem
                dkq = dkq[0, 0, 0] + ptem

    with computation(PARALLEL):
        with interval(0, 1):
            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem = xmfd[0, 0, 0] * buod[0, 0, 0]
                ptem1 = (
                    0.5
                    * (u1[0, 0, 1] - u1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (ucdo[0, 0, 0] + ucdo[0, 0, 1] - u1[0, 0, 0] - u1[0, 0, 1])
                )
                ptem2 = (
                    0.5
                    * (v1[0, 0, 1] - v1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (vcdo[0, 0, 0] + vcdo[0, 0, 1] - v1[0, 0, 0] - v1[0, 0, 1])
                )
            else:
                ptem = 0.0
                ptem1 = 0.0
                ptem2 = 0.0

            buop = 0.5 * (
                gotvx[0, 0, 0] * sflux[0, 0] + (-dkt[0, 0, 0] * bf[0, 0, 0] + ptem)
            )

            shrp = 0.5 * (
                dku[0, 0, 0] * shr2[0, 0, 0]
                + ptem1
                + ptem2
                + (stress[0, 0] * ustar[0, 0] * phim[0, 0] / (vk * zl[0, 0, 0]))
            )

            prod = buop + shrp

        with interval(1, -1):
            tem1_1 = (u1[0, 0, 1] - u1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_1 = (u1[0, 0, 0] - u1[0, 0, -1]) * rdzt[0, 0, -1]
            tem1_2 = (v1[0, 0, 1] - v1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_2 = (v1[0, 0, 0] - v1[0, 0, -1]) * rdzt[0, 0, -1]

            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1_0 = 0.5 * (xmf[0, 0, -1] + xmf[0, 0, 0]) * buou[0, 0, 0]
                ptem1_1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_1 + xmf[0, 0, -1] * tem2_1)
                    * (u1[0, 0, 0] - ucko[0, 0, 0])
                )
                ptem1_2 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_2 + xmf[0, 0, -1] * tem2_2)
                    * (v1[0, 0, 0] - vcko[0, 0, 0])
                )
            else:
                ptem1_0 = 0.0
                ptem1_1 = 0.0
                ptem1_2 = 0.0

            if scuflg[0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                    ptem2_0 = 0.5 * (xmfd[0, 0, -1] + xmfd[0, 0, 0]) * buod[0, 0, 0]
                    ptem2_1 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_1 + xmfd[0, 0, -1] * tem2_1)
                        * (ucdo[0, 0, 0] - u1[0, 0, 0])
                    )
                    ptem2_2 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_2 + xmfd[0, 0, -1] * tem2_2)
                        * (vcdo[0, 0, 0] - v1[0, 0, 0])
                    )
                else:
                    ptem2_0 = 0.0
                    ptem2_1 = 0.0
                    ptem2_2 = 0.0
            else:
                ptem2_0 = 0.0
                ptem2_1 = 0.0
                ptem2_2 = 0.0

            buop = (
                0.5 * ((-dkt[0, 0, -1] * bf[0, 0, -1]) + (-dkt[0, 0, 0] * bf[0, 0, 0]))
                + ptem1_0
                + ptem2_0
            )

            shrp = (
                (
                    0.5
                    * (
                        (dku[0, 0, -1] * shr2[0, 0, -1])
                        + (dku[0, 0, 0] * shr2[0, 0, 0])
                    )
                    + ptem1_1
                    + ptem2_1
                )
                + ptem1_2
                + ptem2_2
            )

            prod = buop + shrp

    with computation(PARALLEL), interval(0, -1):
        rle = ce0 / ele[0, 0, 0]


# Possible stencil name : predict_tke
@gtscript.stencil(backend=backend)
def predict_tke(
    diss: FIELD_FLT,
    prod: FIELD_FLT,
    rle: FIELD_FLT,
    tke: FIELD_FLT,
    dtn: float,
    kk: int,
    tkmin: float,
):
    with computation(PARALLEL), interval(...):
        for n in range(kk):
            diss = max(
                min(
                    rle[0, 0, 0] * tke[0, 0, 0] * sqrt(tke[0, 0, 0]),
                    prod[0, 0, 0] + tke[0, 0, 0] / dtn,
                ),
                0.0,
            )
            tke = max(tke[0, 0, 0] + dtn * (prod[0, 0, 0] - diss[0, 0, 0]), tkmin)


# Possible stencil name : tke_up_down_prop_1
@gtscript.stencil(backend=backend)
def tke_up_down_prop(
    pcnvflg: FIELD_BOOL_IJ,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
    tke: FIELD_FLT,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    xlamue: FIELD_FLT,
    zl: FIELD_FLT,
    ad: FIELD_FLT,
    f1: FIELD_FLT,
    krad: FIELD_INT_IJ,
    mrad: FIELD_INT_IJ,
    xlamde: FIELD_FLT,
    kmpbl: int,
    kmscu: int,
):

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            qcko[0, 0, 0][7] = tke[0, 0, 0]
        if scuflg[0, 0]:
            qcdo[0, 0, 0][7] = tke[0, 0, 0]

    with computation(FORWARD), interval(1, None):
        if mask[0, 0, 0] < kmpbl:
            tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                qcko[0, 0, 0][7] = (
                    (1.0 - tem) * qcko[0, 0, -1][7]
                    + tem * (tke[0, 0, 0] + tke[0, 0, -1])
                ) / (1.0 + tem)

    with computation(BACKWARD), interval(...):
        if mask[0, 0, 0] < kmscu:
            tem = 0.5 * xlamde[0, 0, 0] * (zl[0, 0, 1] - zl[0, 0, 0])
            if (
                scuflg[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
            ):
                qcdo[0, 0, 0][7] = (
                    (1.0 - tem) * qcdo[0, 0, 1][7] + tem * (tke[0, 0, 0] + tke[0, 0, 1])
                ) / (1.0 + tem)

    with computation(PARALLEL), interval(0, 1):
        if mask[0, 0, 0] < kmscu:
            ad = 1.0
            f1 = tke[0, 0, 0]


# Possible stencil name : tke_tridiag_matrix_ele_comp
@gtscript.stencil(backend=backend)
def tke_tridiag_matrix_ele_comp(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkq: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    tke: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dt2: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]
            tem2 = dsig * rdz

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7] + qcko[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * tem2 * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * tem2 * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7] + qcdo[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * tem2 * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * tem2 * xmfd[0, 0, 0]
        with interval(1, -1):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7] + qcko[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * dsig * rdz * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * dsig * rdz * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7] + qcdo[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * dsig * rdz * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * dsig * rdz * xmfd[0, 0, 0]

        with interval(-1, None):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]


@gtscript.stencil(backend=backend)
def part12a(
    rtg: FIELD_FLT_8,
    f1: FIELD_FLT,
    q1: FIELD_FLT_8,
    ad: FIELD_FLT,
    f2: FIELD_FLT_8,
    dtdz1: FIELD_FLT,
    evap: FIELD_FLT_IJ,
    heat: FIELD_FLT_IJ,
    t1: FIELD_FLT,
    rdt: float,
    ntrac1: int,
    ntke: int,
):
    with computation(PARALLEL), interval(...):
        rtg[0, 0, 0][ntke - 1] = (
            rtg[0, 0, 0][ntke - 1] + (f1[0, 0, 0] - q1[0, 0, 0][ntke - 1]) * rdt
        )

    with computation(FORWARD), interval(0, 1):
        ad = 1.0
        f1 = t1[0, 0, 0] + dtdz1[0, 0, 0] * heat[0, 0]
        f2[0, 0, 0][0] = q1[0, 0, 0][0] + dtdz1[0, 0, 0] * evap[0, 0]

        if ntrac1 >= 2:
            for kk in range(1, ntrac1):
                f2[0, 0, 0][kk] = q1[0, 0, 0][kk]


# Possible stencil name : heat_moist_tridiag_mat_ele_comp
@gtscript.stencil(backend=backend)
def heat_moist_tridiag_mat_ele_comp(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    dkt: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    f2: FIELD_FLT_7,
    f2_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    q1: FIELD_FLT_8,
    qcdo: FIELD_FLT_8,
    qcko: FIELD_FLT_8,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    tcdo: FIELD_FLT,
    tcko: FIELD_FLT,
    t1: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dt2: float,
    gocp: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = (
                    qcko[0, 0, 0][0]
                    + qcko[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = (
                    qcdo[0, 0, 0][0]
                    + qcdo[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = (
                    qcko[0, 0, 0][0]
                    + qcko[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = (
                    qcdo[0, 0, 0][0]
                    + qcdo[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


@gtscript.stencil(backend=backend)
def part13a(
    pcnvflg: FIELD_BOOL_IJ,
    mask: FIELD_INT,
    kpbl: FIELD_INT_IJ,
    del_: FIELD_FLT,
    prsl: FIELD_FLT,
    rdzt: FIELD_FLT,
    xmf: FIELD_FLT,
    qcko: FIELD_FLT_8,
    q1: FIELD_FLT_8,
    f2: FIELD_FLT_8,
    scuflg: FIELD_BOOL_IJ,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    xmfd: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    ntrac1: int,
    dt2: float,
):
    with computation(FORWARD), interval(0, -1):
        for kk in range(1, ntrac1):
            if mask[0, 0, 0] > 0:
                if pcnvflg[0, 0] and mask[0, 0, -1] < kpbl[0, 0]:
                    dtodsu = dt2 / del_[0, 0, 0]
                    dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
                    tem = dsig * rdzt[0, 0, -1]
                    ptem = 0.5 * tem * xmf[0, 0, -1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcko[0, 0, -1][kk] + qcko[0, 0, 0][kk]
                    tem2 = q1[0, 0, -1][kk] + q1[0, 0, 0][kk]
                    f2[0, 0, 0][kk] = q1[0, 0, 0][kk] + (tem1 - tem2) * ptem2
                else:
                    f2[0, 0, 0][kk] = q1[0, 0, 0][kk]

                if (
                    scuflg[0, 0]
                    and mask[0, 0, -1] >= mrad[0, 0]
                    and mask[0, 0, -1] < krad[0, 0]
                ):
                    dtodsu = dt2 / del_[0, 0, 0]
                    dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
                    tem = dsig * rdzt[0, 0, -1]
                    ptem = 0.5 * tem * xmfd[0, 0, -1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcdo[0, 0, -1][kk] + qcdo[0, 0, 0][kk]
                    tem2 = q1[0, 0, -1][kk] + q1[0, 0, 0][kk]
                    f2[0, 0, 0][kk] = f2[0, 0, 0][kk] - (tem1 - tem2) * ptem2

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                dtodsd = dt2 / del_[0, 0, 0]
                dtodsu = dt2 / del_[0, 0, 1]
                dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
                tem = dsig * rdzt[0, 0, 0]
                ptem = 0.5 * tem * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcko[0, 0, 0][kk] + qcko[0, 0, 1][kk]
                tem2 = q1[0, 0, 0][kk] + q1[0, 0, 1][kk]
                f2[0, 0, 0][kk] = f2[0, 0, 0][kk] - (tem1 - tem2) * ptem1

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                dtodsd = dt2 / del_[0, 0, 0]
                dtodsu = dt2 / del_[0, 0, 1]
                dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
                tem = dsig * rdzt[0, 0, 0]
                ptem = 0.5 * tem * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcdo[0, 0, 0][kk] + qcdo[0, 0, 1][kk]
                tem2 = q1[0, 0, 0][kk] + q1[0, 0, 1][kk]
                f2[0, 0, 0][kk] = f2[0, 0, 0][kk] + (tem1 - tem2) * ptem1

    with computation(FORWARD), interval(-1, None):
        for kk2 in range(1, ntrac1):
            if pcnvflg[0, 0] and mask[0, 0, -1] < kpbl[0, 0]:
                dtodsu = dt2 / del_[0, 0, 0]
                dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
                tem = dsig * rdzt[0, 0, -1]
                ptem = 0.5 * tem * xmf[0, 0, -1]
                ptem2 = dtodsu * ptem
                tem1 = qcko[0, 0, -1][kk2] + qcko[0, 0, 0][kk2]
                tem2 = q1[0, 0, -1][kk2] + q1[0, 0, 0][kk2]
                f2[0, 0, 0][kk2] = q1[0, 0, 0][kk2] + (tem1 - tem2) * ptem2
            else:
                f2[0, 0, 0][kk2] = q1[0, 0, 0][kk2]


@gtscript.stencil(backend=backend)
def part13b(
    f1: FIELD_FLT,
    t1: FIELD_FLT,
    f2: FIELD_FLT_8,
    q1: FIELD_FLT_8,
    tdt: FIELD_FLT,
    rtg: FIELD_FLT_8,
    dtsfc: FIELD_FLT_IJ,
    del_: FIELD_FLT,
    dqsfc: FIELD_FLT_IJ,
    conq: float,
    cont: float,
    rdt: float,
    ntrac1: int,
):
    with computation(PARALLEL), interval(...):
        tdt = tdt[0, 0, 0] + (f1[0, 0, 0] - t1[0, 0, 0]) * rdt
        rtg[0, 0, 0][0] = rtg[0, 0, 0][0] + (f2[0, 0, 0][0] - q1[0, 0, 0][0]) * rdt

        if ntrac1 >= 2:
            for kk in range(1, ntrac1):
                rtg[0, 0, 0][kk] = rtg[0, 0, 0][kk] + (
                    (f2[0, 0, 0][kk] - q1[0, 0, 0][kk]) * rdt
                )

    with computation(FORWARD), interval(...):
        dtsfc = dtsfc[0, 0] + cont * del_[0, 0, 0] * ((f1[0, 0, 0] - t1[0, 0, 0]) * rdt)
        dqsfc = dqsfc[0, 0] + conq * del_[0, 0, 0] * (
            (f2[0, 0, 0][0] - q1[0, 0, 0][0]) * rdt
        )


# Possible stencil name : moment_tridiag_mat_ele_comp
@gtscript.stencil(backend=backend)
def moment_tridiag_mat_ele_comp(
    ad: FIELD_FLT,
    ad_p1: FIELD_FLT_IJ,
    al: FIELD_FLT,
    au: FIELD_FLT,
    del_: FIELD_FLT,
    diss: FIELD_FLT,
    dku: FIELD_FLT,
    dtdz1: FIELD_FLT,
    f1: FIELD_FLT,
    f1_p1: FIELD_FLT_IJ,
    f2: FIELD_FLT_7,
    f2_p1: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pcnvflg: FIELD_BOOL_IJ,
    prsl: FIELD_FLT,
    rdzt: FIELD_FLT,
    scuflg: FIELD_BOOL_IJ,
    spd1: FIELD_FLT_IJ,
    stress: FIELD_FLT_IJ,
    tdt: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xmfd: FIELD_FLT,
    dspheat: bool,
    dt2: float,
    dspfac: float,
    cp: float,
):

    with computation(PARALLEL), interval(0, -1):
        if dspheat:
            tdt = tdt[0, 0, 0] + dspfac * (diss[0, 0, 0] / cp)

    with computation(PARALLEL), interval(0, 1):
        ad = 1.0 + dtdz1[0, 0, 0] * stress[0, 0] / spd1[0, 0]
        f1 = u1[0, 0, 0]
        f2[0, 0, 0][0] = v1[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2

        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


# Possible stencil name : moment_recover
@gtscript.stencil(backend=backend)
def moment_recover(
    del_: FIELD_FLT,
    du: FIELD_FLT,
    dusfc: FIELD_FLT_IJ,
    dv: FIELD_FLT,
    dvsfc: FIELD_FLT_IJ,
    f1: FIELD_FLT,
    f2: FIELD_FLT_7,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    conw: float,
    rdt: float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] < 1:
            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0][0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0] + conw * del_[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0] + conw * del_[0, 0, 0] * vtend


def mfpblt(
    im,
    ix,
    km,
    kmpbl,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1_gt,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    gdx,
    hpbl,
    kpbl,
    vpert,
    buo,
    xmf,
    tcko,
    qcko,
    ucko,
    vcko,
    xlamue,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
    qtx,
    wu2,
    qtu,
    xlamuem,
    thlu,
    kpblx,
    kpbly,
    rbup,
    rbdn,
    flg,
    hpblx,
    xlamavg,
    sumx,
    scaldfunc,
    ce0,
    cm,
    qmin,
    qlmin,
    alp,
    pgcon,
    a1,
    b1,
    f1,
    fv,
    eps,
    epsm1,
):
    totflag = True

    for i in range(im):
        totflag = totflag and ~cnvflg[i, 0]

    if totflag:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

    mfpblt_s0(
        alp=alp,
        buo=buo,
        cnvflg=cnvflg,
        g=g,
        hpbl=hpbl,
        kpbl=kpbl,
        q1=q1_gt,
        qtu=qtu,
        qtx=qtx,
        thlu=thlu,
        thlx=thlx,
        thvx=thvx,
        vpert=vpert,
        wu2=wu2,
        kpblx=kpblx,
        kpbly=kpbly,
        rbup=rbup,
        rbdn=rbdn,
        hpblx=hpblx,
        xlamavg=xlamavg,
        sumx=sumx,
        ntcw=ntcw - 1,
    )

    mfpblt_s1(
        buo=buo,
        ce0=ce0,
        cm=cm,
        cnvflg=cnvflg,
        elocp=elocp,
        el2orc=el2orc,
        eps=eps,
        epsm1=epsm1,
        flg=flg,
        fv=fv,
        g=g,
        hpbl=hpbl,
        kpbl=kpbl,
        kpblx=kpblx,
        kpbly=kpbly,
        mask=mask,
        pix=pix,
        plyr=plyr,
        qtu=qtu,
        qtx=qtx,
        rbdn=rbdn,
        rbup=rbup,
        thlu=thlu,
        thlx=thlx,
        thvx=thvx,
        wu2=wu2,
        xlamue=xlamue,
        xlamuem=xlamuem,
        zl=zl,
        zm=zm,
        qmin=qmin,
        domain=(im, 1, kmpbl),
    )

    mfpblt_s1a(
        cnvflg=cnvflg,
        hpblx=hpblx,
        kpblx=kpblx,
        mask=mask,
        rbdn=rbdn,
        rbup=rbup,
        zm=zm,
        domain=(im, 1, km),
    )

    mfpblt_s2(
        a1=a1,
        ce0=ce0,
        cm=cm,
        cnvflg=cnvflg,
        dt2=delt,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        gdx=gdx,
        hpbl=hpbl,
        hpblx=hpblx,
        kpbl=kpbl,
        kpblx=kpblx,
        kpbly=kpbly,
        mask=mask,
        pgcon=pgcon,
        pix=pix,
        plyr=plyr,
        qcko=qcko,
        qtu=qtu,
        qtx=qtx,
        scaldfunc=scaldfunc,
        sumx=sumx,
        tcko=tcko,
        thlu=thlu,
        thlx=thlx,
        u1=u1,
        ucko=ucko,
        v1=v1,
        vcko=vcko,
        xlamue=xlamue,
        xlamuem=xlamuem,
        xlamavg=xlamavg,
        xmf=xmf,
        wu2=wu2,
        zl=zl,
        zm=zm,
        qmin=qmin,
        domain=(im, 1, kmpbl),
    )

    mfpblt_s3(
        cnvflg=cnvflg,
        kpbl=kpbl,
        mask=mask,
        xlamue=xlamue,
        qcko=qcko,
        q1_gt=q1_gt,
        zl=zl,
        ntcw=ntcw,
        ntrac1=ntrac1,
        domain=(im, 1, kmpbl),
    )

    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue


@gtscript.stencil(
    **STENCIL_OPTS_2
)
def mfpblt_s3(
    cnvflg: FIELD_BOOL_IJ,
    kpbl: FIELD_INT_IJ,
    mask: FIELD_INT,
    xlamue: FIELD_FLT,
    qcko: FIELD_FLT_8,
    q1_gt: FIELD_FLT_8,
    zl: FIELD_FLT,
    ntcw: int,
    ntrac1: int,
):
    with computation(FORWARD), interval(1, None):
        if ntcw > 2:
            for n in range(1, ntcw):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0, 0, 0][n] = (
                        (1.0 - tem) * qcko[0, 0, -1][n]
                        + tem * (q1_gt[0, 0, 0][n] + q1_gt[0, 0, -1][n])
                    ) / factor

        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n2 in range(ntcw, ntrac1):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0, 0, 0][n2] = (
                        (1.0 - tem) * qcko[0, 0, -1][n2]
                        + tem * (q1_gt[0, 0, 0][n2] + q1_gt[0, 0, -1][n2])
                    ) / factor


@gtscript.stencil(backend=backend)
def mfpblt_s0(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    q1: FIELD_FLT_8,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    vpert: FIELD_FLT_IJ,
    wu2: FIELD_FLT,
    kpblx: FIELD_INT_IJ,
    kpbly: FIELD_INT_IJ,
    rbup: FIELD_FLT_IJ,
    rbdn: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    xlamavg: FIELD_FLT_IJ,
    sumx: FIELD_FLT_IJ,
    alp: float,
    g: float,
    ntcw: int,
):

    with computation(PARALLEL), interval(0, -1):
        if cnvflg[0, 0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw]

    with computation(FORWARD), interval(0, 1):
        kpblx = 0
        kpbly = 0
        rbup = 0.0
        rbdn = 0.0
        hpblx = 0.0
        xlamavg = 0.0
        sumx = 0.0
        if cnvflg[0, 0]:
            ptem = min(alp * vpert[0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = g * ptem / thvx[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfpblt_s1(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    hpbl: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    kpbly: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wu2: FIELD_FLT,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float,
    qmin: float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                xlamue = ce0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(1, None):
            if cnvflg[0, 0]:
                tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
                factor = 1.0 + tem
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

                tlu = thlu[0, 0, 0] / pix[0, 0, 0]
                es = 0.01 * fpvs(tlu)
                qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
                dq = qtu[0, 0, 0] - qs

                if dq > 0.0:
                    gamma = el2orc * qs / (tlu ** 2)
                    qlu = dq / (1.0 + gamma)
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * elocp * qlu) * (
                        1.0 + fv * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + fv * qtu[0, 0, 0])
                buo = g * (thvu / thvx[0, 0, 0] - 1.0)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                wu2 = (4.0 * buo[0, 0, 0] * zm[0, 0, 0]) / (
                    1.0 + (0.5 * 2.0 * xlamue[0, 0, 0] * zm[0, 0, 0])
                )
        with interval(1, None):
            if cnvflg[0, 0]:
                dz = zm[0, 0, 0] - zm[0, 0, -1]
                tem = 0.25 * 2.0 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                wu2 = (((1.0 - tem) * wu2[0, 0, -1]) + (4.0 * buo[0, 0, 0] * dz)) / (
                    1.0 + tem
                )

    with computation(FORWARD):
        with interval(0, 1):
            flg = True
            kpbly = kpbl[0, 0]
            if cnvflg[0, 0]:
                flg = False
                rbup = wu2[0, 0, 0]

        with interval(1, None):
            if flg[0, 0] == False:
                rbdn = rbup[0, 0]
                rbup = wu2[0, 0, 0]
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] <= 0.0


@gtscript.stencil(backend=backend)
def mfpblt_s1a(
    cnvflg: FIELD_BOOL_IJ,
    hpblx: FIELD_FLT_IJ,
    kpblx: FIELD_INT_IJ,
    mask: FIELD_INT,
    rbdn: FIELD_FLT_IJ,
    rbup: FIELD_FLT_IJ,
    zm: FIELD_FLT,
):

    with computation(FORWARD), interval(...):
        rbint = 0.0

        if mask[0, 0, 0] == kpblx[0, 0]:
            if cnvflg[0, 0]:
                if rbdn[0, 0] <= 0.0:
                    rbint = 0.0
                elif rbup[0, 0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0, 0] / (rbdn[0, 0] - rbup[0, 0])

                hpblx = zm[0, 0, -1] + rbint * (zm[0, 0, 0] - zm[0, 0, -1])


@gtscript.stencil(backend=backend)
def mfpblt_s2(
    cnvflg: FIELD_BOOL_IJ,
    gdx: FIELD_FLT_IJ,
    hpbl: FIELD_FLT_IJ,
    hpblx: FIELD_FLT_IJ,
    kpbl: FIELD_INT_IJ,
    kpblx: FIELD_INT_IJ,
    kpbly: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcko: FIELD_FLT_8,
    qtu: FIELD_FLT,
    qtx: FIELD_FLT,
    scaldfunc: FIELD_FLT_IJ,
    sumx: FIELD_FLT_IJ,
    tcko: FIELD_FLT,
    thlu: FIELD_FLT,
    thlx: FIELD_FLT,
    u1: FIELD_FLT,
    ucko: FIELD_FLT,
    v1: FIELD_FLT,
    vcko: FIELD_FLT,
    xmf: FIELD_FLT,
    xlamavg: FIELD_FLT_IJ,
    xlamue: FIELD_FLT,
    xlamuem: FIELD_FLT,
    wu2: FIELD_FLT,
    zl: FIELD_FLT,
    zm: FIELD_FLT,
    a1: float,
    dt2: float,
    ce0: float,
    cm: float,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float,
    qmin: float,
):

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                if kpbl[0, 0] > kpblx[0, 0]:
                    kpbl = kpblx[0, 0]
                    hpbl = hpblx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (kpbly[0, 0] > kpblx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        with interval(1, None):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = a1 * sqrt(wu2[0, 0, 0])
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem = 0.2 / xlamavg[0, 0]
                sigma = min(
                    max((3.14 * tem * tem) / (gdx[0, 0] * gdx[0, 0]), 0.001), 0.999,
                )

                if sigma > a1:
                    scaldfunc = max(min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0)
                else:
                    scaldfunc = 1.0

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            xmf = min(scaldfunc[0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

            tlu = thlu[0, 0, 0] / pix[0, 0, 0]
            es = 0.01 * fpvs(tlu)
            qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
            dq = qtu[0, 0, 0] - qs
            qlu = dq / (1.0 + (el2orc * qs / (tlu ** 2)))

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko[0, 0, 0][0] = qs
                    qcko[0, 0, 0][1] = qlu
                    tcko = tlu + elocp * qlu
                else:
                    qcko[0, 0, 0][0] = qtu[0, 0, 0]
                    qcko[0, 0, 0][1] = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1]
                    + (tem + pgcon) * u1[0, 0, 0]
                    + (tem - pgcon) * u1[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1]
                    + (tem + pgcon) * v1[0, 0, 0]
                    + (tem - pgcon) * v1[0, 0, -1]
                ) / factor


def mfscu(
    im,
    ix,
    km,
    kmscu,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    thlvx,
    gdx,
    thetae,
    radj,
    krad,
    mrad,
    radmin,
    buo,
    xmfd,
    tcdo,
    qcdo,
    ucdo,
    vcdo,
    xlamde,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
    qtx,
    wd2,
    hrad,
    krad1,
    thld,
    qtd,
    thlvd,
    ra1,
    ra2,
    flg,
    xlamdem,
    mradx,
    mrady,
    sumx,
    xlamavg,
    scaldfunc,
    zm_mrad,
    ce0,
    cm,
    pgcon,
    qmin,
    qlmin,
    b1,
    f1,
    a1,
    a2,
    a11,
    a22,
    cldtime,
    actei,
    hvap,
    cp,
    eps,
    epsm1,
    fv,
):

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    mfscu_s0(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        hrad=hrad,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        q1=q1,
        qtd=qtd,
        qtx=qtx,
        ra1=ra1,
        ra2=ra2,
        radmin=radmin,
        radj=radj,
        thetae=thetae,
        thld=thld,
        thlvd=thlvd,
        thlvx=thlvx,
        thlx=thlx,
        thvx=thvx,
        wd2=wd2,
        zm=zm,
        a1=a1,
        a11=a11,
        a2=a2,
        a22=a22,
        actei=actei,
        cldtime=cldtime,
        cp=cp,
        hvap=hvap,
        g=g,
        ntcw=ntcw,
        domain=(im, 1, km),
    )

    mfscu_s1(
        cnvflg=cnvflg,
        flg=flg,
        krad=krad,
        mask=mask,
        mrad=mrad,
        thlvd=thlvd,
        thlvx=thlvx,
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    for i in range(im):
        zm_mrad[i, 0] = zm[i, 0, mrad[i, 0] - 1]

    mfscu_s2(
        zl=zl,
        mask=mask,
        mrad=mrad,
        krad=krad,
        zm=zm,
        zm_mrad=zm_mrad,
        xlamde=xlamde,
        xlamdem=xlamdem,
        hrad=hrad,
        cnvflg=cnvflg,
        ce0=ce0,
        cm=cm,
        domain=(im, 1, kmscu),
    )

    mfscu_s3(
        buo=buo,
        cnvflg=cnvflg,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        fv=fv,
        g=g,
        krad=krad,
        mask=mask,
        pix=pix,
        plyr=plyr,
        thld=thld,
        thlx=thlx,
        thvx=thvx,
        qtd=qtd,
        qtx=qtx,
        xlamde=xlamde,
        zl=zl,
        qmin=qmin,
        domain=(im, 1, kmscu),
    )

    bb1 = 2.0
    bb2 = 4.0

    mfscu_s4(
        buo=buo,
        cnvflg=cnvflg,
        krad1=krad1,
        mask=mask,
        wd2=wd2,
        xlamde=xlamde,
        zm=zm,
        bb1=bb1,
        bb2=bb2,
        domain=(im, 1, km),
    )

    mfscu_s5(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        mradx=mradx,
        mrady=mrady,
        xlamde=xlamde,
        wd2=wd2,
        zm=zm,
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    for i in range(im):
        zm_mrad[i, 0] = zm[i, 0, mrad[i, 0] - 1]

    mfscu_s6(
        zl=zl,
        mask=mask,
        mrad=mrad,
        krad=krad,
        zm=zm,
        zm_mrad=zm_mrad,
        xlamde=xlamde,
        xlamdem=xlamdem,
        hrad=hrad,
        cnvflg=cnvflg,
        mrady=mrady,
        mradx=mradx,
        ce0=ce0,
        cm=cm,
        domain=(im, 1, kmscu),
    )

    mfscu_s7(
        cnvflg=cnvflg,
        dt2=delt,
        gdx=gdx,
        krad=krad,
        mask=mask,
        mrad=mrad,
        ra1=ra1,
        scaldfunc=scaldfunc,
        sumx=sumx,
        wd2=wd2,
        xlamde=xlamde,
        xlamavg=xlamavg,
        xmfd=xmfd,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    mfscu_s8(
        cnvflg=cnvflg, krad=krad, mask=mask, thld=thld, thlx=thlx, domain=(im, 1, km)
    )

    mfscu_s9(
        cnvflg=cnvflg,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        krad=krad,
        mask=mask,
        mrad=mrad,
        pgcon=pgcon,
        pix=pix,
        plyr=plyr,
        qcdo=qcdo,
        qtd=qtd,
        qtx=qtx,
        tcdo=tcdo,
        thld=thld,
        thlx=thlx,
        u1=u1,
        ucdo=ucdo,
        v1=v1,
        vcdo=vcdo,
        xlamde=xlamde,
        xlamdem=xlamdem,
        zl=zl,
        ntcw=ntcw,
        qmin=qmin,
        domain=(im, 1, kmscu),
    )

    mfscu_10(
        cnvflg=cnvflg,
        krad=krad,
        mrad=mrad,
        mask=mask,
        zl=zl,
        xlamde=xlamde,
        qcdo=qcdo,
        q1=q1,
        ntcw=ntcw,
        ntrac1=ntrac1,
        domain=(im, 1, kmscu),
    )

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde


@gtscript.stencil(backend=backend)
def mfscu_s2(
    zl: FIELD_FLT,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    zm: FIELD_FLT,
    zm_mrad: FIELD_FLT_IJ,
    xlamde: FIELD_FLT,
    xlamdem: FIELD_FLT,
    hrad: FIELD_FLT_IJ,
    cnvflg: FIELD_BOOL_IJ,
    ce0: float,
    cm: float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s6(
    zl: FIELD_FLT,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    krad: FIELD_INT_IJ,
    zm: FIELD_FLT,
    zm_mrad: FIELD_FLT_IJ,
    xlamde: FIELD_FLT,
    xlamdem: FIELD_FLT,
    hrad: FIELD_FLT_IJ,
    cnvflg: FIELD_BOOL_IJ,
    mrady: FIELD_INT_IJ,
    mradx: FIELD_INT_IJ,
    ce0: float,
    cm: float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mrady[0, 0] < mradx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0, 0, 0]


@gtscript.stencil(
    **STENCIL_OPTS_2
)
def mfscu_10(
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mrad: FIELD_INT_IJ,
    mask: FIELD_INT,
    zl: FIELD_FLT,
    xlamde: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    q1: FIELD_FLT_8,
    ntcw: int,
    ntrac1: int,
):
    with computation(BACKWARD), interval(...):
        if ntcw > 2:
            for n in range(1, ntcw - 1):
                if (
                    cnvflg[0, 0]
                    and mask[0, 0, 0] < krad[0, 0]
                    and mask[0, 0, 0] >= mrad[0, 0]
                ):
                    dz = zl[0, 0, 1] - zl[0, 0, 0]
                    tem = 0.5 * xlamde[0, 0, 0] * dz
                    factor = 1.0 + tem
                    qcdo[0, 0, 0][n] = (
                        (1.0 - tem) * qcdo[0, 0, 1][n]
                        + tem * (q1[0, 0, 0][n] + q1[0, 0, 1][n])
                    ) / factor

        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n1 in range(ntcw, ntrac1):
                if (
                    cnvflg[0, 0]
                    and mask[0, 0, 0] < krad[0, 0]
                    and mask[0, 0, 0] >= mrad[0, 0]
                ):
                    dz = zl[0, 0, 1] - zl[0, 0, 0]
                    tem = 0.5 * xlamde[0, 0, 0] * dz
                    factor = 1.0 + tem
                    qcdo[0, 0, 0][n1] = (
                        (1.0 - tem) * qcdo[0, 0, 1][n1]
                        + tem * (q1[0, 0, 0][n1] + q1[0, 0, 1][n1])
                    ) / factor


@gtscript.stencil(backend=backend)
def mfscu_s0(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    hrad: FIELD_FLT_IJ,
    krad: FIELD_INT_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    q1: FIELD_FLT_8,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    ra1: FIELD_FLT_IJ,
    ra2: FIELD_FLT_IJ,
    radmin: FIELD_FLT_IJ,
    radj: FIELD_FLT_IJ,
    thetae: FIELD_FLT,
    thld: FIELD_FLT,
    thlvd: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
    a1: float,
    a11: float,
    a2: float,
    a22: float,
    actei: float,
    cldtime: float,
    cp: float,
    hvap: float,
    g: float,
    ntcw: int,
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw - 1]

    with computation(FORWARD), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = mask[0, 0, 0] - 1
                tem1 = max(cldtime * radmin[0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0)
                thld = thlx[0, 0, 0] + tem1
                qtd = qtx[0, 0, 0]
                thlvd = thlvx[0, 0, 0] + tem1
                buo = -g * tem1 / thvx[0, 0, 0]

                ra1 = a1
                ra2 = a11

                tem = thetae[0, 0, 0] - thetae[0, 0, 1]
                tem1 = qtx[0, 0, 0] - qtx[0, 0, 1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = cp * tem / (hvap * tem1)
                    if cteit > actei:
                        ra1 = a2
                        ra2 = a22

                radj = -ra2[0, 0] * radmin[0, 0]

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrad = krad[0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s1(
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    thlvd: FIELD_FLT_IJ,
    thlvx: FIELD_FLT,
):

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0

    with computation(FORWARD), interval(0, 1):
        kk = krad[0, 0] - mrad[0, 0]
        if cnvflg[0, 0]:
            if kk < 1:
                cnvflg[0, 0] = 0


@gtscript.stencil(backend=backend)
def mfscu_s3(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
    thvx: FIELD_FLT,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    xlamde: FIELD_FLT,
    zl: FIELD_FLT,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    fv: float,
    g: float,
    qmin: float,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + fv * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * elocp * qld) * tem1
            else:
                tem1 = 1.0 + fv * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = g * (1.0 - thvd / thvx[0, 0, 0])


@gtscript.stencil(**STENCIL_OPTS)
def mfscu_s4(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    zm: FIELD_FLT,
    bb1: float,
    bb2: float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == krad1[0, 0]:
            if cnvflg[0, 0]:
                dz = zm[0, 0, 1] - zm[0, 0, 0]
                wd2 = (bb2 * buo[0, 0, 1] * dz) / (
                    1.0 + (0.5 * bb1 * xlamde[0, 0, 0] * dz)
                )


@gtscript.stencil(**STENCIL_OPTS)
def mfscu_s5(
    buo: FIELD_FLT,
    cnvflg: FIELD_BOOL_IJ,
    flg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    krad1: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    mradx: FIELD_INT_IJ,
    mrady: FIELD_INT_IJ,
    xlamde: FIELD_FLT,
    wd2: FIELD_FLT,
    zm: FIELD_FLT,
):
    with computation(BACKWARD), interval(...):
        dz = zm[0, 0, 1] - zm[0, 0, 0]
        tem = 0.25 * 2.0 * (xlamde[0, 0, 0] + xlamde[0, 0, 1]) * dz
        ptem1 = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad1[0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrady = mrad[0, 0]
        if flg[0, 0]:
            mradx = krad[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            if mrad[0, 0] < mradx[0, 0]:
                mrad = mradx[0, 0]
            if (krad[0, 0] - mrad[0, 0]) < 1:
                cnvflg = 0


@gtscript.stencil(backend=backend)
def mfscu_s7(
    cnvflg: FIELD_BOOL_IJ,
    gdx: FIELD_FLT_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    ra1: FIELD_FLT_IJ,
    scaldfunc: FIELD_FLT_IJ,
    sumx: FIELD_FLT_IJ,
    wd2: FIELD_FLT,
    xlamde: FIELD_FLT,
    xlamavg: FIELD_FLT_IJ,
    xmfd: FIELD_FLT,
    zl: FIELD_FLT,
    dt2: float,
):
    with computation(FORWARD), interval(0, 1):
        xlamavg = 0.0
        sumx = 0.0

    with computation(BACKWARD), interval(-1, None):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(BACKWARD), interval(...):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if wd2[0, 0, 0] > 0:
                xmfd = ra1[0, 0] * sqrt(wd2[0, 0, 0])
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem1 = (3.14 * (0.2 / xlamavg[0, 0]) * (0.2 / xlamavg[0, 0])) / (
                    gdx[0, 0] * gdx[0, 0]
                )
                sigma = min(max(tem1, 0.001), 0.999)

            if cnvflg[0, 0]:
                if sigma > ra1[0, 0]:
                    scaldfunc = max(min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0)
                else:
                    scaldfunc = 1.0

    with computation(BACKWARD), interval(...):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0] * xmfd[0, 0, 0], xmmx)


@gtscript.stencil(backend=backend)
def mfscu_s8(
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
):

    with computation(PARALLEL), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                thld = thlx[0, 0, 0]


@gtscript.stencil(backend=backend)
def mfscu_s9(
    cnvflg: FIELD_BOOL_IJ,
    krad: FIELD_INT_IJ,
    mask: FIELD_INT,
    mrad: FIELD_INT_IJ,
    pix: FIELD_FLT,
    plyr: FIELD_FLT,
    qcdo: FIELD_FLT_8,
    qtd: FIELD_FLT,
    qtx: FIELD_FLT,
    tcdo: FIELD_FLT,
    thld: FIELD_FLT,
    thlx: FIELD_FLT,
    u1: FIELD_FLT,
    ucdo: FIELD_FLT,
    v1: FIELD_FLT,
    vcdo: FIELD_FLT,
    xlamde: FIELD_FLT,
    xlamdem: FIELD_FLT,
    zl: FIELD_FLT,
    el2orc: float,
    elocp: float,
    eps: float,
    epsm1: float,
    pgcon: float,
    ntcw: int,
    qmin: float,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            tem = 0.5 * xlamde[0, 0, 0] * dz
            factor = 1.0 + tem
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(qmin, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)

        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                qcdo[0, 0, 0][0] = qs
                qcdo[0, 0, 0][ntcw - 1] = qld
                tcdo = tld + elocp * qld
            else:
                qcdo[0, 0, 0] = qtd[0, 0, 0]
                qcdo[0, 0, 0][ntcw - 1] = 0.0
                tcdo = tld

        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0] and mask[0, 0, 0] >= mrad[0, 0]:
            tem = 0.5 * xlamdem[0, 0, 0] * dz
            factor = 1.0 + tem
            ptem = tem - pgcon
            ptem1 = tem + pgcon
            ucdo = (
                (1.0 - tem) * ucdo[0, 0, 1] + ptem * u1[0, 0, 1] + ptem1 * u1[0, 0, 0]
            ) / factor
            vcdo = (
                (1.0 - tem) * vcdo[0, 0, 1] + ptem * v1[0, 0, 1] + ptem1 * v1[0, 0, 0]
            ) / factor


@gtscript.stencil(backend=backend)
def tridit(
    au: FIELD_FLT, cm: FIELD_FLT, cl: FIELD_FLT, f1: FIELD_FLT,
):
    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * au[0, 0, 0]
            f1 = fk * f1[0, 0, 0]
        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * au[0, 0, 0]
            f1 = fkk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])

    with computation(BACKWARD):
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            f1 = fk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])
        with interval(0, -1):
            f1 = f1[0, 0, 0] - au[0, 0, 0] * f1[0, 0, 1]


@gtscript.stencil(
    **STENCIL_OPTS_2
)
def tridin(
    cl: FIELD_FLT,
    cm: FIELD_FLT,
    cu: FIELD_FLT,
    r1: FIELD_FLT,
    r2: FIELD_FLT_7,
    au: FIELD_FLT,
    a1: FIELD_FLT,
    a2: FIELD_FLT_7,
    nt: int,
):
    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * cu[0, 0, 0]
            a1 = fk * r1[0, 0, 0]
            for n0 in range(nt):
                a2[0, 0, 0][n0] = fk * r2[0, 0, 0][n0]

        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * cu[0, 0, 0]
            a1 = fkk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])

            for n1 in range(nt):
                a2[0, 0, 0][n1] = fkk * (
                    r2[0, 0, 0][n1] - cl[0, 0, -1] * a2[0, 0, -1][n1]
                )

        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])

            for n2 in range(nt):
                a2[0, 0, 0][n2] = fk * (
                    r2[0, 0, 0][n2] - cl[0, 0, -1] * a2[0, 0, -1][n2]
                )

    with computation(BACKWARD):
        with interval(0, -1):
            a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
            for n3 in range(nt):
                a2[0, 0, 0][n3] = a2[0, 0, 0][n3] - au[0, 0, 0] * a2[0, 0, 1][n3]


@gtscript.stencil(backend=backend)
def tridi2(
    a1: FIELD_FLT,
    a2: FIELD_FLT_7,
    au: FIELD_FLT,
    cl: FIELD_FLT,
    cm: FIELD_FLT,
    cu: FIELD_FLT,
    r1: FIELD_FLT,
    r2: FIELD_FLT_7,
):

    with computation(PARALLEL), interval(0, 1):
        fk = 1 / cm[0, 0, 0]
        au = fk * cu[0, 0, 0]
        a1 = fk * r1[0, 0, 0]
        a2[0, 0, 0][0] = fk * r2[0, 0, 0][0]

    with computation(FORWARD):
        with interval(1, -1):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fk * cu[0, 0, 0]
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])

    with computation(BACKWARD), interval(0, -1):
        a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
        a2[0, 0, 0][0] = a2[0, 0, 0][0] - au[0, 0, 0] * a2[0, 0, 1][0]


@gtscript.stencil(backend=backend)
def comp_asym_mix_up(
    mask: FIELD_INT,
    mlenflg: FIELD_BOOL_IJ,
    bsum: FIELD_FLT_IJ,
    zlup: FIELD_FLT_IJ,
    thvx_k: FIELD_FLT_IJ,
    tke_k: FIELD_FLT_IJ,
    thvx: FIELD_FLT,
    tke: FIELD_FLT,
    gotvx: FIELD_FLT,
    zl: FIELD_FLT,
    zfmin: float,
    k: int,
):
    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == k:
            mlenflg = True
            zlup = 0.0
            bsum = 0.0
            thvx_k = thvx[0, 0, 0]
            tke_k = tke[0, 0, 0]
        if mlenflg[0, 0] == True:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            ptem = gotvx[0, 0, 0] * (thvx[0, 0, 1] - thvx_k[0, 0]) * dz
            bsum = bsum[0, 0] + ptem
            zlup = zlup[0, 0] + dz
            if bsum[0, 0] >= tke_k[0, 0]:
                if ptem >= 0.0:
                    tem2 = max(ptem, zfmin)
                else:
                    tem2 = min(ptem, -zfmin)
                ptem1 = (bsum[0, 0] - tke_k[0, 0]) / tem2
                zlup = zlup[0, 0] - ptem1 * dz
                zlup = max(zlup[0, 0], 0.0)
                mlenflg = False


@gtscript.stencil(backend=backend)
def comp_asym_mix_dn(
    mask: FIELD_INT,
    mlenflg: FIELD_BOOL_IJ,
    bsum: FIELD_FLT_IJ,
    zldn: FIELD_FLT_IJ,
    thvx_k: FIELD_FLT_IJ,
    tke_k: FIELD_FLT_IJ,
    thvx: FIELD_FLT,
    tke: FIELD_FLT,
    gotvx: FIELD_FLT,
    zl: FIELD_FLT,
    tsea: FIELD_FLT_IJ,
    q1_gt: FIELD_FLT_8,
    zfmin: float,
    fv: float,
    k: int,
    qmin: float,
):
    with computation(BACKWARD), interval(...):
        if mask[0, 0, 0] == k:
            mlenflg = True
            bsum = 0.0
            zldn = 0.0
            thvx_k = thvx[0, 0, 0]
            tke_k = tke[0, 0, 0]
        if mlenflg[0, 0] == True:
            if mask[0, 0, 0] == 0:
                dz = zl[0, 0, 0]
                tem1 = tsea[0, 0] * (1.0 + fv * max(q1_gt[0, 0, 0][0], qmin))
            else:
                dz = zl[0, 0, 0] - zl[0, 0, -1]
                tem1 = thvx[0, 0, -1]
            ptem = gotvx[0, 0, 0] * (thvx_k[0, 0] - tem1) * dz
            bsum = bsum[0, 0] + ptem
            zldn = zldn[0, 0] + dz
            if bsum[0, 0] >= tke_k[0, 0]:
                if ptem >= 0.0:
                    tem2 = max(ptem, zfmin)
                else:
                    tem2 = min(ptem, -zfmin)
                ptem1 = (bsum[0, 0] - tke_k[0, 0]) / tem2
                zldn = zldn[0, 0] - ptem1 * dz
                zldn = max(zldn[0, 0], 0.0)
                mlenflg[0, 0] = False


@gtscript.stencil(backend=backend)
def comp_asym_rlam_ele(
    zi: FIELD_FLT,
    rlam: FIELD_FLT,
    ele: FIELD_FLT,
    zlup: FIELD_FLT_IJ,
    zldn: FIELD_FLT_IJ,
    rlmn: float,
    rlmx: float,
    elmfac: float,
    elmx: float,
    elefac: float,
):
    with computation(FORWARD), interval(...):
        tem = 0.5 * (zi[0, 0, 1] - zi[0, 0, 0])
        tem1 = min(tem, rlmn)

        ptem2 = min(zlup[0, 0], zldn[0, 0])
        rlam = min(max(elmfac * ptem2, tem1), rlmx)

        ptem2 = sqrt(zlup[0, 0] * zldn[0, 0])
        ele = min(max(elefac * ptem2, tem1), elmx)

