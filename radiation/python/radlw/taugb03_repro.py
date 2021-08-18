from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    mod,
)
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from radlw_param import (
    nrates,
    nspa,
    nspb,
    ng03,
    ns03,
    oneminus,
)
from config import *

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"
@stencil(
    backend=backend,
    rebuild=rebuild,
    verbose=True,
    externals={
        "nspa": nspa[2],
        "nspb": nspb[2],
        "ng03": ng03,
        "ns03": ns03,
        "oneminus": oneminus,
    },
)
def taugb03(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng03, 585))],
    absb: Field[(DTYPE_FLT, (ng03, 1175))],
    selfref: Field[(DTYPE_FLT, (ng03, 10))],
    forref: Field[(DTYPE_FLT, (ng03, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng03, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng03, 5))],
    ka_mn2o: Field[(DTYPE_FLT, (ng03, 9, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng03, 5, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jmn2o: FIELD_INT,
    jmn2op: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    ratn2o: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng03, ns03, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  minor gas mapping levels:
            #     lower - n2o, p = 706.272 mbar, t = 278.94 k
            #     upper - n2o, p = 95.58 mbar, t = 215.7 k

            refrat_planck_a = chi_mls[0, 0, 0][0, 8] / chi_mls[0, 0, 0][1, 8]
            refrat_planck_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]
            refrat_m_a = chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][1, 2]
            refrat_m_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            #  --- ...  in atmospheres where the amount of n2O is too great to be considered
            #           a minor species, adjust the column amount of n2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            if specparm < 0.125:
                p = fs - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p = -fs
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0 * 1
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk0 = 1.0 - fs
                fk1 = fs
                fk2 = 0.0
                id000 = ind0 * 1
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 * 1
                id210 = ind0 * 1

            fac000 = fk0 * fac00
            fac100 = fk1 * fac00
            fac200 = fk2 * fac00
            fac010 = fk0 * fac10
            fac110 = fk1 * fac10
            fac210 = fk2 * fac10

            if specparm1 < 0.125:
                p = fs1 - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p = -fs1
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk0 = 1.0 - fs1
                fk1 = fs1
                fk2 = 0.0
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk0 * fac01
            fac101 = fk1 * fac01
            fac201 = fk2 * fac01
            fac011 = fk0 * fac11
            fac111 = fk1 * fac11
            fac211 = fk2 * fac11

            for ig in range(ng03):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2om1 = ka_mn2o[0, 0, 0][ig, jmn2o, indm] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indm]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indm]
                )
                n2om2 = ka_mn2o[0, 0, 0][ig, jmn2o, indmp] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indmp]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
                )

                taug[0, 0, 0][ns03 + ig] = (
                    tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:

            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_b * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 4.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            indf = indfor - 1
            indm = indminor - 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            #  --- ...  in atmospheres where the amount of n2o is too great to be considered
            #           a minor species, adjust the column amount of N2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            fk0 = 1.0 - fs
            fk1 = fs
            fac000 = fk0 * fac00
            fac010 = fk0 * fac10
            fac100 = fk1 * fac00
            fac110 = fk1 * fac10

            fk0 = 1.0 - fs1
            fk1 = fs1
            fac001 = fk0 * fac01
            fac011 = fk0 * fac11
            fac101 = fk1 * fac01
            fac111 = fk1 * fac11

            for ig2 in range(ng03):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                n2om1 = kb_mn2o[0, 0, 0][ig2, jmn2o, indm] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indm]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indm]
                )
                n2om2 = kb_mn2o[0, 0, 0][ig2, jmn2o, indmp] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indmp]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absb[0, 0, 0][ig2, id000]
                    + fac010 * absb[0, 0, 0][ig2, id010]
                    + fac100 * absb[0, 0, 0][ig2, id100]
                    + fac110 * absb[0, 0, 0][ig2, id110]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, id001]
                    + fac011 * absb[0, 0, 0][ig2, id011]
                    + fac101 * absb[0, 0, 0][ig2, id101]
                    + fac111 * absb[0, 0, 0][ig2, id111]
                )

                taug[0, 0, 0][ns03 + ig2] = (
                    tau_major + tau_major1 + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )
