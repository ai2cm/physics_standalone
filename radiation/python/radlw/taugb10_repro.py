import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    FORWARD,
    J,
    PARALLEL,
    Field,
    computation,
    interval,
    stencil,
    mod,
)

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data
from radlw.radlw_param import (
    nspa,
    nspb,
    ng10,
    ns10,
)

rebuild = True
validate = True
backend = "gtc:gt:cpu_ifirst"


@stencil(
    backend=backend,
    rebuild=rebuild,
    verbose=True,
    externals={
        "nspa": nspa[9],
        "nspb": nspb[9],
        "ng10": ng10,
        "ns10": ns10,
    },
)
def taugb10(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
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
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng10, 65))],
    absb: Field[(DTYPE_FLT, (ng10, 235))],
    selfref: Field[(DTYPE_FLT, (ng10, 10))],
    forref: Field[(DTYPE_FLT, (ng10, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng10,))],
    fracrefb: Field[(DTYPE_FLT, (ng10,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng10, ns10

    with computation(PARALLEL), interval(0, -1):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            for ig in range(ng10):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns10 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1

            for ig2 in range(ng10):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )

                taug[0, 0, 0][ns10 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig2] = fracrefb[0, 0, 0][ig2]
