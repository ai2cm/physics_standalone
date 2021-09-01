import sys
import numpy as np
import xarray as xr
from gt4py.gtscript import (
    stencil,
    computation,
    PARALLEL,
    interval,
    log,
    index,
)

sys.path.insert(0, "..")
from config import *
from radsw.radsw_param import ftiny, nbandssw, idxebc, nblow, ngptsw
from radphysparam import iswcliq, iswcice
from util import create_storage_from_array, create_storage_zeros, compare_data


sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "../../fortran/radsw/dump"
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank1")
savepoints = serializer.savepoint_list()

invars = [
    "cfrac",
    "cliqp",
    "reliq",
    "cicep",
    "reice",
    "cdat1",
    "cdat2",
    "cdat3",
    "cdat4",
    "zcf1",
    "dz",
    "delgth",
]

outvars = ["cldfmc", "taucw", "ssacw", "asycw", "cldfrc"]

locvars = [
    "tauliq",
    "tauice",
    "ssaliq",
    "ssaice",
    "ssaran",
    "ssasnw",
    "asyliq",
    "asyice",
    "asyran",
    "asysnw",
    "cldf",
    "dgeice",
    "factor",
    "fint",
    "tauran",
    "tausnw",
    "cldliq",
    "refliq",
    "cldice",
    "refice",
    "cldran",
    "cldsnw",
    "refsnw",
    "extcoliq",
    "ssacoliq",
    "asycoliq",
    "extcoice",
    "ssacoice",
    "asycoice",
    "dgesnw",
    "lcloudy",
    "index",
    "ia",
    "jb",
]

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint["swrad-cldprop-input-000000"])
    if var == "zcf1" or var == "delgth":
        indict[var] = np.tile(tmp[:, None, None], (1, 1, nlay))
    else:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))

indict_gt4py = dict()
for var in invars:
    indict_gt4py[var] = create_storage_from_array(
        indict[var], backend, shape_nlay, DTYPE_FLT
    )

# Read in 2-D array of random numbers used in mcica_subcol, this will change
# in the future once there is a solution for the RNG in python/gt4py
ds = xr.open_dataset("../lookupdata/rand2d_sw.nc")
rand2d = ds["rand2d"].data
cdfunc = np.zeros((npts, nlay, ngptsw))
for n in range(npts):
    cdfunc[n, :, :] = np.reshape(rand2d[n, :], (nlay, ngptsw), order="F")
indict["cdfunc"] = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))

indict_gt4py["cdfunc"] = create_storage_from_array(
    indict["cdfunc"], backend, shape_nlay, type_ngptsw
)

outdict_gt4py = dict()
for var in outvars:
    if var == "cldfrc":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)
    elif var == "cldfmc":
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_ngptsw)
    else:
        outdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_nbdsw)

locdict_gt4py = dict()
for var in locvars:
    if var in [
        "tauliq",
        "tauice",
        "ssaliq",
        "ssaice",
        "ssaran",
        "ssasnw",
        "asyliq",
        "asyice",
        "asyran",
        "asysnw",
    ]:
        locdict_gt4py[var] = create_storage_zeros(
            backend, shape_nlay, type_nbandssw_flt
        )
    elif var == "lcloudy":
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type_ngptsw_bool)
    elif var in ["index", "ia", "jb"]:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_INT)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)

idxebc = np.tile(np.array(idxebc)[None, None, None, :], (npts, 1, nlay, 1))
locdict_gt4py["idxebc"] = create_storage_from_array(
    idxebc, backend, shape_nlay, type_nbandssw_int
)


def loadlookupdata(name):
    """
    Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables
    """
    ds = xr.open_dataset("../lookupdata/radsw_" + name + "_data.nc")

    lookupdict = dict()
    lookupdict_gt4py = dict()

    for var in ds.data_vars.keys():
        # print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :], (npts, 1, nlay, 1)
            )
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :], (npts, 1, nlay, 1, 1)
            )
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :, :], (npts, 1, nlay, 1, 1, 1)
            )

        if len(ds.data_vars[var].shape) >= 1:
            lookupdict_gt4py[var] = create_storage_from_array(
                lookupdict[var], backend, shape_nlay, (DTYPE_FLT, ds[var].shape)
            )
        else:
            lookupdict_gt4py[var] = float(ds[var].data)

    return lookupdict_gt4py


lookupdict = loadlookupdata("cldprtb")

isubcsw = 2

special_externals = {
    "a0r": lookupdict["a0r"],
    "a1r": lookupdict["a1r"],
    "a0s": lookupdict["a0s"],
    "a1s": lookupdict["a1s"],
    "ftiny": ftiny,
    "iswcliq": iswcliq,
    "iswcice": iswcice,
    "isubcsw": isubcsw,
    "nbands": nbandssw,
    "nblow": nblow,
    "ngptsw": ngptsw,
}


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals=special_externals,
)
def cldprop(
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    zcf1: FIELD_FLT,
    dz: FIELD_FLT,
    delgth: FIELD_FLT,
    cldfmc: Field[type_ngptsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    cldfrc: FIELD_FLT,
    tauliq: Field[type_nbandssw_flt],
    tauice: Field[type_nbandssw_flt],
    ssaliq: Field[type_nbandssw_flt],
    ssaice: Field[type_nbandssw_flt],
    ssaran: Field[type_nbandssw_flt],
    ssasnw: Field[type_nbandssw_flt],
    asyliq: Field[type_nbandssw_flt],
    asyice: Field[type_nbandssw_flt],
    asyran: Field[type_nbandssw_flt],
    asysnw: Field[type_nbandssw_flt],
    cldf: FIELD_FLT,
    dgeice: FIELD_FLT,
    factor: FIELD_FLT,
    fint: FIELD_FLT,
    tauran: FIELD_FLT,
    tausnw: FIELD_FLT,
    cldliq: FIELD_FLT,
    refliq: FIELD_FLT,
    cldice: FIELD_FLT,
    refice: FIELD_FLT,
    cldran: FIELD_FLT,
    cldsnw: FIELD_FLT,
    refsnw: FIELD_FLT,
    extcoliq: FIELD_FLT,
    ssacoliq: FIELD_FLT,
    asycoliq: FIELD_FLT,
    extcoice: FIELD_FLT,
    ssacoice: FIELD_FLT,
    asycoice: FIELD_FLT,
    dgesnw: FIELD_FLT,
    lcloudy: Field[type_ngptsw_bool],
    index: FIELD_INT,
    ia: FIELD_INT,
    jb: FIELD_INT,
    idxebc: Field[type_nbandssw_int],
    cdfunc: Field[type_ngptsw],
    extliq1: Field[(DTYPE_FLT, (58, nbands))],
    extliq2: Field[(DTYPE_FLT, (58, nbands))],
    ssaliq1: Field[(DTYPE_FLT, (58, nbands))],
    ssaliq2: Field[(DTYPE_FLT, (58, nbands))],
    asyliq1: Field[(DTYPE_FLT, (58, nbands))],
    asyliq2: Field[(DTYPE_FLT, (58, nbands))],
    extice2: Field[(DTYPE_FLT, (43, nbands))],
    ssaice2: Field[(DTYPE_FLT, (43, nbands))],
    asyice2: Field[(DTYPE_FLT, (43, nbands))],
    extice3: Field[(DTYPE_FLT, (46, nbands))],
    ssaice3: Field[(DTYPE_FLT, (46, nbands))],
    asyice3: Field[(DTYPE_FLT, (46, nbands))],
    fdlice3: Field[(DTYPE_FLT, (46, nbands))],
    abari: Field[(DTYPE_FLT, (5,))],
    bbari: Field[(DTYPE_FLT, (5,))],
    cbari: Field[(DTYPE_FLT, (5,))],
    dbari: Field[(DTYPE_FLT, (5,))],
    ebari: Field[(DTYPE_FLT, (5,))],
    fbari: Field[(DTYPE_FLT, (5,))],
    b0s: Field[(DTYPE_FLT, (nbands,))],
    b1s: Field[(DTYPE_FLT, (nbands,))],
    c0s: Field[(DTYPE_FLT, (nbands,))],
    b0r: Field[(DTYPE_FLT, (nbands,))],
    c0r: Field[(DTYPE_FLT, (nbands,))],
):
    from __externals__ import (
        a0r,
        a1r,
        a0s,
        a1s,
        ftiny,
        iswcliq,
        iswcice,
        nbands,
        nblow,
        isubcsw,
        ngptsw,
    )

    with computation(PARALLEL), interval(...):
        for nb in range(nbdsw):
            ssacw[0, 0, 0][nb] = 1.0

        # Compute cloud radiative properties for a cloudy column.
        if iswcliq > 0:
            if cfrac > ftiny:
                #    - Compute optical properties for rain and snow.
                #      For rain: tauran/ssaran/asyran
                #      For snow: tausnw/ssasnw/asysnw
                #    - Calculation of absorption coefficients due to water clouds
                #      For water clouds: tauliq/ssaliq/asyliq
                #    - Calculation of absorption coefficients due to ice clouds
                #      For ice clouds: tauice/ssaice/asyice
                #    - For Prognostic cloud scheme: sum up the cloud optical property:
                #          taucw=tauliq+tauice+tauran+tausnw
                #          ssacw=ssaliq+ssaice+ssaran+ssasnw
                #          asycw=asyliq+asyice+asyran+asysnw

                cldran = cdat1
                cldsnw = cdat3
                refsnw = cdat4

                dgesnw = 1.0315 * refsnw  # for fu's snow formula

                tauran = cldran * a0r

                #  ---  if use fu's formula it needs to be normalized by snow/ice density
                #       !not use snow density = 0.1 g/cm**3 = 0.1 g/(mu * m**2)
                #       use ice density = 0.9167 g/cm**3 = 0.9167 g/(mu * m**2)
                #       1/0.9167 = 1.09087
                #       factor 1.5396=8/(3*sqrt(3)) converts reff to generalized ice particle size
                #       use newer factor value 1.0315

                if cldsnw > 0.0 and refsnw > 10.0:
                    tausnw = cldsnw * 1.09087 * (a0s + a1s / dgesnw)  # fu's formula
                else:
                    tausnw = 0.0

                for ib in range(nbands):
                    ssaran[0, 0, 0][ib] = tauran * (1.0 - b0r[0, 0, 0][ib])
                    ssasnw[0, 0, 0][ib] = tausnw * (
                        1.0 - (b0s[0, 0, 0][ib] + b1s[0, 0, 0][ib] * dgesnw)
                    )
                    asyran[0, 0, 0][ib] = ssaran[0, 0, 0][ib] * c0r[0, 0, 0][ib]
                    asysnw[0, 0, 0][ib] = ssasnw[0, 0, 0][ib] * c0s[0, 0, 0][ib]

                cldliq = cliqp
                cldice = cicep
                refliq = reliq
                refice = reice

                #  --- ...  calculation of absorption coefficients due to water clouds.

                if cldliq <= 0.0:
                    for ib2 in range(nbands):
                        tauliq[0, 0, 0][ib2] = 0.0
                        ssaliq[0, 0, 0][ib2] = 0.0
                        asyliq[0, 0, 0][ib2] = 0.0

                else:
                    factor = refliq - 1.5
                    index = max(1, min(57, factor)) - 1
                    fint = factor - (index + 1)

                    if iswcliq == 1:
                        for ib3 in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq1[0, 0, 0][index, ib3]
                                + fint
                                * (
                                    extliq1[0, 0, 0][index + 1, ib3]
                                    - extliq1[0, 0, 0][index, ib3]
                                ),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq1[0, 0, 0][index, ib3]
                                    + fint
                                    * (
                                        ssaliq1[0, 0, 0][index + 1, ib3]
                                        - ssaliq1[0, 0, 0][index, ib3]
                                    ),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq1[0, 0, 0][index, ib3]
                                    + fint
                                    * (
                                        asyliq1[0, 0, 0][index + 1, ib3]
                                        - asyliq1[0, 0, 0][index, ib3]
                                    ),
                                ),
                            )

                            tauliq[0, 0, 0][ib3] = cldliq * extcoliq
                            ssaliq[0, 0, 0][ib3] = tauliq[0, 0, 0][ib3] * ssacoliq
                            asyliq[0, 0, 0][ib3] = ssaliq[0, 0, 0][ib3] * asycoliq
                    elif iswcliq == 2:
                        for ib4 in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq2[0, 0, 0][index, ib4]
                                + fint
                                * (
                                    extliq2[0, 0, 0][index + 1, ib4]
                                    - extliq2[0, 0, 0][index, ib4]
                                ),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        ssaliq2[0, 0, 0][index + 1, ib4]
                                        - ssaliq2[0, 0, 0][index, ib4]
                                    ),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        asyliq2[0, 0, 0][index + 1, ib4]
                                        - asyliq2[0, 0, 0][index, ib4]
                                    ),
                                ),
                            )

                            tauliq[0, 0, 0][ib4] = cldliq * extcoliq
                            ssaliq[0, 0, 0][ib4] = tauliq[0, 0, 0][ib4] * ssacoliq
                            asyliq[0, 0, 0][ib4] = ssaliq[0, 0, 0][ib4] * asycoliq

                #  --- ...  calculation of absorption coefficients due to ice clouds.
                if cldice <= 0.0:
                    for ib5 in range(nbands):
                        tauice[0, 0, 0][ib5] = 0.0
                        ssaice[0, 0, 0][ib5] = 0.0
                        asyice[0, 0, 0][ib5] = 0.0
                else:
                    #  --- ...  ebert and curry approach for all particle sizes though somewhat
                    #           unjustified for large ice particles

                    if iswcice == 1:
                        refice = min(130.0, max(13.0, refice))

                        for ib6 in range(nbands):
                            ia = (
                                idxebc[0, 0, 0][ib6] - 1
                            )  # eb_&_c band index for ice cloud coeff

                            extcoice = max(
                                0.0, abari[0, 0, 0][ia] + bbari[0, 0, 0][ia] / refice
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0
                                    - cbari[0, 0, 0][ia]
                                    - dbari[0, 0, 0][ia] * refice,
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ebari[0, 0, 0][ia] + fbari[0, 0, 0][ia] * refice,
                                ),
                            )

                            tauice[0, 0, 0][ib6] = cldice * extcoice
                            ssaice[0, 0, 0][ib6] = tauice[0, 0, 0][ib6] * ssacoice
                            asyice[0, 0, 0][ib6] = ssaice[0, 0, 0][ib6] * asycoice

                    #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                    elif iswcice == 2:
                        refice = min(131.0, max(5.0, refice))

                        factor = (refice - 2.0) / 3.0
                        index = max(1, min(42, factor)) - 1
                        fint = factor - (index + 1)

                        for ib7 in range(nbands):
                            extcoice = max(
                                0.0,
                                extice2[0, 0, 0][index, ib7]
                                + fint
                                * (
                                    extice2[0, 0, 0][index + 1, ib7]
                                    - extice2[0, 0, 0][index, ib7]
                                ),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice2[0, 0, 0][index, ib7]
                                    + fint
                                    * (
                                        ssaice2[0, 0, 0][index + 1, ib7]
                                        - ssaice2[0, 0, 0][index, ib7]
                                    ),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice2[0, 0, 0][index, ib7]
                                    + fint
                                    * (
                                        asyice2[0, 0, 0][index + 1, ib7]
                                        - asyice2[0, 0, 0][index, ib7]
                                    ),
                                ),
                            )

                            tauice[0, 0, 0][ib7] = cldice * extcoice
                            ssaice[0, 0, 0][ib7] = tauice[0, 0, 0][ib7] * ssacoice
                            asyice[0, 0, 0][ib7] = ssaice[0, 0, 0][ib7] * asycoice

                    #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                    #           (generalized effective size from 5 to 140 microns)
                    elif iswcice == 3:
                        dgeice = max(5.0, min(140.0, 1.0315 * refice))

                        factor = (dgeice - 2.0) / 3.0
                        index = max(1, min(45, factor)) - 1
                        fint = factor - (index + 1)

                        for ib8 in range(nbands):
                            extcoice = max(
                                0.0,
                                extice3[0, 0, 0][index, ib8]
                                + fint
                                * (
                                    extice3[0, 0, 0][index + 1, ib8]
                                    - extice3[0, 0, 0][index, ib8]
                                ),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice3[0, 0, 0][index, ib8]
                                    + fint
                                    * (
                                        ssaice3[0, 0, 0][index + 1, ib8]
                                        - ssaice3[0, 0, 0][index, ib8]
                                    ),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice3[0, 0, 0][index, ib8]
                                    + fint
                                    * (
                                        asyice3[0, 0, 0][index + 1, ib8]
                                        - asyice3[0, 0, 0][index, ib8]
                                    ),
                                ),
                            )

                            tauice[0, 0, 0][ib8] = cldice * extcoice
                            ssaice[0, 0, 0][ib8] = tauice[0, 0, 0][ib8] * ssacoice
                            asyice[0, 0, 0][ib8] = ssaice[0, 0, 0][ib8] * asycoice

                for ib9 in range(nbdsw):
                    jb = nblow + ib9 - 16
                    taucw[0, 0, 0][ib9] = (
                        tauliq[0, 0, 0][jb] + tauice[0, 0, 0][jb] + tauran + tausnw
                    )
                    ssacw[0, 0, 0][ib9] = (
                        ssaliq[0, 0, 0][jb]
                        + ssaice[0, 0, 0][jb]
                        + ssaran[0, 0, 0][jb]
                        + ssasnw[0, 0, 0][jb]
                    )
                    asycw[0, 0, 0][ib9] = (
                        asyliq[0, 0, 0][jb]
                        + asyice[0, 0, 0][jb]
                        + asyran[0, 0, 0][jb]
                        + asysnw[0, 0, 0][jb]
                    )
        else:
            if cfrac > ftiny:
                for ib10 in range(nbdsw):
                    taucw[0, 0, 0][ib10] = cdat1
                    ssacw[0, 0, 0][ib10] = cdat1 * cdat2
                    asycw[0, 0, 0][ib10] = ssacw[0, 0, 0][ib10] * cdat3

        # if physparam::isubcsw > 0, call mcica_subcol() to distribute
        # cloud properties to each g-point.

        if isubcsw > 0:
            cldf = 0.0 if cfrac < ftiny else cfrac

    # This section builds mcica_subcol from the fortran into cldprop.
    # Here I've read in the generated random numbers until we figure out
    # what to do with them. This will definitely need to change in future.
    # Only the iovrlw = 1 option is ported from Fortran
    with computation(PARALLEL), interval(1, None):
        tem1 = 1.0 - cldf[0, 0, -1]

        for n in range(ngptsw):
            if cdfunc[0, 0, -1][n] > tem1:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
            else:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(...):
        tem1 = 1.0 - cldf[0, 0, 0]

        for n2 in range(ngptsw):
            if cdfunc[0, 0, 0][n2] >= tem1:
                lcloudy[0, 0, 0][n2] = 1
            else:
                lcloudy[0, 0, 0][n2] = 0

        for n3 in range(ngptsw):
            if lcloudy[0, 0, 0][n3] == 1:
                cldfmc[0, 0, 0][n3] = 1.0
            else:
                cldfmc[0, 0, 0][n3] = 0.0


cldprop(
    indict_gt4py["cfrac"],
    indict_gt4py["cliqp"],
    indict_gt4py["reliq"],
    indict_gt4py["cicep"],
    indict_gt4py["reice"],
    indict_gt4py["cdat1"],
    indict_gt4py["cdat2"],
    indict_gt4py["cdat3"],
    indict_gt4py["cdat4"],
    indict_gt4py["zcf1"],
    indict_gt4py["dz"],
    indict_gt4py["delgth"],
    outdict_gt4py["cldfmc"],
    outdict_gt4py["taucw"],
    outdict_gt4py["ssacw"],
    outdict_gt4py["asycw"],
    outdict_gt4py["cldfrc"],
    locdict_gt4py["tauliq"],
    locdict_gt4py["tauice"],
    locdict_gt4py["ssaliq"],
    locdict_gt4py["ssaice"],
    locdict_gt4py["ssaran"],
    locdict_gt4py["ssasnw"],
    locdict_gt4py["asyliq"],
    locdict_gt4py["asyice"],
    locdict_gt4py["asyran"],
    locdict_gt4py["asysnw"],
    locdict_gt4py["cldf"],
    locdict_gt4py["dgeice"],
    locdict_gt4py["factor"],
    locdict_gt4py["fint"],
    locdict_gt4py["tauran"],
    locdict_gt4py["tausnw"],
    locdict_gt4py["cldliq"],
    locdict_gt4py["refliq"],
    locdict_gt4py["cldice"],
    locdict_gt4py["refice"],
    locdict_gt4py["cldran"],
    locdict_gt4py["cldsnw"],
    locdict_gt4py["refsnw"],
    locdict_gt4py["extcoliq"],
    locdict_gt4py["ssacoliq"],
    locdict_gt4py["asycoliq"],
    locdict_gt4py["extcoice"],
    locdict_gt4py["ssacoice"],
    locdict_gt4py["asycoice"],
    locdict_gt4py["dgesnw"],
    locdict_gt4py["lcloudy"],
    locdict_gt4py["index"],
    locdict_gt4py["ia"],
    locdict_gt4py["jb"],
    locdict_gt4py["idxebc"],
    indict_gt4py["cdfunc"],
    lookupdict["extliq1"],
    lookupdict["extliq2"],
    lookupdict["ssaliq1"],
    lookupdict["ssaliq2"],
    lookupdict["asyliq1"],
    lookupdict["asyliq2"],
    lookupdict["extice2"],
    lookupdict["ssaice2"],
    lookupdict["asyice2"],
    lookupdict["extice3"],
    lookupdict["ssaice3"],
    lookupdict["asyice3"],
    lookupdict["fdlice3"],
    lookupdict["abari"],
    lookupdict["bbari"],
    lookupdict["cbari"],
    lookupdict["dbari"],
    lookupdict["ebari"],
    lookupdict["fbari"],
    lookupdict["b0s"],
    lookupdict["b1s"],
    lookupdict["c0s"],
    lookupdict["b0r"],
    lookupdict["c0r"],
    domain=shape_nlay,
    origin=default_origin,
    validate_args=validate,
)

valdict = dict()
outdict_np = dict()

for var in outvars:
    valdict[var] = serializer.read(
        var, serializer.savepoint["swrad-cldprop-output-000000"]
    )
    outdict_np[var] = outdict_gt4py[var].view(np.ndarray).squeeze()

# compare_data(outdict_np, valdict)

print(f"Python = {outdict_np['ssacw'][0, ...]}")
print(" ")
print(f"Fortran = {valdict['ssacw'][0, ...]}")
print(" ")
print(f"Difference = {outdict_np['ssacw'][0, ...]-valdict['ssacw'][0, ...]}")
