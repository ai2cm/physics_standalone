import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    mod,
)

sys.path.insert(0, "..")
from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *
from phys_const import con_g, con_avgd, con_amd, con_amw, amdw, amdo3
from radphysparam import iswrgas, iswcliq, iovrsw, iswcice

from radsw.radsw_param import ftiny, oneminus, s0, nbandssw

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

isubcsw = 2  # Eventually this will be provided by rad_initialize


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "s0": s0,
        "con_g": con_g,
        "con_avgd": con_avgd,
        "con_amd": con_amd,
        "con_amw": con_amw,
        "amdw": amdw,
        "amdo3": amdo3,
        "iswrgas": iswrgas,
        "iswcliq": iswcliq,
        "iovrsw": iovrsw,
        "nbdsw": nbdsw,
        "ftiny": ftiny,
        "oneminus": oneminus,
    },
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[type_10],
    clouds: Field[type_9],
    aerosols: Field[(DTYPE_FLT, (nbdsw, 3))],
    sfcalb: Field[(DTYPE_FLT, (4,))],
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_2D,
    cosz: FIELD_2D,
    idxday: FIELD_2DBOOL,
    solcon: float,
    cosz1: FIELD_FLT,
    sntz1: FIELD_FLT,
    ssolar: FIELD_FLT,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    tem1: FIELD_FLT,
    tem2: FIELD_FLT,
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    tem0: FIELD_FLT,
    coldry: FIELD_FLT,
    temcol: FIELD_FLT,
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    zcf0: FIELD_2D,
    zcf1: FIELD_2D,
):
    from __externals__ import (
        s0,
        con_g,
        con_avgd,
        con_amd,
        con_amw,
        amdw,
        amdo3,
        iswrgas,
        iswcliq,
        iovrsw,
        nbdsw,
        ftiny,
        oneminus,
    )

    with computation(FORWARD), interval(0, 1):

        s0fac = solcon / s0

        if idxday:
            cosz1 = cosz
            sntz1 = 1.0 / cosz
            ssolar = s0fac * cosz

            # Prepare surface albedo: bm,df - dir,dif; 1,2 - nir,uvv.
            albbm[0, 0, 0][0] = sfcalb[0, 0, 0][0]
            albdf[0, 0, 0][0] = sfcalb[0, 0, 0][1]
            albbm[0, 0, 0][1] = sfcalb[0, 0, 0][2]
            albdf[0, 0, 0][1] = sfcalb[0, 0, 0][3]

            zcf0 = 1.0
            zcf1 = 1.0

    with computation(FORWARD), interval(1, None):
        if idxday:
            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            pavel = plyr
            tavel = tlyr

            h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
            o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delpin / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
            colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
            colamt[0, 0, 0][2] = max(0.0, coldry * o3vmr)  # o3
            colmol = coldry + colamt[0, 0, 0][0]

            #  --- ...  set up gas column amount, convert from volume mixing ratio
            #           to molec/cm2 based on coldry (scaled to 1.0e-20)

            if iswrgas > 0:
                colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                colamt[0, 0, 0][5] = max(temcol, coldry * gasvmr[0, 0, 0][3])  # o2
            else:
                colamt[0, 0, 0][3] = temcol  # n2o
                colamt[0, 0, 0][4] = temcol  # ch4
                colamt[0, 0, 0][5] = temcol

            #  --- ...  set aerosol optical properties
            for ib in range(nbdsw):
                tauae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 0]
                ssaae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 1]
                asyae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 2]

            if iswcliq > 0:  # use prognostic cloud method
                cfrac = clouds[0, 0, 0][0]  # cloud fraction
                cliqp = clouds[0, 0, 0][1]  # cloud liq path
                reliq = clouds[0, 0, 0][2]  # liq partical effctive radius
                cicep = clouds[0, 0, 0][3]  # cloud ice path
                reice = clouds[0, 0, 0][4]  # ice partical effctive radius
                cdat1 = clouds[0, 0, 0][5]  # cloud rain drop path
                cdat2 = clouds[0, 0, 0][6]  # rain partical effctive radius
                cdat3 = clouds[0, 0, 0][7]  # cloud snow path
                cdat4 = clouds[0, 0, 0][8]  # snow partical effctive radius
            else:  # use diagnostic cloud method
                cfrac = clouds[0, 0, 0][0]  # cloud fraction
                cdat1 = clouds[0, 0, 0][1]  # cloud optical depth
                cdat2 = clouds[0, 0, 0][2]  # cloud single scattering albedo
                cdat3 = clouds[0, 0, 0][3]  # cloud asymmetry factor

            # -# Compute fractions of clear sky view:
            #    - random overlapping
            #    - max/ran overlapping
            #    - maximum overlapping

            if iovrsw == 0:
                zcf0 = zcf0 * (1.0 - cfrac)
            elif iovrsw == 1:
                if cfrac > ftiny:  # cloudy layer
                    zcf1 = min(zcf1, 1.0 - cfrac)
                elif zcf1 < 1.0:  # clear layer
                    zcf0 = zcf0 * zcf1
                    zcf1 = 1.0

            elif iovrsw >= 2:
                zcf0 = min(zcf0, 1.0 - cfrac)  # used only as clear/cloudy indicator

    with computation(FORWARD), interval(0, 1):
        if idxday:
            if iovrsw == 1:
                zcf0 = zcf0 * zcf1

            if zcf0 <= ftiny:
                zcf0 = 0.0
            if zcf0 > oneminus:
                zcf0 = 1.0

            zcf1 = 1.0 - zcf0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "ftiny": ftiny,
        "iswcliq": iswcliq,
        "iswcice": iswcice,
        "isubcsw": isubcsw,
        "nbands": nbandssw,
        "nblow": nblow,
        "ngptsw": ngptsw,
    },
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
    zcf1: FIELD_2D,
    dz: FIELD_FLT,
    delgth: FIELD_FLT,
    idxday: FIELD_2DBOOL,
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
    a0r: float,
    a1r: float,
    a0s: float,
    a1s: float,
):
    from __externals__ import (
        ftiny,
        iswcliq,
        iswcice,
        nbands,
        nblow,
        isubcsw,
        ngptsw,
    )

    with computation(PARALLEL), interval(1, None):
        for nb in range(nbdsw):
            if idxday and zcf1 > 0:
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
    with computation(PARALLEL), interval(2, None):
        tem1 = 1.0 - cldf[0, 0, -1]

        for n in range(ngptsw):
            if cdfunc[0, 0, -1][n] > tem1:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
            else:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(1, None):
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
