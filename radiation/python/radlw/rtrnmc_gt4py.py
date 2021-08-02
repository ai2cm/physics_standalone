import gt4py
import os
import sys
import time
import numpy as np
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, BACKWARD, PARALLEL, Field, computation, interval, stencil, exp
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data
from radlw_param import ngb, bpade, ntbl, eps, wtdiff, fluxfac, heatfac

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"

invars = ['semiss', 'delp', 'cldfmc', 'taucld', 'tautot', 'pklay', 'pklev',
          'fracs', 'secdiff', 'nlay', 'nlp1', 'exp_tbl', 'tau_tbl', 'tfn_tbl',
          'totuflux', 'totdflux', 'htr', 'totuclfl', 'totdclfl', 'htrcl', 'htrb']

ddir = '../../fortran/radlw/dump'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Serialized_rank0')

savepoints = serializer.savepoint_list()

indict = dict()
for var in invars:
    tmp = serializer.read(var, serializer.savepoint['lwrad-rtrnmc-input-000000'])
    if var == 'semiss' or var == 'secdiff' or var[-3:] == 'tbl':
        indict[var] = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
    elif var == 'htrb':
        tmp2 = np.append(tmp, np.zeros((1, tmp.shape[1])), axis=0)
        indict[var] = np.tile(tmp2[None, None, :, :], (npts, 1, 1, 1))
    elif var == 'delp' or var[:3] == 'htr':
        tmp2 = np.append(tmp, np.array([0]), axis=0)
        indict[var] = np.tile(tmp2[None, None, :], (npts, 1, 1))
    elif var[:3] == 'tot' or var[:3] == 'htr':
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    elif var == 'pklay' or var == 'pklev':
        indict[var] = np.tile(tmp.T[None, None, :, :], (npts, 1, 1, 1))
    elif tmp.size > 1:
        tmp2 = np.append(tmp, np.zeros((tmp.shape[0], 1)), axis=1)
        indict[var] = np.tile(tmp2.T[None, None, :, :], (npts, 1, 1, 1))
    else:
        indict[var] = tmp

indict_gt4py = dict()
type_ntbl = (np.float64, (ntbl+1,))

for var in invars:
    if var  == 'semiss' or var == 'secdiff':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type_nbands)
    elif var[-3:] == 'tbl':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type_ntbl)
    elif var  == 'htrb':
        indict_gt4py[var] = create_storage_zeros(backend,
                                                 shape_nlp1,
                                                 type_nbands)
    elif var[:3] == 'tot' or var[:3] == 'htr':
        indict_gt4py[var] = create_storage_zeros(backend,
                                                 shape_nlp1,
                                                 type1)
    elif var == 'delp':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type1)
    elif var == 'taucld':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type_nbands)
    elif var == 'fracs' or var == 'tautot' or var == 'cldfmc':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type_ngptlw)
    elif var == 'pklay' or var == 'pklev':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp1,
                                                      type_nbands)
    else:
        indict_gt4py[var] = indict[var]

locvars = ['clrurad', 'clrdrad', 'toturad', 'totdrad', 'gassrcu', 'totsrcu',
           'trngas', 'efclrfr', 'rfdelp', 'fnet', 'fnetc', 'totsrcd',
           'gassrcd', 'tblind', 'odepth',
           'odtot', 'odcld', 'atrtot', 'atrgas', 'reflct', 'totfac',
           'gasfac', 'flxfac', 'plfrac', 'blay', 'bbdgas', 'bbdtot',
           'bbugas', 'bbutot', 'dplnku', 'dplnkd', 'radtotu', 'radclru',
           'radtotd', 'radclrd', 'rad0', 'clfm', 'trng', 'gasu', 'itgas',
           'ittot', 'ib']

locdict_gt4py = dict()

for var in locvars:
    if var[-3:] == 'rad':
        locdict_gt4py[var] = create_storage_zeros(backend,
                                                  shape_nlp1,
                                                  type_nbands)
    elif var == 'fnet' or var == 'fnetc' or var == 'rfdelp':
        locdict_gt4py[var] = create_storage_zeros(backend,
                                                  shape_nlp1,
                                                  type1)
    elif var == 'itgas' or var == 'ittot':
        locdict_gt4py[var] = create_storage_zeros(backend,
                                                  shape_nlp1,
                                                  (np.int32, (ngptlw,)))
    elif var == 'ib':
        locdict_gt4py[var] = create_storage_zeros(backend,
                                                  shape_2D,
                                                  np.int32)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend,
                                                  shape_nlp1,
                                                  type_ngptlw)

ngb = np.array(ngb)
NGB = np.tile(ngb[None, None, :], (npts, 1, 1))
NGB = create_storage_from_array(NGB,
                                backend,
                                shape_2D,
                                (np.int32, (140,)),
                                default_origin=(0, 0))

rec_6 = 0.166667
tblint = ntbl
flxfac = wtdiff * fluxfac
lhlw0 = True

@stencil(backend, rebuild=rebuild, externals={'rec_6': rec_6,
                                              'bpade': bpade,
                                              'tblint': tblint,
                                              'eps': eps,
                                              'flxfac': flxfac,
                                              'heatfac': heatfac,
                                              'lhlw0': lhlw0})
def rtrnmc(semiss: Field[type_nbands],
           secdif: Field[type_nbands],
           delp: FIELD_FLT,
           taucld: Field[type_nbands],
           fracs: Field[type_ngptlw],
           tautot: Field[type_ngptlw],
           cldfmc: Field[type_ngptlw],
           pklay: Field[type_nbands],
           pklev: Field[type_nbands],
           exp_tbl: Field[type_ntbl],
           tau_tbl: Field[type_ntbl],
           tfn_tbl: Field[type_ntbl],
           NGB: Field[gtscript.IJ, (np.int32, (140,))],
           htr: FIELD_FLT,
           htrcl: FIELD_FLT,
           htrb: Field[type_nbands],
           totuflux: FIELD_FLT,
           totdflux: FIELD_FLT,
           totuclfl: FIELD_FLT,
           totdclfl: FIELD_FLT,
           clrurad: Field[type_nbands],
           clrdrad: Field[type_nbands],
           toturad: Field[type_nbands],
           totdrad: Field[type_nbands],
           gassrcu: Field[type_ngptlw],
           totsrcu: Field[type_ngptlw],
           trngas: Field[type_ngptlw],
           efclrfr: Field[type_ngptlw],
           rfdelp: FIELD_FLT,
           fnet: FIELD_FLT,
           fnetc: FIELD_FLT,
           totsrcd: Field[type_ngptlw],
           gassrcd: Field[type_ngptlw],
           tblind: Field[type_ngptlw],
           odepth: Field[type_ngptlw],
           odtot: Field[type_ngptlw],
           odcld: Field[type_ngptlw],
           atrtot: Field[type_ngptlw],
           atrgas: Field[type_ngptlw],
           reflct: Field[type_ngptlw],
           totfac: Field[type_ngptlw],
           gasfac: Field[type_ngptlw],
           flxfac: Field[type_ngptlw],
           plfrac: Field[type_ngptlw],
           blay: Field[type_ngptlw],
           bbdgas: Field[type_ngptlw],
           bbdtot: Field[type_ngptlw],
           bbugas: Field[type_ngptlw],
           bbutot: Field[type_ngptlw],
           dplnku: Field[type_ngptlw],
           dplnkd: Field[type_ngptlw],
           radtotu: Field[type_ngptlw],
           radclru: Field[type_ngptlw],
           radtotd: Field[type_ngptlw],
           radclrd: Field[type_ngptlw],
           rad0: Field[type_ngptlw],
           clfm: Field[type_ngptlw],
           trng: Field[type_ngptlw],
           gasu: Field[type_ngptlw],
           itgas: Field[(np.int32, (ngptlw,))],
           ittot: Field[(np.int32, (ngptlw,))],
           ib: FIELD_2DINT):
    from __externals__ import rec_6, bpade, tblint, eps, flxfac, heatfac, lhlw0
    
    # Downward radiative transfer loop.
    # - Clear sky, gases contribution
    # - Total sky, gases+clouds contribution
    # - Cloudy layer
    # - Total sky radiance
    # - Clear sky radiance
    with computation(FORWARD), interval(-2, -1):
        for ig0 in range(ngptlw):
            ib = NGB[0, 0][ig0]-1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig0] = max(0.0, secdif[0, 0, 0][ib]*tautot[0, 0, 0][ig0])
            if odepth[0, 0, 0][ig0] <= 0.06:
                atrgas[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] - 0.5*odepth[0, 0, 0][ig0]*odepth[0, 0, 0][ig0]
                trng[0, 0, 0][ig0] = 1.0 - atrgas[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = rec_6 * odepth[0, 0, 0][ig0]
            else:
                tblind[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] / (bpade + odepth[0, 0, 0][ig0])
                # Currently itgas needs to be a storage, and can't be a local temporary.
                itgas[0, 0, 0][ig0] = tblint*tblind[0, 0, 0][ig0] + 0.5
                trng[0, 0, 0][ig0]  = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                atrgas[0, 0, 0][ig0] = 1.0 - trng[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                odepth[0, 0, 0][ig0] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]

            plfrac[0, 0, 0][ig0] = fracs[0, 0, 0][ig0]
            blay[0, 0, 0][ig0] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig0] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig0]
            dplnkd[0, 0, 0][ig0] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig0]
            bbdgas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0]*gasfac[0, 0, 0][ig0])
            bbugas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0]*gasfac[0, 0, 0][ig0])
            gassrcd[0, 0, 0][ig0] = bbdgas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            gassrcu[0, 0, 0][ig0] = bbugas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            trngas[0, 0, 0][ig0] = trng[0, 0, 0][ig0]

            # total sky, gases+clouds contribution 
            clfm[0, 0, 0][ig0] = cldfmc[0, 0, 0][ig0]
            if clfm[0, 0, 0][ig0] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig0] = secdif[0, 0, 0][ib] * taucld[0, 0, 0][ib]
                efclrfr[0, 0, 0][ig0] = 1.0 - (1.0 - exp(-odcld[0, 0, 0][ig0]))*clfm[0, 0, 0][ig0]
                odtot[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] + odcld[0, 0, 0][ig0]
                if odtot[0, 0, 0][ig0] < 0.06:
                    totfac[0, 0, 0][ig0] = rec_6 * odtot[0, 0, 0][ig0]
                    atrtot[0, 0, 0][ig0] = odtot[0, 0, 0][ig0]- 0.5*odtot[0, 0, 0][ig0]*odtot[0, 0, 0][ig0]
                else:
                    tblind[0, 0, 0][ig0] = odtot[0, 0, 0][ig0] / (bpade + odtot[0, 0, 0][ig0])
                    ittot[0, 0, 0][ig0] = tblint*tblind[0, 0, 0][ig0] + 0.5
                    totfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]
                    atrtot[0, 0, 0][ig0] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]
                
                bbdtot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0]*totfac[0, 0, 0][ig0])
                bbutot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0]*totfac[0, 0, 0][ig0])
                totsrcd[0, 0, 0][ig0] = bbdtot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]
                totsrcu[0, 0, 0][ig0] = bbutot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]

                # total sky radiance
                radtotd[0, 0, 0][ig0] = radtotd[0, 0, 0][ig0]*trng[0, 0, 0][ig0]*efclrfr[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0] + \
                        clfm[0, 0, 0][ig0]*(totsrcd[0, 0, 0][ig0] - gassrcd[0, 0, 0][ig0])
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = radclrd[0, 0, 0][ig0]*trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]
            else:
                # clear layer

                # total sky radiance 
                radtotd[0, 0, 0][ig0] = radtotd[0, 0, 0][ig0]*trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = radclrd[0, 0, 0][ig0]*trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]
            
            reflct[0, 0, 0][ig0] = 1.0 - semiss[0, 0, 0][ib]
    
    with computation(BACKWARD), interval(0, -2):
        for ig in range(ngptlw):
            ib = NGB[0, 0][ig]-1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig] = max(0.0, secdif[0, 0, 0][ib]*tautot[0, 0, 0][ig])
            if odepth[0, 0, 0][ig] <= 0.06:
                atrgas[0, 0, 0][ig] = odepth[0, 0, 0][ig] - 0.5*odepth[0, 0, 0][ig]*odepth[0, 0, 0][ig]
                trng[0, 0, 0][ig] = 1.0 - atrgas[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = rec_6 * odepth[0, 0, 0][ig]
            else:
                tblind[0, 0, 0][ig] = odepth[0, 0, 0][ig] / (bpade + odepth[0, 0, 0][ig])
                itgas[0, 0, 0][ig] = tblint*tblind[0, 0, 0][ig] + 0.5
                trng[0, 0, 0][ig]  = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                atrgas[0, 0, 0][ig] = 1.0 - trng[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                odepth[0, 0, 0][ig] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig]]

            plfrac[0, 0, 0][ig] = fracs[0, 0, 0][ig]
            blay[0, 0, 0][ig] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig]
            dplnkd[0, 0, 0][ig] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig]
            bbdgas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig]*gasfac[0, 0, 0][ig])
            bbugas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig]*gasfac[0, 0, 0][ig])
            gassrcd[0, 0, 0][ig] = bbdgas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            gassrcu[0, 0, 0][ig] = bbugas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            trngas[0, 0, 0][ig] = trng[0, 0, 0][ig]
 
            # total sky, gases+clouds contribution 
            clfm[0, 0, 0][ig] = cldfmc[0, 0, 0][ig]
            if clfm[0, 0, 0][ig] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig] = secdif[0, 0, 0][ib] * taucld[0, 0, 0][ib]
                efclrfr[0, 0, 0][ig] = 1.0 - (1.0 - exp(-odcld[0, 0, 0][ig]))*clfm[0, 0, 0][ig]
                odtot[0, 0, 0][ig] = odepth[0, 0, 0][ig] + odcld[0, 0, 0][ig]
                if odtot[0, 0, 0][ig] < 0.06:
                    totfac[0, 0, 0][ig] = rec_6 * odtot[0, 0, 0][ig]
                    atrtot[0, 0, 0][ig] = odtot[0, 0, 0][ig]- 0.5*odtot[0, 0, 0][ig]*odtot[0, 0, 0][ig]
                else:
                    tblind[0, 0, 0][ig] = odtot[0, 0, 0][ig] / (bpade + odtot[0, 0, 0][ig])
                    ittot[0, 0, 0][ig] = tblint*tblind[0, 0, 0][ig] + 0.5
                    totfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig]]
                    atrtot[0, 0, 0][ig] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig]]
                
                bbdtot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig]*totfac[0, 0, 0][ig])
                bbutot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig]*totfac[0, 0, 0][ig])
                totsrcd[0, 0, 0][ig] = bbdtot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]
                totsrcu[0, 0, 0][ig] = bbutot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]

                # total sky radiance
                radtotd[0, 0, 0][ig] = radtotd[0, 0, 1][ig]*trng[0, 0, 0][ig]*efclrfr[0, 0, 0][ig] + gassrcd[0, 0, 0][ig] + \
                        clfm[0, 0, 0][ig]*(totsrcd[0, 0, 0][ig] - gassrcd[0, 0, 0][ig])
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = radclrd[0, 0, 1][ig]*trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]
            else:
                # clear layer

                # total sky radiance 
                radtotd[0, 0, 0][ig] = radtotd[0, 0, 1][ig]*trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = radclrd[0, 0, 1][ig]*trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]
            
            reflct[0, 0, 0][ig] = 1.0 - semiss[0, 0, 0][ib]

    # Compute spectral emissivity & reflectance, include the
    # contribution of spectrally varying longwave emissivity and
    # reflection from the surface to the upward radiative transfer.
    # note: spectral and Lambertian reflection are identical for the
    #       diffusivity angle flux integration used here.

    with computation(FORWARD), interval(0, 1):
        for ig2 in range(ngptlw):
            ib = NGB[0, 0][ig2]-1
            rad0[0, 0, 0][ig2] = semiss[0, 0, 0][ib] * fracs[0, 0, 0][ig2] * pklay[0, 0, 0][ib]
   
            # Compute total sky radiance
            radtotu[0, 0, 0][ig2] = rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2]*radtotd[0, 0, 0][ig2]
            toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, 0][ig2]
   
            # Compute clear sky radiance
            radclru[0, 0, 0][ig2] = rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2]*radclrd[0, 0, 0][ig2]
            clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, 0][ig2]

    # Upward radiative transfer loop
    # - Compute total sky radiance
    # - Compute clear sky radiance

    # toturad holds summed radiance for total sky stream
    # clrurad holds summed radiance for clear sky stream

    with computation(FORWARD), interval(0, 1):
        for ig3 in range(ngptlw):
            ib = NGB[0, 0][ig3]-1
            clfm[0, 0, 0][ig3] = cldfmc[0, 0, 0][ig3]
            trng[0, 0, 0][ig3] = trngas[0, 0, 0][ig3]
            gasu[0, 0, 0][ig3] = gassrcu[0, 0, 0][ig3]

            if clfm[0, 0, 0][ig3] > eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = radtotu[0, 0, 0][ig3]*trng[0, 0, 0][ig3]*efclrfr[0, 0, 0][ig3] + gasu[0, 0, 0][ig3] + \
                    clfm[0, 0, 0][ig3]*(totsrcu[0, 0, 0][ig3] - gasu[0, 0, 0][ig3])

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = radclru[0, 0, 0][ig3]*trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = radtotu[0, 0, 0][ig3]*trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = radclru[0, 0, 0][ig3]*trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]

    with computation(FORWARD), interval(1, None):
        for ig4 in range(ngptlw):
            ib = NGB[0, 0][ig4]-1
            clfm[0, 0, 0][ig4] = cldfmc[0, 0, 0][ig4]
            trng[0, 0, 0][ig4] = trngas[0, 0, 0][ig4]
            gasu[0, 0, 0][ig4] = gassrcu[0, 0, 0][ig4]

            if clfm[0, 0, 0][ig4] > eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = radtotu[0, 0, -1][ig4]*trng[0, 0, 0][ig4]*efclrfr[0, 0, 0][ig4] + gasu[0, 0, 0][ig4] + \
                    clfm[0, 0, 0][ig4]*(totsrcu[0, 0, 0][ig4] - gasu[0, 0, 0][ig4])
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = radclru[0, 0, -1][ig4]*trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = radtotu[0, 0, -1][ig4]*trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = radclru[0, 0, -1][ig4]*trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]

    # Process longwave output from band for total and clear streams.
    # Calculate upward, downward, and net flux.
    with computation(PARALLEL), interval(...):
        for nb in range(nbands):
            totuflux = totuflux + toturad[0, 0, 0][nb]
            totdflux = totdflux + totdrad[0, 0, 0][nb]
            totuclfl = totuclfl + clrurad[0, 0, 0][nb]
            totdclfl = totdclfl + clrdrad[0, 0, 0][nb]

        totuflux = totuflux * flxfac
        totdflux = totdflux * flxfac
        totuclfl = totuclfl * flxfac
        totdclfl = totdclfl * flxfac

    # calculate net fluxes and heating rates (fnet, htr)
    # also compute optional clear sky heating rates (fnetc, htrcl)
    with computation(FORWARD), interval(0, 1):
        fnet = totuflux - totdflux
        if lhlw0:
            fnetc = totuclfl - totdclfl

    with computation(PARALLEL), interval(1, None):
        fnet = totuflux - totdflux
        if lhlw0:
            fnetc = totuclfl - totdclfl

    with computation(PARALLEL), interval(0, -1):
        rfdelp = heatfac / delp
        htr = (fnet - fnet[0, 0, 1]) * rfdelp
        if lhlw0:
            htrcl = (fnetc - fnetc[0, 0, 1])*rfdelp


start = time.time()
rtrnmc(indict_gt4py['semiss'],
       indict_gt4py['secdiff'],
       indict_gt4py['delp'],
       indict_gt4py['taucld'],
       indict_gt4py['fracs'],
       indict_gt4py['tautot'],
       indict_gt4py['cldfmc'],
       indict_gt4py['pklay'],
       indict_gt4py['pklev'],
       indict_gt4py['exp_tbl'],
       indict_gt4py['tau_tbl'],
       indict_gt4py['tfn_tbl'],
       NGB,
       indict_gt4py['htr'],
       indict_gt4py['htrcl'],
       indict_gt4py['htrb'],
       indict_gt4py['totuflux'],
       indict_gt4py['totdflux'],
       indict_gt4py['totuclfl'],
       indict_gt4py['totdclfl'],
       locdict_gt4py['clrurad'],
       locdict_gt4py['clrdrad'],
       locdict_gt4py['toturad'],
       locdict_gt4py['totdrad'],
       locdict_gt4py['gassrcu'],
       locdict_gt4py['totsrcu'],
       locdict_gt4py['trngas'],
       locdict_gt4py['efclrfr'],
       locdict_gt4py['rfdelp'],
       locdict_gt4py['fnet'],
       locdict_gt4py['fnetc'],
       locdict_gt4py['totsrcd'],
       locdict_gt4py['gassrcd'],
       locdict_gt4py['tblind'],
       locdict_gt4py['odepth'],
       locdict_gt4py['odtot'],
       locdict_gt4py['odcld'],
       locdict_gt4py['atrtot'],
       locdict_gt4py['atrgas'],
       locdict_gt4py['reflct'],
       locdict_gt4py['totfac'],
       locdict_gt4py['gasfac'],
       locdict_gt4py['flxfac'],
       locdict_gt4py['plfrac'],
       locdict_gt4py['blay'],
       locdict_gt4py['bbdgas'],
       locdict_gt4py['bbdtot'],
       locdict_gt4py['bbugas'],
       locdict_gt4py['bbutot'],
       locdict_gt4py['dplnku'],
       locdict_gt4py['dplnkd'],
       locdict_gt4py['radtotu'],
       locdict_gt4py['radclru'],
       locdict_gt4py['radtotd'],
       locdict_gt4py['radclrd'],
       locdict_gt4py['rad0'],
       locdict_gt4py['clfm'],
       locdict_gt4py['trng'],
       locdict_gt4py['gasu'],
       locdict_gt4py['itgas'],
       locdict_gt4py['ittot'],
       locdict_gt4py['ib'],
       domain=(npts, 1, nlp1),
       origin=default_origin,
       validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

outvars = ['totuflux', 'totdflux', 'htr', 'totuclfl', 'totdclfl', 'htrcl', 'htrb']


def view_gt4py_storage(gt4py_dict):

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        if backend == "gtcuda":
            data.synchronize()

        np_dict[var] = np.squeeze(data.view(np.ndarray))

    return np_dict

outvars = ['totuflux', 'totdflux', 'htr', 'totuclfl', 'totdclfl', 'htrcl', 'htrb']
outdict_gt4py = dict()

for var in outvars:
    outdict_gt4py[var] = indict_gt4py[var]
outdict_np = view_gt4py_storage(outdict_gt4py)

for var in outdict_np.keys():
    if var == 'htr' or var == 'htrcl':
        outdict_np[var] = outdict_np[var][-1, :-1].squeeze()
    elif var == 'htrb':
        outdict_np[var] = outdict_np[var][-1, :-1, :].squeeze()
    else:
        outdict_np[var] = outdict_np[var][-1, :]

outdict_val = dict()
for var in outvars:
    outdict_val[var] = serializer.read(var, serializer.savepoint['lwrad-rtrnmc-output-000000'])

compare_data(outdict_np, outdict_val)