import gt4py
import os
import sys
from gt4py import type_hints
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval, stencil, floor
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from radlw_param import a0, a1, a2, ngptlw, nbands, maxgas, maxxsec, abssnow0, absrain, ipat, cldmin
from radphysparam import ilwcice, isubclw
from util import (view_gt4py_storage, compare_data, create_storage_from_array,
                 create_storage_zeros, create_storage_ones)
from config import *

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Serialized_rank0')
savepoints = serializer.savepoint_list()

amdw = con_amd/con_amw
amdo3 = con_amd/con_amo3

semiss0_np = np.ones(nbands)

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"

invars = ['nlay', 'nlp1', 'ipseed', 'cldfrc', 'clwp', 'relw', 'ciwp', 'reiw',
          'cda1', 'cda2', 'cda3', 'cda4', 'dz', 'delgth']
nlay_vars = ['cliqp', 'reliq', 'cicep', 'reice', 'cdat1', 'cdat2',
             'cdat3', 'cdat4', 'dz']

indict = dict()
for var in invars:
    tmp = serializer.read(var, savepoints[9])
    if var in nlay_vars or var == 'cldfrc':
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    elif var == 'delgth':
        indict[var] = np.tile(tmp[:, None], (1, 1))
    else:
        indict[var] = tmp


indict_gt4py = dict()

for var in invars:
    if var in nlay_vars:
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type1)
    elif var == 'delgth':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_2D,
                                                      type1)
    elif var == 'cfrac':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlp2,
                                                      type1)
    elif var == 'cldfmc':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type_ngptlw)
    elif var == 'taucld':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type_nbands)
    else:
        indict_gt4py[var] = indict[var]


locvars = ['tauliq', 'tauice', 'cldf', 'dgeice', 'factor', 'fint', 'tauran',
           'tausnw', 'cldliq', 'refliq', 'cldice', 'refice']
bandvars = ['tauliq', 'tauice']

locdict_gt4py = dict()

for var in locvars:
    if var in bandvars:
        locdict_gt4py[var] = create_storage_zeros(backend, shape, type_nbands)
    elif var == 'cldf':
        locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, type1)
    else:
        locdict_gt4py[var] = create_storage_zeros(backend, shape_2D, type1)

ds = xr.open_dataset('../lookupdata/radlw_cldprlw_data.nc')

absliq1 = ds['absliq1']
absice0 = ds['absice0']
absice1 = ds['absice1']
absice2 = ds['absice2']
absice3 = ds['absice3']

lookup_dict = dict()
lookup_dict['absliq1'] = create_storage_from_array(absliq1, backend, (1, 1, 1), (np.float64, (58, nbands)))
lookup_dict['absice0'] = create_storage_from_array(absice0, backend, (1, 1, 1), (np.float64, (2,)))
lookup_dict['absice1'] = create_storage_from_array(absice1, backend, (1, 1, 1), (np.float64, (2, 5)))
lookup_dict['absice2'] = create_storage_from_array(absice2, backend, (1, 1, 1), (np.float64, (43, nbands)))
lookup_dict['absice3'] = create_storage_from_array(absice3, backend, (1, 1, 1), (np.float64, (46, nbands)))
lookup_dict['ipat'] = create_storage_from_array(ipat, backend, (1, 1, 1), type_nbands)

@gtscript.function
def mcica_subcol(cldf, nlay, ipseed, dz, de_lgth):
    lcloudy = True
    return lcloudy

@gtscript.function
def cldprop(cfrac, cliqp, reliq, cicep, reice, cdat1, cdat3, cdat4,
            ipseed, dz, de_lgth,
            absliq1, absice1, absice3, ipat,
            tauliq, tauice, cldf, dgeice, factor, fint, tauran,
            tausnw, cldliq, refliq, cldice, refice, indx,
            cldfmcin, taucldin):

    cldfmc = cldfmcin
    taucld = taucldin

    if ilwcliq > 0:
        if cfrac > cldmin:
            tauran = absrain * cdat1

            if cdat3 > 0.0 and cdat4 > 10.0:
                tausnw = abssnow0*1.05756*cdat3/cdat4
            else:
                tausnw = 0.0

            cldliq = cliqp
            cldice = cicep
            refliq = reliq
            refice = reice

            if cldliq <= 0.0:
                for i in range(nbands):
                    tauliq[0, 0, 0][i] = 0.0
            #else:
            #    if ilwcliq == 1:
            #        factor = refliq - 1.5
                    # if factor > 1:
                    #     if factor < 57:
                    #         indx = floor(factor)
                    #     else:
                    #         indx = 57
                    # else:
                    #     indx = 1
                    #index = max(1, min(57, floor(min(factor))))
                    #fint = factor - indx
#
    #                for i in range(nbands):
    #                    tauliq[0, 0, 0][i] = max(0.0, cldliq*(absliq1[0, 0, 0][index, i]
    #                        + fint*(absliq1[0, 0, 0][index+1, i]-absliq1[0, 0, 0][index, i])))
#
    #        if cldice <= 0.0:
    #            for i in range(nbands):
    #                tauice[0, 0, 0][i] = 0.0
    #        else:
    #            if ilwcice == 1:
    #                refice = min(130., max(13., refice))
#
    #                for i in range(nbands):
    #                    ia = ipat[i]
    #                    tauice[0, 0, 0][i] = max(0., cldice*(absice1[0, 0, 0][0, i] + absice1[0, 0, 0][1, i]/refice))
    #            elif ilwcice == 2:
    #                dgeice = max(5.0, 1.0315*refice)
    #                factor = (dgeice - 2.0)/3.0
    #                index = max(1, min(45, floor(factor)))
    #                fint = factor - index
#
    #                for i in range(nbands):
    #                    tauice[0, 0, 0][i] = max(0., cldice*(absice3[0, 0, 0][index, i] + fint*(absice3[0, 0, 0][index+1, i] - absice3[0, 0, 0][index, i])))
#
    #        for i in range(nbands):
    #            taucld[0, 0, 0][i] = tauice[0, 0, 0][i] + tauliq[0, 0, 0][i] + \
    #                tauran[0, 0, 0][i] + tausnw[0, 0, 0][i]
    #else:
    #    if cfrac > cldmin:
    #        for i in range(nbands):
    #            taucld[0, 0, 0][i] = cdat1
    #
    #if isubclw > 0:
    #    if cfrac < cldmin:
    #        cldf = 0.0
    #    else:
    #        cldf = cfrac
    #
    #lcloudy = mcica_subcol(cldf, nlay, ipseed, dz, de_lgth)
#
    #for g in range(ngptlw):
    #    if lcloudy[0, 0, 0][g]:
    #        cldfmc[0, 0, 0][g] = 1.0
    #    else:
    #        cldfmc[0, 0, 0][g] = 0.0
#
    return cldfmc, taucld

@gtscript.stencil(backend=backend, rebuild=True)
def run_cldprop(cfrac: FIELD_FLT, 
                cliqp: FIELD_FLT, 
                reliq: FIELD_FLT, 
                cicep: FIELD_FLT, 
                reice: FIELD_FLT, 
                cdat1: FIELD_FLT,  
                cdat3: FIELD_FLT, 
                cdat4: FIELD_FLT,
                ipseed: FIELD_INT, 
                dz: FIELD_FLT, 
                de_lgth: FIELD_2D,
                absliq1: Field[(np.float64, (58, nbands))],  
                absice1: Field[(np.float64, (2, 5))], 
                absice3: Field[(np.float64, (46, nbands))],
                ipat: Field[type_nbands],
                tauliq: Field[type_nbands], 
                tauice: Field[type_nbands], 
                cldf: FIELD_FLT, 
                dgeice: FIELD_2D, 
                factor: FIELD_2D, 
                fint: FIELD_2D,
                tauran: FIELD_2D,
                tausnw: FIELD_2D, 
                cldliq: FIELD_2D, 
                refliq: FIELD_2D,
                cldice: FIELD_2D, 
                refice: FIELD_2D,
                indx: FIELD_2DINT,
                cldfmc: Field[type_ngptlw], 
                taucld: Field[type_nbands],
                cldfmcout: Field[type_ngptlw], 
                taucldout: Field[type_nbands]):
    with computation(PARALLEL), interval(...):
        cldfmcout, taucldout = cldprop(
            cfrac, cliqp, reliq, cicep, reice, cdat1, cdat3, cdat4,
            ipseed, dz, de_lgth,
            absliq1, absice1, absice3, ipat,
            tauliq, tauice, cldf, dgeice, factor, fint, tauran,
            tausnw, cldliq, refliq, cldice, refice, indx,
            cldfmc, taucld)
