import gt4py
import os
import sys
import time
import numpy as np
from numpy.lib.npyio import load
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, BACKWARD, PARALLEL, Field, computation, interval, stencil, exp
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data
from radlw_param import nrates, nspa, nspb, ng01, ng02, ns02

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

invars = ['laytrop', 'pavel', 'coldry', 'colamt', 'colbrd', 'wx', 'tauaer',
          'rfrate', 'fac00', 'fac01', 'fac10', 'fac11', 'jp', 'jt', 'jt1',
          'selffac', 'selffrac', 'indself', 'forfac', 'forfrac', 'indfor',
          'minorfrac', 'scaleminor', 'scaleminorn2', 'indminor', 'nlay',
          'fracs', 'tautot']

integervars = ['jp', 'jt', 'jt1', 'indself', 'indfor', 'indminor']
fltvars = ['pavel', 'coldry', 'colamt', 'colbrd', 'fac00', 'fac01', 'fac10', 'fac11',
           'selffac', 'selffrac', 'forfac', 'forfrac', 'minorfrac', 'scaleminor',
           'scaleminorn2']

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Serialized_rank0')

savepoints = serializer.savepoint_list()
#print(savepoints)

indict = dict()

for var in invars:
    tmp = serializer.read(var, savepoints[10])
    if var == 'colamt' or var == 'wx':
        indict[var] = np.tile(tmp[None, None, :, :], (npts, 1, 1, 1))
    elif var == 'tauaer' or var == 'fracs' or var == 'tautot':
        indict[var] = np.tile(tmp.T[None, None, :, :], (npts, 1, 1, 1))
    elif var == 'rfrate':
        indict[var] = np.tile(tmp[None, None, :, :, :], (npts, 1, 1, 1, 1))
    elif var in integervars or var in fltvars:
        indict[var] = np.tile(tmp[None, None, :], (npts, 1, 1))
    else:
        indict[var] = tmp[0]
        
indict_gt4py = dict()

for var in invars:
    if var == 'colamt':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type_maxgas)
    elif var == 'wx':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type_maxxsec)
    elif var == 'tauaer':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      type_nbands)
    elif var == 'rfrate':
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      (np.float64, (nrates, 2)))
    elif var in integervars:
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      DTYPE_INT)
    elif var in fltvars:
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
                                                      DTYPE_FLT)
    elif var == 'fracs' or var == 'tautot':
        indict_gt4py[var] = create_storage_zeros(backend,
                                                 shape_nlay,
                                                 type_ngptlw)
    else:
        indict_gt4py[var] = indict[var]

taug = create_storage_zeros(backend,
                            shape_nlay,
                            type_ngptlw)

locdict_gt4py = dict()

locvars_int = ['ind0', 'ind0p', 'ind1', 'ind1p', 'inds', 'indsp', 'indf', 'indfp',
               'indm', 'indmp']
locvars_flt = ['pp', 'corradj', 'scalen2', 'tauself', 'taufor', 'taun2']

for var in locvars_int:
    locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_INT)

for var in locvars_flt:
    locdict_gt4py[var] = create_storage_zeros(backend, shape_nlay, DTYPE_FLT)

lookupdict = dict()
lookupdict_gt4py = dict()


def loadlookupdata(name):
    ds = xr.open_dataset('../lookupdata/radlw_'+name+'_data.nc')

    for var in ds.data_vars.keys():
        print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(ds[var].data[None, None, None, :], (npts, 1, nlay, 1))
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(ds[var].data[None, None, None, :, :], (npts, 1, nlay, 1, 1))

        lookupdict_gt4py[var] = create_storage_from_array(lookupdict[var],
                                                          backend,
                                                          shape_nlay,
                                                          (np.float64, ds.data_vars[var].shape))
    return lookupdict_gt4py


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[0],
                                                      'nspb': nspb[0],
                                                      'laytrop': indict['laytrop'],
                                                      'ng01': ng01,
                                                      'nlay': nlay})
def taugb01(pavel: FIELD_FLT,
            coldry: FIELD_FLT,
            colamt: Field[type_maxgas],
            colbrd: FIELD_FLT,
            wx: Field[type_maxxsec],
            tauaer: Field[type_nbands],
            rfrate: Field[(np.float64, (nrates, 2))],
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
            scaleminor: FIELD_FLT,
            scaleminorn2: FIELD_FLT,
            indminor: FIELD_INT,
            fracs: Field[type_ngptlw],
            tautot: Field[type_ngptlw],
            taug: Field[type_ngptlw],
            absa: Field[(DTYPE_FLT, (10, 65))],
            absb: Field[(DTYPE_FLT, (10, 235))],
            selfref: Field[(DTYPE_FLT, (10, 10))],
            forref: Field[(DTYPE_FLT, (10, 4))],
            fracrefa: Field[(DTYPE_FLT, (10,))],
            fracrefb: Field[(DTYPE_FLT, (10,))],
            ka_mn2: Field[(DTYPE_FLT, (10, 19))],
            kb_mn2: Field[(DTYPE_FLT, (10, 19))],
            ind0: FIELD_INT,
            ind0p: FIELD_INT,
            ind1: FIELD_INT,
            ind1p: FIELD_INT,
            inds: FIELD_INT,
            indsp: FIELD_INT,
            indf: FIELD_INT,
            indfp: FIELD_INT,
            indm: FIELD_INT,
            indmp: FIELD_INT,
            pp: FIELD_FLT,
            corradj: FIELD_FLT,
            scalen2: FIELD_FLT,
            tauself: FIELD_FLT,
            taufor: FIELD_FLT,
            taun2: FIELD_FLT):
    from __externals__ import nspa, nspb, laytrop, ng01, nlay
    with computation(PARALLEL), interval(0, laytrop):
        ind0 = ((jp-1)*5 + (jt-1)) * nspa
        ind1 = (jp*5 + (jt1-1)) * nspa
        inds = indself - 1
        indf = indfor - 1
        indm = indminor - 1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        pp = pavel
        scalen2 = colbrd * scaleminorn2
        if pp < 250.0:
            corradj = 1.0 - 0.15 * (250.0-pp) / 154.4
        else:
            corradj = 1.0

        for ig in range(ng01):
            tauself = selffac * (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] -  forref[0, 0, 0][ig, indf])) 
            taun2 = scalen2 * (ka_mn2[0, 0, 0][ig, indm] + minorfrac * \
                (ka_mn2[0, 0, 0][ig, indmp] - ka_mn2[0, 0, 0][ig, indm]))

            taug[0, 0, 0][ig] = corradj * (colamt[0, 0, 0][0] * \
                (fac00*absa[0, 0 ,0][ig, ind0] + fac10*absa[0, 0 ,0][ig, ind0p] + \
                 fac01*absa[0, 0, 0][ig, ind1] + fac11*absa[0, 0, 0][ig, ind1p]) + \
                tauself + taufor + taun2)

            fracs[0, 0, 0][ig] = fracrefa[0, 0, 0][ig]

    with computation(PARALLEL), interval(laytrop, nlay):
        ind0 = ((jp-13)*5 + (jt -1)) * nspb
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb
        indf = indfor-1
        indm = indminor-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1
        indmp = indm + 1

        scalen2 = colbrd * scaleminorn2
        corradj = 1.0 - 0.15 * (pavel / 95.6)

        for ig2 in range(ng01):
            taufor = forfac * (forref[0, 0, 0][ig2, indf] + forfrac * \
                (forref[0, 0 ,0][ig2, indfp] - forref[0, 0, 0][ig2, indf])) 
            taun2 = scalen2 * (kb_mn2[0, 0, 0][ig2, indm] + minorfrac * \
                (kb_mn2[0, 0, 0][ig2, indmp] - kb_mn2[0, 0, 0][ig2, indm]))

            taug[0, 0, 0][ig2] = corradj * (colamt[0, 0, 0][0] * \
                (fac00*absb[0, 0, 0][ig2, ind0] + fac10*absb[0, 0, 0][ig2, ind0p] + \
                    fac01*absb[0, 0, 0][ig2, ind1] + fac11*absb[0, 0, 0][ig2, ind1p]) + \
                taufor + taun2)

            fracs[0, 0, 0][ig2] = fracrefb[0, 0, 0][ig2]


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[1],
                                                      'nspb': nspb[1],
                                                      'laytrop': indict['laytrop'],
                                                      'ng02': ng02,
                                                      'ns02': ns02,
                                                      'nlay': nlay})
def taugb02(pavel: FIELD_FLT,
            coldry: FIELD_FLT,
            colamt: Field[type_maxgas],
            colbrd: FIELD_FLT,
            wx: Field[type_maxxsec],
            tauaer: Field[type_nbands],
            rfrate: Field[(np.float64, (nrates, 2))],
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
            scaleminor: FIELD_FLT,
            scaleminorn2: FIELD_FLT,
            indminor: FIELD_INT,
            fracs: Field[type_ngptlw],
            tautot: Field[type_ngptlw],
            taug: Field[type_ngptlw],
            absa: Field[(DTYPE_FLT, (10, 65))],
            absb: Field[(DTYPE_FLT, (10, 235))],
            selfref: Field[(DTYPE_FLT, (10, 10))],
            forref: Field[(DTYPE_FLT, (10, 4))],
            fracrefa: Field[(DTYPE_FLT, (10,))],
            fracrefb: Field[(DTYPE_FLT, (10,))],
            ind0: FIELD_INT,
            ind0p: FIELD_INT,
            ind1: FIELD_INT,
            ind1p: FIELD_INT,
            inds: FIELD_INT,
            indsp: FIELD_INT,
            indf: FIELD_INT,
            indfp: FIELD_INT,
            indm: FIELD_INT,
            indmp: FIELD_INT,
            pp: FIELD_FLT,
            corradj: FIELD_FLT,
            scalen2: FIELD_FLT,
            tauself: FIELD_FLT,
            taufor: FIELD_FLT,
            taun2: FIELD_FLT):
    from __externals__ import nspa, nspb, laytrop, ng02, nlay, ns02
    with computation(PARALLEL), interval(0, laytrop):
        ind0 = ((jp-1)*5 + (jt-1)) * nspa
        ind1 = (jp*5 + (jt1-1)) * nspa
        inds = indself - 1
        indf = indfor - 1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        corradj = 1.0 - 0.05 * (pavel - 100.0) / 900.0

        for ig in range(ng02):
            tauself = selffac * (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] -  forref[0, 0, 0][ig, indf])) 

            taug[0, 0, 0][ns02+ig] = corradj * (colamt[0, 0, 0][0] * \
                (fac00*absa[0, 0 ,0][ig, ind0] + fac10*absa[0, 0 ,0][ig, ind0p] + \
                 fac01*absa[0, 0, 0][ig, ind1] + fac11*absa[0, 0, 0][ig, ind1p]) + \
                tauself + taufor)

            fracs[0, 0, 0][ns02+ig] = fracrefa[0, 0, 0][ig]

    with computation(PARALLEL), interval(laytrop, nlay):
        ind0 = ((jp-13)*5 + (jt -1)) * nspb
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb
        indf = indfor-1

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1

        for ig2 in range(ng02):
            taufor = forfac * (forref[0, 0, 0][ig2, indf] + forfrac * \
                (forref[0, 0 ,0][ig2, indfp] - forref[0, 0, 0][ig2, indf])) 

            taug[0, 0, 0][ns02+ig2] = colamt[0, 0, 0][0] * \
                (fac00*absb[0, 0, 0][ig2, ind0] + fac10*absb[0, 0, 0][ig2, ind0p] + \
                    fac01*absb[0, 0, 0][ig2, ind1] + fac11*absb[0, 0, 0][ig2, ind1p]) + \
                taufor

            fracs[0, 0, 0][ns02+ig2] = fracrefb[0, 0, 0][ig2]

lookupdict_gt4py = loadlookupdata('kgb01')

start = time.time()
taugb01(indict_gt4py['pavel'],
        indict_gt4py['coldry'],
        indict_gt4py['colamt'],
        indict_gt4py['colbrd'],
        indict_gt4py['wx'],
        indict_gt4py['tauaer'],
        indict_gt4py['rfrate'],
        indict_gt4py['fac00'],
        indict_gt4py['fac01'],
        indict_gt4py['fac10'],
        indict_gt4py['fac11'],
        indict_gt4py['jp'],
        indict_gt4py['jt'],
        indict_gt4py['jt1'],
        indict_gt4py['selffac'],
        indict_gt4py['selffrac'],
        indict_gt4py['indself'],
        indict_gt4py['forfac'],
        indict_gt4py['forfrac'],
        indict_gt4py['indfor'],
        indict_gt4py['minorfrac'],
        indict_gt4py['scaleminor'],
        indict_gt4py['scaleminorn2'],
        indict_gt4py['indminor'],
        indict_gt4py['fracs'],
        indict_gt4py['tautot'],
        taug,
        lookupdict_gt4py['absa'],
        lookupdict_gt4py['absb'],
        lookupdict_gt4py['selfref'],
        lookupdict_gt4py['forref'],
        lookupdict_gt4py['fracrefa'],
        lookupdict_gt4py['fracrefb'],
        lookupdict_gt4py['ka_mn2'],
        lookupdict_gt4py['kb_mn2'],
        locdict_gt4py['ind0'],
        locdict_gt4py['ind0p'],
        locdict_gt4py['ind1'],
        locdict_gt4py['ind1p'],
        locdict_gt4py['inds'],
        locdict_gt4py['indsp'],
        locdict_gt4py['indf'],
        locdict_gt4py['indfp'],
        locdict_gt4py['indm'],
        locdict_gt4py['indmp'],
        locdict_gt4py['pp'],
        locdict_gt4py['corradj'],
        locdict_gt4py['scalen2'],
        locdict_gt4py['tauself'],
        locdict_gt4py['taufor'],
        locdict_gt4py['taun2'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

lookupdict_gt4py = loadlookupdata('kgb02')

start = time.time()
taugb02(indict_gt4py['pavel'],
        indict_gt4py['coldry'],
        indict_gt4py['colamt'],
        indict_gt4py['colbrd'],
        indict_gt4py['wx'],
        indict_gt4py['tauaer'],
        indict_gt4py['rfrate'],
        indict_gt4py['fac00'],
        indict_gt4py['fac01'],
        indict_gt4py['fac10'],
        indict_gt4py['fac11'],
        indict_gt4py['jp'],
        indict_gt4py['jt'],
        indict_gt4py['jt1'],
        indict_gt4py['selffac'],
        indict_gt4py['selffrac'],
        indict_gt4py['indself'],
        indict_gt4py['forfac'],
        indict_gt4py['forfrac'],
        indict_gt4py['indfor'],
        indict_gt4py['minorfrac'],
        indict_gt4py['scaleminor'],
        indict_gt4py['scaleminorn2'],
        indict_gt4py['indminor'],
        indict_gt4py['fracs'],
        indict_gt4py['tautot'],
        taug,
        lookupdict_gt4py['absa'],
        lookupdict_gt4py['absb'],
        lookupdict_gt4py['selfref'],
        lookupdict_gt4py['forref'],
        lookupdict_gt4py['fracrefa'],
        lookupdict_gt4py['fracrefb'],
        locdict_gt4py['ind0'],
        locdict_gt4py['ind0p'],
        locdict_gt4py['ind1'],
        locdict_gt4py['ind1p'],
        locdict_gt4py['inds'],
        locdict_gt4py['indsp'],
        locdict_gt4py['indf'],
        locdict_gt4py['indfp'],
        locdict_gt4py['indm'],
        locdict_gt4py['indmp'],
        locdict_gt4py['pp'],
        locdict_gt4py['corradj'],
        locdict_gt4py['scalen2'],
        locdict_gt4py['tauself'],
        locdict_gt4py['taufor'],
        locdict_gt4py['taun2'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

outdict_gt4py = {'fracs': indict_gt4py['fracs'][-1, :, :, :].squeeze().T,
                 'tautot': indict_gt4py['tautot'][-1, :, :, :].squeeze().T,
                 'taug': taug[-1, :, :, :].squeeze().T}

print(outdict_gt4py['fracs'].shape)
    
outvars = ['fracs', 'tautot', 'taug']

print(f"savepoint = {savepoints[11]}")

outdict_val = dict()
for var in outvars:
    outdict_val[var] = serializer.read(var, savepoints[11])

compare_data(outdict_val, outdict_gt4py)

# print(f"Fortran = {outdict_val['taug'][0, :]}")
# print(f"Python = {outdict_gt4py['taug'][0, :]}")
# 
# print(f"Difference = {(outdict_val['taug'] - outdict_gt4py['taug']).max()}")