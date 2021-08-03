import gt4py
import os
import sys
import time
import numpy as np
import xarray as xr
import gt4py.gtscript as gtscript
from gt4py.gtscript import (FORWARD,
                            BACKWARD,
                            PARALLEL,
                            Field,
                            computation,
                            interval,
                            stencil,
                            exp,
                            mod)
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from config import *
from util import create_storage_from_array, create_storage_zeros, compare_data
from radlw.radlw_param import (nrates, nspa, nspb, ng01, ng02, ng03, ng04, ng05,
                               ng06, ng07, ng08, ng09, ng10, ng11, ng12, ng13, ng14,
                               ng15, ng16,
                               ns02, ns03, ns04, ns05, ns06, ns07, ns08, ns09, ns10,
                               ns11, ns12, ns13, ns14, ns15, ns16, oneminus)

np.set_printoptions(precision=15)

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

rebuild = False
validate = False
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

ddir = '../../fortran/radlw/dump'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Serialized_rank0')

savepoints = serializer.savepoint_list()
#print(savepoints)

indict = dict()

for var in invars:
    tmp = serializer.read(var, serializer.savepoint['lwrad-taumol-input-000000'])
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
               'indm', 'indmp', 'js', 'js1', 'jmn2o', 'jmn2op', 'jpl', 'jplp',
               'id000', 'id010', 'id100', 'id110', 'id200', 'id210', 'id001',
               'id011', 'id101', 'id111', 'id201', 'id211', 'jmo3', 'jmo3p',
               'jmco2', 'jmco2p']
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
        # print(f"{var} = {ds.data_vars[var].shape}")
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(ds[var].data[None, None, None, :], (npts, 1, nlay, 1))
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(ds[var].data[None, None, None, :, :], (npts, 1, nlay, 1, 1))
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(ds[var].data[None, None, None, :, :, :], (npts, 1, nlay, 1, 1, 1))

        lookupdict_gt4py[var] = create_storage_from_array(lookupdict[var],
                                                          backend,
                                                          shape_nlay,
                                                          (np.float64, ds[var].shape))

    ds2 = xr.open_dataset('../lookupdata/radlw_ref_data.nc')
    tmp = np.tile(ds2['chi_mls'].data[None, None, None, :, :], (npts, 1, nlay, 1, 1))

    lookupdict_gt4py['chi_mls'] = create_storage_from_array(tmp,
                                                            backend,
                                                            shape_nlay,
                                                            (np.float64, ds2['chi_mls'].shape))

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


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[2],
                                                      'nspb': nspb[2],
                                                      'laytrop': indict['laytrop'],
                                                      'ng03': ng03,
                                                      'ns03': ns03,
                                                      'nlay': nlay,
                                                      'oneminus': oneminus})
def taugb03(pavel: FIELD_FLT,
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
            taun2: FIELD_FLT,
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
            id211: FIELD_INT):
    from __externals__ import nspa, nspb, laytrop, ng03, nlay, ns03, oneminus
    with computation(PARALLEL), interval(0, laytrop):
        #  --- ...  minor gas mapping levels:
        #     lower - n2o, p = 706.272 mbar, t = 278.94 k
        #     upper - n2o, p = 95.58 mbar, t = 215.7 k

        refrat_planck_a = chi_mls[0, 0, 0][0, 8]/chi_mls[0, 0, 0][1, 8]
        refrat_planck_b = chi_mls[0, 0, 0][0, 12]/chi_mls[0, 0, 0][1, 12]
        refrat_m_a = chi_mls[0, 0, 0][0, 2]/chi_mls[0, 0, 0][1, 2]
        refrat_m_b = chi_mls[0, 0, 0][0, 12]/chi_mls[0, 0, 0][1, 12]

        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-1)*5 + (jt-1)) * nspa + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = (jp*5 + (jt1-1)) * nspa + js1 - 1

        speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a*colamt[0, 0, 0][1]
        specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
        specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
        jmn2o = 1 + specmult_mn2o - 1
        fmn2o = mod(specmult_mn2o, 1.0)

        speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a*colamt[0, 0, 0][1]
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
        jmn2op= jmn2o+ 1
        jplp  = jpl  + 1

        #  --- ...  in atmospheres where the amount of n2O is too great to be considered
        #           a minor species, adjust the column amount of n2O by an empirical factor
        #           to obtain the proper contribution.

        p = coldry * chi_mls[0, 0, 0][3, jp]
        ratn2o = colamt[0, 0 ,0][3] / p
        if ratn2o > 1.5:
            adjfac = 0.5 + (ratn2o - 0.5)**0.65
            adjcoln2o = adjfac * p
        else:
            adjcoln2o = colamt[0, 0, 0][3]

        id000 = id000
        id010 = id010
        id100 = id100
        id110 = id110
        id200 = id200
        id210 = id210

        id001 = id001
        id011 = id011
        id101 = id101
        id111 = id111
        id201 = id201
        id211 = id211

        if specparm < 0.125:
            p = fs - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p = -fs
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0*1
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk0 = 1.0 - fs
            fk1 = fs
            fk2 = 0.0
            id000 = ind0*1
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0*1
            id210 = ind0*1

        fac000 = fk0*fac00
        fac100 = fk1*fac00
        fac200 = fk2*fac00
        fac010 = fk0*fac10
        fac110 = fk1*fac10
        fac210 = fk2*fac10

        if specparm1 < 0.125:
            p = fs1 - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p = -fs1
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1 + 1
            id011 = ind1 +10
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
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk0*fac01
        fac101 = fk1*fac01
        fac201 = fk2*fac01
        fac011 = fk0*fac11
        fac111 = fk1*fac11
        fac211 = fk2*fac11

        for ig in range(ng03):
            tauself = selffac* (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor  = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf]))
            n2om1   = ka_mn2o[0, 0, 0][ig, jmn2o, indm] + fmn2o * \
                (ka_mn2o[0, 0, 0][ig, jmn2op, indm] - ka_mn2o[0, 0, 0][ig, jmn2o, indm])
            n2om2   = ka_mn2o[0, 0, 0][ig, jmn2o, indmp] + fmn2o * \
                (ka_mn2o[0, 0, 0][ig, jmn2op, indmp] - ka_mn2o[0, 0, 0][ig, jmn2o, indmp])
            absn2o  = n2om1 + minorfrac * (n2om2 - n2om1)

            tau_major = speccomb * \
                (fac000*absa[0, 0, 0][ig, id000] + fac010*absa[0, 0, 0][ig, id010] + \
                 fac100*absa[0, 0, 0][ig, id100] + fac110*absa[0, 0, 0][ig, id110] + \
                fac200*absa[0, 0, 0][ig, id200] + fac210*absa[0, 0, 0][ig, id210])

            tau_major1 = speccomb1 * \
                (fac001*absa[0, 0, 0][ig, id001] + fac011*absa[0, 0, 0][ig, id011] + \
                 fac101*absa[0, 0, 0][ig, id101] + fac111*absa[0, 0, 0][ig, id111] + \
                fac201*absa[0, 0, 0][ig, id201] + fac211*absa[0, 0, 0][ig, id211])

            taug[0, 0, 0][ns03+ig] = tau_major + tau_major1 + tauself + taufor + \
                adjcoln2o*absn2o

            fracs[0, 0, 0][ns03+ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * \
                (fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl])

    with computation(PARALLEL), interval(laytrop, nlay):

        refrat_planck_a = chi_mls[0, 0, 0][0, 8]/chi_mls[0, 0, 0][1, 8]
        refrat_planck_b = chi_mls[0, 0, 0][0, 12]/chi_mls[0, 0, 0][1, 12]
        refrat_m_a = chi_mls[0, 0, 0][0, 2]/chi_mls[0, 0, 0][1, 2]
        refrat_m_b = chi_mls[0, 0, 0][0, 12]/chi_mls[0, 0, 0][1, 12]

        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-13)*5 + (jt-1)) * nspb + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb + js1 - 1

        speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_b*colamt[0, 0, 0][1]
        specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
        specmult_mn2o = 4.0 * min(specparm_mn2o, oneminus)
        jmn2o = 1 + specmult_mn2o - 1
        fmn2o = mod(specmult_mn2o, 1.0)

        speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_b*colamt[0, 0, 0][1]
        specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + specmult_planck - 1
        fpl = mod(specmult_planck, 1.0)

        indf = indfor - 1
        indm = indminor - 1
        indfp = indf + 1
        indmp = indm + 1
        jmn2op= jmn2o+ 1
        jplp  = jpl  + 1

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
            adjfac = 0.5 + (ratn2o - 0.5)**0.65
            adjcoln2o = adjfac * p
        else:
            adjcoln2o = colamt[0, 0, 0][3]

        fk0 = 1.0 - fs
        fk1 = fs
        fac000 = fk0*fac00
        fac010 = fk0*fac10
        fac100 = fk1*fac00
        fac110 = fk1*fac10

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0*fac01
        fac011 = fk0*fac11
        fac101 = fk1*fac01
        fac111 = fk1*fac11

        for ig2 in range(ng03):
            taufor = forfac * (forref[0, 0, 0][ig2, indf] + forfrac * \
                (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])) 
            n2om1  = kb_mn2o[0, 0, 0][ig2, jmn2o, indm] + fmn2o * \
                (kb_mn2o[0, 0, 0][ig2, jmn2op, indm] - kb_mn2o[0, 0, 0][ig2, jmn2o, indm])
            n2om2  = kb_mn2o[0, 0, 0][ig2, jmn2o, indmp] + fmn2o * \
                (kb_mn2o[0, 0, 0][ig2, jmn2op, indmp] - kb_mn2o[0, 0, 0][ig2, jmn2o, indmp])
            absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

            tau_major = speccomb * \
                (fac000*absb[0, 0, 0][ig2, id000] + fac010*absb[0, 0, 0][ig2, id010] + \
                 fac100*absb[0, 0, 0][ig2, id100] + fac110*absb[0, 0, 0][ig2, id110])

            tau_major1 = speccomb1 * \
                (fac001*absb[0, 0, 0][ig2, id001] + fac011*absb[0, 0, 0][ig2, id011] + \
                 fac101*absb[0, 0, 0][ig2, id101] + fac111*absb[0, 0, 0][ig2, id111])

            taug[0, 0, 0][ns03+ig2] = tau_major + tau_major1 + taufor + adjcoln2o*absn2o            

            fracs[0, 0, 0][ns03+ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * \
                (fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl])


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[3],
                                                      'nspb': nspb[3],
                                                      'laytrop': indict['laytrop'],
                                                      'ng04': ng04,
                                                      'ns04': ns04,
                                                      'nlay': nlay,
                                                      'oneminus': oneminus})
def taugb04(pavel: FIELD_FLT,
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
            absa: Field[(DTYPE_FLT, (ng04, 585))],
            absb: Field[(DTYPE_FLT, (ng04, 1175))],
            selfref: Field[(DTYPE_FLT, (ng04, 10))],
            forref: Field[(DTYPE_FLT, (ng04, 4))],
            fracrefa: Field[(DTYPE_FLT, (ng04, 9))],
            fracrefb: Field[(DTYPE_FLT, (ng04, 5))],
            chi_mls: Field[(DTYPE_FLT, (7, 59))],
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
            taun2: FIELD_FLT,
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
            id211: FIELD_INT):
    from __externals__ import nspa, nspb, laytrop, ng04, nlay, ns04, oneminus
    with computation(PARALLEL), interval(...):
        refrat_planck_a = chi_mls[0, 0, 0][0, 10]/chi_mls[0, 0, 0][1, 10]     # P = 142.5940 mb
        refrat_planck_b = chi_mls[0, 0, 0][2, 12]/chi_mls[0, 0, 0][1, 12]     # P = 95.58350 mb

    with computation(PARALLEL), interval(0, laytrop):
        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-1)*5 + (jt-1)) * nspa + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = ( jp*5 + (jt1-1)) * nspa + js1 - 1

        speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a*colamt[0, 0, 0][1]
        specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
        specmult_planck = 8.0 * min(specparm_planck, oneminus)
        jpl = 1 + specmult_planck - 1
        fpl = mod(specmult_planck, 1.0)

        inds = indself - 1
        indf = indfor - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp  = jpl  + 1

        id000 = id000
        id010 = id010
        id100 = id100
        id110 = id110
        id200 = id200
        id210 = id210

        if specparm < 0.125:
            p = fs - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p = -fs
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk0 = 1.0 - fs
            fk1 = fs
            fk2 = 0.0
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk0*fac00
        fac100 = fk1*fac00
        fac200 = fk2*fac00
        fac010 = fk0*fac10
        fac110 = fk1*fac10
        fac210 = fk2*fac10

        id001 = id001
        id011 = id011
        id101 = id101
        id111 = id111
        id201 = id201
        id211 = id211

        if specparm1 < 0.125:
            p = fs1 - 1.0
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p = -fs1
            p4 = p**4
            fk0 = p4
            fk1 = 1.0 - p - 2.0*p4
            fk2 = p + p4
            id001 = ind1 + 1
            id011 = ind1 +10
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
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk0*fac01
        fac101 = fk1*fac01
        fac201 = fk2*fac01
        fac011 = fk0*fac11
        fac111 = fk1*fac11
        fac211 = fk2*fac11

        for ig in range(ng04):
            tauself = selffac * (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor  = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])) 

            tau_major = speccomb * \
                (fac000*absa[0, 0, 0][ig, id000] + fac010*absa[0, 0, 0][ig, id010] + \
                 fac100*absa[0, 0, 0][ig, id100] + fac110*absa[0, 0, 0][ig, id110] + \
                fac200*absa[0, 0, 0][ig, id200] + fac210*absa[0, 0, 0][ig, id210])

            tau_major1 = speccomb1 * \
                (fac001*absa[0, 0, 0][ig, id001] + fac011*absa[0, 0, 0][ig, id011] + \
                 fac101*absa[0, 0, 0][ig, id101] + fac111*absa[0, 0, 0][ig, id111] + \
                fac201*absa[0, 0, 0][ig, id201] + fac211*absa[0, 0, 0][ig, id211])

            taug[0, 0, 0][ns04+ig] = tau_major + tau_major1 + tauself + taufor

            fracs[0, 0, 0][ns04+ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * \
                (fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl])

    with computation(PARALLEL), interval(laytrop, nlay):
        speccomb = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][2] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-13)*5 + (jt-1)) * nspb + js - 1

        speccomb1 = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][2] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb + js1 - 1

        speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b*colamt[0, 0, 0][1]
        specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + specmult_planck - 1
        fpl = mod(specmult_planck, 1.0)
        jplp = jpl + 1

        id000 = ind0
        id010 = ind0 + 5
        id100 = ind0 + 1
        id110 = ind0 + 6
        id001 = ind1
        id011 = ind1 + 5
        id101 = ind1 + 1
        id111 = ind1 + 6

        fk0 = 1.0 - fs
        fk1 = fs
        fac000 = fk0*fac00
        fac010 = fk0*fac10
        fac100 = fk1*fac00
        fac110 = fk1*fac10

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0*fac01
        fac011 = fk0*fac11
        fac101 = fk1*fac01
        fac111 = fk1*fac11

        for ig2 in range(ng04):
            tau_major =  speccomb * \
                (fac000*absb[0, 0, 0][ig2, id000] + fac010*absb[0, 0, 0][ig2, id010] + \
                 fac100*absb[0, 0, 0][ig2, id100] + fac110*absb[0, 0, 0][ig2, id110])
            tau_major1 = speccomb1 * \
                (fac001*absb[0, 0, 0][ig2, id001] + fac011*absb[0, 0, 0][ig2, id011] + \
                 fac101*absb[0, 0, 0][ig2, id101] + fac111*absb[0, 0, 0][ig2, id111])

            taug[0, 0, 0][ns04+ig2] = tau_major + tau_major1

            fracs[0, 0, 0][ns04+ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * \
                (fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl])

        taug[0, 0, 0][ns04+ 7] = taug[0, 0, 0][ns04+ 7] * 0.92
        taug[0, 0, 0][ns04+ 8] = taug[0, 0, 0][ns04+ 8] * 0.88
        taug[0, 0, 0][ns04+ 9] = taug[0, 0, 0][ns04+ 9] * 1.07
        taug[0, 0, 0][ns04+10] = taug[0, 0, 0][ns04+10] * 1.1
        taug[0, 0, 0][ns04+11] = taug[0, 0, 0][ns04+11] * 0.99
        taug[0, 0, 0][ns04+12] = taug[0, 0, 0][ns04+12] * 0.88
        taug[0, 0, 0][ns04+13] = taug[0, 0, 0][ns04+13] * 0.943


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[4],
                                                      'nspb': nspb[4],
                                                      'laytrop': indict['laytrop'],
                                                      'ng05': ng05,
                                                      'ns05': ns05,
                                                      'nlay': nlay,
                                                      'oneminus': oneminus})
def taugb05(pavel: FIELD_FLT,
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
            absa: Field[(DTYPE_FLT, (ng05, 585))],
            absb: Field[(DTYPE_FLT, (ng05, 1175))],
            selfref: Field[(DTYPE_FLT, (ng05, 10))],
            forref: Field[(DTYPE_FLT, (ng05, 4))],
            fracrefa: Field[(DTYPE_FLT, (ng05, 9))],
            fracrefb: Field[(DTYPE_FLT, (ng05, 5))],
            ka_mo3: Field[(DTYPE_FLT, (ng05, 9, 19))],
            ccl4: Field[(DTYPE_FLT, (ng05,))],
            chi_mls: Field[(DTYPE_FLT, (7, 59))],
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
            taun2: FIELD_FLT,
            js: FIELD_INT,
            js1: FIELD_INT,
            jmn2o: FIELD_INT,
            jmn2op: FIELD_INT,
            jpl: FIELD_INT,
            jplp: FIELD_INT,
            jmo3: FIELD_INT,
            jmo3p: FIELD_INT,
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
            id211: FIELD_INT):
    from __externals__ import nspa, nspb, laytrop, ng05, nlay, ns05, oneminus
    with computation(PARALLEL), interval(...):
        refrat_planck_a = chi_mls[0, 0, 0][0, 4]/chi_mls[0, 0, 0][1, 4]     # P = 142.5940 mb
        refrat_planck_b = chi_mls[0, 0, 0][2, 42]/chi_mls[0, 0, 0][1, 42]     # P = 95.58350 mb
        refrat_m_a = chi_mls[0, 0, 0][0, 6]/chi_mls[0, 0, 0][1, 6]

    with computation(PARALLEL), interval(0, laytrop):
        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-1)*5 + (jt-1)) * nspa + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = (jp*5 + (jt1-1)) * nspa + js1 - 1

        speccomb_mo3 = colamt[0, 0, 0][0] + refrat_m_a*colamt[0, 0, 0][1]
        specparm_mo3 = colamt[0, 0, 0][0] / speccomb_mo3
        specmult_mo3 = 8.0 * min(specparm_mo3, oneminus)
        jmo3 = 1 + specmult_mo3 - 1
        fmo3 = mod(specmult_mo3, 1.0)

        speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a*colamt[0, 0, 0][1]
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
        jplp  = jpl  + 1
        jmo3p = jmo3 + 1

        id000 = id000
        id010 = id010
        id100 = id100
        id110 = id110
        id200 = id200
        id210 = id210

        if specparm < 0.125:
            p0   = fs - 1.0
            p40  = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0   = -fs
            p40  = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        id001 = id001
        id011 = id011
        id101 = id101
        id111 = id111
        id201 = id201
        id211 = id211

        fac000 = fk00 * fac00
        fac100 = fk10 * fac00
        fac200 = fk20 * fac00
        fac010 = fk00 * fac10
        fac110 = fk10 * fac10
        fac210 = fk20 * fac10

        if specparm1 < 0.125:
            p1   = fs1 - 1.0
            p41  = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1   = -fs1
            p41  = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01
        fac101 = fk11 * fac01
        fac201 = fk21 * fac01
        fac011 = fk01 * fac11
        fac111 = fk11 * fac11
        fac211 = fk21 * fac11

        for ig in range(ng05):
            tauself = selffac * (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor  = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf]))
            o3m1    = ka_mo3[0, 0, 0][ig, jmo3, indm] + fmo3 * \
                (ka_mo3[0, 0, 0][ig, jmo3p, indm] - ka_mo3[0, 0, 0][ig, jmo3, indm])
            o3m2    = ka_mo3[0, 0, 0][ig, jmo3, indmp] + fmo3 * \
                (ka_mo3[0, 0, 0][ig, jmo3p, indmp] - ka_mo3[0, 0, 0][ig, jmo3, indmp])
            abso3   = o3m1 + minorfrac*(o3m2 - o3m1)

            taug[0, 0, 0][ns05+ig] = speccomb * \
                (fac000*absa[0, 0, 0][ig, id000] + fac010*absa[0, 0, 0][ig, id010] + \
                 fac100*absa[0, 0, 0][ig, id100] + fac110*absa[0, 0, 0][ig, id110] + \
                 fac200*absa[0, 0, 0][ig, id200] + fac210*absa[0, 0, 0][ig, id210]) + \
                speccomb1 * \
                (fac001*absa[0, 0, 0][ig, id001] + fac011*absa[0, 0, 0][ig, id011] + \
                 fac101*absa[0, 0, 0][ig, id101] + fac111*absa[0, 0, 0][ig, id111] + \
                 fac201*absa[0, 0, 0][ig, id201] + fac211*absa[0, 0, 0][ig, id211]) + \
                tauself + taufor+abso3*colamt[0, 0, 0][2]+wx[0, 0, 0][0]*ccl4[0, 0, 0][ig]

            fracs[0, 0, 0][ns05+ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * \
                (fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl])

    with computation(PARALLEL), interval(laytrop, nlay):
        speccomb = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 0]*colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][2] / speccomb
        specmult = 4.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-13)*5 + (jt-1)) * nspb + js - 1

        speccomb1 = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 1]*colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][2] / speccomb1
        specmult1 = 4.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb + js1 - 1

        speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b*colamt[0, 0, 0][1]
        specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
        specmult_planck = 4.0 * min(specparm_planck, oneminus)
        jpl = 1 + specmult_planck - 1
        fpl = mod(specmult_planck, 1.0)
        jplp= jpl + 1

        id000 = ind0
        id010 = ind0 + 5
        id100 = ind0 + 1
        id110 = ind0 + 6
        id001 = ind1
        id011 = ind1 + 5
        id101 = ind1 + 1
        id111 = ind1 + 6

        fk00 = 1.0 - fs
        fk10 = fs

        fk01 = 1.0 - fs1
        fk11 = fs1

        fac000 = fk00 * fac00
        fac010 = fk00 * fac10
        fac100 = fk10 * fac00
        fac110 = fk10 * fac10

        fac001 = fk01 * fac01
        fac011 = fk01 * fac11
        fac101 = fk11 * fac01
        fac111 = fk11 * fac11

        for ig2 in range(ng05):
            taug[0, 0, 0][ns05+ig2] = speccomb * \
                (fac000*absb[0, 0, 0][ig2, id000] + fac010*absb[0, 0, 0][ig2, id010] + \
                 fac100*absb[0, 0, 0][ig2, id100] + fac110*absb[0, 0, 0][ig2, id110]) + \
                speccomb1 * \
                (fac001*absb[0, 0, 0][ig2, id001] + fac011*absb[0, 0, 0][ig2, id011] + \
                 fac101*absb[0, 0, 0][ig2, id101] + fac111*absb[0, 0, 0][ig2, id111]) + \
                wx[0, 0, 0][0] * ccl4[0, 0, 0][ig2]

            fracs[0, 0, 0][ns05+ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * \
                (fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl])


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[5],
                                                      'nspb': nspb[5],
                                                      'laytrop': indict['laytrop'],
                                                      'ng06': ng06,
                                                      'ns06': ns06,
                                                      'nlay': nlay,
                                                      'oneminus': oneminus})
def taugb06(pavel: FIELD_FLT,
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
            absa: Field[(DTYPE_FLT, (ng06, 65))],
            selfref: Field[(DTYPE_FLT, (ng06, 10))],
            forref: Field[(DTYPE_FLT, (ng06, 4))],
            fracrefa: Field[(DTYPE_FLT, (ng06,))],
            ka_mco2: Field[(DTYPE_FLT, (ng06, 19))],
            cfc11adj: Field[(DTYPE_FLT, (ng06,))],
            cfc12: Field[(DTYPE_FLT, (ng06,))],
            chi_mls: Field[(DTYPE_FLT, (7, 59))],
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
            taufor: FIELD_FLT):
    from __externals__ import nspa, nspb, laytrop, ng06, nlay, ns06, oneminus
    with computation(PARALLEL), interval(0, laytrop):
        ind0 = ((jp-1)*5 + (jt -1)) * nspa
        ind1 = ( jp   *5 + (jt1-1)) * nspa

        inds = indself - 1
        indf = indfor - 1
        indm = indminor - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        temp   = coldry * chi_mls[0, 0, 0][1, jp]
        ratco2 = colamt[0, 0, 0][1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.77
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[0, 0, 0][1]

        for ig in range(ng06):
            tauself = selffac * (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor  = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf]))
            absco2  = ka_mco2[0, 0, 0][ig, indm] + minorfrac * \
                (ka_mco2[0, 0, 0][ig, indmp] - ka_mco2[0, 0, 0][ig, indm])

            taug[0, 0, 0][ns06+ig] = colamt[0, 0, 0][0] * \
                (fac00*absa[0, 0, 0][ig, ind0] + fac10*absa[0, 0, 0][ig, ind0p] + \
                 fac01*absa[0, 0, 0][ig, ind1] + fac11*absa[0, 0, 0][ig, ind1p]) + \
                tauself + taufor + adjcolco2*absco2 + \
                wx[0, 0, 0][1]*cfc11adj[0, 0, 0][ig] + wx[0, 0, 0][2]*cfc12[0, 0, 0][ig]

            fracs[0, 0, 0][ns06+ig] = fracrefa[0, 0, 0][ig]

    with computation(PARALLEL), interval(laytrop, nlay):
        for ig2 in range(ng06):
            taug[0, 0, 0][ns06+ig2] = wx[0, 0, 0][1]*cfc11adj[0, 0, 0][ig2] + wx[0, 0, 0][2]*cfc12[0, 0, 0][ig2]

            fracs[0, 0, 0][ns06+ig2] = fracrefa[0, 0, 0][ig2]


@stencil(backend=backend, rebuild=rebuild, externals={'nspa': nspa[6],
                                                      'nspb': nspb[6],
                                                      'laytrop': indict['laytrop'],
                                                      'ng07': ng07,
                                                      'ns07': ns07,
                                                      'nlay': nlay,
                                                      'oneminus': oneminus})
def taugb07(pavel: FIELD_FLT,
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
            absa: Field[(DTYPE_FLT, (ng07, 585))],
            absb: Field[(DTYPE_FLT, (ng07, 235))],
            selfref: Field[(DTYPE_FLT, (ng07, 10))],
            forref: Field[(DTYPE_FLT, (ng07, 4))],
            fracrefa: Field[(DTYPE_FLT, (ng07, 9))],
            fracrefb: Field[(DTYPE_FLT, (ng07,))],
            ka_mco2: Field[(DTYPE_FLT, (ng07, 9, 19))],
            kb_mco2: Field[(DTYPE_FLT, (ng07, 19))],
            chi_mls: Field[(DTYPE_FLT, (7, 59))],
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
            taun2: FIELD_FLT,
            js: FIELD_INT,
            js1: FIELD_INT,
            jmco2: FIELD_INT,
            jmco2p: FIELD_INT,
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
            id211: FIELD_INT):
    from __externals__ import nspa, nspb, laytrop, ng07, nlay, ns07, oneminus
    with computation(PARALLEL), interval(...):
        refrat_planck_a = chi_mls[0, 0, 0][0, 2]/chi_mls[0, 0, 0][2, 2]     # P = 706.2620 mb
        refrat_m_a = chi_mls[0, 0, 0][0, 2]/chi_mls[0, 0, 0][2, 2]          # P = 706.2720 mb

    with computation(PARALLEL), interval(0, laytrop):
        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 0]*colamt[0, 0, 0][2]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp-1)*5 + (jt-1)) * nspa + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 1]*colamt[0, 0, 0][2]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = (jp*5 + (jt1-1)) * nspa + js1 - 1

        speccomb_mco2 = colamt[0, 0, 0][0] + refrat_m_a*colamt[0, 0, 0][2]
        specparm_mco2 = colamt[0, 0, 0][0] / speccomb_mco2
        specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
        jmco2 = 1 + specmult_mco2 - 1
        fmco2 = mod(specmult_mco2, 1.0)

        speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a*colamt[0, 0, 0][2]
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
        jplp  = jpl  + 1
        jmco2p= jmco2+ 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        temp   = coldry * chi_mls[0, 0, 0][1, jp]
        ratco2 = colamt[0, 0, 0][1] / temp
        if ratco2 > 3.0:
            adjfac = 3.0 + (ratco2-3.0)**0.79
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[0, 0, 0][1]

        id000 = id000
        id010 = id010
        id100 = id100
        id110 = id110
        id200 = id200
        id210 = id210

        if specparm < 0.125:
            p0 = fs - 1.0
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0 + 2
            id210 = ind0 +11
        elif specparm > 0.875:
            p0 = -fs
            p40 = p0**4
            fk00 = p40
            fk10 = 1.0 - p0 - 2.0*p40
            fk20 = p0 + p40

            id000 = ind0 + 1
            id010 = ind0 +10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk00 = 1.0 - fs
            fk10 = fs
            fk20 = 0.0

            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 +10
            id200 = ind0
            id210 = ind0

        fac000 = fk00 * fac00
        fac100 = fk10 * fac00
        fac200 = fk20 * fac00
        fac010 = fk00 * fac10
        fac110 = fk10 * fac10
        fac210 = fk20 * fac10

        id001 = id001
        id011 = id011
        id101 = id101
        id111 = id111
        id201 = id201
        id211 = id211

        if specparm1 < 0.125:
            p1 = fs1 - 1.0
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1 + 2
            id211 = ind1 +11
        elif specparm1 > 0.875:
            p1 = -fs1
            p41 = p1**4
            fk01 = p41
            fk11 = 1.0 - p1 - 2.0*p41
            fk21 = p1 + p41

            id001 = ind1 + 1
            id011 = ind1 +10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk01 = 1.0 - fs1
            fk11 = fs1
            fk21 = 0.0

            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 +10
            id201 = ind1
            id211 = ind1

        fac001 = fk01 * fac01
        fac101 = fk11 * fac01
        fac201 = fk21 * fac01
        fac011 = fk01 * fac11
        fac111 = fk11 * fac11
        fac211 = fk21 * fac11

        for ig in range(ng07):
            tauself = selffac* (selfref[0, 0, 0][ig, inds] + selffrac * \
                (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds]))
            taufor  = forfac * (forref[0, 0, 0][ig, indf] + forfrac * \
                (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])) 
            co2m1   = ka_mco2[0, 0, 0][ig, jmco2, indm] + fmco2 * \
                (ka_mco2[0, 0, 0][ig, jmco2p, indm] - ka_mco2[0, 0, 0][ig, jmco2, indm])
            co2m2   = ka_mco2[0, 0, 0][ig, jmco2, indmp] + fmco2 * \
                (ka_mco2[0, 0, 0][ig, jmco2p, indmp] - ka_mco2[0, 0, 0][ig, jmco2, indmp])
            absco2  = co2m1 + minorfrac * (co2m2 - co2m1)

            taug[0, 0, 0][ns07+ig] = speccomb * \
                (fac000*absa[0, 0, 0][ig, id000] + fac010*absa[0, 0, 0][ig, id010] + \
                 fac100*absa[0, 0, 0][ig, id100] + fac110*absa[0, 0, 0][ig, id110] + \
                 fac200*absa[0, 0, 0][ig, id200] + fac210*absa[0, 0, 0][ig, id210]) + \
                speccomb1 * \
                (fac001*absa[0, 0, 0][ig, id001] + fac011*absa[0, 0, 0][ig, id011] + \
                 fac101*absa[0, 0, 0][ig, id101] + fac111*absa[0, 0, 0][ig, id111] + \
                 fac201*absa[0, 0, 0][ig, id201] + fac211*absa[0, 0, 0][ig, id211]) + \
                tauself + taufor + adjcolco2*absco2

            fracs[0, 0, 0][ns07+ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * \
                (fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl])
    
    with computation(PARALLEL), interval(laytrop, nlay):
        temp   = coldry * chi_mls[0, 0, 0][1, jp]
        ratco2 = colamt[0, 0, 0][1] / temp
        if ratco2 > 3.0:
            adjfac = 2.0 + (ratco2-2.0)**0.79
            adjcolco2 = adjfac * temp
        else:
            adjcolco2 = colamt[0, 0, 0][1]

        ind0 = ((jp-13)*5 + (jt -1)) * nspb
        ind1 = ((jp-12)*5 + (jt1-1)) * nspb

        indm = indminor - 1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig2 in range(ng07):
            absco2 = kb_mco2[0, 0, 0][ig2, indm] + minorfrac * \
                (kb_mco2[0, 0, 0][ig2, indmp] - kb_mco2[0, 0, 0][ig2, indm])

            taug[0, 0, 0][ns07+ig2] = colamt[0, 0, 0][2] * \
                (fac00*absb[0, 0, 0][ig2, ind0] + fac10*absb[0, 0, 0][ig2, ind0p] + \
                 fac01*absb[0, 0, 0][ig2, ind1] + fac11*absb[0, 0, 0][ig2, ind1p]) + \
                adjcolco2 * absco2

            fracs[0, 0, 0][ns07+ig2] = fracrefb[0, 0, 0][ig2]

        taug[0, 0, 0][ns07+ 5] = taug[0, 0, 0][ns07+ 5] * 0.92
        taug[0, 0, 0][ns07+ 6] = taug[0, 0, 0][ns07+ 6] * 0.88
        taug[0, 0, 0][ns07+ 7] = taug[0, 0, 0][ns07+ 7] * 1.07
        taug[0, 0, 0][ns07+ 8] = taug[0, 0, 0][ns07+ 8] * 1.1
        taug[0, 0, 0][ns07+ 9] = taug[0, 0, 0][ns07+ 9] * 0.99
        taug[0, 0, 0][ns07+10] = taug[0, 0, 0][ns07+10] * 0.855



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

lookupdict_gt4py = loadlookupdata('kgb03')
# print(lookupdict_gt4py)

start = time.time()
taugb03(indict_gt4py['pavel'],
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
        lookupdict_gt4py['ka_mn2o'],
        lookupdict_gt4py['kb_mn2o'],
        lookupdict_gt4py['chi_mls'],
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
        locdict_gt4py['js'],
        locdict_gt4py['js1'],
        locdict_gt4py['jmn2o'],
        locdict_gt4py['jmn2op'],
        locdict_gt4py['jpl'],
        locdict_gt4py['jplp'],
        locdict_gt4py['id000'],
        locdict_gt4py['id010'],
        locdict_gt4py['id100'],
        locdict_gt4py['id110'],
        locdict_gt4py['id200'],
        locdict_gt4py['id210'],
        locdict_gt4py['id001'],
        locdict_gt4py['id011'],
        locdict_gt4py['id101'],
        locdict_gt4py['id111'],
        locdict_gt4py['id201'],
        locdict_gt4py['id211'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

lookupdict_gt4py = loadlookupdata('kgb04')

start = time.time()
taugb04(indict_gt4py['pavel'],
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
        lookupdict_gt4py['chi_mls'],
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
        locdict_gt4py['js'],
        locdict_gt4py['js1'],
        locdict_gt4py['jmn2o'],
        locdict_gt4py['jmn2op'],
        locdict_gt4py['jpl'],
        locdict_gt4py['jplp'],
        locdict_gt4py['id000'],
        locdict_gt4py['id010'],
        locdict_gt4py['id100'],
        locdict_gt4py['id110'],
        locdict_gt4py['id200'],
        locdict_gt4py['id210'],
        locdict_gt4py['id001'],
        locdict_gt4py['id011'],
        locdict_gt4py['id101'],
        locdict_gt4py['id111'],
        locdict_gt4py['id201'],
        locdict_gt4py['id211'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

lookupdict_gt4py = loadlookupdata('kgb05')

start = time.time()
taugb05(indict_gt4py['pavel'],
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
        lookupdict_gt4py['ka_mo3'],
        lookupdict_gt4py['ccl4'],
        lookupdict_gt4py['chi_mls'],
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
        locdict_gt4py['js'],
        locdict_gt4py['js1'],
        locdict_gt4py['jmn2o'],
        locdict_gt4py['jmn2op'],
        locdict_gt4py['jpl'],
        locdict_gt4py['jplp'],
        locdict_gt4py['jmo3'],
        locdict_gt4py['jmo3p'],
        locdict_gt4py['id000'],
        locdict_gt4py['id010'],
        locdict_gt4py['id100'],
        locdict_gt4py['id110'],
        locdict_gt4py['id200'],
        locdict_gt4py['id210'],
        locdict_gt4py['id001'],
        locdict_gt4py['id011'],
        locdict_gt4py['id101'],
        locdict_gt4py['id111'],
        locdict_gt4py['id201'],
        locdict_gt4py['id211'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")


lookupdict_gt4py = loadlookupdata('kgb06')

start = time.time()
taugb06(indict_gt4py['pavel'],
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
        lookupdict_gt4py['selfref'],
        lookupdict_gt4py['forref'],
        lookupdict_gt4py['fracrefa'],
        lookupdict_gt4py['ka_mco2'],
        lookupdict_gt4py['cfc11adj'],
        lookupdict_gt4py['cfc12'],
        lookupdict_gt4py['chi_mls'],
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
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

lookupdict_gt4py = loadlookupdata('kgb07')

start = time.time()
taugb07(indict_gt4py['pavel'],
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
        lookupdict_gt4py['ka_mco2'],
        lookupdict_gt4py['kb_mco2'],
        lookupdict_gt4py['chi_mls'],
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
        locdict_gt4py['js'],
        locdict_gt4py['js1'],
        locdict_gt4py['jmco2'],
        locdict_gt4py['jmco2p'],
        locdict_gt4py['jpl'],
        locdict_gt4py['jplp'],
        locdict_gt4py['id000'],
        locdict_gt4py['id010'],
        locdict_gt4py['id100'],
        locdict_gt4py['id110'],
        locdict_gt4py['id200'],
        locdict_gt4py['id210'],
        locdict_gt4py['id001'],
        locdict_gt4py['id011'],
        locdict_gt4py['id101'],
        locdict_gt4py['id111'],
        locdict_gt4py['id201'],
        locdict_gt4py['id211'],
        domain=domain2,
        origin=default_origin,
        validate_args=validate)
end = time.time()
print(f"Elapsed time = {end-start}")

outdict_gt4py = {'fracs': indict_gt4py['fracs'][-1, :, :, :].squeeze().T,
                 'tautot': indict_gt4py['tautot'][-1, :, :, :].squeeze().T,
                 'taug': taug[-1, :, :, :].squeeze().T}
    
outvars = ['fracs', 'tautot', 'taug']

outdict_val = dict()
for var in outvars:
    outdict_val[var] = serializer.read(var, serializer.savepoint['lwrad-taugb07-output-000000'])

compare_data(outdict_val, outdict_gt4py)

#print(f"ind0 = {locdict_gt4py['ind0']}")
# print(f"Python = {outdict_gt4py['taug'][0, :]}")
# 
# print(indict_gt4py['laytrop'])
#print(f"Difference = {(outdict_val['taug'][ns07, :] - outdict_gt4py['taug'][ns07, :])}")
# print(' ')
# print(f"Fortran = {outdict_val['fracs'][ns05, :]}")
# print(' ')
# print(f"Python = {outdict_gt4py['fracs'][ns05, :]}")
# print(' ')
# # 
# print(f"jpl = {locdict_gt4py['tau_major'][0, :, :40]}")