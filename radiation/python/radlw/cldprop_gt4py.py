import gt4py
import os
import sys
import numpy as np
import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval, stencil
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from radlw_param import a0, a1, a2, ngptlw, nbands, maxgas, maxxsec
from util import view_gt4py_storage, compare_data, create_storage_from_array, create_storage_zeros, create_storage_ones
from config import (npts, nlay, nlp1, ilwrgas, ilwcliq, DTYPE_FLT, DTYPE_INT,
                    FIELD_FLT, FIELD_2D, FIELD_INT, shape0, shape, shape_2D,
                    shape_nlay, shape_nlp1, shape_nlp2, default_origin,
                    type1, type_nbands, type_nbands3, type_maxgas, type_maxxsec,
                    type_9, type_10, domain, domain2)

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

amdw = con_amd/con_amw
amdo3 = con_amd/con_amo3

semiss0_np = np.ones(nbands)

rebuild = False
validate = False
backend = "gtc:gt:cpu_ifirst"

invars = ['nlay', 'nlp1', 'ipseed', 'cfrac', 'cliqp', 'reliq', 'cicep', 'reice',
          'cdat1', 'cdat2', 'cdat3', 'cdat4', 'dz', 'de_lgth', 'cldfmc', 'taucld']
nlay_vars = ['cliqp', 'reliq', 'cicep', 'reice', 'cdat1', 'cdat2',
             'cdat3', 'cdat4', 'dz']

indict = dict()
indict_gt4py = dict()

for var in invars:
    if var in nlay_vars:
        indict_gt4py[var] = create_storage_from_array(indict[var],
                                                      backend,
                                                      shape_nlay,
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



locvars = ['tauliq', 'tauice', 'cldf']

locdict_gt4py = dict()

locdict_gt4py['tauliq'] = create_storage_zeros(backend,
                                               shape,
                                               type_nbands)
locdict_gt4py['tauice'] = create_storage_zeros(backend,
                                               shape,
                                               type_nbands)
locdict_gt4py['cldf'] = create_storage_zeros(backend,
                                             shape_nlay,
                                             type_nbands)

