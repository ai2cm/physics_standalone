from dataclasses import fields
import gt4py
import os
import sys
import numpy as np
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval, stencil
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from util import view_gt4py_storage, compare_data

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

eps = 1e-6
npts = 24
nbands = 16
nlay = 63
nlp1 = 64
maxgas = 7
ilwrgas = 1
ilwcliq = 1
maxxsec = 4

domain = (npts, 1, 1)
domain2 = (npts, 1, nlay)

amdw = con_amd/con_amw
amdo3 = con_amd/con_amo3

semiss0_np = np.ones(nbands)

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = Field[DTYPE_INT]
FIELD_FLT = Field[DTYPE_FLT]

rebuild = True
backend = "gtc:gt:cpu_ifirst"
shape0 = (1, 1, 1)
shape = (npts, 1, 1)
shape_nlay = (npts, 1, nlay)
shape_nlp1 = (npts, 1, nlp1)
shape_nlp2 = (npts, 1, nlp1+1)
default_origin = (0, 0, 0)
type1 = np.float64
type_nbands = (np.float64, (nbands,))
type_nbands3 = (np.float64, (nbands, 3))
type_maxgas = (np.float64, (maxgas,))
type_maxxsec = (np.float64, (maxxsec,))
type_9 = (np.float64, (9,))
type_10 = (np.float64, (10,))

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Generator_rank0')

savepoints = serializer.savepoint_list()

print(savepoints[2])

invars = ['plyr', 'plvl', 'tlyr', 'tlvl', 'qlyr', 'olyr', 'gasvmr',
          'clouds', 'icsdlw', 'faerlw', 'semis', 'tsfg',
          'dz', 'delp', 'de_lgth', 'im', 'lmk', 'lmp', 'lprnt']
nlay_vars = ['plyr', 'tlyr', 'qlyr', 'olyr', 'dz', 'delp']
nlp1_vars = ['plvl', 'tlvl']

print('Loading input vars...')
indict = dict()

for var in invars:
    tmp = serializer.read(var, savepoints[2])
    if var in nlay_vars or var in nlp1_vars:
        indict[var] = np.tile(tmp[:, None, :], (1, 1, 1))
    elif var == 'faerlw':
        indict[var] = np.tile(tmp[:, None, :, :, :], (1, 1, 1, 1, 1))
    elif var == 'semis':
        indict[var] = np.tile(tmp[:, None, None], (1, 1, 1))
    elif var == 'gasvmr' or var == 'clouds':
        indict[var] = np.tile(tmp[:, None, :, :], (1, 1, 1, 1))
    else:
        indict[var] = tmp

print('Done')
print(' ')
print('Creating input storages...')

indict_gt4py = dict()

for var in invars:
    if var in nlay_vars:
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type1)
    elif var in nlp1_vars:
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlp1,
                                dtype=type1)
    elif var == 'faerlw':
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type_nbands3)
    elif var == 'semis':
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape,
                                dtype=type1)
    elif var == 'gasvmr':
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type_10)
    elif var == 'clouds':
        indict_gt4py[var] = gt4py.storage.from_array(
                                indict[var],
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type_9)
    else:
        indict_gt4py[var] = indict[var]

print('Done')
print(' ')
print('Creating local storages...')

locvars = ['pavel', 'tavel', 'delp', 'colbrd', 'h2ovmr',
           'o3vmr', 'coldry', 'colamt', 'temcol', 'tauaer',
           'taucld', 'semiss0', 'semiss', 'tz', 'dz', 'wx',
           'cldfrc', 'clwp', 'ciwp', 'relw', 'reiw', 'cda1', 'cda2',
           'cda3', 'cda4', 'pwvcm']

locdict_gt4py = dict()

for var in locvars:
    if var  == 'colamt':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype = type_maxgas)
    elif var == 'wx':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type_maxxsec)
    elif var == 'pwvcm':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape,
                                dtype=type1)
    elif var == 'tauaer' or var == 'taucld':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype=type_nbands)
    elif var == 'semiss0':
        locdict_gt4py[var] = gt4py.storage.ones(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape,
                                dtype=type_nbands)
    elif var == 'semiss':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape,
                                dtype=type_nbands)
    elif var == 'tz':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlp1,
                                dtype=type1)
    elif var == 'cldfrc':
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlp2,
                                dtype=type1)
    else:
        locdict_gt4py[var] = gt4py.storage.zeros(
                                backend=backend,
                                default_origin=default_origin,
                                shape=shape_nlay,
                                dtype = type1)

print('Done')

@stencil(backend=backend, rebuild=rebuild)
def get_surface_emissivity(sfemis: FIELD_FLT,
                           semiss0: Field[type_nbands],
                           semiss: Field[type_nbands],
                           value: int):
    with computation(PARALLEL), interval(...):
        if sfemis[0, 0, 0] > eps and sfemis[0, 0, 0] <= 1.:
            semiss[0, 0, 0][value] = sfemis[0, 0 ,0]
        else:
            semiss[0, 0, 0][value] = semiss0[0, 0, 0][value]


@stencil(backend=backend, rebuild=rebuild)
def set_aerosols(aerosols: Field[type_nbands3],
                 tauaer: Field[type_nbands],
                 value: int):
    with computation(PARALLEL), interval(...):
        tauaer[0, 0, 0][value] = aerosols[0, 0, 0][value, 0]*(1. - aerosols[0, 0, 0][value, 1])

tem1 = 100.*con_g
tem2 = 1.0e-20 * 1.0e3 * con_avgd

@stencil(backend=backend, rebuild=rebuild)
def set_absorbers(plyr: FIELD_FLT,
                  delpin: FIELD_FLT,
                  tlyr: FIELD_FLT,
                  dzlyr: FIELD_FLT,
                  qlyr: FIELD_FLT,
                  olyr: FIELD_FLT,
                  gasvmr: Field[type_10],
                  pavel: FIELD_FLT,
                  delp: FIELD_FLT,
                  tavel: FIELD_FLT,
                  dz: FIELD_FLT,
                  h2ovmr: FIELD_FLT,
                  o3vmr: FIELD_FLT,
                  coldry: FIELD_FLT,
                  temcol: FIELD_FLT,
                  tem0: FIELD_FLT,
                  colamt: Field[type_maxgas],
                  wx: Field[type_maxxsec]):
    with computation(PARALLEL), interval(...):

        pavel[0, 0, 0] = plyr[0, 0, 0]
        delp[0, 0, 0] = delpin[0, 0, 0]
        tavel[0, 0, 0] = tlyr[0, 0, 0]
        dz[0, 0, 0] = dzlyr[0, 0, 0]

        h2ovmr[0, 0, 0] = max(0., qlyr[0, 0, 0] * \
            amdw/(1.-qlyr[0, 0, 0]))
        o3vmr[0, 0, 0] = max(0., olyr[0, 0, 0]*amdo3)

        tem0[0, 0, 0] = (1. - h2ovmr[0, 0, 0])*con_amd + \
            h2ovmr[0, 0, 0]*con_amw
        coldry[0, 0, 0] = tem2*delp[0, 0, 0] / \
            (tem1*tem0[0, 0, 0]*(1.+h2ovmr[0, 0, 0]))
        temcol[0, 0, 0] = 1.e-12*coldry[0, 0 ,0]

        colamt[0, 0, 0][0] = max(0., coldry[0, 0, 0]*h2ovmr[0, 0, 0])
        colamt[0, 0, 0][1] = max(temcol[0, 0, 0], coldry[0, 0, 0]*gasvmr[0, 0, 0][0])
        colamt[0, 0, 0][2] = max(temcol[0, 0, 0], coldry[0 ,0, 0]*o3vmr[0, 0, 0])

        if ilwrgas > 0:
            colamt[0, 0, 0][3] = max(temcol[0, 0, 0], coldry[0, 0, 0]*gasvmr[0, 0, 0][1])  # n2o
            colamt[0, 0, 0][4] = max(temcol[0, 0, 0], coldry[0, 0, 0]*gasvmr[0, 0, 0][2])  # ch4
            colamt[0, 0, 0][5] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][3])  # o2
            colamt[0, 0, 0][6] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][4])  # co
            
            wx[0, 0, 0][0] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][8])   # ccl4
            wx[0, 0, 0][1] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][5])   # cf11
            wx[0, 0, 0][2] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][6])   # cf12
            wx[0, 0, 0][3] = max(0.0, coldry[0, 0, 0]*gasvmr[0, 0, 0][7])   # cf22

@stencil(backend=backend, rebuild=rebuild)
def set_clouds(clouds: Field[type_9],
               cldfrc: FIELD_FLT,
               clwp: FIELD_FLT,
               relw: FIELD_FLT,
               ciwp: FIELD_FLT,
               reiw: FIELD_FLT,
               cda1: FIELD_FLT,
               cda2: FIELD_FLT,
               cda3: FIELD_FLT,
               cda4: FIELD_FLT):
    with computation(PARALLEL), interval(1, None):
        if ilwcliq > 0:
            cldfrc = clouds[0, 0, -1][0]
    with computation(PARALLEL), interval(...):
        if ilwcliq > 0:
            clwp[0, 0, 0] = clouds[0, 0, 0][1]
            relw[0, 0, 0] = clouds[0, 0, 0][2]
            ciwp[0, 0, 0] = clouds[0, 0, 0][3]
            reiw[0, 0, 0] = clouds[0, 0, 0][4]
            cda1[0, 0, 0] = clouds[0, 0, 0][5]
            cda2[0, 0, 0] = clouds[0, 0, 0][6]
            cda3[0, 0, 0] = clouds[0, 0, 0][7]
            cda4[0, 0, 0] = clouds[0, 0, 0][8]
        else:
            cda1[0, 0, 0] = clouds[0, 0, 0][1]

@stencil(backend=backend, rebuild=rebuild)
def compute_temps_for_pwv(tem1: FIELD_FLT,
                          tem2: FIELD_FLT,
                          coldry: FIELD_FLT,
                          colamt: Field[type_maxgas]):
    with computation(FORWARD), interval(...):
        tem1[0, 0, 0] += coldry[0, 0, 0] + colamt[0, 0, 0][0]
        tem2[0, 0, 0] += colamt[0, 0, 0][0]


# Execute code from here

for j in range(nbands):
    get_surface_emissivity(indict_gt4py['semis'],
                           locdict_gt4py['semiss0'],
                           locdict_gt4py['semiss'],
                           value=j,
                           origin=default_origin,
                           domain=domain,
                           validate_args=True)


locdict_gt4py['tz'] = indict_gt4py['tlvl']

tem0 = gt4py.storage.zeros(backend=backend,
                           default_origin=default_origin,
                           shape=shape_nlay,
                           dtype = type1)

set_absorbers(indict_gt4py['plyr'],
              indict_gt4py['delp'],
              indict_gt4py['tlyr'],
              indict_gt4py['dz'],
              indict_gt4py['qlyr'],
              indict_gt4py['olyr'],
              indict_gt4py['gasvmr'],
              locdict_gt4py['pavel'],
              locdict_gt4py['delp'],
              locdict_gt4py['tavel'],
              locdict_gt4py['dz'],
              locdict_gt4py['h2ovmr'],
              locdict_gt4py['o3vmr'],
              locdict_gt4py['coldry'],
              locdict_gt4py['temcol'],
              tem0,
              locdict_gt4py['colamt'],
              locdict_gt4py['wx'],
              origin=default_origin,
              domain=domain2,
              validate_args=True
              )

for j in range(nbands):
    set_aerosols(indict_gt4py['faerlw'],
                 locdict_gt4py['tauaer'],
                 value=j,
                 origin=default_origin,
                 domain=domain2,
                 validate_args=True)

set_clouds(indict_gt4py['clouds'],
           locdict_gt4py['cldfrc'],
           locdict_gt4py['clwp'],
           locdict_gt4py['relw'],
           locdict_gt4py['ciwp'],
           locdict_gt4py['reiw'],
           locdict_gt4py['cda1'],
           locdict_gt4py['cda2'],
           locdict_gt4py['cda3'],
           locdict_gt4py['cda4'],
           domain=(npts, 1, 63),
           origin=default_origin,
           validate_args=True)

locdict_gt4py['cldfrc'][:, 0, 0] = 1.0

tem1 = gt4py.storage.zeros(backend=backend,
                           default_origin=default_origin,
                           shape=shape,
                           dtype=type1)
tem2 = gt4py.storage.zeros(backend=backend,
                           default_origin=default_origin,
                           shape=shape,
                           dtype=type1)

# This stencil below didn't work, but I realized it's just a sum over k. 

#compute_temps_for_pwv(tem1,
#                      tem2,
#                      locdict_gt4py['coldry'],
#                      locdict_gt4py['colamt'],
#                      origin=default_origin,
#                      domain=domain,
#                      validate_args=True)

tem1 = np.sum(locdict_gt4py['coldry'], 2)[:, :, None] + np.sum(locdict_gt4py['colamt'][:, :, :, 0], 2)[:, :, None]
tem2 = np.sum(locdict_gt4py['colamt'][:, :, :, 0], 2)[:, :, None]

tem0 = 10.0 * tem2 / (amdw * tem1 * con_g)
locdict_gt4py['pwvcm'] = tem0 * indict_gt4py['plvl'][:, :, 0][:, :, None]

# Load serialized data to validate against
ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'
serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, 'Serialized_rank0')

savepoints = serializer.savepoint_list()

sp = savepoints[6]

valdict = dict()

for var in locvars:
    valdict[var] = serializer.read(var, sp)

locdict_np = view_gt4py_storage(locdict_gt4py)

compare_data(valdict, locdict_np)


