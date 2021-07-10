import gt4py
import os
import sys
import numpy as np
import xarray as xr
from copy import deepcopy
from gt4py import gtscript
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import nbands, nrates, delwave

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]

DEFAULT_ORIGIN = (0, 0, 0)
DEFAULT_ORIGIN4 = (0, 0, 0, 0)
DEFAULT_ORIGIN5 = (0, 0, 0, 0, 0)
BACKEND = 'numpy'
INT_VARS = []

def numpy_dict_to_gt4py_dict(np_dict):

    # shape = np_dict["qi"].shape
    gt4py_dict = {}

    for var in np_dict:

        data = np_dict[var]
        if isinstance(data, np.ndarray):
            ndim = data.ndim
        else:
            ndim = 0

        if (ndim > 0) and (ndim <= 3) and (data.size >= 2):

            if ndim == 1:
                shape = (1, 1, len(data))
            elif ndim == 2:
                shape = (1, 1, data.shape[0], data.shape[1])

            reshaped_data = np.empty(shape)

            if ndim == 1:  # 1D array (i-dimension)
                reshaped_data[...] = data[np.newaxis, np.newaxis, :]
                origin = DEFAULT_ORIGIN
            elif ndim == 2:  # 2D array (i-dimension, j-dimension)
                reshaped_data[...] = data[np.newaxis, np.newaxis, :, :]
                origin = DEFAULT_ORIGIN4

            dtype =  DTYPE_FLT
            gt4py_dict[var] = gt4py.storage.from_array(
                reshaped_data, BACKEND, origin, dtype=dtype
            )

        else:  # Scalars

            gt4py_dict[var] = deepcopy(data)

    return gt4py_dict

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()

# print(savepoints)

invars = ['pavel', 'tavel', 'tz', 'stemp', 'h2ovmr', 'colamt',
          'coldry', 'colbrd', 'nlay', 'nlp1']

indict = dict()

for var in invars:
    tmp = serializer.read(var, savepoints[0])

    if tmp.size == 1:
        indict[var] = tmp[0]
    else:
        indict[var] = tmp

shape1 = (1, 1, indict['nlay'])
shape2 = (1, 1, indict['nlp1'])
shape_bands = (1, 1, indict['nlp1'], nbands)
shape_rf = (1, 1, indict['nlay'], nrates, 2)


def setcoef(invars):

    gtdict = dict()

    gtdict = numpy_dict_to_gt4py_dict(invars)

    pavel = gtdict['pavel']
    tavel = gtdict['tavel']
    tz = gtdict['tz']
    stemp = gtdict['stemp']
    h2ovmr = gtdict['h2ovmr']
    colamt = gtdict['colamt']
    coldry = gtdict['coldry']
    colbrd = gtdict['colbrd']
    nlay = gtdict['nlay']
    nlp1 = gtdict['nlp1']

    dfile = '../lookupdata/totplnk.nc'
    pfile = '../lookupdata/radlw_ref_data.nc'
    totplnk_np = xr.open_dataset(dfile)['totplnk'].data
    preflog_np = xr.open_dataset(pfile)['preflog'].data
    tref_np = xr.open_dataset(pfile)['tref'].data
    chi_mls_np = xr.open_dataset(pfile)['chi_mls'].data

    totplnk = gt4py.storage.from_array(totplnk_np,
                                       BACKEND,
                                       DEFAULT_ORIGIN)
    preflog = gt4py.storage.from_array(preflog_np,
                                       BACKEND,
                                       DEFAULT_ORIGIN)
    tref = gt4py.storage.from_array(tref_np,
                                    BACKEND,
                                    DEFAULT_ORIGIN)
    chi_mls = gt4py.storage.from_array(chi_mls_np,
                                       BACKEND,
                                       DEFAULT_ORIGIN)

    pklay = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN4, shape_bands, dtype=DTYPE_FLT)
    pklev = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN4, shape_bands, dtype=DTYPE_FLT)

    jp = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)
    jt = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)
    jt1 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)

    fac00 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    fac01 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    fac10 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    fac11 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)

    forfac = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    forfrac = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    selffac = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    scaleminor = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    scaleminorn2 = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    indminor = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    minorfrac = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    indfor = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    indself = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    selffrac = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    rfrate = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN5, shape_rf, dtype=DTYPE_FLT)

    indlay = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)
    indlev = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)
    tlyrfr = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)
    tlvlfr = gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_FLT)

    f_zero = 0.0
    f_one = 1.0
    stpfac  = 296.0/1013.0

    indlay[0, 0, 0] = np.minimum(180, np.maximum(1, int(stemp-159.0)))
    indlev[0, 0, 0] = np.minimum(180, np.maximum(1, int(tz[0, 0, 0]-159.0) ))
    tlyrfr[0, 0, 0] = stemp - int(stemp)
    tlvlfr[0, 0, 0] = tz[0, 0, 0] - int(tz[0, 0, 0])

    for i in range(nbands):
        tem1 = totplnk[indlay[0, 0, 0], i] - totplnk[indlay[0, 0, 0]-1, i]
        tem2 = totplnk[indlev[0, 0, 0], i] - totplnk[indlev[0, 0, 0]-1, i]
        pklay[:, :, 0, i] = delwave[i] * (totplnk[indlay[0, 0, 0]-1, i] + tlyrfr[0, 0, 0]*tem1)
        pklev[:, :, 0, i] = delwave[i] * (totplnk[indlev[0, 0, 0]-1, i] + tlvlfr[0, 0, 0]*tem2)
    
    #tavel_int - gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)
    #tz_int - gt4py.storage.empty(BACKEND, DEFAULT_ORIGIN, shape1, dtype=DTYPE_INT)

    tavel_int = gt4py.storage.from_array(indict['tavel'].astype(DTYPE_INT),
                                         BACKEND,
                                         DEFAULT_ORIGIN,
                                         shape1,
                                         dtype=DTYPE_INT)
    tz_int = gt4py.storage.from_array(indict['tz'].astype(DTYPE_INT),
                                      BACKEND,
                                      DEFAULT_ORIGIN,
                                      shape2,
                                      dtype=DTYPE_INT)
   
    @gtscript.stencil(backend=BACKEND)
    def compute_ind(tavel: FIELD_FLT,
                    tavel_int: FIELD_INT,
                    tz: FIELD_FLT,
                    tz_int: FIELD_INT,
                    indlay: FIELD_INT,
                    tlyrfr: FIELD_FLT,
                    indlev: FIELD_INT,
                    tlvlfr: FIELD_FLT):
        with computation(PARALLEL), interval(...):
            indlay[0, 0, 0] = min(180, max(1, tavel_int[0, 0, 0]-159.0))
            tlyrfr[0, 0, 0] = tavel[0, 0, 0] - tavel_int[0, 0, 0]

            indlev[0, 0, 0] = min(180, max(1, tz_int[0, 0, 1]-159.0))
            tlyrfr[0, 0, 0] = tz[0, 0, 1] - tz_int[0, 0, 1]

    def compute_pk(delwave: FIELD_FLT,
                   totplnk: FIELD_FLT,
                   tlyrfr: FIELD_FLT,
                   tlvlfr: FIELD_FLT,
                   indlay: FIELD_INT,
                   indlev: FIELD_INT):
        with computation(PARALLEL), interval(...):
            pklay[0, 0, 1, 0] = delwave * (totplnk[0, 0, -1, 0] + tlyfr*(totplnk[0, 0, 0, 0]-totplnk[0, 0, -1, 0]))
            pklev[0, 0, 1, 0] = delwave * (totplnk[0, 0, -1, 0] + tlyfr*(totplnk[0, 0, 0, 0]-totplnk[0, 0, -1, 0]))


    compute_ind(tavel, tavel_int, tz, tz_int, indlay, tlyrfr, indlev, tlvlfr)

setcoef(indict)


