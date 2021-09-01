import numpy as np
import sys
import os

IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True

if IS_DOCKER:
    sys.path.insert(0, "/work/radiation/python")
else:
    sys.path.insert(
        0, "/Users/andrewp/Documents/work/physics_standalone/radiation/python/radlw"
    )
from radlw.radlw_param import nbands, maxgas, maxxsec, ngptlw
from radsw.radsw_param import ngptsw, nbhgh, nblow, nbdsw
from gt4py import gtscript
from gt4py.gtscript import Field

if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/Users/andrewp/Documents/code/serialbox2/install"

npts = 24

nlay = 63
nlp1 = 64

ilwrgas = 1
ilwcliq = 1

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
DTYPE_BOOL = bool
FIELD_INT = Field[DTYPE_INT]
FIELD_FLT = Field[DTYPE_FLT]
FIELD_BOOL = Field[DTYPE_BOOL]
FIELD_2D = Field[gtscript.IJ, DTYPE_FLT]
FIELD_2DINT = Field[gtscript.IJ, DTYPE_INT]
FIELD_2DBOOL = Field[gtscript.IJ, DTYPE_BOOL]

shape = (npts, 1, 1)
shape_2D = (npts, 1)
shape_nlay = (npts, 1, nlay)
shape_nlp1 = (npts, 1, nlp1)
shape_nlp2 = (npts, 1, nlp1 + 1)
default_origin = (0, 0, 0)

type_nbands = (DTYPE_FLT, (nbands,))
type_nbandssw_int = (DTYPE_INT, (nbhgh - nblow + 1,))
type_nbandssw_flt = (DTYPE_FLT, (nbhgh - nblow + 1,))
type_ngptlw = (DTYPE_FLT, (ngptlw,))
type_ngptsw = (DTYPE_FLT, (ngptsw,))
type_ngptsw_bool = (DTYPE_BOOL, (ngptsw,))
type_nbands3 = (DTYPE_FLT, (nbands, 3))
type_maxgas = (DTYPE_FLT, (maxgas,))
type_maxxsec = (DTYPE_FLT, (maxxsec,))
type_nbdsw = (DTYPE_FLT, (nbdsw,))
type_9 = (DTYPE_FLT, (9,))
type_10 = (DTYPE_FLT, (10,))
