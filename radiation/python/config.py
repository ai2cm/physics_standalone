import numpy as np
import sys

sys.path.insert(
    0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python/radlw"
)
from radlw_param import nbands, maxgas, maxxsec, ngptlw
from radsw_param import ngptsw, nbhgh, nblow
from gt4py import gtscript
from gt4py.gtscript import Field

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

domain = (npts, 1, 1)
domain2 = (npts, 1, nlay)

shape0 = (1, 1, 1)
shape = (npts, 1, 1)
shape_2D = (npts, 1)
shape_nlay = (npts, 1, nlay)
shape_nlp1 = (npts, 1, nlp1)
shape_nlp2 = (npts, 1, nlp1 + 1)
default_origin = (0, 0, 0)
type_nbands = (DTYPE_FLT, (nbands,))
type_nbandssw_flt = (DTYPE_FLT, (nbhgh - nblow + 1,))
type_nbandssw_int = (DTYPE_INT, (nbhgh - nblow + 1,))
type_ngptlw = (DTYPE_FLT, (ngptlw,))
type_ngptsw = (DTYPE_FLT, (ngptsw,))
type_nbands3 = (DTYPE_FLT, (nbands, 3))
type_maxgas = (DTYPE_FLT, (maxgas,))
type_maxxsec = (DTYPE_FLT, (maxxsec,))
type_9 = (DTYPE_FLT, (9,))
type_10 = (DTYPE_FLT, (10,))
