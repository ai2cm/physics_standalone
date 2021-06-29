import os
import numpy as np
from gt4py import gtscript

prefix = "get_prs_fv3"
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "numpy"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else True
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
DTYPE_BOOL = bool
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]
FIELD_FLT_IJ = gtscript.Field[gtscript.IJ, DTYPE_FLT]
FIELD_BOOL = gtscript.Field[DTYPE_BOOL]
# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"

IN_VARS = ["ix", "levs", "ntrac", "phii", "prsi", "tgrs", "qgrs", "del", "del_gz"]
OUT_VARS = ["del", "del_gz"]

# Physics Constants used from physcon module

con_rv = 4.6150e2
con_rd = 2.8705e2
con_fvirt = con_rv/con_rd - 1.0