import os
import numpy as np
from gt4py import gtscript

prefix = "phifv3"
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "numpy"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else True
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True

DTYPE_FLT = np.float64
FIELD_FLT = gtscript.Field[DTYPE_FLT]
# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/s1053/install/serialbox/gnu"

IN_VARS = ["ix", "levs", "ntrac", "gt0", "gq0", "del_gz", "phii", "phil"]
OUT_VARS = ["del_gz", "phii", "phil"]

# Physics Constants used from physcon module

con_rv = 4.6150e2
con_rd = 2.8705e2
con_fvirt = con_rv/con_rd - 1.0