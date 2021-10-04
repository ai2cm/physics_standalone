import os
import numpy as np
from gt4py import gtscript

prefix = "cloud_mp"
### SERIALIZATION ###

IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False

# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/s1053/install/serialbox/gnu"


# Names of the input variables
IN_VARS = [
    "iie",
    "kke",
    "kbot",
    "qv",
    "ql",
    "qr",
    "qg",
    "qa",
    "qn",
    "pt",
    "uin",
    "vin",
    "dz",
    "delp",
    "area",
    "dt_in",
    "land",
    "seconds",
    "p",
    "lradar",
    "reset",
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]

# Names of the output variables
OUT_VARS = [
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]

# Names of the integer and boolean variables
INT_VARS = ["iie", "kke", "kbot", "seconds", "lradar", "reset"]

# Names of the float variables
FLT_VARS = [
    "qv",
    "ql",
    "qr",
    "qg",
    "qa",
    "qn",
    "pt",
    "uin",
    "vin",
    "dz",
    "delp",
    "area",
    "dt_in",
    "land",
    "p",
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]


### UTILITY ###

# Predefined types of variables and fields
DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]

# GT4PY parameters
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "gtx86"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else False
DEBUG_MODE = (
    (os.getenv("DEBUG_MODE") == "True") if ("DEBUG_MODE" in os.environ) else False
)
DEFAULT_ORIGIN = (0, 0, 0)

# Stencils mode
STENCILS = (
    str(os.getenv("STENCILS"))
    if (("STENCILS" in os.environ) and USE_GT4PY)
    else "normal"
)
