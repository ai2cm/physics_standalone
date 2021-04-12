import os
import numpy as np
from gt4py import gtscript


### SERIALIZATION ###

IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True

# Path of serialbox directory
if IS_DOCKER: SERIALBOX_DIR = "/usr/local/serialbox"
else:         SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"

# Path of the serialized data directory
DATA_PATH     = "./data"
DATA_REF_PATH = "./data_ref"

# Names of the input variables
IN_VARS = [ "iie", "kke", "kbot",
            "qv", "ql", "qr", "qg", "qa", "qn",
            "pt", "uin", "vin", "dz", "delp",
            "area", "dt_in", "land",
            "seconds", "p", "lradar",
            "reset",
            "qi", "qs",
            "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
            "pt_dt", "w", "udt", "vdt",
            "rain", "snow", "ice", "graupel",
            "refl_10cm" ]
            
# Names of the output variables
OUT_VARS = [ "qi", "qs",
             "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
             "pt_dt", "w", "udt", "vdt",
             "rain", "snow", "ice", "graupel",
             "refl_10cm" ]

# Names of the integer and boolean variables
INT_VARS = [ "iie", "kke", "kbot", "seconds", 
             "lradar", "reset" ]

# Names of the float variables
FLT_VARS = [ "qv", "ql", "qr", "qg", "qa", "qn",
             "pt", "uin", "vin", "dz", "delp",
             "area", "dt_in", "land",
             "p",
             "qi", "qs",
             "qv_dt", "ql_dt", "qr_dt", "qi_dt", "qs_dt", "qg_dt", "qa_dt",
             "pt_dt", "w", "udt", "vdt",
             "rain", "snow", "ice", "graupel",
             "refl_10cm" ]


### UTILITY ###

# Predefined types of variables and fields
DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]

# GT4PY parameters
BACKEND        = str(os.getenv("VERSION")) if ("VERSION" in os.environ) else "gtx86"
REBUILD        = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else True
DEBUG_MODE     = (os.getenv("DEBUG_MODE") == "True") if ("DEBUG_MODE" in os.environ) else False
DEFAULT_ORIGIN = (0, 0, 0)

# Python or GT4Py
USE_GT4PY = (BACKEND != "python")

# Output mode
PROGRESS_MODE = (os.getenv("PROGRESS_MODE") == "True") if ("PROGRESS_MODE" in os.environ) else True

# Execution mode
RUN_MODE   = str(os.getenv("RUN_MODE")) if ("RUN_MODE" in os.environ) else "validation"
VALIDATION = (RUN_MODE == "validation")
BENCHMARK  = (RUN_MODE != "validation")
NORMAL     = (RUN_MODE == "normal")
WEAK       = (RUN_MODE == "weak")
STRONG     = (RUN_MODE == "strong")

# Stencils mode
STENCILS = str(os.getenv("STENCILS")) if (("STENCILS" in os.environ) and USE_GT4PY) else "normal"

# Repetitions of the computations to get the average timing
REPS = int(os.getenv("REPS")) if ("REPS" in os.environ) else 1

# Get the number of threads
N_TH = int(os.getenv("OMP_NUM_THREADS")) if ("OMP_NUM_THREADS" in os.environ) else 12

CN = int(os.getenv("CN")) if ("CN" in os.environ) else 48