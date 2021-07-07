import numpy as np
from gt4py import gtscript
import os

prefix = "samfshalcnv"
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "numpy"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else False
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else True
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/c14/install/daint/serialbox2_master/gnu_debug"
REBUILD = False
BACKEND_OPTS = {}  # {'verbose': True} if BACKEND.startswith('gt') else {}
default_origin = (0, 0, 0)

DTYPE_INT = np.int32
DTYPE_FLOAT = np.float64

FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLOAT = gtscript.Field[DTYPE_FLOAT]

IN_VARS = [
    "im",
    "ix",
    "km",
    "itc",
    "ntc",
    "ntk",
    "ntr",
    "ncloud",
    "clam",
    "c0s",
    "c1",
    "asolfac",
    "pgcon",
    "delt",
    "islimsk",
    "psp",
    "delp",
    "prslp",
    "garea",
    "hpbl",
    "dot",
    "phil",  # "fscav", (not used)
    "kcnv",
    "kbot",
    "ktop",
    "qtr",
    "q1",
    "t1",
    "u1",
    "v1",
    "rn",
    "cnvw",
    "cnvc",
    "ud_mf",
    "dt_mf",
]
OUT_VARS = [
    "kcnv",
    "kbot",
    "ktop",
    "qtr",
    "q1",
    "t1",
    "u1",
    "v1",
    "rn",
    "cnvw",
    "cnvc",
    "ud_mf",
    "dt_mf",
]


def change_backend(backend):
    global BACKEND
    BACKEND = backend
