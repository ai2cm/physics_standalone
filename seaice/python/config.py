import os
import numpy as np
from gt4py import gtscript

prefix = "sfc_sice"
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "numpy"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else False
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]
# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/s1053/install/serialbox/gnu"

IN_VARS = [
    "im",
    "km",
    "ps",
    "t1",
    "q1",
    "delt",
    "sfcemis",
    "dlwflx",
    "sfcnsw",
    "sfcdsw",
    "srflag",
    "cm",
    "ch",
    "prsl1",
    "prslki",
    "islimsk",
    "wind",
    "flag_iter",
    "lprnt",
    "ipr",
    "cimin",
    "hice",
    "fice",
    "tice",
    "weasd",
    "tskin",
    "tprcp",
    "stc",
    "ep",
    "snwdph",
    "qsurf",
    "snowmt",
    "gflux",
    "cmm",
    "chh",
    "evap",
    "hflx",
]

OUT_VARS = [
    "hice",
    "fice",
    "tice",
    "weasd",
    "tskin",
    "tprcp",
    "stc",
    "ep",
    "snwdph",
    "qsurf",
    "snowmt",
    "gflux",
    "cmm",
    "chh",
    "evap",
    "hflx",
]
