import os
import numpy as np
from gt4py import gtscript

prefix = "sfc_drv"
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

IN_VARS = [
    "im",
    "km",
    "ps",
    "t1",
    "q1",
    "soiltyp",
    "vegtype",
    "sigmaf",
    "sfcemis",
    "dlwflx",
    "dswsfc",
    "snet",
    "delt",
    "tg3",
    "cm",
    "ch",
    "prsl1",
    "prslki",
    "zf",
    "land",
    "wind",
    "slopetyp",
    "shdmin",
    "shdmax",
    "snoalb",
    "sfalb",
    "flag_iter",
    "flag_guess",
    "lheatstrg",
    "isot",
    "ivegsrc",
    "bexppert",
    "xlaipert",
    "vegfpert",
    "pertvegf",
    "weasd",
    "snwdph",
    "tskin",
    "tprcp",
    "srflag",
    "smc",
    "stc",
    "slc",
    "canopy",
    "trans",
    "tsurf",
    "zorl",
    "sncovr1",
    "qsurf",
    "gflux",
    "drain",
    "evap",
    "hflx",
    "ep",
    "runoff",
    "cmm",
    "chh",
    "evbs",
    "evcw",
    "sbsno",
    "snowc",
    "stm",
    "snohf",
    "smcwlt2",
    "smcref2",
    "wet1",
]


IN_VARS_FPVS = ["c1xpvs", "c2xpvs", "tbpvs"]

OUT_VARS = [
    "weasd",
    "snwdph",
    "tskin",
    "tprcp",
    "srflag",
    "smc",
    "stc",
    "slc",
    "canopy",
    "trans",
    "tsurf",
    "zorl",
    "sncovr1",
    "qsurf",
    "gflux",
    "drain",
    "evap",
    "hflx",
    "ep",
    "runoff",
    "cmm",
    "chh",
    "evbs",
    "evcw",
    "sbsno",
    "snowc",
    "stm",
    "snohf",
    "smcwlt2",
    "smcref2",
    "wet1",
]
