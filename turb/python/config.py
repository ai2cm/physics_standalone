import os
import numpy as np
from gt4py import gtscript

prefix = "satmedmfvdif"
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

IN_VARS = [
    "ix",
    "im",
    "km",
    "ntrac",
    "ntcw",
    "ntiw",
    "ntke",
    "dv",
    "du",
    "tdt",
    "rtg",
    "u1",
    "v1",
    "t1",
    "q1",
    "swh",
    "hlw",
    "xmu",
    "garea",
    "psk",
    "rbsoil",
    "zorl",
    "u10m",
    "v10m",
    "fm",
    "fh",
    "tsea",
    "heat",
    "evap",
    "stress",
    "spd1",
    "kpbl",
    "prsi",
    "del",
    "prsl",
    "prslk",
    "phii",
    "phil",
    "delt",
    "dspheat",
    "dusfc",
    "dvsfc",
    "dtsfc",
    "dqsfc",
    "hpbl",
    "kinver",
    "xkzm_m",
    "xkzm_h",
    "xkzm_s",
]

OUT_VARS = [
    "dv",
    "du",
    "tdt",
    "rtg",
    "kpbl",
    "dusfc",
    "dvsfc",
    "dtsfc",
    "dqsfc",
    "hpbl",
]

# Physics Constants used from physcon module

grav = 9.80665e0
rd = 2.87050e2
cp = 1.00460e3
rv = 4.61500e2
hvap = 2.50000e6
hfus = 3.33580e5
wfac = 7.0
cfac = 4.5
gamcrt = 3.0
sfcfrac = 0.1
vk = 0.4
rimin = -100.0
rbcr = 0.25
zolcru = -0.02
tdzmin = 1.0e-3
rlmn = 30.0
rlmx = 500.0
elmx = 500.0
prmin = 0.25
prmax = 4.0
prtke = 1.0
prscu = 0.67
f0 = 1.0e-4
crbmin = 0.15
crbmax = 0.35
tkmin = 1.0e-9
dspfac = 0.5
qmin = 1.0e-8
qlmin = 1.0e-12
zfmin = 1.0e-8
aphi5 = 5.0
aphi16 = 16.0
elmfac = 1.0
elefac = 1.0
cql = 100.0
dw2min = 1.0e-4
dkmax = 1000.0
xkgdx = 25000.0
qlcr = 3.5e-5
zstblmax = 2500.0
xkzinv = 0.15
h1 = 0.33333333
ck0 = 0.4
ck1 = 0.15
ch0 = 0.4
ch1 = 0.15
ce0 = 0.4
rchck = 1.5
cdtn = 25.0
xmin = 180.0
xmax = 330.0

con_ttp = 2.7316e2
con_cvap = 1.8460e3
con_cliq = 4.1855e3
con_hvap = 2.5000e6
con_rv = 4.6150e2
con_csol = 2.1060e3
con_hfus = 3.3358e5
con_psat = 6.1078e2