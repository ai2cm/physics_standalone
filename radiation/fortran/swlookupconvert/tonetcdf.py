import sys
import numpy as np
import os
import xarray as xr

sys.path.insert(0, "/work/radiation/python")
from config import *

import serialbox as ser

ddir = "saveData"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "radsw_cldprtb")
savepoints = serializer.savepoint_list()

refvars = [
    "extliq1",
    "extliq2",
    "ssaliq1",
    "ssaliq2",
    "asyliq1",
    "asyliq2",
    "extice2",
    "ssaice2",
    "asyice2",
    "extice3",
    "ssaice3",
    "asyice3",
    "fdlice3",
    "abari",
    "bbari",
    "cbari",
    "dbari",
    "ebari",
    "fbari",
    "b0s",
    "b1s",
    "c0s",
    "b0r",
    "c0r",
]


dims = [
    ("n", "nbands"),
    ("n", "nbands"),
    ("n", "nbands"),
    ("n", "nbands"),
    ("n", "nbands"),
    ("n", "nbands"),
    ("m", "nbands"),
    ("m", "nbands"),
    ("m", "nbands"),
    ("p", "nbands"),
    ("p", "nbands"),
    ("p", "nbands"),
    ("p", "nbands"),
    "s",
    "s",
    "s",
    "s",
    "s",
    "s",
    "nbands",
    "nbands",
    "nbands",
    "nbands",
    "nbands",
]

dataout = dict()
for n, key in enumerate(refvars):
    dataout[key] = (dims[n], serializer.read(key, serializer.savepoint["mySavepoint"]))

dataout["a0r"] = 3.07e-3
dataout["a1r"] = 0.0
dataout["a0s"] = 0.0
dataout["a1s"] = 1.5

dsout = xr.Dataset(dataout)
print(dsout)

dsout.to_netcdf(os.path.join(LOOKUP_DIR, "radsw_cldprtb_data.nc"))
