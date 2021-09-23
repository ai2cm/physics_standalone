import sys
import os
import xarray as xr

sys.path.insert(0, "/work/radiation/python")
from config import *

import serialbox as ser

ddir = "saveData"

# refvars = ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"]

# dims = [
#     ("NG16", "MSA16"),
#     ("NG16", "MSB16"),
#     ("NG16", "MSF16"),
#     ("NG16", "MFR16"),
#     ("NG16", "MAF16"),
#     "NG16",
# ]

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "radlw_avplnk")
savepoints = serializer.savepoint_list()

print(savepoints)

totplnk = serializer.read("totplnk", serializer.savepoint["mySavepoint"])

dsout = xr.Dataset({"totplnk": (("nplnk", "nbands"), totplnk)})
print(dsout)
# dataout = dict()
# for n, key in enumerate(refvars):
#    dataout[key] = (dims[n], serializer.read(key, savepoints[0]))

# dsout = xr.Dataset(dataout)
# print(dsout)

dsout.to_netcdf(os.path.join(LOOKUP_DIR, "totplnk.nc"))
