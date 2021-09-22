import sys
import numpy as np
import os
import xarray as xr

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()

print(savepoints[1])

cline = 'SURFACE EMISSIVITY INDEX, IDM,JDM: 360 180    NOTE: DATA FROM N TO S'
idxems = serializer.read('idxems', savepoints[1])

print(cline)
print(idxems.shape)

NCLINE = 80
IMXEMS = 360
JMXEMS = 180

ds = xr.Dataset({'cline': cline,
                 'idxems': (('IMXEMS', 'JMXEMS'), idxems)})

dout = dout = '/Users/AndrewP/Documents/work/physics_standalone/radiation/python/radlw/semisdata.nc'

print(ds)

ds.to_netcdf(dout)