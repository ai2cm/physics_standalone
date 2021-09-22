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

ddir = '/Users/AndrewP/Documents/work/lookupconvert/saveData'

refvars = ['absa',
           'absb',
           'selfref',
           'forref',
           'fracrefa',
           'fracrefb']

dims = [('NG16', 'MSA16'),
        ('NG16', 'MSB16'),
        ('NG16', 'MSF16'),
        ('NG16', 'MFR16'),
        ('NG16', 'MAF16'),
        'NG16']

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "radlw_avplnk")
savepoints=serializer.savepoint_list()

print(savepoints)

totplnk = serializer.read('totplnk', savepoints[0])

dsout = xr.Dataset({'totplnk': (('nplnk', 'nbands'), totplnk)})
print(dsout)
#dataout = dict()
#for n, key in enumerate(refvars):
#    dataout[key] = (dims[n], serializer.read(key, savepoints[0]))

#dsout = xr.Dataset(dataout)
#print(dsout)

dsout.to_netcdf('saveData/totplnk.nc')



