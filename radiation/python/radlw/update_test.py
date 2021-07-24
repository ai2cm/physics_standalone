import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
import numpy as np
import os
import xarray as xr

from radupdate import radupdate
from radlw_param import ntbl

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")
savepoints = serializer.savepoint_list()

print(savepoints[1])

invars = ['idat', 'jdat', 'fhswr', 'dtf', 'lsswr',
          'slag', 'sdec', 'cdec', 'solcon']

flgvars = ['ictm', 'isol', 'ntoz', 'ico2', 'iaer']

indict = dict()
indict['me'] = 0
indict['month0'] = 0
indict['iyear0'] = 0
indict['monthd'] = 0
indict['kyrsav'] = 0
indict['loz1st'] = True
indict['kyrstr'] = 1
indict['kyrend'] = 1
indict['iyr_sav'] = 0

for var in invars:
    indict[var] = serializer.read(var, savepoints[1])

for var in flgvars:
    indict[var] = serializer.read(var, savepoints[0])

print(f"ntoz = {indict['ntoz']}")

soldict, aerdict, gasdict, loz1st = radupdate(indict)