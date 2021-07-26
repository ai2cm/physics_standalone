import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
import os

from radupdate import radupdate
from radlw_param import ntbl
from util import compare_data

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data'
ddir2 = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")
serializer2 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Serialized_rank0")
savepoints = serializer.savepoint_list()
savepoints2 = serializer2.savepoint_list()

print(savepoints2)

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

solvars = ['slag', 'sdec', 'cdec', 'solcon']
aervars = ['kprfg', 'idxcg', 'cmixg', 'denng', 'ivolae']
gasvars = ['co2vmr_sav', 'gco2cyc']

soldict_val = dict()
for var in solvars:
    soldict_val[var] = serializer2.read(var, savepoints2[-3])

aerdict_val = dict()
for var in aervars:
    aerdict_val[var] = serializer2.read(var, savepoints2[-2])

gasdict_val = dict()
for var in gasvars:
    gasdict_val[var] = serializer2.read(var, savepoints2[-1])

compare_data(soldict, soldict_val)
compare_data(aerdict, aerdict_val)
compare_data(gasdict, gasdict_val)