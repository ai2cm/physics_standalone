import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
import os

from radupdate import radupdate
from radlw_param import ntbl
from util import compare_data
from config import *

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

import serialbox as ser

ddir = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/data/LW"
ddir2 = "/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")
serializer2 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Serialized_rank0")
serializer3 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Init_rank0")
savepoints = serializer.savepoint_list()
savepoints2 = serializer2.savepoint_list()
savepoints3 = serializer3.savepoint_list()

invars = ["idat", "jdat", "fhswr", "dtf", "lsswr", "slag", "sdec", "cdec", "solcon"]

flgvars = ["ictm", "isol", "ntoz", "ico2", "iaer"]

indict = dict()
indict["me"] = 0
indict["month0"] = 0
indict["iyear0"] = 0
indict["monthd"] = 0
indict["kyrsav"] = 0
indict["loz1st"] = True
indict["kyrstr"] = 1
indict["kyrend"] = 1
indict["iyr_sav"] = 0

for var in invars:
    indict[var] = serializer.read(var, serializer.savepoint["rad-update"])

for var in flgvars:
    indict[var] = serializer.read(var, serializer.savepoint["rad-initialize"])

print(f"ntoz = {indict['ntoz']}")

soldict, aerdict, gasdict, loz1st = radupdate(indict)

solvars = ["slag", "sdec", "cdec", "solcon"]
aervars = ["kprfg", "idxcg", "cmixg", "denng", "ivolae"]
gasvars = ["co2vmr_sav", "gco2cyc"]

print(savepoints3)

soldict_val = dict()
for var in solvars:
    soldict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_sol_update_out000000"]
    )

aerdict_val = dict()
for var in aervars:
    aerdict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_aer_update_out000000"]
    )

gasdict_val = dict()
for var in gasvars:
    gasdict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_gas_update_out000000"]
    )

print(f"isolar = {indict['isol']}")
print(f"Python = {soldict['solcon']}")
print(f"Fortran = {soldict_val['solcon']}")

compare_data(soldict, soldict_val)
compare_data(aerdict, aerdict_val)
compare_data(gasdict, gasdict_val)
