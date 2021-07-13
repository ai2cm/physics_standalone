import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
import numpy as np
import os
import xarray as xr

from rad_initialize import rad_initialize
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

invars = ['si', 'levr', 'ictm', 'isol', 'ico2', 'iaer', 'ialb', 'iems', 'ntcw',
          'num_p2d', 'num_p3d', 'npdf3d', 'ntoz', 'iovr_sw', 'iovr_lw',
          'isubc_sw', 'isubc_lw', 'icliq_sw', 'crick_proof', 'ccnorm',
          'imp_physics', 'norad_precip', 'idate', 'iflip', 'me']

indict = dict()

for var in invars:
    if var != 'levr' and var != 'me':
        indict[var] = serializer.read(var, savepoints[0])
    elif var == 'levr':
        indict[var] = serializer.read(var, savepoints[2])

indict['me'] = 0
indict['exp_tbl'] = np.zeros((ntbl+1))
indict['tau_tbl'] = np.zeros((ntbl+1))
indict['tfn_tbl'] = np.zeros((ntbl+1))

rad_initialize(indict)

