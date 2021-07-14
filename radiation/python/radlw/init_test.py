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
ddir2 = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Generator_rank0")
serializer2 = ser.Serializer(ser.OpenModeKind.Read, ddir2, "Serialized_rank0")
savepoints = serializer.savepoint_list()
savepoints2 = serializer2.savepoint_list()

print(savepoints2)

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

(aer_dict,
 sol_dict,
 gas_dict,
 sfc_dict,
 cld_dict,
 rlw_dict,
 rsw_dict,
 ipsd0) = rad_initialize(indict)

aervars = ['extrhi',
           'scarhi',
           'ssarhi',
           'asyrhi',
           'extstra',
           'extrhd',
           'scarhd',
           'ssarhd',
           'asyrhd',
           'prsref',
           'haer',
           'eirfwv',
           'solfwv']

sfcvars = ['idxems']

solvars = ['solar_fname']

cldvars = ['llyr']

lwvars = ['semiss0',
          'fluxfac',
          'heatfac',
          'exp_tbl',
          'tau_tbl',
          'tfn_tbl']

swvars = ['heatfac', 'exp_tbl']

aerdict_out = dict()
soldict_out = dict()
sfcdict_out = dict()
clddict_out = dict()
lwdict_out = dict()
swdict_out = dict()

def compare_data(data, ref_data, explicit=True, blocking=True):

    wrong = []
    flag = True

    for var in data:

        # Fix indexing for fortran vs python
        if var != 'cline':
            if not np.allclose(
                data[var], ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
            ):

                wrong.append(var)
                flag = False

            else:

                if explicit:
                    print(f"Successfully validated {var}!")

    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag:
            print(f"Output data does not match reference data for field {wrong}!")

for var in solvars:
    print(var)
    soldict_out[var] = serializer2.read(var, savepoints2[0])
    test = [chr(i) for i in soldict_out[var]]
    tmp = ""
    test = tmp.join(test)
    soldict_out[var] = test.strip()

for var in aervars:
    aerdict_out[var] = serializer2.read(var, savepoints2[2])

for var in sfcvars:
    sfcdict_out[var] = serializer2.read(var, savepoints2[3])



for var in cldvars:
    clddict_out[var] = serializer2.read(var, savepoints2[4])

for var in lwvars:
    lwdict_out[var] = serializer2.read(var, savepoints2[5])

for var in swvars:
    swdict_out[var] = serializer2.read(var, savepoints2[6])


# Sol init only outputs a string, which we can't validate
compare_data(aer_dict, aerdict_out)
compare_data(sfc_dict, sfcdict_out) # Good
# Gas init doesn't output anything
compare_data(cld_dict, clddict_out) # Good
compare_data(rsw_dict, swdict_out) # Good
compare_data(rlw_dict, lwdict_out) # Good





