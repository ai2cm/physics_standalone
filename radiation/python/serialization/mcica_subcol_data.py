import sys
import numpy as np
import os
import xarray as xr

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

os.environ["DYLD_LIBRARY_PATH"] = "/Users/andrewp/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/andrewp/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/andrewp/Documents/work/physics_standalone/radiation/fortran/radsw/dump"
ddir2 = "/Users/andrewp/Documents/work/physics_standalone/radiation/fortran/data/SW"

for tile in range(6):
    serializer2 = ser.Serializer(
        ser.OpenModeKind.Read, ddir2, "Generator_rank" + str(tile)
    )

    nday = serializer2.read("nday", serializer2.savepoint["swrad-in-000000"])[0]
    print(f"nday = {nday}")

    if nday > 0:
        serializer = ser.Serializer(
            ser.OpenModeKind.Read, ddir, "Serialized_rank" + str(tile)
        )
        savepoints = serializer.savepoint_list()

        rnlist = list()
        for pt in savepoints:
            if "random_number-output-000000" in pt.name:
                rnlist.append(pt)
                print(pt)

        nlay = 63
        ngptsw = 112
        rand2d = np.zeros((24, nlay * ngptsw))
        rand2d2 = np.zeros((24, nlay * ngptsw))

        for n, sp in enumerate(rnlist):
            tmp = serializer.read("rand2d", sp)
            lat = int(sp.name[-2:]) - 1
            rand2d[lat, :] = tmp

        # rand2d = np.insert(rand2d, 22, np.zeros(nlay*ngptlw), axis=0)
        # rand2d2 = np.insert(rand2d2, 22, np.zeros(nlay*ngptlw), axis=0)

        # print(rand2d-rand2d2)

        ds = xr.Dataset({"rand2d": (("iplon", "n"), rand2d)})

        dout = (
            "/Users/andrewp/Documents/work/physics_standalone/radiation/python/lookupdata/rand2d_tile"
            + str(tile)
            + "_sw.nc"
        )
        print(dout)

        ds.to_netcdf(dout)
