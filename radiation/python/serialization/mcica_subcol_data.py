import sys
import numpy as np
import os
import xarray as xr

sys.path.insert(0, "..")
from config import *

import serialbox as ser

ddir = "../../fortran/radsw/dump"
ddir2 = "../../fortran/data/SW"

scheme = "LW"
smallscheme = scheme.lower()

for tile in range(6):
    serializer2 = ser.Serializer(
        ser.OpenModeKind.Read,
        os.path.join(FORTRANDATA_DIR, scheme),
        "Generator_rank" + str(tile),
    )

    if scheme == "SW":
        nday = serializer2.read(
            "nday", serializer2.savepoint[smallscheme + "rad-in-000000"]
        )[0]

        if nday > 0:
            serializer = ser.Serializer(
                ser.OpenModeKind.Read, SW_SERIALIZED_DIR, "Serialized_rank" + str(tile)
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

            for n, sp in enumerate(rnlist):
                tmp = serializer.read("rand2d", sp)
                lat = int(sp.name[-2:]) - 1
                rand2d[lat, :] = tmp
    elif scheme == "LW":
        serializer = ser.Serializer(
            ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Serialized_rank" + str(tile)
        )
        savepoints = serializer.savepoint_list()

        rnlist = list()
        for pt in savepoints:
            if "random_number-output-000000" in pt.name:
                rnlist.append(pt)
                print(pt)

        nlay = 63
        ngptlw = 140
        rand2d = np.zeros((24, nlay * ngptlw))

        for n, sp in enumerate(rnlist):
            tmp = serializer.read("rand2d", sp)
            lat = int(sp.name[-2:]) - 1
            rand2d[lat, :] = tmp

        # rand2d = np.insert(rand2d, 22, np.zeros(nlay*ngptlw), axis=0)
        # rand2d2 = np.insert(rand2d2, 22, np.zeros(nlay*ngptlw), axis=0)

        # print(rand2d-rand2d2)

    ds = xr.Dataset({"rand2d": (("iplon", "n"), rand2d)})

    dout = "../lookupdata/rand2d_tile" + str(tile) + "_" + smallscheme + ".nc"
    print(dout)

    # ds.to_netcdf(dout)
