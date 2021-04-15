#!/usr/bin/env python3

import os
import sys
import numpy as np
import turb_gt as turb
import time as tm
from utility import *
from utility.serialization import *

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser


# IN_VARS2 = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]


SELECT_SP = None
# SELECT_SP = {"tile": 2, "savepoint": "satmedmfvdif-in-iter2-000000"}


t_tot_start = tm.perf_counter()
print(
    "VERSION: ",
    BACKEND,
    "STENCILS: ",
    STENCILS,
    "RUN MODE: ",
    RUN_MODE,
    "REBUILD: ",
    REBUILD,
    "REPS: ",
    REPS,
    "CN: ",
    CN,
)


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
        d[var] = data
    return d


timings = {"driver": [0.0], "tot": 0.0}
region_timings = np.zeros(9)
ser_count = 18
for tile in range(1):

    # Read serialized data
    serializer = ser.Serializer(
        ser.OpenModeKind.Read, "../fortran/data", "Generator_rank" + str(tile)
    )
    savepoints = serializer.savepoint_list()
    in_data = data_dict_from_var_list(IN_VARS, serializer, savepoints[0])

    # Get original iie and kke
    i_x = in_data["ix"]
    k_m = in_data["km"]
    compare_dict = {}
    for n in range(REPS + 1):
        # Scale the dataset gridsize
        scaled_data = scale_dataset_to_N(in_data, CN)

        # Start driver timer
        if n > 0:
            t_driver_start = tm.perf_counter()
        out_data = turb.run(scaled_data, compare_dict)
        if n > 0:
            # Stop driver timer
            t_driver_end = tm.perf_counter()
            timings["driver"][0] += t_driver_end - t_driver_start
output_file = open(
    "./out/timings_benchmark_{}_{}_C{}.dat".format(BACKEND, STENCILS, CN), "w"
)
for var in timings:
    if var != "tot":
        timings[var][0] /= REPS
for var in timings:
    if var != "tot":
        output_file.write(str(timings[var][0]) + " ")
        print(var, str(timings[var][0]))
output_file.close()
t_tot_end = tm.perf_counter()
timings["tot"] = t_tot_end - t_tot_start
print("\n>> Total elapsed time: {:.3f} seconds".format(timings["tot"]))