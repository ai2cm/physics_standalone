from utility import *
from utility.serialization import *

if STENCILS == "merged":
    from microphys.drivers.gfdl_cloud_microphys_gt4py import (
        gfdl_cloud_microphys_driver_merged as gfdl_cloud_microphys_driver,
    )
else:
    from microphys.drivers.gfdl_cloud_microphys_gt4py import (
        gfdl_cloud_microphys_driver_split as gfdl_cloud_microphys_driver,
    )
from microphys.drivers.gfdl_cloud_microphys_gt4py import gfdl_cloud_microphys_init
import time as tm

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

ser_count_max = 11
num_tiles = 6

gfdl_cloud_microphys_init()
timings = {
    "warm_rain_1_call": [0.0],
    "warm_rain_1_run": [0.0],
    "sedimentation_call": [0.0],
    "sedimentation_run": [0.0],
    "warm_rain_2_call": [0.0],
    "warm_rain_2_run": [0.0],
    "icloud_call": [0.0],
    "icloud_run": [0.0],
    "main_loop_call": [0.0],
    "main_loop_run": [0.0],
    "driver": [0.0],
    "tot": 0.0,
}
i_e = None
k_e = None
ser_count = 0
for tile in range(1):
    # Read serialized data
    input_data = read_data("../data", tile, ser_count, True)

    # Get original iie and kke
    i_e = input_data["iie"]
    k_e = input_data["kke"]

    for n in range(REPS + 1):
        # Scale the dataset gridsize
        scaled_data = scale_dataset_to_N(input_data, CN)
        scaled_data = numpy_dict_to_gt4py_dict(scaled_data)

        # Start driver timer
        if n > 0:
            t_driver_start = tm.perf_counter()

        # Main algorithm
        (
            qi,
            qs,
            qv_dt,
            ql_dt,
            qr_dt,
            qi_dt,
            qs_dt,
            qg_dt,
            qa_dt,
            pt_dt,
            w,
            udt,
            vdt,
            rain,
            snow,
            ice,
            graupel,
            refl_10cm,
        ) = gfdl_cloud_microphys_driver(
            scaled_data, False, True, 0, 0, timings, 0, (n > 0)
        )

        if n > 0:
            # Stop driver timer
            t_driver_end = tm.perf_counter()
            timings["driver"][0] += t_driver_end - t_driver_start

print("Scaled data size: ", scaled_data["ql"].shape)
output_file = open(
    "./out/timings_benchmark_{}_{}_C{}.dat".format(BACKEND, STENCILS, CN), "w"
)
for var in timings:
    if var != "tot":
        timings[var][0] /= REPS
if STENCILS != "merged":
    timings["main_loop_call"][0] = (
        timings["warm_rain_1_call"][0]
        + timings["sedimentation_call"][0]
        + timings["warm_rain_2_call"][0]
        + timings["icloud_call"][0]
    )
    timings["main_loop_run"][0] = (
        timings["warm_rain_1_run"][0]
        + timings["sedimentation_run"][0]
        + timings["warm_rain_2_run"][0]
        + timings["icloud_run"][0]
    )
for var in timings:
    if var != "tot":
        output_file.write(str(timings[var][0]) + " ")
        print(var, str(timings[var][0]))
output_file.close()
t_tot_end = tm.perf_counter()
timings["tot"] = t_tot_end - t_tot_start
print("\n>> Total elapsed time: {:.3f} seconds".format(timings["tot"]))