from config import *
from utility.serialization import *

if USE_GT4PY:

    if STENCILS == "merged":
        from microphys.drivers.gfdl_cloud_microphys_gt4py import (
            gfdl_cloud_microphys_driver_merged as gfdl_cloud_microphys_driver,
        )
    else:
        from microphys.drivers.gfdl_cloud_microphys_gt4py import (
            run as gfdl_cloud_microphys_driver,
        )

    from microphys.drivers.gfdl_cloud_microphys_gt4py import gfdl_cloud_microphys_init

else:

    from microphys.drivers.gfdl_cloud_microphys_py import *

import time as tm


if __name__ == "__main__":

    # Start total timer
    t_tot_start = tm.perf_counter()

    ser_count_max = 11
    num_tiles = 6

    if NORMAL:
        # factors = [ (1./12., 12), (1./6.,   12), (0.25,    12), (1./3., 12),
        #             (5./12., 12), (0.5,     12), (7./12.,  12), (2./3., 12),
        #             (0.75,   12), (10./12., 12), (11./12., 12), (1.,    12) ]
        factors = [(1.0 / 12.0, 12), (1.0 / 6.0, 12), (0.25, 12)]
    elif WEAK:
        factors = [(1.0, N_TH)]
    elif STRONG:
        factors = [(1.0, 12)]
    else:
        factors = [(1.0, 1)]

    # Initialization
    gfdl_cloud_microphys_init()

    if PROGRESS_MODE:
        print("### Current progress ###")

    # Initialize timers
    n_timings = len(factors)
    timings = {
        "warm_rain_1_call": np.zeros(n_timings),
        "warm_rain_1_run": np.zeros(n_timings),
        "sedimentation_call": np.zeros(n_timings),
        "sedimentation_run": np.zeros(n_timings),
        "warm_rain_2_call": np.zeros(n_timings),
        "warm_rain_2_run": np.zeros(n_timings),
        "icloud_call": np.zeros(n_timings),
        "icloud_run": np.zeros(n_timings),
        "main_loop_call": np.zeros(n_timings),
        "main_loop_run": np.zeros(n_timings),
        "driver": np.zeros(n_timings),
        "tot": 0.0,
    }

    i_e = None
    k_e = None

    for tile in range(1):
        for ser_count in range(1):

            if PROGRESS_MODE:
                percentage = (tile * ser_count_max + ser_count) / (
                    num_tiles * ser_count_max
                )
                print(
                    "{:<23}".format("|" + int(22 * percentage) * "=")
                    + f"| {int(100 * percentage)}%    ",
                    end="",
                )
            else:
                print("##################################")
                print(f"# Comparing tile {tile}, ser_count {ser_count}")
                print("##################################")

            # Read serialized data
            input_data = read_data(DATA_PATH, tile, ser_count, True)
            output_data = read_data(DATA_REF_PATH, tile, ser_count, False)

            # Get original iie and kke
            i_e = input_data["iie"]
            k_e = input_data["kke"]

            n_iter = 0
            for factor in factors:

                for n in range(REPS + 1):

                    # Scale the dataset gridsize
                    scaled_data = scale_dataset(input_data, factor)

                    if USE_GT4PY:
                        scaled_data = numpy_dict_to_gt4py_dict(scaled_data)

                    # Start driver timer
                    if BENCHMARK and (n > 0):
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
                        scaled_data, False, True, 0, 0, timings, n_iter, (n > 0)
                    )

                    if BENCHMARK and (n > 0):

                        # Stop driver timer
                        t_driver_end = tm.perf_counter()

                        timings["driver"][n_iter] += t_driver_end - t_driver_start

                n_iter += 1

            if VALIDATION:

                if USE_GT4PY:

                    data = view_gt4py_storage(
                        {
                            "qi": qi[:, :, :],
                            "qs": qs[:, :, :],
                            "qv_dt": qv_dt[:, :, :],
                            "ql_dt": ql_dt[:, :, :],
                            "qr_dt": qr_dt[:, :, :],
                            "qi_dt": qi_dt[:, :, :],
                            "qs_dt": qs_dt[:, :, :],
                            "qg_dt": qg_dt[:, :, :],
                            "qa_dt": qa_dt[:, :, :],
                            "pt_dt": pt_dt[:, :, :],
                            "w": w[:, :, :],
                            "udt": udt[:, :, :],
                            "vdt": vdt[:, :, :],
                            "rain": rain[:, :, 0],
                            "snow": snow[:, :, 0],
                            "ice": ice[:, :, 0],
                            "graupel": graupel[:, :, 0],
                            "refl_10cm": refl_10cm[:, :, :],
                        }
                    )
                else:

                    data = {
                        "qi": qi[:, :, :],
                        "qs": qs[:, :, :],
                        "qv_dt": qv_dt[:, :, :],
                        "ql_dt": ql_dt[:, :, :],
                        "qr_dt": qr_dt[:, :, :],
                        "qi_dt": qi_dt[:, :, :],
                        "qs_dt": qs_dt[:, :, :],
                        "qg_dt": qg_dt[:, :, :],
                        "qa_dt": qa_dt[:, :, :],
                        "pt_dt": pt_dt[:, :, :],
                        "w": w[:, :, :],
                        "udt": udt[:, :, :],
                        "vdt": vdt[:, :, :],
                        "rain": rain[:, :],
                        "snow": snow[:, :],
                        "ice": ice[:, :],
                        "graupel": graupel[:, :],
                        "refl_10cm": refl_10cm[:, :, :],
                    }

                ref_data = {
                    "qi": output_data["qi"],
                    "qs": output_data["qs"],
                    "qv_dt": output_data["qv_dt"],
                    "ql_dt": output_data["ql_dt"],
                    "qr_dt": output_data["qr_dt"],
                    "qi_dt": output_data["qi_dt"],
                    "qs_dt": output_data["qs_dt"],
                    "qg_dt": output_data["qg_dt"],
                    "qa_dt": output_data["qa_dt"],
                    "pt_dt": output_data["pt_dt"],
                    "w": output_data["w"],
                    "udt": output_data["udt"],
                    "vdt": output_data["vdt"],
                    "rain": output_data["rain"],
                    "snow": output_data["snow"],
                    "ice": output_data["ice"],
                    "graupel": output_data["graupel"],
                    "refl_10cm": output_data["refl_10cm"],
                }

                compare_data(
                    data, ref_data, explicit=(not PROGRESS_MODE), blocking=True
                )

            if PROGRESS_MODE:
                print("\r", end="")

    if PROGRESS_MODE:
        print("|" + 22 * "=" + "| 100% Success!")
    else:
        print("Success!")

    if BENCHMARK:

        # Output running timings
        if NORMAL:
            output_file = open(
                "./out/timings_benchmark_{}_{}{}.dat".format(
                    BACKEND, STENCILS, "_debug" if DEBUG_MODE else ""
                ),
                "w",
            )
        elif WEAK:
            output_file = open(
                "./out/timings_weak_{}_{}{}.dat".format(
                    BACKEND, STENCILS, "_debug" if DEBUG_MODE else ""
                ),
                "a",
            )
        elif STRONG:
            output_file = open(
                "./out/timings_strong_{}_{}{}.dat".format(
                    BACKEND, STENCILS, "_debug" if DEBUG_MODE else ""
                ),
                "a",
            )

        n_iter = 0
        for factor in factors:

            i_dim = DTYPE_INT(i_e * factor[0] * factor[1])

            for var in timings:
                if var != "tot":
                    timings[var][n_iter] /= REPS

            print("\nFor gridsize ({}x1x{}):".format(i_dim, k_e))

            if USE_GT4PY:

                if STENCILS != "merged":

                    timings["main_loop_call"][n_iter] = (
                        timings["warm_rain_1_call"][n_iter]
                        + timings["sedimentation_call"][n_iter]
                        + timings["warm_rain_2_call"][n_iter]
                        + timings["icloud_call"][n_iter]
                    )
                    timings["main_loop_run"][n_iter] = (
                        timings["warm_rain_1_run"][n_iter]
                        + timings["sedimentation_run"][n_iter]
                        + timings["warm_rain_2_run"][n_iter]
                        + timings["icloud_run"][n_iter]
                    )

                    print(
                        "> Warm rain processes (1st pass) => Call: {:.3f} seconds  ---  Run: {:.3f} seconds".format(
                            timings["warm_rain_1_call"][n_iter],
                            timings["warm_rain_1_run"][n_iter],
                        )
                    )
                    print(
                        "> Sedimentation                  => Call: {:.3f} seconds  ---  Run: {:.3f} seconds".format(
                            timings["sedimentation_call"][n_iter],
                            timings["sedimentation_run"][n_iter],
                        )
                    )
                    print(
                        "> Warm rain processes (2nd pass) => Call: {:.3f} seconds  ---  Run: {:.3f} seconds".format(
                            timings["warm_rain_2_call"][n_iter],
                            timings["warm_rain_2_run"][n_iter],
                        )
                    )
                    print(
                        "> Ice-phase microphysics         => Call: {:.3f} seconds  ---  Run: {:.3f} seconds".format(
                            timings["icloud_call"][n_iter],
                            timings["icloud_run"][n_iter],
                        )
                    )

                print(
                    "> Main loop                      => Call: {:.3f} seconds  ---  Run: {:.3f} seconds".format(
                        timings["main_loop_call"][n_iter],
                        timings["main_loop_run"][n_iter],
                    )
                )

            else:

                timings["main_loop_run"][n_iter] = (
                    timings["warm_rain_1_run"][n_iter]
                    + timings["sedimentation_run"][n_iter]
                    + timings["warm_rain_2_run"][n_iter]
                    + timings["icloud_run"][n_iter]
                )

                print(
                    "> Warm rain processes (1st pass) => {:.3f} seconds".format(
                        timings["warm_rain_1_run"][n_iter]
                    )
                )
                print(
                    "> Sedimentation                  => {:.3f} seconds".format(
                        timings["sedimentation_run"][n_iter]
                    )
                )
                print(
                    "> Warm rain processes (2nd pass) => {:.3f} seconds".format(
                        timings["warm_rain_2_run"][n_iter]
                    )
                )
                print(
                    "> Ice-phase microphysics         => {:.3f} seconds".format(
                        timings["icloud_run"][n_iter]
                    )
                )
                print(
                    "> Main loop                      => {:.3f} seconds".format(
                        timings["main_loop_run"][n_iter]
                    )
                )

            print(
                "> Whole driver                   => {:.3f} seconds".format(
                    timings["driver"][n_iter]
                )
            )

            output_file.write(
                str(factor[0])
                + " "
                + str(factor[1])
                + " "
                + str(i_dim * k_e)
                + " "
                + str(N_TH)
                + " "
            )
            for var in timings:
                if var != "tot":
                    output_file.write(str(timings[var][n_iter]) + " ")

            output_file.write("\n")

            n_iter += 1

        output_file.close()

    # Stop total timer
    t_tot_end = tm.perf_counter()

    timings["tot"] = t_tot_end - t_tot_start

    print("\n>> Total elapsed time: {:.3f} seconds".format(timings["tot"]))
