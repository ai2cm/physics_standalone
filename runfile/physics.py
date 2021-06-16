#!/usr/bin/env python3

"""physics.py: Validate a parameterization with a specified GT4Py backend"""

from argparse import ArgumentParser
import numpy as np
import sys
import os


def parse_args():
    usage = "usage: python %(prog)s parameterization backend [--data_dir=path] [--which_tile=selection] [--which_savepoint=name]"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "parameterization",
        type=str,
        action="store",
        help="which parameterization to run",
    )

    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="which gt4py backend to use",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )

    parser.add_argument(
        "--select_tile",
        type=str,
        action="store",
        help="which tile to validate, None for all tiles",
        default="All",
    )

    parser.add_argument(
        "--select_sp",
        type=str,
        action="store",
        help="which savepoint to validate, None for all savepoints",
        default="All",
    )

    parser.add_argument(
        "--verbose",
        action="store_false",
    )

    return parser.parse_args()


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
        d[var] = data
    return d


def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(
        ref_data.keys()
    ), "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        print(key)
        ind = np.array(
            np.nonzero(~np.isclose(exp_data[key], ref_data[key], equal_nan=True))
        )
        if ind.size > 0:
            diff = abs(exp_data[key] - ref_data[key])
            max_diff_ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
            print(
                "FAIL at ",
                key,
                max_diff_ind,
                exp_data[key][max_diff_ind],
                ref_data[key][max_diff_ind],
                "Number of fails:",
                ind.shape[1],
            )
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), (
            "Data does not match for field " + key
        )


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        print(
            args.parameterization,
            args.data_dir,
            args.backend,
            args.select_tile,
            args.select_sp,
        )
    os.environ["BACKEND"] = args.backend
    if args.data_dir is None:
        args.data_dir = "../" + args.parameterization + "/data"

    if args.parameterization == "seaice":
        SEAICE_DIR = "../seaice/python/"
        sys.path.append(SEAICE_DIR)
        from config import *
        import sea_ice_gt4py as phy

    elif args.parameterization == "shalconv":
        SHALCONV_DIR = "../shalconv/python/"
        sys.path.append(SHALCONV_DIR)
        from shalconv.config import *
        import shalconv.samfshalcnv as phy

    elif args.parameterization == "turb":
        TURB_DIR = "../turb/python/"
        sys.path.append(TURB_DIR)
        from config import *
        import turb_gt as phy

    elif args.parameterization == "microph":
        MPH_DIR = "../microph/python/"
        sys.path.append(MPH_DIR)
        from config import *
        import microphys.drivers.gfdl_cloud_microphys_gt4py as phy

    else:
        raise Exception(f"Parameterization {args.parameterization} is not supported")

    sys.path.append(SERIALBOX_DIR + "/python")
    import serialbox as ser

    if (args.select_tile != "All") and (args.select_sp != "All"):
        select_sp = {"tile": int(args.select_tile), "savepoint": args.select_sp}
    else:
        select_sp = None
    timings = {"elapsed_time": 0, "run_time": 0}
    for tile in range(6):
        if select_sp is not None:
            if tile != select_sp["tile"]:
                continue

        serializer = ser.Serializer(
            ser.OpenModeKind.Read, args.data_dir, "Generator_rank" + str(tile)
        )

        savepoints = serializer.savepoint_list()

        isready = False
        for sp in savepoints:

            if select_sp is not None:
                if sp.name != select_sp["savepoint"] and sp.name != select_sp[
                    "savepoint"
                ].replace("-in-", "-out-"):
                    continue

            if sp.name.startswith(prefix + "-in"):

                if isready:
                    raise Exception("out-of-order data enountered: " + sp.name)

                print("> running ", f"tile-{tile}", sp)

                # read serialized input data
                in_data = data_dict_from_var_list(IN_VARS, serializer, sp)

                out_data = phy.run(in_data, timings)

                isready = True

            if sp.name.startswith(prefix + "-out"):

                if not isready:
                    raise Exception("out-of-order data encountered: " + sp.name)

                print("> validating ", f"tile-{tile}", sp)

                # read serialized output data
                ref_data = data_dict_from_var_list(OUT_VARS, serializer, sp)

                # check result
                compare_data(out_data, ref_data)

                isready = False

    print("SUCCESS")

    write_benchmark = False

    if write_benchmark:
        if (args.select_tile == "None") and (args.select_sp == "None"):
            timings["elapsed_time"] = timings["elapsed_time"] / (6 * len(savepoints))
            timings["run_time"] = timings["run_time"] / (6 * len(savepoints))
        output_file = open("timings_{}_{}.dat".format(args.parameterization, BACKEND), "w")
        output_file.write(str(timings["elapsed_time"]) + " " + str(timings["run_time"]))
        output_file.close()
