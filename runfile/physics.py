#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import sys


def parse_args():
    usage = "usage: python %(prog)s <which_physics> <data_dir> <backend>"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "which_physics",
        type=str,
        action="store",
        help="which physics to run",
    )

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )

    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="which gt4py backend to use",
    )

    parser.add_argument(
        "select_tile",
        type=str,
        action="store",
        help="which tile to validate, None for all tiles",
    )

    parser.add_argument(
        "select_sp",
        type=str,
        action="store",
        help="which savepoint to validate, None for all savepoints",
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
        ind = np.array(
            np.nonzero(~np.isclose(exp_data[key], ref_data[key], equal_nan=True))
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i, exp_data[key][i], ref_data[key][i])
        assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), (
            "Data does not match for field " + key
        )


if __name__ == "__main__":
    args = parse_args()
    print(
        args.which_physics,
        args.data_dir,
        args.backend,
        args.select_tile,
        args.select_sp,
    )
    if args.which_physics == "seaice":
        SEAICE_DIR = "../seaice/python/"
        sys.path.append(SEAICE_DIR)
        import sea_ice_gt4py as phy
        from config import *
    elif args.which_physics == "shalconv":
        SHALCONV_DIR = "../shalconv/python/"
        sys.path.append(SHALCONV_DIR)
        import shalconv.samfshalcnv as phy
        from shalconv.config import *

    sys.path.append(SERIALBOX_DIR + "/python")
    import serialbox as ser

    if (args.select_tile != "None") and (args.select_sp != "None"):
        SELECT_SP = {"tile": int(args.select_tile), "savepoint": args.select_sp}
    else:
        SELECT_SP = None
    timings = {"elapsed_time": 0, "run_time": 0}
    for tile in range(6):
        if SELECT_SP is not None:
            if tile != SELECT_SP["tile"]:
                continue

        serializer = ser.Serializer(
            ser.OpenModeKind.Read, args.data_dir, "Generator_rank" + str(tile)
        )

        savepoints = serializer.savepoint_list()

        isready = False
        for sp in savepoints:

            if SELECT_SP is not None:
                if sp.name != SELECT_SP["savepoint"] and sp.name != SELECT_SP[
                    "savepoint"
                ].replace("-in-", "-out-"):
                    continue

            if sp.name.startswith(prefix + "-in"):

                if isready:
                    raise Exception("out-of-order data enountered: " + sp.name)

                print("> running ", f"tile-{tile}", sp)

                # read serialized input data
                in_data = data_dict_from_var_list(IN_VARS, serializer, sp)
                # run Python version
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
    if (args.select_tile == "None") and (args.select_sp == "None"):
        timings["elapsed_time"] = timings["elapsed_time"] / (6 * len(savepoints))
        timings["run_time"] = timings["run_time"] / (6 * len(savepoints))
    output_file = open("timings_{}_{}.dat".format(args.which_physics, BACKEND), "w")
    output_file.write(str(timings["elapsed_time"]) + " " + str(timings["run_time"]))
    output_file.close()
