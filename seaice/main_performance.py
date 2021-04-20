#!/usr/bin/env python3

import os
import pickle
import numpy as np
import sea_ice_timer as si_py
import sea_ice_gt4py as si_gt4py

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# BACKEND = ["python", "numpy", "gtx86", "gtcuda"]
BACKEND = ["gtx86"]

IN_VARS = [
    "im",
    "km",
    "ps",
    "t1",
    "q1",
    "delt",
    "sfcemis",
    "dlwflx",
    "sfcnsw",
    "sfcdsw",
    "srflag",
    "cm",
    "ch",
    "prsl1",
    "prslki",
    "islimsk",
    "wind",
    "flag_iter",
    "lprnt",
    "ipr",
    "cimin",
    "hice",
    "fice",
    "tice",
    "weasd",
    "tskin",
    "tprcp",
    "stc",
    "ep",
    "snwdph",
    "qsurf",
    "snowmt",
    "gflux",
    "cmm",
    "chh",
    "evap",
    "hflx",
]

OUT_VARS = [
    "hice",
    "fice",
    "tice",
    "weasd",
    "tskin",
    "tprcp",
    "stc",
    "ep",
    "snwdph",
    "qsurf",
    "snowmt",
    "gflux",
    "cmm",
    "chh",
    "evap",
    "hflx",
]

SCALAR_VARS = ["delt", "cimin", "im", "km"]

TWOD_VARS = ["stc"]
BOOL_VARS = ["flag_iter"]
INT_VARS = ["islimsk"]
ITER = 10

GP = [128 * 128]
FP = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]


def save_obj(obj, name):
    with open("obj/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open("obj/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def init_dict(num_gridp, frac_gridp):
    d = {}
    sea_ice_point = load_obj("sea_ice_point")
    land_point = load_obj("land_point")

    num_sea_ice = int(num_gridp * frac_gridp)
    for var in IN_VARS:
        if var in SCALAR_VARS:
            d[var] = sea_ice_point[var]
        elif var in TWOD_VARS:
            d[var] = np.empty((num_gridp, 4))
            d[var][:num_sea_ice, :] = sea_ice_point[var]
            d[var][num_sea_ice:, :] = land_point[var]
        elif var in BOOL_VARS:
            d[var] = np.ones(num_gridp, dtype=bool)
            d[var][:num_sea_ice] = sea_ice_point[var]
            d[var][num_sea_ice:] = land_point[var]
        elif var in INT_VARS:
            d[var] = np.ones(num_gridp, dtype=np.int32)
            d[var][:num_sea_ice] = sea_ice_point[var]
            d[var][num_sea_ice:] = land_point[var]
        else:
            d[var] = np.empty(num_gridp)
            d[var][:num_sea_ice] = sea_ice_point[var]
            d[var][num_sea_ice:] = land_point[var]

    d["im"] = num_gridp

    return d


if os.path.exists("time.p"):
    time = pickle.load(open("time.p", "rb"))
else:
    time = {}
flag = False
for implement in BACKEND:
    print("Implementation: ", implement)
    if implement not in time:
        time[implement] = {}
    for frac in [1]:
        if float(frac) not in time[implement]:
            time[implement][float(frac)] = {}
        for grid_points in GP:
            if flag:  # float(grid_points) in time[implement][float(frac)]:
                print(
                    "Skipping ",
                    grid_points,
                    "gridpoints with ",
                    100 * frac,
                    "% sea_ice",
                )
                if grid_points == 32768 * 64 and frac == 1:
                    print(
                        f"DEBUG {implement} {grid_points} {frac} {time[implement][frac][float(grid_points)]}"
                    )
            else:
                print(
                    "Running ", grid_points, "gridpoints with ", 100 * frac, "% sea_ice"
                )
                elapsed_time = np.empty(ITER)
                for i in range(ITER):
                    if implement == "gtcuda" and grid_points >= 32768 * 128:
                        elapsed_time[i] = np.nan
                    else:
                        in_dict = init_dict(grid_points, frac)
                        if implement == "python":
                            elapsed_time[i] = 1.0
                            out_data, elapsed_time[i] = si_py.run(in_dict)
                        else:
                            out_data, elapsed_time[i] = si_gt4py.run(
                                in_dict, backend=implement
                            )
                time[implement][frac][float(grid_points)] = np.median(elapsed_time)
                print(implement, frac, float(grid_points), np.median(elapsed_time))
                pickle.dump(time, open("time.p", "wb"))


# for implement in BACKEND:
#     sorted_time = sorted(time[implement].items())
#     ys = []
#     plt.figure(figsize=(8,6))
#     for key in sorted_time:
#         lists = sorted(key[1].items())
#         x, y = zip(*lists)
#         ys.append(y)
#     colors = cm.rainbow(np.linspace(0,1, len(ys)))
#     for y,c,k in zip(ys, colors, sorted_time):
#         plt.scatter(x, y, label=str(k[0]), s=8, color=c)

#     plt.title('Performance Overview for ' + implement.capitalize())
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('number of grid points')
#     plt.ylabel('elapsed time [s]')
#     plt.xlim(1.e1, 1.e7)
#     plt.ylim(3.e-6, 1.e1)
#     plt.grid()
#     plt.legend(title='fraction of sea ice points', ncol=3, loc=2)
#     plt.savefig("perf_" + implement + ".png")
