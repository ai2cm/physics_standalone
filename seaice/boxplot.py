#!/usr/bin/env python3

import pickle
import numpy as np
import sea_ice_timer as si_py
import sea_ice_gt4py_3sten as si_gt4py
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
BACKEND = ['python', 'numpy','gtx86', 'gtcuda']

IN_VARS = ["im", "km", "ps", "t1", "q1", "delt", "sfcemis", "dlwflx", \
           "sfcnsw", "sfcdsw", "srflag", "cm", "ch", "prsl1", "prslki", \
           "islimsk", "wind", "flag_iter", "lprnt", "ipr", "cimin", \
           "hice", "fice", "tice", "weasd", "tskin", "tprcp", "stc", \
           "ep", "snwdph", "qsurf", "snowmt", "gflux", "cmm", "chh", \
           "evap", "hflx"]

OUT_VARS = ["hice", "fice", "tice", "weasd", "tskin", "tprcp", "stc", \
            "ep", "snwdph", "qsurf", "snowmt", "gflux", "cmm", "chh", \
            "evap", "hflx"]

SCALAR_VARS = ["delt", "cimin", 'im', 'km']

TWOD_VARS = ['stc']
BOOL_VARS = ['flag_iter']
INT_VARS = ['islimsk']
ITER = 10

GP = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,  65536, 131072, 262144, 524288, 1048576]
FP = [0., 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]

  

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def init_dict(num_gridp, frac_gridp):
    d = {}
    sea_ice_point = load_obj('sea_ice_point')
    land_point = load_obj('land_point')
    
    num_sea_ice = int(num_gridp*frac_gridp)
    for var in IN_VARS:
        if var in SCALAR_VARS:
            d[var] = sea_ice_point[var]
        elif var in TWOD_VARS:
            d[var] = np.empty((num_gridp, 4))
            d[var][:num_sea_ice,:] = sea_ice_point[var]
            d[var][num_sea_ice:,:] = land_point[var]
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


    d['im'] = num_gridp

    return d

# define dataframe
index = np.arange(0,np.size(BACKEND) * np.size(FP) * np.size(GP) * ITER - 1)  
data_errorbar = pd.DataFrame(columns = ['BACKEND', 'Ice Fraction', 'Gridpoints', 'Median', 'Sigma', 'Iteration', 'Time'], index = index)

# set counter
counter = 0  


for implement in BACKEND:
    print('Implementation: ', implement)
    time = {}

    for frac in FP:
        time[float(frac)] = {}
        for grid_points in GP:
            print('Running ', grid_points, 'gridpoints with ', 100*frac, '% sea_ice')
            elapsed_time = np.empty(ITER)
            for i in range(ITER):
                in_dict = init_dict(grid_points, frac)
                if implement == 'python':
                    out_data, elapsed_time[i] = si_py.run(in_dict)
                else:
                    out_data, elapsed_time[i] = si_gt4py.run(in_dict, backend=implement)
                    
                data_errorbar.loc[counter,'BACKEND'] = implement
                data_errorbar.loc[counter,'Gridpoints'] = grid_points
                data_errorbar.loc[counter,'Ice Fraction'] = frac
                data_errorbar.loc[counter,'Iteration'] = i
                data_errorbar.loc[counter,'Ice Fraction'] = frac
                data_errorbar.loc[counter,'Time'] = elapsed_time[i]
               
                counter = counter + 1
            #time[frac][float(grid_points)] = np.median(elapsed_time)
            
            # calculate standarddeviation 
            #sigma = np.std(elapsed_time)

            # read data into dataframe
            #data_errorbar.loc[counter,'BACKEND'] = implement
            #data_errorbar.loc[counter,'Gridpoints'] = grid_points
            #data_errorbar.loc[counter,'Ice Fraction'] = frac
            #data_errorbar.loc[counter,'Median'] = np.median(elapsed_time) 
            #data_errorbar.loc[counter,'Sigma'] = np.std(elapsed_time)
            
           
            
            
    
    # select backend
    data_errorbar = data_errorbar[data_errorbar.BACKEND == implement]
    # select 0.25 ice fraction 
    data_errorbar = data_errorbar[data_errorbar['Ice Fraction'] == 0.25] 
    
    #plt.figure(figsize=(8,6))
    

    
 
    #plt.legend(title='fraction of sea ice points', ncol=3, loc=2)
    #plt.savefig("perf_errorbars_" + implement + ".png")
    coff = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    array =  np.asarray(coff) * 0.3

    fig, ax = plt.subplots(figsize=(8,6))
    data_errorbar.boxplot(ax=ax, by =['Gridpoints'],positions=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,  65536, 131072, 262144, 524288, 1048576),widths=array,column = ['Time'])
    plt.title('25% Sea Ice Fraction Error for ' + implement.capitalize())
    plt.suptitle("")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of grid points')
    ax.set_ylabel('elapsed time [s]')
    ax.set_xlim(2*1e1, 1*1e5)
    ax.set_ylim(5e-7, 5e0)
    #ax.loglog()
    plt.grid()
    plt.tight_layout()
    fig.savefig("perf_boxplot_" + implement + ".png")
   
data_errorbar.to_csv('output_python.csv',index=False)
