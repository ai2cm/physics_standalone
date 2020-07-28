import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.loadtxt('results_optimized.dat')
df = pd.DataFrame(data, columns=['iter', 'gridpoints', 'ice_fraction', 'time'])

BACKEND = ['python', 'numpy','gtx86', 'gtcuda']
gp = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
# ii = [1.0, 0.99, 0.95, 0.75, 0.50, 0.25, 0.1, 0.05, 0.01, 0.0]
fp = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]

fig = plt.subplots(figsize=(8, 6))
plt.grid()
ys = []


# select 0.25 ice fraction 
df = df[df['ice_fraction'] == 0.25]
df.time = df.time / 1e3

# define position for boxplot
pos = np.array((32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576))

# define width for boxplot 
width =  pos * 0.3


fig, ax = plt.subplots(figsize=(8,6))
df.boxplot(ax=ax, by =['gridpoints'],positions=pos \
                          ,widths=width,column = ['time'])
plt.title('Boxplot for 25% sea ice fraction with Optimized Fortran')
plt.suptitle("")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('number of grid points')
ax.set_ylabel('elapsed time [s]')
ax.set_xlim(2e1, 2e6)
ax.set_ylim(5e-7, 5e0)
plt.grid()
plt.tight_layout()
fig.savefig("boxplot_fortran_opt.png")
    






