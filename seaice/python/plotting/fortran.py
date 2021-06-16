#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.loadtxt('results_opt.dat')
df = pd.DataFrame(data, columns=['iter', 'gridpoints', 'ice_fraction', 'time'])

# ii = [1.0, 0.99, 0.95, 0.75, 0.50, 0.25, 0.1, 0.05, 0.01, 0.0]
ii = [0., 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.]

plt.subplots(figsize=(8, 6))
ys = []
for i in ii:
    x = df[df.ice_fraction == i].groupby(df['gridpoints']).median().gridpoints
    yy = df[df.ice_fraction == i].groupby(df['gridpoints']).median().time / 1.e3
    ys.append(yy)
colors = cm.rainbow(np.linspace(0, 1, 10))
for y, c, i in zip(ys, colors, ii):
    plt.scatter(x, y, label=str(i), s=8, color=c)

plt.title('Performance Overview for ' + 'FORTRAN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of grid points')
plt.ylabel('elapsed time [s]')
plt.xlim(1.e1, 1.e7)
plt.ylim(3.e-6, 1.e1)
plt.grid()
plt.legend(title='fraction of sea ice points', ncol=3, loc=2)
plt.savefig("perf_" + 'fortran' + ".png")
