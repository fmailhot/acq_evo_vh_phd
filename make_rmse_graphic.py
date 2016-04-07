#!/usr/bin/python
import sys
import numpy as np
from glob import glob
import matplotlib as mpl
mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
import pylab as plt
params = {'legend.fontsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'text.usetex': True}
           #'backend': 'ps'}

plt.rcParams.update(params)

if len(sys.argv) != 2:
    print "\nUsage: %s <path/to/rmse/graphs\n" % sys.argv[0]
    sys.exit(2)

for f in sorted(glob(sys.argv[1] + "/F2*")):
    plt.plot(np.loadtxt(f), "o", markersize=8,
             label=r"\textbf{Learner " + f.split("_")[-1].split(".")[0] + "}")

plt.xlim(0, 120)
plt.ylim(300, 1000)
plt.xlabel(r"\textbf{Lexicon size/10}", size=16)
plt.ylabel(r"\textbf{RMSE per word}", size=16)
plt.legend(numpoints=1, shadow=True)
xtick_locs = range(0, 121, 10)
plt.xticks(xtick_locs, [r"$\mathbf{%s}$" % x for x in xtick_locs])
ytick_locs = range(300, 1001, 100)
plt.yticks(ytick_locs, [r"$\mathbf{%s}$" % x for x in ytick_locs])
plt.show()
