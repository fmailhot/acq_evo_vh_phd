#!/usr/bin/python
import sys
import numpy as np
import matplotlib as mpl
from glob import glob

mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
import pylab as plt
#params = {#'backend': 'ps',
#            'legend.fontsize': 14,
#            'xtick.labelsize':14,
#            'ytick.labelsize':14,
#            'text.usetex': True}
#plt.rcParams.update(params)

if sys.argv[1] == "trans":
    title = "Transparent neutrality"
else:
    title = "Opaque neutrality"

for i in zip(range(221, 225), sorted(glob("trans/*1000*"))):
    plt.subplot(i[0])
    for f in glob(i[1] + "/F2*"):
        plt.plot(np.loadtxt(f), "o", markersize=5,
                 label="Learner " + f.split("_")[-1].split(".")[0])
    plt.legend(numpoints=1)
    if "vp" in i[1]:
        subtit = "Vertical flow with peers"
    elif "op" in i[1]:
        subtit = "Oblique flow with peers"
    elif "o" in i[1]:
        subtit = "Oblique flow"
    else:
        subtit = "Vertical flow"
    plt.title(subtit)

plt.suptitle(title, fontsize=16, weight="bold")
plt.show()
