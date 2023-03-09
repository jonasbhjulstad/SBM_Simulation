import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

fname = "/home/man/Documents/Sycl_Graph_T/data/SIR_sim/traj.csv"
#read df without header
#numpy read csv
traj = np.genfromtxt(fname, delimiter=',')


fig, ax = plt.subplots(3)
fig.suptitle('SIR Simulation')
for i in range(len(ax)):
    ax[i].plot(traj[:,i::3])

plt.show()
