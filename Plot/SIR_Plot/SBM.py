import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

trajname = "/home/man/Documents/Sycl_Graph_Old/data/SIR_sim/SBM_traj.csv"
groupname = "/home/man/Documents/Sycl_Graph_Old/data/SIR_sim/SBM_group_traj.csv"
#read df without header
df = pd.read_csv(trajname, header=None)
df2 = pd.read_csv(groupname, header=None)
#plot
t = np.arange(0, len(df[0]))
fig, ax = plt.subplots()
ax.plot(t, df[0], label='S')
ax.plot(t, df[1], label='I')
ax.plot(t, df[2], label='R')
ax.legend()

fig2, ax2 = plt.subplots()
df2.plot(ax=ax2)

plt.show()
