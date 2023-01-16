import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

fname = "/home/man/Documents/Sycl_Graph/data/SIR_sim/traj.csv"
#read df without header
df = pd.read_csv(fname, header=None)
#plot
t = np.arange(0, len(df[0]))
fig, ax = plt.subplots()
ax.plot(t, df[0], label='S')
ax.plot(t, df[1], label='I')
ax.plot(t, df[2], label='R')
ax.legend()

plt.show()
