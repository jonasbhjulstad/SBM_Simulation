import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

fname = "/home/man/Documents/Bernoulli_MC/data/SIR_sim/traj.csv"
#read df without header
df = pd.read_csv(fname, header=None)
#plot
fig, ax = plt.subplots()
ax.plot(df[0], df[0], label='S')
ax.plot(df[0], df[1], label='I')
ax.plot(df[0], df[2], label='R')

plt.show()
