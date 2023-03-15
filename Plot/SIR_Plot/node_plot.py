import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

fname = "/home/man/Documents/Sycl_Graph_Old/data/SIR_sim/traj.csv"
#read df without header
#numpy read csv
traj = np.genfromtxt(fname, delimiter=',')

#3
fix, ax = plt.subplots(3)
#plot
_ = [x.plot(traj[:,i::3]) for i, x in enumerate(ax)]
plt.show()
