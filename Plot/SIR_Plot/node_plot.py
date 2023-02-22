import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pathlib
filedir = pathlib.Path(__file__).parent.absolute()

fname = "/home/man/Documents/Sycl_Graph/data/SIR_sim/traj.csv"
#read df without header
#numpy read csv
traj = np.genfromtxt(fname, delimiter=',')

#plot
plt.plot(traj)
plt.show()
