import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import sys
from itertools import repeat
import os
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
Graphs_dir = Data_dir + "Graphs/"

def get_single_p_entropy(p_out):
    return np.genfromtxt(Graphs_dir + str(p_out)[:4] + "/nmis.csv", delimiter=',')

p_out = np.linspace(0, 1, 21)
entropies = [get_single_p_entropy(po) for po in p_out]
e_std = [np.std(e) for e in entropies]
e_mean = [np.mean(e) for e in entropies]


# sns.violinplot(data=entropies, scale='width', cut=0)
plt.plot(p_out, e_mean)

plt.show()

# f, ax = plt.subplots()
# plt.errorbar(p_out, e_mean, yerr=e_std)


# plt.show()
