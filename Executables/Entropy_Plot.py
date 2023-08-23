import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
Graphs_dir = Data_dir + "Graphs/"

def get_single_p_entropy(p_out):
    return np.genfromtxt(Graphs_dir + str(p_out)[:3] + "/entropies.csv", delimiter=',')

p_out = np.arange(0.0, 1.0, 0.1)
entropies = [get_single_p_entropy(po) for po in p_out]


#plot multiple violins
fig, ax = plt.subplots()
#scale width
sns.violinplot(data=entropies, ax=ax, scale='width')

ax.set_xticklabels(p_out)
ax.set_xlabel("p_out")
ax.set_ylabel("Entropy")
#log y
# ax.set_yscale('log')
#only 2 decimal places on x
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.show()
