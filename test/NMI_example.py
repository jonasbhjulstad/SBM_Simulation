import json
from collections import Counter
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import sys
import graph_tool.all as gt
import os
from itertools import repeat
# def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
#     args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
#     return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
sys.path.append(Binder_path)
from SIR_SBM import *
Data_dir = Project_root + "/data/SIR_sim/"
Graphs_dir = Data_dir + "Graphs/"

N_tot = 1e3
N_communities = 3
N_pop = int(N_tot/N_communities)

avg_degree = 3

eps = np.linspace(0.5, 3.0, 11)
p_in = lambda p_out: p_out + eps/N_tot
vcm = [[i]*N_pop for i in range(N_communities)]
vcm = [item for sublist in vcm for item in sublist]

Gs = []
for lin, lrs in zip(lbd_in, lbd_rs):
    probs = np.identity(N_communities)*lin*2
    probs[probs==0] = lrs
    Gs.append(next(gt.generate_sbm(vcm, probs, directed=False) for _ in range(len(p_out))))

states = []
pool = mp.Pool(int(mp.cpu_count()/2))
states = starmap_with_kwargs(pool, gt.minimize_blockmodel_dl, zip(Gs), repeat(dict(state_args={'deg_corr': True})))
# for i, g in enumerate(Gs):
    # print("Graph " + str(i))
    # states.append(gt.minimize_blockmodel_dl(g))
labels = [gt.align_partition_labels(list(state.get_state()), vcm) for state in states]

nmis = [gt.mutual_information(vcm, lab, norm=True) for lab in labels]

a = 1


plt.plot(eps, nmis)

plt.show()
