import sys
import os
Project_root = "/home/man/Documents/ER_Bernoulli_Robust_MPC/"
Binder_path = Project_root + "/build/Binders/"
Data_dir = Project_root + "data/SIR_sim/"
sys.path.append(Binder_path)
from SIR_SBM import *

def get_p_dirs(base_dir):
      p_dirs = []
      for d in os.listdir(base_dir):
         if (os.path.isdir(base_dir + d) and d.startswith("p_out")):
               p_dirs.append(base_dir + d + "/")
      return p_dirs

def get_graph_dirs(base_dir):
      graph_dirs = []
      for d in os.listdir(base_dir):
         if (os.path.isdir(base_dir + d) and d.startswith("Graph")):
               graph_dirs.append(base_dir + d + "/")
      return graph_dirs
