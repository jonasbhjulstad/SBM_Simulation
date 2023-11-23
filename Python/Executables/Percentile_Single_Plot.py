from SBM_Routines.Path_Config import *
from SBM_Routines.SBM_Optimization import *
from Database.Simulation_Tables import *
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import os
import json






if __name__ == '__main__':
    fig, ax = plt.subplots(3,1)
    engine = create_engine("postgresql://postgres:postgres@localhost:5432")

    sim_type = "/Excitation/"
    #json load sim_params
    plot_SIR_percentile_trajectory(0,0, ax, engine)
    # plot_LS_predictions(d_dir, ax, sim_params["Nt"], sim_params["N_sims"])
    [x.grid() for x in ax]
    plt.show()
