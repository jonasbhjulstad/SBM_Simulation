import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import sys

CoMix_fname = "/home/man/Documents/Bernoulli_MC/data/Network_Datasets/CoMix/CoMix_fr_contact_common.csv"

if __name__ == '__main__':
    #create a graph from CoMix_fname with part_id and cont_id as nodes
    #and frequency_multi as edge weight
    CoMix_df = pd.read_csv(CoMix_fname)
    G = nx.from_pandas_edgelist(CoMix_df, source='part_id', target='cont_id', edge_attr='frequency_multi')
    #plot
    nx.draw(G, with_labels=True)
    plt.show()

