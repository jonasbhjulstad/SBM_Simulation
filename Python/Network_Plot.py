import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

p = .8
N = 20
g = nx.Graph()
# g = nx.newman_watts_strogatz_graph(N, 3, 0.1)
pos = [(np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)) for i  in range(N)]
    
G = nx.newman_watts_strogatz_graph(N, 3, .2, )

fig, ax = plt.subplots(1)


# illustration_model = nx.erdos_renyi_graph(N, p)
nx.draw(G, pos=pos, ax=ax,  width=1.0, style='dashed', edge_color='k', node_color='w', edgecolors='k')
plt.show()
# fig.savefig("Barabasi_Albert_Illustration_{pop}_{pBA}.svg".format(pop=N, pBA=p), format='svg')