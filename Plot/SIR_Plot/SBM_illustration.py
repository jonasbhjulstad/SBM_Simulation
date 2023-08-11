import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
fpath = "/home/man/Documents/Sycl_Graph_Old/build/"
N_communities = 10
N_pop_cluster = 30
G = nx.planted_partition_graph(N_communities, N_pop_cluster, 1.0, 1e-3, seed=42)

fig, ax = plt.subplots()

pos = nx.spring_layout(G)
#draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color='w', edgecolors='k')
#draw edges
nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.5)




#remove axis
ax.axis('off')
#remove ticks
ax.set_xticks([])
ax.set_yticks([])
#remove frame
ax.set_frame_on(False)

fig.savefig(fpath + "big_SBM.svg", format="svg")




#find N_communities in G
clusters = [list(G.nodes)[N_pop_cluster*i:N_pop_cluster*(i+1)] for i in range(N_communities)]
#find center position of clusters
pos_clusters = [np.mean([pos[i] for i in cluster], axis=0) for cluster in clusters]
N_nodes = len(clusters)
#Create self-looped complete graph
G2 = nx.complete_graph(N_nodes)
G_self = nx.Graph()
G_self.add_nodes_from(G2.nodes)
#draw nodes at pos_clusters
fig2, ax2 = plt.subplots()
edgelist = [(i,i) for i in range(N_communities)]
G_self.add_edges_from(edgelist)
nx.draw_networkx_nodes(G2, pos_clusters, ax=ax2, node_color='w', edgecolors='k')
#draw edges
nx.draw_networkx_edges(G2, pos_clusters, ax=ax2, width=0.3, alpha=0.5)
#draw self-loops

nx.draw_networkx_edges(G_self, pos_clusters, ax=ax2,width=0.8, edgelist=edgelist, arrowstyle="<|-")
#remove axis
ax2.axis('off')
#remove ticks
ax2.set_xticks([])
ax2.set_yticks([])
#remove frame
ax2.set_frame_on(False)
fig2.savefig(fpath + "weighted_SBM.svg", format="svg")

plt.show()
