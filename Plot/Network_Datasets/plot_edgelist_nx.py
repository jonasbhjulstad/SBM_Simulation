import networkx as nx
import matplotlib.pyplot as plt
filename = "/home/man/Documents/SBM_Simulation/data/Edgelists/merge_graph.csv"
G = nx.read_edgelist(filename, delimiter=',', nodetype=int)
nx.draw(G, with_labels=True)
#print number of nodes and edges
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
plt.show()
