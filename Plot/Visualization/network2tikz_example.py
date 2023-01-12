nodes = ['a','b','c','d']
edges = [('a','b'), ('a','c'), ('c','d'),('d','b')]
gender = ['f', 'm', 'f', 'm']
colors = {'m': 'blue', 'f': 'red'}

style = {}
style['node_label'] = ['Alice', 'Bob', 'Claire', 'Dennis']
style['node_color'] = [colors[g] for g in gender]
style['node_opacity'] = .5
style['edge_curved'] = .1

from network2tikz import plot
plot((nodes,edges),'network.tex',**style)