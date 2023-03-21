#!/usr/bin/env python
"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

filepath = "/home/man/Documents/ER_Bernoulli/build/"

class LayeredNetworkGraph(object):

    def __init__(self, cluster_graphs,  graphs, node_labels=None, layout=nx.spring_layout, ax=None, y_offset=0):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """
        self.y_offset = y_offset
        self.Gc = cluster_graphs
        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])


    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.
        # TODO: implement FR in 3D with layer constraints.

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        N_pop_0 = self.Gc[0].number_of_nodes()
        pos_0 = self.layout(self.Gc[0], *args, **kwargs)
        pos_1 = self.layout(self.Gc[1], *args, **kwargs)
        pos = dict()
        for i in range(N_pop_0):
            pos[i] = pos_0[i]
        for i in range(N_pop_0, N_pop_0 + self.Gc[1].number_of_nodes()):
            pos[i] = (pos_1[i - N_pop_0][0], pos_1[i - N_pop_0][1])

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})

        # pos = self.layout(self.graphs[0], k=100, *args, **kwargs)
        # # pos = self.layout(composition, *args, **kwargs)

        # self.node_positions = dict()
        # for z, g in enumerate(self.graphs):
        #     self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})
        # self.node_positions = dict()
        # for z, g in enumerate(self.graphs):
        #     pos = self.layout(g, scale=2,*args, **kwargs)
        #     self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})


    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        #add y offset
        y = [y[i] + self.y_offset for i in range(len(y))]
        self.ax.scatter(x, y, z, *args, **kwargs)


    def draw_edges(self, edges, alphas, *args, **kwargs):
        for z, (G, a) in enumerate(zip(self.graphs, alphas)):
            segment = [(self.node_positions[(source,z)], self.node_positions[(target,z)]) for source, target in G.edges()]
            #add y offset
            segment = [(self.node_positions[(source,z)], (self.node_positions[(target,z)][0], self.node_positions[(target,z)][1] + self.y_offset, self.node_positions[(target,z)][2])) for source, target in G.edges()]
            coll = Line3DCollection(segment, alpha=a, *args, **kwargs)
            self.ax.add_collection3d(coll)


    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)


    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        ymin = ymin + self.y_offset
        ymax = ymax + self.y_offset
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)


    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)


    def draw(self, color='white', alpha=1, plane_alpha=0.2, plane_color='gray', draw_node_labels=False, edge_alpha=0.2):

        alphas = np.linspace(1, 0.1, self.total_layers)
        self.draw_edges(self.edges_within_layers,  alphas, color='k', linestyle='-', zorder=2)
        # self.draw_edges(self.edges_between_layers, color='k', alpha=edge_alpha, linestyle='--', zorder=2)
        for z in range(self.total_layers):
            self.draw_plane(z, alpha=alphas[z]*plane_alpha,color=plane_color, zorder=1)
            #increase node edge thickness
            # self.draw_nodes([node for node in self.nodes if node[1]==z], s=300, zorder=3, marker="$\u25EF$", color='k', linewidth=1)
            # self.draw_nodes([node for node in self.nodes if node[1]==z], s=300, zorder=3, color=color, alpha=1)
            self.draw_nodes(self.nodes, s=300, zorder=4, marker="$\u25EF$", color='k', linewidth=1, alpha=alphas[z])
            self.draw_nodes(self.nodes, s=300, zorder=3, color=color, alpha=alphas[z])
            

        if self.node_labels and draw_node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)


if __name__ == '__main__':

    N_pop = 10

    #linear decrease of p_self, increase of p_other
    p_connect = np.linspace(0.1, 0.7, 5)
    # p_connect = np.logspace(-1, 0, 5)
    # p_other = np.linspace(0.05, 0.6, 3)
    G = []
    #create two complete graphs with N_pop

    G0 = nx.complete_graph(int(N_pop))
    #id offset N_pop
    G1 = nx.complete_graph(int(N_pop))
    G1 = nx.relabel_nodes(G1, {k: k + N_pop for k in G1.nodes()})

    G0_pos = nx.spring_layout(G0)
    G1_pos = nx.spring_layout(G1)
    #combine and add id offset
    #concatenate G0_pos to G1_pos
    #offset G1_pos keys by N_pop
    #offset x by width of G0
    G1_pos = {k: (v[0] + 5, v[1]) for k, v in G1_pos.items()}


    G_pos = {**G0_pos, **G1_pos}

    G_cons = [[nx.erdos_renyi_graph(N_pop*2, p) for i in range(5)] for p in p_connect] 
    #compose with
    G_complete = nx.compose(G0, G1)

    Gs = [[nx.compose(G_complete, Gc) for Gc in Gcp] for Gcp in G_cons]

    #draw all Gs to svg
    for i, Gp in enumerate(Gs):
        for j, G in enumerate(Gp):
            pos=nx.kamada_kawai_layout(G)
            # nx.draw(G, with_labels=False, node_color='white', node_size=300, font_size=8, width=0.5, edge_color='black', pos=nx.spring_layout(G))
            
            #draw without border

            code = nx.to_latex(G, G_pos)

            #code to file
            codepath = filepath + 'G_{i}_{j}.tex'.format(i=i, j=j)
            with open(codepath, 'w') as f:
                f.write(code)


            # nx.draw_networkx_nodes(G, pos=G_pos, node_color='white', node_size=300, edgecolors='black')
            # nx.draw_networkx_edges(G, pos=G_pos, width=0.5, edge_color='black')
            # nx.draw_networkx_edges(G1, pos=G1_pos, width=0.5, edge_color='black')

            # #remove all figure borders
            # plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['left'].set_visible(False)
            # plt.gca().spines['bottom'].set_visible(False)
            # plt.gca().spines['top'].set_visible(False)


            # # plt.savefig(filepath + 'G_{i}_{j}.svg'.format(i=i, j=j))
            # plt.savefig(filepath + 'G_{i}_{j}.pdf'.format(i=i, j=j))
            
            # plt.close()

    # node_labels = {nn : str(nn) for nn in range(len(G)*N_pop*2)}

    # yspace = 10

    # # initialise figure and plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Layer_0 = LayeredNetworkGraph([G0, G1], Gs[0], node_labels=node_labels, ax=ax, layout=nx.spring_layout, y_offset=0)
    # y_extent = Layer_0.get_extent()[-1][-1]
    # Layer_1 = LayeredNetworkGraph([G0, G1], Gs[1], node_labels=node_labels, ax=ax, layout=nx.spring_layout, y_offset=y_extent + yspace)
    # ax.set_axis_off()
    #create a 3d gray box around the graphs

    #rotate view
    # ax.view_init(azim=0, elev=90)
    # plt.show()
    #import bpy
    # import svg in blender using bpy


