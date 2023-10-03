from SBM_Routines.Path_Config import *
from collections import Counter
from itertools import repeat
import graph_tool.all as gt
import numpy as np


def complete_graph_max_edges(N):
    def n_choose_2(n): return n*(n-1)/2
    return int(n_choose_2(N) + N)


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def remap(G, state, idx):
    community_map = list(state.project_partition(idx, 0))
    count = Counter(community_map)
    N_count_keys = len(count.keys())
    new_keys = list(range(N_count_keys))
    idx_map = {k: v for k, v in zip(count.keys(), new_keys)}
    # replace indices with linspace(0, N_count_keys, N_count_keys)
    for i, c in enumerate(community_map):
        community_map[i] = int(idx_map[c])

    edge_idx_map = []
    for e in state.get_levels()[idx+1].g.get_edges():
        edge_idx_map.append(Edge_t(idx_map[e[0]], idx_map[e[1]]))

    G.remap(community_map, edge_idx_map)
    return G


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def create_weighted_graph(nodelists, edgelists):
    N_connections = len(edgelists)
    N_communities = len(nodelists)
    N_pop = len(nodelists[0])
    assert complete_graph_max_edges(N_communities) == N_connections

    G = gt.complete_graph(N_communities, directed=False, self_loops=True)
    weights = [len(e) for e in edgelists]
    G.ep['weight'] = G.new_edge_property('int', vals=weights)
    G.vp['vweight'] = G.new_vertex_property(
        'int', vals=[len(n) for n in nodelists])
    # correct weights with self loops
    for i, e in enumerate(G.edges()):
        if e.source() == e.target():
            G.ep['weight'][e] += N_pop
    return G


def index_relabel_ascending(indices):
    unique = np.unique(indices)
    indices = np.array(indices)
    new_indices = list(range(len(unique)))
    nidx_offset = 0
    old_indices = []
    result = np.zeros_like(indices)

    for old_idx in indices:
        if old_idx not in old_indices:
            old_indices.append(old_idx)
            result[indices == old_idx] = new_indices[nidx_offset]
            nidx_offset += 1
    return result
# Project vertex community map from level 1 to level 0


def project_mapping(map_1, map_0):
    result = map_0.copy()
    for j, map_0_elem in enumerate(map_0):
        result[j] = map_1[map_0_elem]
    return result


def structural_inference(edgelists, nodelists):
    G = create_weighted_graph(nodelists, edgelists)
    weight = G.ep['weight']
    state = gt.minimize_blockmodel_dl(G, state_args=dict(
        recs=[weight], rec_types=["discrete-poisson"]))
    edges = G.get_edges()
    vcm = index_relabel_ascending(list(state.get_state()))
    N_blocks = state.get_nonempty_B()
    entropy = state.entropy()
    return N_blocks, entropy, vcm


def multiple_structural_inference(edgelists, nodelists):
    N_blocks, entropies, ecms, vcms = [], [], [], []
    for elist, nodelist in zip(edgelists, nodelists):
        Nb, entropy, vcm = structural_inference(elist, nodelist)
        N_blocks.append(Nb)
        entropies.append(entropy)
        vcms.append(vcm)
    return N_blocks, entropies, vcms


def flatten_sublists(l):
    result = []
    for sublist in l:
        sublist_flat = [item for sublist in sublist for item in sublist]
        result.append(sublist_flat)
    return result


def create_graphs(p):
    edgelists, vertexlists, base_ecm, base_vcm = generate_N_SBM_graphs(
        p.N_pop, p.N_communities, p.p_in, p.p_out, p.seed, p.N_graphs)
    return edgelists, vertexlists, base_ecm, base_vcm


def multiple_structural_inference_over_p_out(sim_params):
    seeds = np.random.randint(0, 100000, sim_params[0].N_graphs)
    entropies, N_blocks = [], []
    for p in sim_params:
        print(p.p_out)
        elist, vlist, _, _ = create_graphs(p)
        Nb, entropy, _ = multiple_structural_inference(elist, vlist)
        N_blocks.append(Nb)
        entropies.append(entropy)
    return N_blocks, entropies


def inference_over_p_out(sim_params):
    seeds = np.random.randint(0, 100000, sim_params[0].N_graphs)

    edgelists, vertex_lists, entropies, N_blocks, ecms, vcms = [], [], [], [], [], []
    for p in sim_params:
        elist, vlist, base_ecms, base_vcms = create_graphs(p)
        elist_flat = flatten_sublists(elist)
        edgelists.append(elist_flat)
        vertex_lists.append(vlist)
        Nb, entropy, inferred_vcms = multiple_structural_inference(
            elist, vlist)
        N_blocks.append(Nb)
        entropies.append(entropy)
        vcms.append([(vcm_0, vcm_1)
                    for vcm_1, vcm_0 in zip(inferred_vcms, base_vcms)])
    return edgelists, vertex_lists, N_blocks, entropies, vcms, sim_params
