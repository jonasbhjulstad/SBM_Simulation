//
// Created by arch on 9/14/22.
//

#ifndef FROLS_GRAPH_GENERATION_HPP
#define FROLS_GRAPH_GENERATION_HPP

#include <graph_lite.h>
#include <itertools.hpp>
#include <random>

namespace Network_Models {
    template<typename Graph, typename RNG, typename dType=float>
    void random_connect(Graph &G, dType p_ER, RNG &rng) {
        std::bernoulli_distribution d_ER(p_ER);
        for (auto &&v_idx: iter::combinations(FROLS::range(0, Graph::MAX_VERTICES), 2)) {
            if (d_ER(rng))
                G.add_edge(v_idx[0], v_idx[1]);
        }
    }

    template<typename Graph, typename RNG, typename dType=float>
    Graph generate_erdos_renyi(size_t N_pop,
                               dType p_ER,
                               const typename Graph::Vertex_Prop_t& node_prop,
                               RNG &rng) {
        Graph G;
        for (int i = 0; i < N_pop; i++) {
            G.add_vertex(i, node_prop);
        }
        random_connect(G, p_ER, rng);
        return G;
    }
}
#endif //FROLS_GRAPH_GENERATION_HPP
