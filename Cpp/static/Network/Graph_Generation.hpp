//
// Created by arch on 9/14/22.
//

#ifndef FROLS_GRAPH_GENERATION_HPP
#define FROLS_GRAPH_GENERATION_HPP

#include <graph_lite.h>
#include <itertools.hpp>

namespace Network_Models {
    template<typename RNG, typename NodeProp>
    void random_connect(graph_lite::Graph<int, NodeProp> &G, double p_ER, RNG &rng) {
        std::bernoulli_distribution d_ER(p_ER);
        for (auto &&nodes: iter::combinations(G, 2)) {
            if (d_ER(rng))
                G.add_edge(nodes[0], nodes[1]);
        }
    }

    template<typename RNG, typename NodeProp>
    graph_lite::Graph<int, NodeProp> generate_erdos_renyi(size_t N_pop,
                                                          double p_ER,
                                                          RNG &rng) {
        using namespace graph_lite;
        Graph<int, NodeProp> G;
        NodeProp prop;
        for (int i = 0; i < N_pop; i++) {
            G.add_node_with_prop(i, prop);
        }
        random_connect(G, p_ER, rng);
        return G;
    }
}
#endif //FROLS_GRAPH_GENERATION_HPP
