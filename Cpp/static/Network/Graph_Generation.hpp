//
// Created by arch on 9/14/22.
//

#ifndef FROLS_GRAPH_GENERATION_HPP
#define FROLS_GRAPH_GENERATION_HPP

#include <itertools.hpp>
#include <memory>
#include <random>

namespace Network_Models {
    template<typename Graph, typename RNG, typename dType=float>
    void random_connect(Graph &G, dType p_ER, RNG &rng) {
        FROLS::random::uniform_real_distribution d_ER;
        uint32_t N_edges = 0;
        for (auto &&v_idx: iter::combinations(FROLS::range(0, Graph::MAX_VERTICES), 2)) {
            if (d_ER(rng) < p_ER)
            {
                G.add_edge(v_idx[0], v_idx[1]);
                N_edges++;
                if (N_edges == Graph::MAX_EDGES)
                {
                    std::cout << "Warning: max edges reached" << std::endl;
                    return;
                }
            }
        }
    }

    template<typename Graph, typename RNG, typename dType=float>
    Graph generate_erdos_renyi(uint32_t N_pop,
                               dType p_ER,
                               const typename Graph::Vertex_Prop_t& node_prop,
                               RNG &rng,
                               std::array<std::shared_ptr<std::mutex>, Graph::MAX_VERTICES+1>& v_mx,
                               std::array<std::shared_ptr<std::mutex>, Graph::MAX_EDGES+1>& e_mx) {
        Graph G(v_mx, e_mx);
        for (int i = 0; i < N_pop; i++) {
            G.add_vertex(i, node_prop);
        }
        random_connect(G, p_ER, rng);
        return G;
    }
}
#endif //FROLS_GRAPH_GENERATION_HPP
