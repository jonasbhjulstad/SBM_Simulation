//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <itertools.hpp>
#include <memory>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/Math/math.hpp>

namespace Network_Models
{
    template <typename Graph, typename RNG, typename dType = float>
    void random_connect(Graph &G, dType p_ER, RNG &rng)
    {
        Sycl_Graph::random::uniform_real_distribution d_ER;
        uint32_t N_edges = 0;
        for (auto &&v_idx : iter::combinations(Sycl_Graph::range(0, G.NV_max), 2))
        {
            if (d_ER(rng) < p_ER)
            {
                G.add(v_idx[0], v_idx[1]);
                N_edges++;
                if (N_edges == G.NE_max)
                {
                    std::cout << "Warning: max edges reached" << std::endl;
                    return;
                }
            }
        }
    }

    template <typename Graph, typename RNG, typename dType = float>
    void generate_erdos_renyi(
        Graph &G,
        uint32_t N_pop,
        dType p_ER,
        const typename Graph::Vertex_Prop_t &node_prop,
        RNG &rng)
    {
        for (int i = 0; i < N_pop; i++)
        {
            G.add(i, node_prop);
        }
        random_connect(G, p_ER, rng);
    }
}
#endif // FROLS_GRAPH_GENERATION_HPP
