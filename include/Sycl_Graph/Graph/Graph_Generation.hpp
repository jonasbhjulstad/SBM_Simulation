//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <itertools.hpp>
#include <memory>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/Math/math.hpp>

namespace Sycl_Graph::Dynamic::Network_Models
{
    template <typename Graph, typename RNG, typename dType = float>
    void random_connect(Graph &G, dType p_ER, RNG &rng)
    {
        Sycl_Graph::random::uniform_real_distribution d_ER;
        uint32_t N_edges = 0;
        std::vector<typename Graph::Edge_t> edges;
        edges.reserve(G.N_vertices * G.N_vertices);
        std::vector<uint32_t> from;
        for (auto &&v_idx : iter::combinations(Sycl_Graph::range(0, G.NV_max), 2))
        {
            if (d_ER(rng) < p_ER)
            {
                edges.push_back(typename Graph::Edge_t({}, v_idx[0], v_idx[1]));
                N_edges++;
                if (N_edges == G.NE_max)
                {
                    std::cout << "Warning: max edges reached" << std::endl;
                    return;
                }
            }
        }
        G.add(edges);
    }

    template <typename Graph, typename RNG, typename dType = float>
    void generate_erdos_renyi(
        Graph &G,
        uint32_t N_pop,
        dType p_ER,
        const typename Graph::Vertex_Prop_t &node_prop,
        RNG &rng)
    {
        auto vertex_ids = Sycl_Graph::range(0, N_pop);
        std::vector<typename Graph::Vertex_Prop_t> vertex_props(N_pop, node_prop);
        assert(G.add(vertex_ids, vertex_props));
        random_connect(G, p_ER, rng);
    }
}
#endif // FROLS_GRAPH_GENERATION_HPP
