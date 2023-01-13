//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <itertools.hpp>
#include <memory>
#include <sycl/CL/sycl.hpp>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <random>

namespace Sycl_Graph::Dynamic::Network_Models
{
    template <typename Graph, typename RNG, typename dType = float>
    void random_connect(Graph &G, dType p_ER, RNG &rng)
    {
        Sycl_Graph::random::uniform_real_distribution d_ER;
        uint32_t N_edges = 0;
        std::vector<typename Graph::uInt_t> from;
        std::vector<typename Graph::uInt_t> to;
        uint32_t N_edges_max = Sycl_Graph::n_choose_k(G.N_vertices(), 2);

        from.reserve(N_edges_max);
        to.reserve(N_edges_max);
        std::vector<typename Graph::Edge_Prop_t> edges(N_edges_max);
        for (auto &&v_idx : iter::combinations(Sycl_Graph::range(0, G.NV), 2))
        {
            if (d_ER(rng) < p_ER)
            {
                from.push_back(v_idx[0]);
                to.push_back(v_idx[1]);

                N_edges++;
                if (N_edges == G.NE)
                {
                    std::cout << "Warning: max edges reached" << std::endl;
                    return;
                }
            }
        }
        G.add_edge(to, from, edges);
    }

    template <typename Graph, typename dType = float, typename RNG = std::mt19937>
    Graph generate_erdos_renyi(
        sycl::queue &q,
        uint32_t NV,
        dType p_ER,
        RNG rng = std::mt19937(std::random_device()()),
        uint32_t NE = 0)
    {
        NE = NE == 0 ? 1.2*Sycl_Graph::n_choose_k(NV, 2) : NE;
        Graph G(q, NV, NE);
        random_connect(G, p_ER, rng);
        return G;
    }


    template <typename Graph, typename RNG, typename dType = float>
    void generate_erdos_renyi(
        Graph &G,
        dType p_ER,
        const std::vector<typename Graph::Vertex_Prop_t> &vertex_props,
        RNG &rng)
    {
        auto vertex_ids = Sycl_Graph::range(0, vertex_props.size());
        G.add_vertex(vertex_ids, vertex_props);
        random_connect(G, p_ER, rng);
    }
    template <typename Graph, typename RNG, typename dType = float>
    void generate_erdos_renyi(
        Graph &G,
        uint32_t NV,
        dType p_ER,
        const typename Graph::Vertex_Prop_t vertex_prop = typename Graph::Vertex_Prop_t(),
        RNG rng = std::mt19937(std::random_device()()))
        {
            std::vector<typename Graph::Vertex_Prop_t> vertex_props(NV, vertex_prop);
            generate_erdos_renyi(G, p_ER, vertex_props, rng);
        }

}
#endif // FROLS_GRAPH_GENERATION_HPP
