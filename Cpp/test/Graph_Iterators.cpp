//
// Created by arch on 9/29/22.
//


#include "../static/Graph/FROLS_Graph.hpp"
#include <array>
#include <algorithm>
#include <iostream>
#include <gtest/gtest.h>
constexpr size_t N_edges = 20;
constexpr size_t N_nodes = 100;

TEST(FROLS_Graph_iterators, default_iterate)
{
    using namespace FROLS::Graph;
    std::array<Edge<double>, N_edges> edges;

    std::array<Vertex<size_t>, N_nodes> nodes;
    std::generate(nodes.begin(), nodes.end(), [n = 0]() mutable {return Vertex<size_t>{(size_t)n++};});

    Graph G(nodes, edges);
    size_t n = 0;
    for (const auto v : G)
    {
        EXPECT_EQ(v.data, nodes[n].data);
        n++;
    }
}

TEST(FROLS_Graph_iterators, neighbor_test)
{
    using namespace FROLS::Graph;
    std::array<Edge<double>, N_edges> edges;

    std::array<Vertex<size_t>, N_nodes> nodes;
    std::generate(nodes.begin(), nodes.end(), [n = 0]() mutable {return Vertex<size_t>{(size_t)n++};});

    edges[0] = Edge<double>{.4, 25, 24};
    edges[1] = Edge<double>{.1, 25, 1};
    Graph G(nodes, edges);
//    size_t n = 0;
//    for (const auto& nv : G.get_neighbors(G.get_vertex_iterator(25)))
//    {
//
//    }

}

int main()
{


    return 0;
}





