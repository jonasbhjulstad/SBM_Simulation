#ifndef SYCL_GRAPH_META_EDGE_BUFFER_HPP
#define SYCL_GRAPH_META_EDGE_BUFFER_HPP
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Meta/Edge_Buffer.hpp>
#include <vector>
namespace Sycl_Graph::Meta
{
    template <typename E, std::unsigned_integer uI_t>
    struct Edge_Buffer: public Edge_Buffer_Base<E, uI_t, Edge_Buffer<E, uI_t>>
    {
        std::vector<Edge<E*, uI_t>> edges; 
        uI_t size() const
        {
            return edges.size();
        }

        void add(const E* data, uI_t to, uI_t from)
        {
            edges.push_back(Edge<E*, uI_t>(data, to, from));
        }

        std::vector<Edge<E*, uI_t>> get_edges()
        {
            //filter out invalid edges
            std::vector<Edge<E*, uI_t>> valid_edges;
            valid_edges.reserve(edges.size());
            for (auto& edge : edges)
            {
                if (edge.to != Edge<E*, uI_t>::invalid_id && edge.from != Edge<E*, uI_t>::invalid_id)
                {
                    valid_edges.push_back(edge);
                }
            }
            return valid_edges;
        }


        void remove(uI_t index)
        {
            edges[index].to = Edge<E*, uI_t>::invalid_id;
            edges[index].from = Edge<E*, uI_t>::invalid_id;
        }

    };
}

#endif