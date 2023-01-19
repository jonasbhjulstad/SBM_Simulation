#ifndef SYCL_GRAPH_META_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_META_VERTEX_BUFFER_HPP
#include <vector>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
namespace Sycl_Graph::Meta
{
    template <typename V, typename uI_t>
    struct Vertex_Buffer: public Vertex_Buffer_Base<V, uI_t, Vertex_Buffer<V, uI_t>>
    {
        std::vector<Vertex<V*, uI_t>> vertices;

        Vertex_Buffer(const std::vector<Vertex<V*, uI_t>>& vertices)
        {
            this->vertices = vertices;
        }

        uI_t size() const
        {
            return vertices.size();
        }

        void add(const V* data)
        {
            vertices.push_back(Vertex<V*, uI_t>(data));
        }

        std::vector<Vertex<V*, uI_t>> get_vertices()
        {
            //filter out invalid vertices
            std::vector<Vertex<V*, uI_t>> valid_vertices;
            valid_vertices.reserve(vertices.size());
            for (auto& vertex : vertices)
            {
                if (vertex.id != Vertex<V*, uI_t>::invalid_id)
                {
                    valid_vertices.push_back(vertex);
                }
            }
            return valid_vertices;
        }

        void remove(uI_t index)
        {
            vertices[index].id = Vertex<V*, uI_t>::invalid_id;
        }
    };
}

#endif