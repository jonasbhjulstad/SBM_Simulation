#ifndef SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#define SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#include <Sycl_Graph/Graph/Graph_Types.hpp>
namespace Sycl_Graph::Buffer::Base
{
    template <Vertex_type V, typename Derived>
    struct Vertex_Buffer
    {

        typedef V Vertex_t;
        typedef typename V::uI_t uI_t;

        auto size() const
        {
            return static_cast<const Derived *>(this)->size();
        }
        void add(const std::vector<Vertex_t> &&vertices)
        {
            static_cast<Derived *>(this)->add(vertices);
        }
        std::vector<Vertex_t> get_vertices()
        {
            return static_cast<Derived *>(this)->get_vertices();
        }

        void remove(uI_t index)
        {
            static_cast<Derived *>(this)->remove(index);
        }
    };

    template <typename T>
    concept Vertex_Buffer_type =
        Vertex_type<typename T::Vertex_t> &&
        std::unsigned_integral<typename T::uI_t> &&
        requires(T t) {
            // t = T(const std::vector<typename T::Vertex_t>& vertices);
            t.size();
            t.add(std::vector<typename T::Vertex_t>());
            t.get_vertices();
            t.remove(T::uI_t());
        };

}
#endif