#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <vector>
namespace Sycl_Graph
{
    template <typename D, std::unsigned_integral ID_t>
    struct Vertex
    {
        Vertex() = default;
        //sycl device copyable   
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t id = std::numeric_limits<ID_t>::max();
        D data;
    };

    template <typename D, std::unsigned_integral ID_t>
    struct Edge
    {
        Edge(const D& data, ID_t to, ID_t from)
            : data(data), to(to), from(from) {}
        Edge(ID_t to, ID_t from)
            : to(to), from(from) {}
        D data;
        static constexpr ID_t invalid_id = std::numeric_limits<ID_t>::max();
        ID_t to = invalid_id;
        ID_t from = invalid_id;
    };

} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP