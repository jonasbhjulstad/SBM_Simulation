#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <concepts>
#include <numeric>
#include <vector>
#include <Sycl_Graph/type_helpers.hpp>
#include <Sycl_Graph/Math/math.hpp>
namespace Sycl_Graph
{
    enum Graph_Connection_Type
    {
        directed,
        undirected
    };

    template <typename D, std::unsigned_integral _uI_t = uint32_t>
    struct Vertex
    {
        Vertex(_uI_t id, const D& data): id(id), data(data) {}        
        typedef D Data_t;
        typedef _uI_t uI_t;
        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        uI_t id = std::numeric_limits<uI_t>::max();
        D data;
    };

    template <typename T>
    concept Vertex_type = 
    std::unsigned_integral<typename T::uI_t> &&
    requires(T t)
    {
        {t.id} -> std::convertible_to<typename T::uI_t>;
        {t.data} -> std::convertible_to<typename T::Data_t>;
    };

    template <typename D, typename uI_t>
    std::vector<Vertex<D, uI_t>> make_vertices(const std::vector<D> &data, const std::vector<uI_t> &ids)
    {
        std::vector<Vertex<D, uI_t>> vertices(data.size());
        vertices.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            vertices[i] = {ids[i], data[i]};
        }
        return vertices;
    }

    template <typename D, std::unsigned_integral _uI_t = uint32_t>
    struct Edge
    {
        typedef D Data_t;
        typedef _uI_t uI_t;
        Edge(const D &data, uI_t to, uI_t from)
            : data(data), to(to), from(from) {}
        Edge(uI_t to, uI_t from)
            : to(to), from(from), D{} {}
        D data;
        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        uI_t to = invalid_id;
        uI_t from = invalid_id;

    };

    template <typename T>
    concept Edge_type = requires(T t)
    {
        t.data;
        t.to;
        t.from;
    };

    template <Edge_type E, Vertex_type _To, Vertex_type _From>
    struct Invariant_Edge: public E
    {
        typedef _To To;
        typedef _From From;
    };

    template <typename T>
    concept Invariant_Edge_type = Edge_type<T> && Vertex_type<typename T::To> && Vertex_type<typename T::From>;

} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP