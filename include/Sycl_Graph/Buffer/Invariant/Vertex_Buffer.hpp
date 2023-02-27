#ifndef SYCL_GRAPH_INVARIANT_VERTEX_BUFFER_HPP
#include <concepts>
#include <Sycl_Graph/Graph/Invariant/Buffer.hpp>
#include <Sycl_Graph/Graph/Vertex_Buffer_Base.hpp>

namespace Sycl_Graph::Invariant
{

    template <Vertex_Buffer_type... VBs>
    struct Vertex_Buffer : public Buffer<VBs...>
    {
        Vertex_Buffer(VBs &&...buffers) : buffers(buffers...) {}

        std::tuple<VBs...> buffers;

        typedef Buffer<VBs...> Base_t;
        typedef typename std::tuple_element_t<0, std::tuple<VBs...>>::uI_t uI_t;
        typedef std::tuple<typename VBs::Vertex_t...> Vertex_t;
        typedef std::tuple<typename VBs::Vertex_t::Data_t...> Data_t;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

        template <typename V>
        using Vertex_type = Base_t::Container_type;
        template <typename D>
        using Vertex_Data_type = Base_t::Container_Data_type;


        template <typename V>
            requires Vertex_type<V>::value
        void add(const std::vector<uI_t> &&ids, const std::vector<V> &&data)
        {
            // create vector of vertices
            std::vector<Vertex<V, uI_t>> vertices(ids.size());
            vertices.reserve(ids.size());
            std::transform(ids.begin(), ids.end(), data.begin(), vertices.begin(), [](auto &&id, auto &&data)
                           { return Vertex<V, uI_t>{id, data}; });
            add(vertices);
        }

        template <typename D>
            requires Vertex_Data_type<D>::value
        void add(const std::vector<D> &&data)
        {
            std::vector<Vertex<D, uI_t>> vertices(data.size());
            std::vector<uI_t> ids = get_buffer<D>().get_available_ids(data.size());
            vertices.reserve(data.size());
            for (uI_t i = 0; i < data.size(); ++i)
                vertices.emplace_back(ids[i], data[i]);
            get_buffer<Vertex<D, uI_t>>().add(vertices);
        }

        template <typename... Ds>
            requires(Vertex_Data_type<Ds>::value && ...)
        void add(const std::vector<Ds> &&...data)
        {
            (add(data), ...);
        }

        template <typename D>
            requires Vertex_Data_type<D>::value
        void add(const std::vector<uI_t> &&ids)
        {
            std::vector<Vertex<D, uI_t>> vertices(ids.size());
            vertices.reserve(ids.size());
            for (uI_t i = 0; i < ids.size(); ++i)
                vertices.emplace_back(ids[i], D{});
            add(vertices);
        }

        template <typename V>
            requires Vertex_type<V>::value
        void remove(const std::vector<uI_t> &&ids)
        {
            get_buffer<V>().remove(ids);
        }

        template <typename V>
            requires Vertex_type<V>::value
        auto get_vertices()
        {
            return get_buffer<V>().get_vertices();
        }

        template <typename V>
            requires Vertex_type<V>::value
        auto get_vertices(const std::vector<uI_t> &ids)
        {
            return get_buffer<V>().get_vertices(ids);
        }

        auto get_vertices()
        {
            return std::apply([](auto &&...buffers)
                              { return std::tuple_cat(buffers.get_vertices()...); },
                              buffers);
        }
    };

    template <typename T>
    concept Vertex_Buffer_type =
        // variadic constraint
        std::unsigned_integral<typename T::uI_t> &&
        requires(T t) {
            t.buffers;
            t.size();
            t.add(std::vector<typename T::Vertex_t>());
            t.get_vertices();
            t.remove(std::vector<typename T::Vertex_t>());
        };

} // namespace Sycl_Graph::Invariant
#endif