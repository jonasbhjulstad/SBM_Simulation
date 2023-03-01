#ifndef SYCL_GRAPH_INVARIANT_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_INVARIANT_VERTEX_BUFFER_HPP
#include <concepts>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>

namespace Sycl_Graph::Invariant
{


    template <Sycl_Graph::Base::Vertex_Buffer_type... VBs>
    struct Vertex_Buffer : public Buffer<VBs...>
    {
        typedef Buffer<VBs...> Base_t;
        typedef typename std::tuple_element_t<0, std::tuple<VBs...>>::uI_t uI_t;
        typedef std::tuple<typename VBs::Vertex_t::Data_t...> Data_t;

        typedef Sycl_Graph::Base::Vertex<std::tuple<typename VBs::Vertex_t ...>, uI_t> Vertex_t;

        Vertex_Buffer(const VBs &...buffers) : buffers(buffers...) {}
        Vertex_Buffer(const VBs &&...buffers) : buffers(buffers...) {}
        // Vertex_Buffer(const std::vector<typename VBs::Vertex_t>&& ... vertices): buffers(vertices ...) {}
        std::tuple<VBs...> buffers;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        template <typename V>
        using Vertex = Sycl_Graph::Base::Vertex<V, uI_t>;

        template <typename V>
        void add(const std::vector<uI_t> &&ids, const std::vector<V> &&data)
        {
            // create vector of vertices
            std::vector<V> vertices(ids.size());
            vertices.reserve(ids.size());
            std::transform(ids.begin(), ids.end(), data.begin(), vertices.begin(), [](auto &&id, auto &&data)
                           { return V{id, data}; });
            add(vertices);
        }

        template <typename D>
        void add(const std::vector<D> &&data)
        {
            std::vector<Vertex<D>> vertices(data.size());
            std::vector<uI_t> ids = get_buffer<D>().get_available_ids(data.size());
            vertices.reserve(data.size());
            for (uI_t i = 0; i < data.size(); ++i)
                vertices.emplace_back(ids[i], data[i]);
            get_buffer<Vertex<D>>().add(vertices);
        }

        template <typename... Ds>
        void add(const std::vector<Ds> &&...data)
        {
            (add(data), ...);
        }

        template <typename D>
        void add(const std::vector<uI_t> &&ids)
        {
            std::vector<Vertex<D>> vertices(ids.size());
            vertices.reserve(ids.size());
            for (uI_t i = 0; i < ids.size(); ++i)
                vertices.emplace_back(ids[i], D{});
            add(vertices);
        }

        template <typename V>
        void remove(const std::vector<uI_t> &&ids)
        {
            get_buffer<V>().remove(ids);
        }

        template <typename V>
        auto get_vertices()
        {
            return get_buffer<V>().get_vertices();
        }

        template <typename V>
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
        std::unsigned_integral<typename T::uI_t>;

} // namespace Sycl_Graph::Invariant
#endif