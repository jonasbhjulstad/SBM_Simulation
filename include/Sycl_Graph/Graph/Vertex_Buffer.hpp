#ifndef SYCL_GRAPH_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_VERTEX_BUFFER_HPP
#include <Sycl_Graph/Graph/Graph_Types.hpp>
namespace Sycl_Graph
{
    template <Vertex_type V, typename Derived>
    struct Vertex_Buffer_Base
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
    requires(T t)
    {
        // t = T(const std::vector<typename T::Vertex_t>& vertices);
        t.size();
        t.add(std::vector<typename T::Vertex_t>());
        t.get_vertices();
        t.remove(T::uI_t());
    };
    namespace Invariant
    {


    template <Vertex_Buffer_type ... VBs>
    struct Vertex_Buffer
    {
        Vertex_Buffer(VBs &&... buffers): buffers(buffers ...) {}
        
        std::tuple<VBs ...> buffers;

        typedef typename std::tuple_element_t<0, std::tuple<VBs ...>>::uI_t uI_t;
        typedef std::tuple<typename VBs::Vertex_t ...> Vertex_t;
        typedef std::tuple<typename VBs::Vertex_t::Data_t ...> Data_t;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        
        template <typename T>
        struct Vertex_type
        {
            static constexpr bool value = (std::is_same_v<T, typename VBs::Vertex_t> || ...);
        };
        template <typename T>
        struct Vertex_Data_type
        {
            static constexpr bool value = (std::is_same_v<T, typename VBs::Vertex_t::Data_t> || ...);
        };
        template <typename V> requires Vertex_type<V>::value
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<V, typename VBs::Vertex_t ...>();
        }

        template <typename ... Vs> requires (Vertex_type<Vs>::value && ...)
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Vs)>{type_index<Vs>() ...};
        }

        template <typename V> requires Vertex_type<V>::value
        auto&& get_buffer()
        {
            //get index of buffer
            constexpr uI_t index = get_buffer_index<V>();
            return std::get<index>(buffers);
        }
        template <typename ... Vs> requires (Vertex_type<Vs>::value && ...)
        auto&& get_buffers()
        {
            return std::array{get_buffer<Vs>() ...};
        }

        template <typename D> requires Vertex_Data_type<D>::value
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<Vertex<D, uI_t>, typename VBs::Vertex_t ...>();
        }

        template <typename ... Ds> requires (Vertex_Data_type<Ds>::value && ...)
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Ds)>{type_index<Vertex<Ds, uI_t>>() ...};
        }

        template <typename D> requires Vertex_Data_type<D>::value
        auto&& get_buffer()
        {
            //get index of buffer
            constexpr uI_t index = get_buffer_index<D>();
            return std::get<index>(buffers);
        }
        template <typename ... Ds> requires (Vertex_Data_type<Ds>::value && ...)
        auto&& get_buffers()
        {
            return std::array{get_buffer<Ds>() ...};
        }

        auto size() const
        {
            return std::apply([](auto &&... buffers) {
                return (buffers.size() + ...);
            }, buffers);
        }

        template <typename  V> requires Vertex_type<V>::value
        auto size() const
        {
            return get_buffer<V>().size();
        }

        template <typename ... Vs> requires (Vertex_type<Vs>::value && ...)
        void add(const std::vector<Vertex<Vs, uI_t>> && ... vertices)
        {
            (get_buffer<Vs>().add(vertices), ...);
        }

        template <typename V> requires Vertex_type<V>::value
        void add(const std::vector<uI_t> &&ids, const std::vector<V> &&data)
        {
            //create vector of vertices
            std::vector<Vertex<V, uI_t>> vertices(ids.size());
            vertices.reserve(ids.size());
            std::transform(ids.begin(), ids.end(), data.begin(), vertices.begin(), [](auto &&id, auto &&data) {
                return Vertex<V, uI_t>{id, data};
            });
            add(vertices);
        }


        template <typename D> requires Vertex_Data_type<D>::value
        void add(const std::vector<D>&& data)
        {
            std::vector<Vertex<D, uI_t>> vertices(data.size());
            std::vector<uI_t> ids = get_buffer<D>().get_available_ids(data.size());
            vertices.reserve(data.size());
            for(uI_t i = 0; i < data.size(); ++i)
                vertices.emplace_back(ids[i], data[i]);
            get_buffer<Vertex<D, uI_t>>().add(vertices);
        }

        template <typename ... Ds> requires (Vertex_Data_type<Ds>::value && ...)
        void add(const std::vector<Ds>&& ... data)
        {
            (add(data), ...);
        }

        template <typename D> requires Vertex_Data_type<D>::value
        void add(const std::vector<uI_t>&& ids)
        {
            std::vector<Vertex<D, uI_t>> vertices(ids.size());
            vertices.reserve(ids.size());
            for(uI_t i = 0; i < ids.size(); ++i)
                vertices.emplace_back(ids[i], D{});
            add(vertices);
        }

        template <typename ... Vs> requires (Vertex_type<Vs>::value && ...)
        void remove(const std::vector<Vertex<Vs, uI_t>>&&... vertices)
        {
            ((get_buffer<Vs>().remove(vertices), ...));
        }

        template <typename V>   requires Vertex_type<V>::value
        void remove(const std::vector<uI_t>&& ids)
        {
            get_buffer<V>().remove(ids);
        }

        template <typename V> requires Vertex_type<V>::value
        auto get_vertices()
        {
            return get_buffer<V>().get_vertices();
        }

        template <typename V> requires Vertex_type<V>::value
        auto get_vertices(const std::vector<uI_t>& ids)
        {
            return get_buffer<V>().get_vertices(ids);
        }

        auto get_vertices()
        {
            return std::apply([](auto &&... buffers) {
                return std::tuple_cat(buffers.get_vertices() ...);
            }, buffers);
        }

    };

    template <typename T>
    concept Invariant_Vertex_Buffer_type =
    //variadic constraint
    (Vertex_Buffer_type<VBs> && ...) &&
    std::unsigned_integral<typename T::uI_t> &&
    requires(T t)
    {
        T(VBs &&... buffers) -> std::convertible_to<T>;
        t.buffers;
        t.size();
        t.add(std::vector<typename T::Vertex_t>());
        t.get_vertices();
        t.remove(std::vector<typename T::Vertex_t>());
    };

}
#endif