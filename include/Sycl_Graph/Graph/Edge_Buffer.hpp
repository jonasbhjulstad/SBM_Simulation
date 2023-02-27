#ifndef SYCL_GRAPH_EDGE_BUFFER_HPP
#define SYCL_GRAPH_EDGE_BUFFER_HPP
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
namespace Sycl_Graph
{

    template <Edge_type E, typename Derived>
    struct Edge_Buffer_Base
    {
        typedef E Edge_t;
        typedef Edge_t::uI_t uI_t;

        uI_t size() const
        {
            return static_cast<const Derived *>(this)->size();
        }
        void add(const std::vector<Edge_t> &&edges)
        {
            static_cast<Derived *>(this)->add(edges);
        }
        std::vector<Edge_t> get_edges()
        {
            return static_cast<Derived *>(this)->get_edges();
        }

        void remove(const std::vector<uI_t>&& ids)
        {
            static_cast<Derived *>(this)->remove(ids);
        }

        void remove(const std::vector<Edge_t>&& edges)
        {
            std::vector<uI_t> ids;
            ids.reserve(edges.size());
            for (const auto& edge : edges)
            {
                ids.push_back(edge.id);
            }
            static_cast<Derived *>(this)->remove(ids);
        }

        void remove(uI_t index)
        {
            remove({index});
        }

        Derived &operator=(const Derived &other)
        {
            static_cast<Derived *>(this)->operator=(other);
            return *this;
        }

        Derived& operator+(const Derived& other)
        {
            static_cast<Derived *>(this)->operator+(other);
            return *this;
        }

    };

    template <typename T>
    concept Edge_Buffer_type =
    Edge_type<typename T::Edge_t> &&
    std::unsigned_integral<typename T::uI_t> && 
    requires(T t)
    {
        T(const std::vector<typename T::Edge_t>& edges) -> std::convertible_to<T>;
        t.size();
        t.add(std::vector<T::Edge_t>());
        t.get_edges();
        t.remove(T::uI_t());
    };

    template <Edge_Buffer_type ... EBs> requires (Invariant_Edge_type<EBs::Edge_t> && ...)
    struct Invariant_Edge_Buffer
    {
        Invariant_Edge_Buffer(EBs &&... buffers): buffers(buffers ...) {}
        
        std::tuple<EBs ...> buffers;

        template <typename T>
        concept Edge_Data_type = (std::is_same_v<T, EBs::Edge_t::Data_t> || ...);

        typedef _uI_t uI_t;
        typedef std::tuple<EBs::Edge_t ...> Edge_t;
        typedef std::tuple<EBs::Edge_t::Data_t ...> Data_t;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        
        template <Invariant_Edge_type E>
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<E, EBs::Edge_t ...>;
        }

        template <Invariant_Edge_type ... Es>
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Es)>{type_index<Es>() ...};
        }

        template <Invariant_Edge_type E>
        static constexpr auto&& get_buffer()
        {
            return std::get<get_buffer_index<E>()>(buffers);
        }

        template <Invariant_Edge_type ... Es>
        static constexpr auto get_buffers()
        {
            return std::array{get_buffer<Es>() ...};
        }

        auto size() const
        {
            return std::apply([](auto &&... buffers) {
                return (buffers.size() + ...);
            }, buffers);
        }

        template <Invariant_Edge_type E>
        auto size() const
        {
            return get_buffer<E>().size();
        }

        template <Invariant_Edge_type ... Es>
        void add(const std::vector<Edge<Es, uI_t>> &&edges ...)
        {
            ((get_buffer<Es>().add(edges), ...), ...);
        }

        template <Invariant_Edge_type E>
        void add(const std::vector<uI_t> &&ids, const std::vector<E> &&data)
        {
            //create vector of edges
            std::vector<Edge<E, uI_t>> edges(ids.size());
            edges.reserve(ids.size());
            std::transform(ids.begin(), ids.end(), data.begin(), edges.begin(), [](auto &&id, auto &&data) {
                return Edge<E, uI_t>{id, data};
            });
            add(edges);
        }


        template <Edge_Data_type D>
        void add(const std::vector<D>&& data)
        {
            std::vector<Edge<D, uI_t>> edges(data.size());
            std::vector<uI_t> ids = std::get<idx>(buffers).get_available_ids(data.size());
            edges.reserve(data.size());
            for(uI_t i = 0; i < data.size(); ++i)
                edges.emplace_back(ids[i], data[i]);
            get_buffer<Edge<D, uI_t>>().add(edges);
        }

        template <Edge_Data_type ... Ds>
        void add(const std::vector<Ds>&& ... data)
        {
            (add(data), ...);
        }

        template <Edge_Data_type D>
        void add(const std::vector<uI_t>&& ids)
        {
            std::vector<Edge<E, uI_t>> edges(ids.size());
            edges.reserve(ids.size());
            for(uI_t i = 0; i < ids.size(); ++i)
                edges.emplace_back(ids[i], D{});
            add(edges);
        }

        template <Invariant_Edge_type ... Es>
        void remove(const std::vector<Edge<Es, uI_t>>&&... edges)
        {
            ((get_buffer<Es>().remove(edges), ...));
        }

        template <Invariant_Edge_type E>
        void remove(const std::vector<uI_t>&& ids)
        {
            get_buffer<E>().remove(ids);
        }

        void remove(const std::vector<uI_t>&& ids)
        {
            ((remove<Es>(ids), ...));
        }

        template <Invariant_Edge_type E>
        void remove(uI_t id)
        {
            get_buffer<E>().remove(id);
        }

        void remove(uI_t id)
        {
            ((remove<Es>(id), ...));
        }


        template <Invariant_Edge_type E>
        auto get_edges()
        {
            return get_buffer<E>().get_edges();
        }

        template <Invariant_Edge_type E>
        auto get_edges(const std::vector<uI_t>&& ids)
        {
            return get_buffer<E>().get_edges(ids);
        }

        auto get_edges()
        {
            return std::apply([](auto &&... buffers) {
                return std::tuple_cat(buffers.get_edges() ...);
            }, buffers);
        }
        
        //Buffer operations

        auto& operator=(Invariant_Edge_Buffer&& other)
        {
            buffers = std::move(other.buffers);
            return *this;
        }

        auto copy() const
        {
            return Invariant_Edge_Buffer(buffers);
        }

        auto& operator+(const Invariant_Edge_Buffer& other)
        {
            std::apply([](auto &&... buffers) {
                ((buffers + other.buffers), ...);
            }, buffers);
            return *this;
        }

    }
}

#endif