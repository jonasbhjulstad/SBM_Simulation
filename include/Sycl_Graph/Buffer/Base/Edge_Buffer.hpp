#ifndef SYCL_GRAPH_EDGE_BUFFER_HPP
#define SYCL_GRAPH_EDGE_BUFFER_HPP
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
namespace Sycl_Graph::Buffer::Base
{

    template <Edge_type E, typename Derived>
    struct Edge_Buffer
    {
        typedef E Edge_t;
        typedef typename Edge_t::uI_t uI_t;
        typedef typename Edge_t::Data_t Data_t;

        uI_t size() const
        {
            return static_cast<const Derived *>(this)->size();
        }
        void add(const std::vector<Edge_t> &&edges)
        {
            static_cast<Derived *>(this)->add(edges);
        }

        void add(const std::vector<uI_t>&& to_ids, const std::vector<uI_t>&& from_ids, const std::vector<Data_t>&& data)
        {
            std::vector<Edge_t> edges;
            edges.reserve(to_ids.size());
            for (size_t i = 0; i < to_ids.size(); i++)
            {
                edges.push_back({to_ids[i], from_ids[i], data[i]});
            }
            add(edges);
        }

        void add(uI_t to_id, uI_t from_id)
        {
            add({Edge_t(to_id, from_id)});
        }
        std::vector<Edge_t> get_edges()
        {
            return static_cast<Derived *>(this)->get_edges();
        }

        std::vector<Edge_t> get_edges(const std::vector<uI_t> &&to_ids, const std::vector<uI_t> &&from_ids)
        {
            return static_cast<Derived *>(this)->get_edges(to_ids, from_ids);
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

        void remove(uI_t to_id, uI_t from_id)
        {
            remove({Edge_t(to_id, from_id)});
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
        t.size();
        t.add(std::vector<typename T::Edge_t>());
        t.get_edges();
        t.remove(T::uI_t());
    };

}

#endif