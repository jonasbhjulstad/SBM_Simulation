#ifndef SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#define SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#include <Sycl_Graph/Graph/Invariant/Buffer.hpp>
#include <Sycl_Graph/Graph/Invariant/Graph_Types.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>

namespace Sycl_Graph::Invariant
{
    template <Sycl_Graph::Edge_Buffer_type... EBs>
        requires(Edge_type<typename EBs::Edge_t> && ...)
    struct Edge_Buffer: public Buffer<EBs...>
    {
        Edge_Buffer(EBs &&...buffers) : buffers(buffers...) {}

        std::tuple<EBs...> buffers;

        template <typename V>
        using Edge_type = Base_t::Container_type;
        template <typename D>
        using Edge_Data_type = Base_t::Container_Data_type;


        typedef Buffer<EBs...> Base_t;
        typedef typename Base_t::uI_t uI_t;
        typedef std::tuple<typename EBs::Edge_t...> Edge_t;
        typedef std::tuple<typename EBs::Edge_t::Data_t...> Data_t;

        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

        template <Edge_type E>
        auto get_edges()
        {
            return get_buffer<E>().get_edges();
        }

        template <Edge_type E>
        auto get_edges(const std::vector<uI_t> &&to_ids, const std::vector<uI_t> &&from_ids)
        {
            return get_buffer<E>().get_edges(to_ids, from_ids);
        }

        auto get_edges()
        {
            return std::apply([](auto &&...buffers)
                              { return std::tuple_cat(buffers.get_edges()...); },
                              buffers);
        }
    }
}
#endif