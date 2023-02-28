#ifndef SYCL_GRAPH_BUFFER_SYCL_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/buffer_routines.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <tuple>
namespace Sycl_Graph::Sycl
{

    struct Edge_Accessor
    {
        typedef typename Edge_t::uI_t uI_t;

        Edge_Accessor(sycl::buffer<Edge_t, 1> &edge_buf, sycl::buffer<uI_t, 1> &to_buf,
                      sycl::buffer<uI_t, 1> &from_buf, sycl::handler &h,
                      sycl::property_list props = {})
            : data(edge_buf, h, props), to(to_buf, h, props),
              from(from_buf, h, props) {}
        sycl::accessor<Edge_t, 1, Mode> data;
        sycl::accessor<uI_t, 1, Mode> to;
        sycl::accessor<uI_t, 1, Mode> from;
    };

    template <typename... Ds>
    struct Buffer_Accessor
    {
        Buffer_Accessor(sycl::buffer<Ds, 1> &...bufs, sycl::handler &h,
                        sycl::property_list props = {})
            : bufs(bufs, h, props)...
        {
        }
        sycl::accessor<Ds, 1, Mode>... bufs;
    };

    template <std::unsigned_integral uI_t, typename... Ds>
    struct Buffer
    {
        sycl::queue &q;
        std::tuple<sycl::buffer<Ds, 1>...> bufs;
        uI_t N_elements = 0;
        Buffer(sycl::queue &q, uI_t N, const sycl::property_list &props = {})
            : to_buf(sycl::range<1>(N), props), from_buf(sycl::range<1>(NE), props),
              data_buf(sycl::range<1>(N), props), N(N), q(q) {}

        Buffer(sycl::queue &q, const std::vector<Ds> &data...,
               const sycl::property_list &props = {})
            : bufs(sycl::buffer<Ds, 1>(data, props)...), q(q){}
        {
        }

        uI_t size() const { return std::get<0>(data).size(); }
        uI_t N_elements() const { return N_elements; }

        // returns a buffer accessor with all types
        template <sycl::access::mode Mode>
        Buffer_Accessor<Ds...> get_access(sycl::handler &h)
        {
            return Buffer_Accessor<Ds...>(bufs, h);
        }

        // returns a buffer accessor with only the specified types
        template <sycl::access::mode Mode, typename... D_subset>
        Buffer_Accessor<D_subset...> get_access(sycl::handler &h)
        {
            return Buffer_Accessor<D_subset...>((std::get<index_of_type<D_subset>(Ds...)>(bufs)...), h);
        }

        void resize(uI_t size)
        {
            (buffer_resize(bufs, size, q), ...);
            N_elements = std::max(N_elements, size);
        }

        void add(const std::vector<Ds> &&data..., uI_t offset = 0)
        {
            (host_buffer_add(bufs, data, q, offset), ...);
            N_elements += std::get<0>(data).size();
        }

        void remove(uI_t offset, uI_t size)
        {
            (device_buffer_remove(bufs, q, offset, size), ...);
            N_elements -= size;
        }

        void remove(const std::vector<uI_t> &indices)
        {
            (device_buffer_remove(bufs, q, indices), ...);
            N_elements -= indices.size();
        }

        Buffer<uI_t, Ds...> &operator=(const Buffer<uI_t, Ds...> &other)
        {
            bufs = other.bufs;
            return *this;
        }

        Buffer<uI_t, Ds...> &operator+(const Buffer<uI_t, Ds...> &other)
        {
            (device_buffer_combine(bufs, other.bufs, q), ...);
            return *this;
        }
        size_t byte_size()
        {
            return to_buf.size() * sizeof(uI_t) +
                   from_buf.size() * sizeof(uI_t) +
                   data_buf.size() * sizeof(Data_t);
        }
    };

} // namespace Sycl_Graph::Sycl
#endif