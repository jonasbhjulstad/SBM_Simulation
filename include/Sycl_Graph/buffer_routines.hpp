#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <vector>
#include <string>
namespace Sycl_Graph
{
    template <typename T>
    inline void host_buffer_copy(sycl::buffer<T, 1> &buf, const std::vector<T> &vec, sycl::handler &h)
    {
        h.parallel_for(vec.size(), [=](sycl::id<1> i)
                       { buf[i] = vec[i]; });
    }

    template <typename T>
    class host_buffer_copy_kernel;

    template <typename T, typename uI_t = uint32_t>
    inline void host_buffer_add(sycl::buffer<T, 1> &buf, const std::vector<T> &vec, sycl::queue &q, uI_t offset = 0)
    {
        if (vec.size() == 0)
        {
            return;
        }
        if constexpr (sizeof(T) > 0)
        {
            sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
            q.submit([&](sycl::handler &h)
                     {
        auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
        auto acc = buf.template get_access<sycl::access::mode::write>(h);


        h.parallel_for<host_buffer_copy_kernel<T>>(vec.size(), [=](sycl::id<1> i)
        {
            acc[i + offset] = tmp_acc[i];
        }); });
        //submit with kernel_name
        }
    }

    template <typename T, typename uI_t = uint32_t>
    inline void host_buffer_add(std::vector<sycl::buffer<T, 1> &> &bufs, const std::vector<const std::vector<T> &> &vecs, sycl::queue &q, const std::vector<uI_t> &offsets)
    {
        for (uI_t i = 0; i < vecs.size(); ++i)
        {
            host_buffer_add(bufs[i], vecs[i], q, offsets[i]);
        }
    }
}
#endif