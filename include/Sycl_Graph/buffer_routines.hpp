#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <vector>
namespace Sycl_Graph
{
    template <typename T>
    inline void host_buffer_copy(cl::sycl::buffer<T, 1>& buf, const std::vector<T>& vec, cl::sycl::handler& h)
    {
        h.parallel_for(vec.size(), [=](sycl::id<1> i)
        {
            buf[i] = vec[i];
        });
    }
    template <typename T, typename uI_t>
    inline void host_buffer_add(cl::sycl::buffer<T, 1>& buf, const std::vector<T>& vec, cl::sycl::handler& h, uI_t offset = 0)
    {
        auto acc = buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(vec.size(), [=](sycl::id<1> i)
        {
            acc[i + offset] += vec[i];
        });
    }
}
#endif