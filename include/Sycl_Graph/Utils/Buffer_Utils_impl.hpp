#ifndef BUFFER_UTILS_IMPL_HPP
#define BUFFER_UTILS_IMPL_HPP
#include <fstream>
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <stdexcept>


template <typename T, std::size_t N>
sycl::event read_buffer(sycl::buffer<T,N>& buf, sycl::queue& q, std::vector<T>& result, std::vector<sycl::event>& dep_events)
{
    if (result.size() < buf.size())
    {
        throw std::runtime_error("Result vector is too small to hold the buffer");
        result.resize(buf.size());
    }
    return q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto acc = sycl::accessor<T, N, sycl::access_mode::read>(buf, h, sycl::range<N>(buf.get_range()));
        h.copy(acc, result.data());
    });
}

template <typename T, std::size_t N>
sycl::event read_buffer(sycl::buffer<T,N>& buf, sycl::queue& q, std::vector<T>& result, std::vector<sycl::event>& dep_events, sycl::range<N> range, sycl::range<N> offset)
{
    if (result.size() < range.size())
    {
        throw std::runtime_error("Result vector is too small to hold the buffer");
        result.resize(buf.size());
    }
    return q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto acc = sycl::accessor<T, N, sycl::access_mode::read>(buf, h, range, offset);
        h.copy(acc, result.data());
    });
}


template <typename T, std::size_t N = 1>
sycl::event initialize_device_buffer(sycl::queue& q, const std::vector<T> &vec, cl::sycl::buffer<T, N>& buf)
{
    return q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(vec.data(), acc);
    });
}

template <typename T, std::size_t N = 1>
cl::sycl::buffer<T> create_local_buffer(sycl::queue& q, const std::vector<T> &host_buffer, sycl::event& event)
{
    cl::sycl::buffer<T, N> result(host_buffer.size());
    event = q.submit([&](sycl::handler& h)
    {
        auto acc = sycl::local_accessor<T, N>(host_buffer, h);
        h.copy(host_buffer.data(), acc);
    });
    return result;
}

template <typename T, std::size_t N>
cl::sycl::buffer<T, N> create_device_buffer(sycl::queue& q, const std::vector<T> &host_buffer, const sycl::range<N>& range, sycl::event& event)
{
    cl::sycl::buffer<T, N> result(range);
    event = q.submit([&](sycl::handler& h)
    {
        auto acc = result.template get_access<sycl::access::mode::discard_write>(h);
        h.copy(host_buffer.data(), acc);
    });
    return result;
}


template <typename T>
std::vector<std::vector<T>> diff(const std::vector<std::vector<T>> &v)
{
    std::vector<std::vector<T>> res(v.size() - 1, std::vector<T>(v[0].size()));
    for (int i = 0; i < v.size() - 1; i++)
    {
        for (int j = 0; j < v[i].size(); j++)
        {
            res[i][j] = v[i + 1][j] - v[i][j];
        }
    }
    return res;
}
#endif
