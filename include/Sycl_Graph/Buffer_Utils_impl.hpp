#ifndef BUFFER_UTILS_IMPL_HPP
#define BUFFER_UTILS_IMPL_HPP
#include <fstream>
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>


template <typename T>
std::vector<T> merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}
template <typename T, std::size_t N>
std::vector<T> read_buffer(sycl::buffer<T,N>& buf, sycl::queue& q)
{
    std::vector<T> host_data(buf.size());
    auto event = q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, host_data.data());
    });
    event.wait();
    return host_data;
}

template <typename T>
std::vector<T> read_buffer(sycl::buffer<T>& buf, sycl::queue& q)
{
    std::vector<T> host_data(buf.get_count());
    auto event = q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, host_data.data());
    });
    event.wait();
    return host_data;
}


template <typename T, std::size_t N = 1>
cl::sycl::buffer<T> create_device_buffer(sycl::queue& q, const std::vector<T> &host_buffer, sycl::event& event)
{
    cl::sycl::buffer<T, N> result(host_buffer.size());
    event = q.submit([&](sycl::handler& h)
    {
        auto acc = result.template get_access<sycl::access::mode::discard_write>(h);
        h.copy(host_buffer.data(), acc);
    });
    event.wait();
    return result;
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
