#ifndef SYCL_GRAPH_UTILES_BUFFER_UTILS_HPP
#define SYCL_GRAPH_UTILES_BUFFER_UTILS_HPP
#include <CL/sycl.hpp>
#include <vector>
#include <Static_RNG/distributions.hpp>
#include <fstream>
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
sycl::event read_buffer(sycl::buffer<T,N>& buf, sycl::queue& q, std::vector<T>& result, sycl::event& dep_event)
{
    if (result.size() < buf.size())
    {
        throw std::runtime_error("Result vector is too small to hold the buffer");
        result.resize(buf.size());
    }
    return q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_event);
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
sycl::event initialize_device_buffer(sycl::queue& q, const std::vector<T> &vec, sycl::buffer<T, N>& buf)
{
    return q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(vec.data(), acc);
    });
}

template <typename T, std::size_t N = 1>
sycl::event initialize_device_buffer(sycl::queue& q, const std::vector<T> &&vec, sycl::buffer<T, N>& buf)
{
    return q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(h);
        h.copy(vec.data(), acc);
    });
}

template <typename T, std::size_t N = 3>
sycl::event clear_buffer(sycl::queue& q, sycl::buffer<T, N>& buf, std::vector<sycl::event>& dep_events)
{
    return q.submit([&](sycl::handler& h)
    {
        h.depends_on(dep_events);
        auto buf_acc = buf.template get_access<sycl::access::mode::write>(h);
        h.fill(buf_acc, (T)0);
    });
}


template <typename T, std::size_t N = 1>
sycl::buffer<T> create_local_buffer(sycl::queue& q, const std::vector<T> &host_buffer, sycl::event& event)
{
    sycl::buffer<T, N> result(host_buffer.size());
    event = q.submit([&](sycl::handler& h)
    {
        auto acc = sycl::local_accessor<T, N>(host_buffer, h);
        h.copy(host_buffer.data(), acc);
    });
    return result;
}

template <typename T, std::size_t N>
sycl::buffer<T, N> create_device_buffer(sycl::queue& q, const std::vector<T> &host_buffer, const sycl::range<N>& range, sycl::event& event)
{
    sycl::buffer<T, N> result(range);
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

void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter);

void linewrite(std::ofstream &file, const std::vector<float> &val);

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter);

std::vector<uint32_t> generate_seeds(uint32_t N_rng, uint32_t seed);
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed, sycl::event& event);

sycl::buffer<Static_RNG::default_rng> generate_rngs(sycl::queue& q, uint32_t N_rng, uint32_t seed, sycl::event& event);

sycl::buffer<Static_RNG::default_rng, 2> generate_rngs(sycl::queue& q, sycl::range<2> size, uint32_t seed, sycl::event& event);





std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist);

std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec);
std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t seed);

std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t N_rngs, uint32_t seed);

sycl::event generate_floats(sycl::queue& q, std::vector<float>& result, float min, float max, uint32_t seed);

#endif
