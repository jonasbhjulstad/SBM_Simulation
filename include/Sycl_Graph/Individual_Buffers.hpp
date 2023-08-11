#ifndef INDIVIDUAL_BUFFERS_HPP
#define INDIVIDUAL_BUFFERS_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <cstdint>
#include <vector>

struct Individual_Buffer_Data
{
    std::vector<std::vector<std::vector<SIR_State>>> vertex_state;
    std::vector<std::vector<std::vector<uint32_t>>> from_events;
    std::vector<std::vector<std::vector<uint32_t>>> to_events;
};
struct Individual_Buffers
{
    Individual_Buffers(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, uint32_t N_connections, uint32_t Nt, uint32_t N_communities, uint32_t N_pop, uint32_t N_edges, uint32_t N_wg, uint32_t seed);

    // default copy constructor
    Individual_Buffers(const Individual_Buffers &) = default;
    std::vector<sycl::event> events = std::vector<sycl::event>(4);

    sycl::buffer<Static_RNG::default_rng> rngs;
    sycl::buffer<SIR_State, 3> trajectory;
    sycl::buffer<uint32_t, 3> events_from;
    sycl::buffer<uint32_t, 3> events_to;
    sycl::buffer<uint8_t, 2> infections_from;
    sycl::buffer<uint8_t, 2> infections_to;
    sycl::buffer<float, 3> p_Is;
    std::vector<sycl::event> events = std::vector<sycl::event>(4);

    const uint32_t N_connections, Nt, N_communities, N_pop, N_edges, N_wg, seed;

    sycl::event initialize_trajectory(sycl::queue &q, float p_I0, float p_R0, std::vector<sycl::event> dep_event);

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<std::vector<float>>> &p_Is_vec, , const std::vector<uint32_t> &init_seeds);

    static std::vector<Individual_Buffers> make(sycl::queue &q, const std::vector<std::vector<float>> &p_Is, const std::vector<uint32_t> &init_seeds);
    Individual_Buffer_Data read_buffers(sycl::queue &q, auto dep_event);
};

template <typename T>
std::vector<T> read_pointer(const std::shared_ptr<T> &p, uint32_t N)
{
    std::vector<T> result(N);
    std::copy(p.get(), p.get() + N, result.begin());
    return result;
}

template <typename T>
std::vector<std::vector<T>> read_pointer(const std::shared_ptr<T> &p, uint32_t rows, uint32_t cols)
{
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols));
    for (int i = 0; i < rows; i++)
    {
        std::copy(p.get() + i * cols, p.get() + (i + 1) * cols, result[i].begin());
    }
    return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> read_pointer(const std::shared_ptr<T> &p, sycl::range<3> size)
{
    std::vector<std::vector<std::vector<T>>> result(size[0]);
    std::generate(result.begin(), result.end(), [cols]() { return std::vector<std::vector<T>>(size[1], std::vector<T>(size[2])); });
    for(int i = 0; i < size[0]; i++)
    {
        for(int j = 0; j < size[1]; j++)
        {
            std::copy(p.get() + i * size[1] * size[2] + j * size[2], p.get() + i * size[1] * size[2] + (j + 1) * size[2], result[i][j].begin());
        }
    }
    return result;
}

#endif
