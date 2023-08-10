#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP
#include <vector>
#include <cstdint>
#include <CL/sycl.hpp>
#include "Buffer_Utils_impl.hpp"
void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter);

void linewrite(std::ofstream &file, const std::vector<float> &val);

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter);

std::vector<uint32_t> pairlist_to_vec(const std::vector<std::pair<uint32_t, uint32_t>> &pairlist);
std::vector<std::pair<uint32_t, uint32_t>> vec_to_pairlist(const std::vector<uint32_t> &vec);
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed, sycl::event&);
std::vector<uint32_t> generate_seeds(uint32_t N_rng, uint32_t seed);

extern template std::vector<uint32_t> merge_vectors<uint32_t>(const std::vector<std::vector<uint32_t>> &);
extern template std::vector<float> merge_vectors<float>(const std::vector<std::vector<float>> &);
extern template sycl::buffer<uint32_t> buffer_create_1D<uint32_t>(sycl::queue &, const std::vector<uint32_t> &, sycl::event &);
extern template sycl::buffer<float> buffer_create_1D<float>(sycl::queue &, const std::vector<float> &, sycl::event &);
extern template sycl::buffer<uint32_t, 2> buffer_create_2D<uint32_t>(sycl::queue &, const std::vector<std::vector<uint32_t>> &, sycl::event &);
extern template sycl::buffer<float, 2> buffer_create_2D<float>(sycl::queue &, const std::vector<std::vector<float>> &, sycl::event &);
extern template sycl::buffer<SIR_State, 2> buffer_create_2D<SIR_State>(sycl::queue &, const std::vector<std::vector<SIR_State>> &, sycl::event &);

// extern template std::vector<std::vector<uint32_t>> read_buffer<uint32_t>(sycl::queue &q, sycl::buffer<uint32_t, 2> &buf,
//                                                                          sycl::event events);
// extern template std::vector<std::vector<float>> read_buffer<float>(sycl::queue &q, sycl::buffer<float, 2> &buf, sycl::event events);

// extern template std::vector<std::vector<uint32_t>> read_buffer<uint32_t>(sycl::queue &q, sycl::buffer<uint32_t, 2> &buf,
//                                                         sycl::event events, std::ofstream&);
// extern template std::vector<std::vector<float>> read_buffer<float>(sycl::queue &q, sycl::buffer<float, 2> &buf, sycl::event events, std::ofstream&);

template <typename T>
std::shared_ptr<T> read_buffer(sycl::queue &q, sycl::buffer<T, 2> &buf,
                                                        sycl::event &dep_event, sycl::event& event)
{

    auto range = buf.get_range();
    auto rows = range[0];
    auto cols = range[1];

    std::shared_ptr<T> p_data(new T[cols * rows]);

    event = q.submit([&](sycl::handler &h)
                          {
        //create accessor
        h.depends_on(dep_event);
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, p_data); });

    return p_data;
}

extern template std::vector<std::vector<uint32_t>> diff<uint32_t>(const std::vector<std::vector<uint32_t>> &v);
extern template std::vector<std::vector<int>> diff<int>(const std::vector<std::vector<int>> &v);

extern template std::shared_ptr<sycl::buffer<uint32_t, 1>> shared_buffer_create_1D<uint32_t>(sycl::queue &q, const std::vector<uint32_t> &data, sycl::event &res_event);
extern template std::shared_ptr<sycl::buffer<float, 1>> shared_buffer_create_1D<float>(sycl::queue &q, const std::vector<float> &data, sycl::event &res_event);

std::vector<std::vector<float>> generate_floats(uint32_t rows, uint32_t cols, float min, float max, uint32_t seed);
std::vector<std::vector<std::vector<float>>> generate_floats(uint32_t N0, uint32_t N1, uint32_t N2, float min, float max, uint32_t seed);



#endif
