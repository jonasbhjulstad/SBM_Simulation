#ifndef SBM_SIMULATION_UTILS_MATH_HPP
#define SBM_SIMULATION_UTILS_MATH_HPP
#include <vector>
#include <CL/sycl.hpp>
#include <cstdint>
std::vector<float> make_linspace(float start, float end, float step);
SYCL_EXTERNAL uint32_t ceil_div(uint32_t a, uint32_t b);
SYCL_EXTERNAL uint32_t floor_div(uint32_t a, uint32_t b);
std::vector<uint32_t> make_iota(uint32_t N);
std::vector<uint32_t> make_iota(uint32_t start, uint32_t end);
#endif
