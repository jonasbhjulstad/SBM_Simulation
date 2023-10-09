#ifndef SYCL_GRAPH_UTILS_MATH_HPP
#define SYCL_GRAPH_UTILS_MATH_HPP
#include <vector>
#include <cstdint>
std::vector<float> make_linspace(float start, float end, float step);

std::vector<uint32_t> make_iota(uint32_t N);
#endif
