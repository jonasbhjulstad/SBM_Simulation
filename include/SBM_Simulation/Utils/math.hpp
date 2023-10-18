#ifndef SBM_SIMULATION_UTILS_MATH_HPP
#define SBM_SIMULATION_UTILS_MATH_HPP
#include <vector>
#include <cstdint>
std::vector<float> make_linspace(float start, float end, float step);

std::vector<uint32_t> make_iota(uint32_t N);
std::vector<uint32_t> make_iota(uint32_t start, uint32_t end);
#endif
