#ifndef VECTOR_REMAP_HPP
#define VECTOR_REMAP_HPP

#include <Sycl_Graph/SIR_Types.hpp>
std::vector<std::vector<std::vector<State_t>>> vector_remap(std::vector<State_t> &input, size_t N0, size_t N1, size_t N2);

std::vector<std::vector<std::vector<uint32_t>>> vector_remap(std::vector<uint32_t> &input, size_t N0, size_t N1, size_t N2);

#endif
