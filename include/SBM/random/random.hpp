#pragma once

#include <cstdint>
#include <oneapi/dpl/random>
#include <random>
#include <vector>

namespace SBM {
std::vector<oneapi::dpl::ranlux48> generate_rngs_dpl(uint32_t seed, uint32_t N);
std::vector<std::mt19937> generate_rngs(uint32_t seed, uint32_t N);

std::vector<uint32_t> repeat_N_indices(const std::vector<uint32_t> weights);
std::vector<uint32_t>
discrete_finite_sample(std::mt19937 &rng, const std::vector<uint32_t> &weights,
                       uint32_t N_samples);

} // namespace SBM
