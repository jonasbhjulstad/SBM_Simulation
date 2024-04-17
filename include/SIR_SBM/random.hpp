#pragma once
#include <SIR_SBM/common.hpp>
#include <sycl.hpp>
#include <oneapi/dpl/random>
template <typename RNG>
std::vector<RNG> generate_rngs(uint32_t seed, size_t N)
{
    RNG seeder(seed);
    std::vector<RNG> rngs(N);
    std::generate_n(rngs.begin(), N, [&seeder](){
        uint32_t seed = seeder();
        return RNG(seed);});
    return rngs;
}

