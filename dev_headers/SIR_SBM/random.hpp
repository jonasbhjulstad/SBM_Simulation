#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <oneapi/dpl/random>
#include <random>
#include <sycl.hpp>
#end
namespace SIR_SBM {
template <typename RNG>
std::vector<RNG> generate_rngs(uint32_t seed, size_t N) {
  RNG seeder(seed);
  std::vector<RNG> rngs(N);
  std::generate_n(rngs.begin(), N, [&seeder]() {
    uint32_t seed = seeder();
    return RNG(seed);
  });
  return rngs;
}


std::vector<uint32_t> discrete_finite_sample(std::mt19937_64& rng, const std::vector<uint32_t>& weights, size_t N_samples) {
    std::vector<double> vals(weights.size());
    std::uniform_real_distribution<double> dist(0, 1);
    std::transform(weights.begin(), weights.end(), vals.begin(), [&](auto w){return std::pow(dist(rng), 1. / w);});
    std::vector<std::pair<int, double>> valsWithIndices;
    for (size_t iter = 0; iter < vals.size(); iter++) {
        valsWithIndices.emplace_back(iter, vals[iter]);
    }
    std::sort(valsWithIndices.begin(), valsWithIndices.end(), [](auto x, auto y) {return x.second > y.second; });
    std::vector<uint32_t> samples(N_samples);
    std::transform(valsWithIndices.begin(), valsWithIndices.begin() + N_samples, samples.begin(), [](auto x) {return x.first; });
    return samples;
}


} // namespace SIR_SBM
