#pragma once
#hdr
#include <vector>
#include <cstdint>
#end
#src
#include <unordered_map>
#include <SIR_SBM/vector/routines.hpp>
#end
namespace SIR_SBM
{
    std::vector<uint32_t> count_occurrences(const std::vector<uint32_t>& samples, uint32_t N_bins)
    {
        auto counts = std::unordered_map<uint32_t, uint32_t>();
        std::vector<uint32_t> result(N_bins, 0);
        for (uint32_t i = 0; i < samples.size(); i++)
        {
            counts[samples[i]]++;
        }
        for (uint32_t i = 0; i < N_bins; i++)
        {
            result[i] = counts[i];
        }
        return result;
    }
}