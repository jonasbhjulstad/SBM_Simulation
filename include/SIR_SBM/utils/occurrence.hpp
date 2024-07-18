#pragma once

#include <vector>
#include <cstdint>


namespace SIR_SBM
{
    std::vector<uint32_t> count_occurrences(const std::vector<uint32_t>& samples, uint32_t N_bins);
}