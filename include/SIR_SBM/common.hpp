#pragma once
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <oneapi/tbb/info.h>
namespace SIR_SBM
{
    const size_t N_DEFAULT_THREADS = oneapi::tbb::info::default_concurrency();
    template <typename T>
    using Vec3D = std::vector<std::vector<std::vector<T>>>;
} 