#pragma once
#hdr
#include <algorithm>
#include <boost/multi_array.hpp>
#include <cmath>
#include <limits>
#include <oneapi/tbb/info.h>
#include <tuple>
#include <vector>
#include <cstddef>
namespace SIR_SBM {
const uint32_t N_DEFAULT_THREADS = oneapi::tbb::info::default_concurrency();
template <typename T> using Vec3D = std::vector<std::vector<std::vector<T>>>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;
template <typename T> using Vec1D = std::vector<T>;
} // namespace SIR_SBM
#end
