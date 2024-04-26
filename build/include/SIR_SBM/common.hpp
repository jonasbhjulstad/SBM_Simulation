// common.hpp
//

#ifndef LZZ_common_hpp
#define LZZ_common_hpp
#include <algorithm>
#include <cmath>
#include <limits>
#include <oneapi/tbb/info.h>
#include <tuple>
#include <vector>
namespace SIR_SBM {
const size_t N_DEFAULT_THREADS = oneapi::tbb::info::default_concurrency();
template <typename T> using Vec3D = std::vector<std::vector<std::vector<T>>>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;
} // namespace SIR_SBM
#define LZZ_INLINE inline
#undef LZZ_INLINE
#endif
