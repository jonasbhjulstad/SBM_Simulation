// common.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_common_hpp
#define LZZ_SIR_SBM_LZZ_common_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//common.hpp"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <cmath>
#include <limits>
#include <oneapi/tbb/info.h>
#include <tuple>
#include <vector>
#include <cstddef>
namespace SIR_SBM {
const size_t N_DEFAULT_THREADS = oneapi::tbb::info::default_concurrency();
template <typename T> using Vec3D = std::vector<std::vector<std::vector<T>>>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;
template <typename T> using Vec1D = std::vector<T>;
} // namespace SIR_SBM
#define LZZ_INLINE inline
#undef LZZ_INLINE
#endif
