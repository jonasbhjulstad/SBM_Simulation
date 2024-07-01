// vector.cpp
//

#include "vector.hpp"
#define LZZ_INLINE inline
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
namespace SIR_SBM
{
#line 29 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
  std::vector <uint32_t> make_iota (size_t N)
#line 30 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//vector.hpp"
{
    std::vector<uint32_t> result(N);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
}
#undef LZZ_INLINE
