// combination.hpp
//

#ifndef LZZ_combination_hpp
#define LZZ_combination_hpp
#include <SIR_SBM/common.hpp>
#include <string>
#include <unordered_map>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  size_t n_choose_k (size_t n, size_t k);
}
namespace SIR_SBM
{
  Vec2D <int> combinations_with_replacement (int n, int k);
}
#undef LZZ_INLINE
#endif
