// reshape.hpp
//

#ifndef LZZ_reshape_hpp
#define LZZ_reshape_hpp
#include <stdexcept>
#include <vector>
#include <SIR_SBM/common.hpp>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  template <typename T>
  Vec3D <T> reshape (std::vector <uint32_t> const & vec, uint32_t N0, uint32_t N1, uint32_t N2);
}
namespace SIR_SBM
{
  template <typename T>
  Vec3D <T> reshape (std::vector <uint32_t> const & vec, uint32_t N0, uint32_t N1, uint32_t N2)
                              {
  if (vec.size() != N0 * N1 * N2) {
    throw std::runtime_error("Invalid size for reshape");
  }

  std::vector<std::vector<std::vector<T>>> res;
  res.reserve(N0);
  for (uint32_t i = 0; i < N0; i++) {
    std::vector<std::vector<T>> res1;
    res1.reserve(N1);
    for (uint32_t j = 0; j < N1; j++) {
      std::vector<T> res2;
      res2.reserve(N2);
      for (uint32_t k = 0; k < N2; k++) {
        res2.push_back(vec[i * N1 * N2 + j * N2 + k]);
      }
      res1.push_back(res2);
    }
    res.push_back(res1);
  }
  return res;
}
}
#undef LZZ_INLINE
#endif
