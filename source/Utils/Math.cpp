#include <SBM_Graph/Utils/Math.hpp>
#include <numeric>
namespace SBM_Graph {
  std::vector<uint32_t> make_iota(uint32_t N)
  {
    std::vector<uint32_t> iota(N);
    std::iota(iota.begin(), iota.end(), 0);
    return iota;
  }
}
