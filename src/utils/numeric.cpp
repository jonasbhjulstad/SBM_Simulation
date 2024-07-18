#include <SIR_SBM/utils/numeric.hpp>
#include <numeric>
namespace SIR_SBM {
std::vector<uint32_t> make_iota(uint32_t N) {
  std::vector<uint32_t> result(N);
  std::iota(result.begin(), result.end(), 0);
  return result;
}
} // namespace SIR_SBM