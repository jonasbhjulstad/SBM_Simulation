#include <SIR_SBM/epidemiological/infection_count.hpp>
#include <SIR_SBM/graph/indices.hpp>
#include <algorithm>
namespace SIR_SBM {

uint32_t get_new_infections(
    const std::vector<Population_Count> &pop_count, uint32_t sim_idx,
    uint32_t p_idx, uint32_t N_partitions, uint32_t t_idx, uint32_t Nt) {
  Population_Count pop_t_1 =
      pop_count[get_partition_idx(sim_idx, p_idx, t_idx + 1, N_partitions, Nt)];
  Population_Count pop_t = pop_count[get_partition_idx(sim_idx, p_idx, t_idx, N_partitions, Nt)];
  int dI = (int)pop_t_1.I - (int)pop_t.I;
  int dR = (int)pop_t_1.R - (int)pop_t.R;
  return std::max<int>({dI + dR, 0});
}
} // namespace SIR_SBM