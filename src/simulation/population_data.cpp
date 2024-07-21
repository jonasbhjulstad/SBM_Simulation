#include <SIR_SBM/simulation/population_data.hpp>
#include <SIR_SBM/graph/indices.hpp>
#include <fstream>
namespace SIR_SBM {
Population_Data::Population_Data(uint32_t N_sims, uint32_t N_partitions,
                                 uint32_t Nt)
    : std::vector<Population_Count>(N_sims * N_partitions * Nt),
      N_sims(N_sims), N_partitions(N_partitions), Nt(Nt) {}

Population_Data::Population_Data(const std::vector<Population_Count> &data,
                                 uint32_t N_sims, uint32_t N_partitions,
                                 uint32_t Nt)
    : std::vector<Population_Count>(data), N_sims(N_sims),
      N_partitions(N_partitions), Nt(Nt) {}
uint32_t Population_Data::get_partition_idx(uint32_t sim_idx,
                                             uint32_t partition_idx,
                                             uint32_t t) const {
  return sim_idx * N_partitions * Nt + partition_idx * Nt + t;
}

Population_Count Population_Data::operator()(uint32_t sim_idx,
                                       uint32_t partition_idx, uint32_t t) const {
  return this->operator[](get_partition_idx(sim_idx, partition_idx, t));
}
void Population_Data::write(const std::filesystem::path &fname) const {
  std::ofstream f;
  std::filesystem::create_directories(fname.parent_path());
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    std::filesystem::path p = fname;
    p += "_" + std::to_string(sim_idx) + ".csv";
    f.open(p);
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        auto pop = this->operator()(sim_idx, p_idx, t_idx);
        f << pop.S << "," << pop.I << "," << pop.R << ",";
      }
      f << std::endl;
    }
    f.close();
  }
}
} // namespace SIR_SBM