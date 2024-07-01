#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sim_param.hpp>
#include <SIR_SBM/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <execution>
#include <filesystem>
#include <fstream>
#end
namespace SIR_SBM {
struct Sim_Result {
  explicit Sim_Result(const Sim_Param &p, const SBM_Graph &G)
      : contact_events(p.N_sims, G.N_connections() * 2, p.Nt),
        population_count(p.N_sims, G.N_partitions(), p.Nt + 1),
        N_partitions(G.N_partitions()), N_connections(G.N_connections()),
        N_sims(p.N_sims), Nt(p.Nt) {}
  void resize(const Sim_Param &p, const SBM_Graph &G) {
    contact_events = LinVec3D<uint32_t>(p.N_sims, G.N_connections() * 2, p.Nt);
    population_count =
        LinVec3D<Population_Count>(p.N_sims, G.N_partitions(), p.Nt + 1);
    N_partitions = G.N_partitions();
    N_connections = G.N_connections();
    N_sims = p.N_sims;
    Nt = p.Nt;
  }
  LinVec3D<uint32_t> contact_events;
  LinVec3D<Population_Count> population_count;
  // Vec3D<uint32_t> contact_events;
  // LinearVector3D<Population_Count> population_count;

  void write(const std::filesystem::path &dir) {
    write_contact_events(dir);
    write_population_count(dir);
  }

  void write_contact_events(const std::filesystem::path &dir) {
    std::ofstream f;
    std::filesystem::create_directories(dir);

    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
      f.open(dir / ("contact_events_" + std::to_string(sim_idx) + ".csv"));
      for (int t_idx = 0; t_idx < Nt; t_idx++) {
        for (int c_idx = 0; c_idx < 2 * N_connections; c_idx++) {
          f << contact_events(sim_idx, c_idx, t_idx) << ",";
        }
        f << std::endl;
      }
      f.close();
    }
  }

  void write_population_count(const std::filesystem::path &dir) {
    std::ofstream f;
    std::filesystem::create_directories(dir);
    uint32_t idx;
    Population_Count pc;
    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
      f.open(dir / ("population_count_" + std::to_string(sim_idx) + ".csv"));
      for (int t_idx = 0; t_idx < Nt; t_idx++) {
        for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
          pc = population_count(sim_idx, p_idx, t_idx);
          f << pc.S << "," << pc.I << "," << pc.R << ",";
        }
        f << std::endl;
      }
      f.close();
    }
  }

  void validate() const {

    auto sim_vec = make_iota<uint32_t>(N_sims);
    auto t_vec = make_iota<uint32_t>(Nt);
    std::for_each(sim_vec.begin(), sim_vec.end(),
                  [&](uint32_t sim_idx) { validate_partition_size(sim_idx); });

    std::for_each(sim_vec.begin(), sim_vec.end(), [&](uint32_t sim_idx) {
      std::for_each(t_vec.begin(), t_vec.end(),
                    [&](uint32_t t_idx) { validate_step(sim_idx, t_idx); });
    });
  }

  uint32_t N_partitions, N_connections, N_sims, Nt;

private:
  void validate_partition_size(uint32_t sim_idx) const {
    std::vector<uint32_t> start_pop_size;
    std::transform(population_count.data(),
                   population_count.data() + population_count.size(),
                   std::back_inserter(start_pop_size),
                   [](Population_Count pc) { return pc.S + pc.I + pc.R; });
    for (int t = 0; t < Nt + 1; t++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        auto pc = population_count(sim_idx, p_idx, t);
        if (pc.S + pc.I + pc.R != start_pop_size[p_idx]) {
          std::string msg = "Inconsistent population count for partition " +
                            std::to_string(p_idx) + " at time " +
                            std::to_string(t);
          throw std::runtime_error(msg);
        }
      }
    }
  }

  std::vector<int> get_t_dI(uint32_t sim_idx, uint32_t t_idx) const {
    std::vector<int> t_dI(N_partitions, 0);
    for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
      auto R_diff = population_count(sim_idx, p_idx, t_idx + 1).R -
                    population_count(sim_idx, p_idx, t_idx).R;
      auto I_diff = population_count(sim_idx, p_idx, t_idx + 1).I -
                    population_count(sim_idx, p_idx, t_idx).I;
      t_dI[p_idx] = I_diff + R_diff;
    }
    return t_dI;
  }

  void validate_step(uint32_t sim_idx, uint32_t t) const {
    auto con_infs = merge_connection_infections(sim_idx, t);
    auto t_dI = get_t_dI(sim_idx, t);
    for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
      if (t_dI[p_idx] > con_infs[p_idx]) {
        std::string msg =
            "Inconsistent dI count for partition " + std::to_string(p_idx) +
            " at time " + std::to_string(t) + " for sim " +
            std::to_string(sim_idx) + ": dI = " + std::to_string(t_dI[p_idx]) +
            ", con_infs = " + std::to_string(con_infs[p_idx]);
        throw std::runtime_error(msg);
      }
    }
  }

  std::vector<uint32_t> merge_connection_infections(uint32_t sim_idx,
                                                    uint32_t t_idx) const {
    std::vector<uint32_t> connection_infections(N_connections, 0);
    uint32_t con_idx = 0;
    for (auto comb :
         iter::combinations_with_replacement(make_iota(N_partitions), 2)) {
      // forward
      auto from = comb[0];
      auto to = comb[1];

      connection_infections[to] += contact_events(sim_idx, 2 * con_idx, t_idx);
      connection_infections[from] +=
          contact_events(sim_idx, 2 * con_idx + 1, t_idx);
      con_idx++;
    }
    return connection_infections;
  }
};

} // namespace SIR_SBM