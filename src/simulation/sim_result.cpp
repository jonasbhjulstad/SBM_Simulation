#include <SIR_SBM/epidemiological/epidemiological.hpp>
#include <SIR_SBM/epidemiological/infection_count.hpp>
#include <SIR_SBM/graph/indices.hpp>
#include <SIR_SBM/math/numeric.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <SIR_SBM/utils/csv.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <execution>
#include <fstream>

namespace SIR_SBM {
Sim_Result::Sim_Result(const Sim_Param &p, const SBM_Graph &G)
    : contact_events(p.N_sims, G.N_connections(), p.Nt),
      population_count(p.N_sims, G.N_partitions(), (p.Nt + 1)),
      N_partitions(G.N_partitions()), N_connections(G.N_connections()),
      N_sims(p.N_sims), Nt(p.Nt), N_pops(G.N_partition_vertices()),
      N_contact_events(p.N_sims * G.N_connections() * 2 * p.Nt) {}
void Sim_Result::resize(const Sim_Param &p, const SBM_Graph &G) {
  contact_events = Connection_Data(p.N_sims, G.N_connections(), p.Nt);
  population_count = Population_Data(p.N_sims, G.N_partitions(), (p.Nt + 1));
  N_partitions = G.N_partitions();
  N_connections = G.N_connections();
  N_sims = p.N_sims;
  N_pops = G.N_partition_vertices();
  N_contact_events = p.N_sims * G.N_connections() * 2 * p.Nt;

  Nt = p.Nt;
}
uint32_t Sim_Result::partition_idx(uint32_t sim_idx, uint32_t p_idx,
                                   uint32_t t_idx) const {
  return get_partition_idx(sim_idx, p_idx, t_idx, N_partitions, Nt + 1);
}
uint32_t Sim_Result::from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                         uint32_t t_idx) const {
  return get_from_connection_idx(sim_idx, con_idx, t_idx, N_connections, Nt);
}
uint32_t Sim_Result::to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                       uint32_t t_idx) const {
  return get_to_connection_idx(sim_idx, con_idx, t_idx, N_connections, Nt);
}

void Sim_Result::write(const std::filesystem::path &dir) {
  std::filesystem::path c_name = dir;
  c_name += "/contact_events";
  std::filesystem::path p_name = dir;
  p_name += "/population_count";
  write_contact_events(contact_events, c_name, N_sims, N_connections, Nt);
  write_population_count(population_count, p_name, N_sims, N_partitions,
                         Nt + 1);
  std::filesystem::path i_name = dir;
  i_name += "/partition_infections";
  write_partition_infections(population_count, i_name, N_sims, N_partitions,
                             Nt);

  std::filesystem::path pc_name = dir;
  i_name += "/partition_contacts";
  write_partition_contacts(contact_events, pc_name, N_sims, N_partitions, Nt);
}

void Sim_Result::validate() const {

  auto sim_vec = make_iota(N_sims);
  auto t_vec = make_iota(Nt);
  std::for_each(sim_vec.begin(), sim_vec.end(),
                [&](uint32_t sim_idx) { validate_partition_size(sim_idx); });

  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      validate_step(sim_idx, t_idx);
    }
  }
}

void Sim_Result::validate_partition_size(uint32_t sim_idx) const {
  for (int t = 0; t < Nt + 1; t++) {
    for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
      auto pc = population_count[partition_idx(sim_idx, p_idx, t)];
      if (pc.S + pc.I + pc.R != N_pops[p_idx]) {
        std::string msg = "Inconsistent population count for partition " +
                          std::to_string(p_idx) + " at time " +
                          std::to_string(t);
        throw std::runtime_error(msg);
      }
    }
  }
}

std::vector<uint32_t> Sim_Result::get_t_infections(uint32_t sim_idx,
                                                   uint32_t t) const {
  std::vector<uint32_t> t_infs(N_partitions, 0);
  for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
    t_infs[p_idx] = get_new_infections(population_count, sim_idx, p_idx,
                                       N_partitions, t, Nt + 1);
  }
  return t_infs;
}

void Sim_Result::validate_step(uint32_t sim_idx, uint32_t t) const {
  auto con_infs = merge_connection_infections(sim_idx, t);
  auto t_dI = get_t_infections(sim_idx, t);
  for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
    if (t_dI[p_idx] > con_infs[p_idx]) {
      auto p_inf = con_infs[p_idx];
      auto dI = t_dI[p_idx];
      std::string msg =
          "Inconsistent dI count for partition " + std::to_string(p_idx) +
          " at time " + std::to_string(t) + " for sim " +
          std::to_string(sim_idx) + ": dI = " + std::to_string(dI) +
          ", con_infs = " + std::to_string(p_inf);
      throw std::runtime_error(msg);
    }
  }
}

std::vector<uint32_t>
Sim_Result::merge_connection_infections(uint32_t sim_idx,
                                        uint32_t t_idx) const {
  std::vector<uint32_t> connection_infections(N_connections, 0);
  uint32_t con_idx = 0;
  for (auto comb :
       iter::combinations_with_replacement(make_iota(N_partitions), 2)) {
    // forward
    auto from = comb[0];
    auto to = comb[1];

    connection_infections[to] +=
        contact_events[to_connection_idx(sim_idx, con_idx, t_idx)];
    connection_infections[from] +=
        contact_events[from_connection_idx(sim_idx, con_idx, t_idx)];
    con_idx++;
  }
  return connection_infections;
}

} // namespace SIR_SBM