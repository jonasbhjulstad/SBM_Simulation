module;

import SIR_SBM.epidemiological;
import SIR_SBM.graph;
import SIR_SBM.simulation.parameters;
import SIR_SBM.vector;
#include <cppitertools/combinations_with_replacement.hpp>
#include <filesystem>
#include <fstream>

export module SIR_SBM.simulation.result;



struct Sim_Result {
  explicit Sim_Result(const Sim_Param &p, const SBM_Graph &G)
      : contact_events(make_shared_array<uint32_t>(
            p.N_sims * G.N_connections() * 2 * p.Nt)),
        population_count(make_shared_array<Population_Count>(
            p.N_sims * G.N_partitions() * p.Nt + 1)),
        N_partitions(G.N_partitions()), N_connections(G.N_connections()),
        N_sims(p.N_sims), Nt(p.Nt), N_pop_count(p.N_sims*G.N_partitions()*(p.Nt+1)), N_contact_events(p.N_sims*G.N_connections()*2*p.Nt) {}
  void resize(const Sim_Param &p, const SBM_Graph &G) {
    contact_events =
        make_shared_array<uint32_t>(p.N_sims * G.N_connections() * 2 * p.Nt);
    population_count = make_shared_array<Population_Count>(
        p.N_sims * G.N_partitions() * p.Nt + 1);
    N_partitions = G.N_partitions();
    N_connections = G.N_connections();
    N_sims = p.N_sims;
    N_pop_count = p.N_sims * G.N_partitions() * (p.Nt + 1);
    N_contact_events = p.N_sims * G.N_connections() * 2 * p.Nt;

    Nt = p.Nt;
  }
  std::shared_ptr<uint32_t> contact_events;
  std::shared_ptr<Population_Count> population_count;
  uint32_t N_pop_count;
  uint32_t N_contact_events;
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
          f << contact_events.get()[get_linear_idx(
                   {sim_idx, c_idx, t_idx}, {N_sims, 2 * N_connections, Nt})]
            << ",";
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
          pc = population_count.get()[get_linear_idx(
              {sim_idx, p_idx, t_idx}, {N_sims, N_partitions, Nt})];
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
    std::vector<uint32_t> start_pop_size(N_pop_count);
    for (int t = 0; t < Nt + 1; t++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        auto pc = population_count.get()[get_linear_idx(
            {sim_idx, p_idx, t}, {N_sims, N_partitions, Nt})];
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
      auto idx_t =
          get_linear_idx({sim_idx, p_idx, t_idx}, {N_sims, N_partitions, Nt});
      auto idx_t1 = get_linear_idx({sim_idx, p_idx, t_idx + 1},
                                   {N_sims, N_partitions, Nt});
      auto R_diff = population_count.get()[idx_t1].R - population_count.get()[idx_t].R;
      auto I_diff = population_count.get()[idx_t1].I - population_count.get()[idx_t].I;
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
    for (auto comb : iter::combinations_with_replacement(
             make_iota<uint32_t>(N_partitions), 2)) {
      // forward
      auto from = comb[0];
      auto to = comb[1];

      auto con_to_idx = get_linear_idx({sim_idx, 2 * con_idx, t_idx},
                                       {N_sims, 2 * N_connections, Nt});
      auto con_from_idx = get_linear_idx({sim_idx, 2 * con_idx + 1, t_idx},
                                         {N_sims, 2 * N_connections, Nt});
      connection_infections[to] += contact_events.get()[con_to_idx];
      connection_infections[from] += contact_events.get()[con_from_idx];
      con_idx++;
    }
    return connection_infections;
  }
};

