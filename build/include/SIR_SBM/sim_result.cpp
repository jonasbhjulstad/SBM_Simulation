// sim_result.cpp
//

#include "sim_result.hpp"
#define LZZ_INLINE inline
namespace SIR_SBM
{
  Sim_Result::Sim_Result (Sim_Param const & p, SBM_Graph const & G)
    : infected_count (std::vector<uint32_t>(G.N_connections() * 2 * p.N_sims *
                                             (p.Nt + 1))), population_count (std::vector<Population_Count>(G.N_partitions() *
                                                       p.N_sims * (p.Nt + 1))), N_partitions (G.N_partitions()), N_connections (G.N_connections()), N_sims (p.N_sims), Nt (p.Nt)
                                   {}
}
namespace SIR_SBM
{
  void Sim_Result::resize (Sim_Param const & p, SBM_Graph const & G)
                                                      {
    infected_count.resize(G.N_connections() * 2 * p.N_sims * (p.Nt + 1));
    population_count.resize(G.N_partitions() * p.N_sims * (p.Nt + 1));
  }
}
namespace SIR_SBM
{
  void Sim_Result::write (std::filesystem::path const & dir)
                                               {
    write_infected_count(dir);
    write_population_count(dir);
  }
}
namespace SIR_SBM
{
  void Sim_Result::write_infected_count (std::filesystem::path const & dir)
                                                              {
    std::ofstream f;
    std::filesystem::create_directories(dir);

    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
      f.open(dir / ("infected_count_" + std::to_string(sim_idx) + ".csv"));
      for (int t_idx = 0; t_idx < Nt + 1; t_idx++) {
        for (int c_idx = 0; c_idx < 2 * N_connections; c_idx++) {
          auto idx = get_connection_linear_idx(sim_idx, t_idx, c_idx);
          f << infected_count[idx] << ",";
        }
        f << std::endl;
      }
      f.close();
    }
  }
}
namespace SIR_SBM
{
  void Sim_Result::write_population_count (std::filesystem::path const & dir)
                                                                {
    std::ofstream f;
    std::filesystem::create_directories(dir);
    size_t idx;
    Population_Count pc;
    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
      f.open(dir / ("population_count_" + std::to_string(sim_idx) + ".csv"));
      for (int t_idx = 0; t_idx < Nt; t_idx++) {
        for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
          idx = get_partition_linear_idx(sim_idx, t_idx, p_idx);
          pc = population_count[idx];
          f << pc.S << "," << pc.I << "," << pc.R << ",";
        }
        f << std::endl;
      }
      f.close();
    }
  }
}
namespace SIR_SBM
{
  void Sim_Result::validate () const
                        {

    auto sim_vec = make_iota<uint32_t>(N_sims);
    auto t_vec = make_iota<uint32_t>(Nt);
    std::for_each(sim_vec.begin(), sim_vec.end(),
                  [&](uint32_t sim_idx) { validate_partition_size(sim_idx); });

    std::for_each(sim_vec.begin(), sim_vec.end(), [&](uint32_t sim_idx) {
      std::for_each(t_vec.begin(), t_vec.end(),
                    [&](uint32_t t_idx) { validate_step(sim_idx, t_idx); });
    });
  }
}
namespace SIR_SBM
{
  uint32_t Sim_Result::get_partition_linear_idx (uint32_t sim_idx, uint32_t t_idx, uint32_t p_idx) const
                                                          {
    return p_idx * N_sims * (Nt + 1) + sim_idx * (Nt + 1) + t_idx;
  }
}
namespace SIR_SBM
{
  uint32_t Sim_Result::get_connection_linear_idx (uint32_t sim_idx, uint32_t t_idx, uint32_t con_idx) const
                                                             {
    return con_idx * N_sims * (Nt + 1) + sim_idx * (Nt + 1) + t_idx;
  }
}
namespace SIR_SBM
{
  void Sim_Result::validate_partition_size (uint32_t sim_idx) const
                                                       {
    std::vector<uint32_t> start_pop_size(N_partitions, 0);
    std::transform(population_count.begin(), population_count.end(),
                   start_pop_size.begin(),
                   [](Population_Count pc) { return pc.S + pc.I + pc.R; });
    for (int t = 0; t < Nt + 1; t++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        auto idx = get_partition_linear_idx(sim_idx, t, p_idx);
        auto pc = population_count[idx];
        if (pc.S + pc.I + pc.R != start_pop_size[p_idx]) {
          std::string msg = "Inconsistent population count for partition " +
                            std::to_string(p_idx) + " at time " +
                            std::to_string(t);
          throw std::runtime_error(msg);
        }
      }
    }
  }
}
namespace SIR_SBM
{
  std::vector <uint32_t> Sim_Result::get_t_dI (uint32_t sim_idx, uint32_t t_idx) const
                                                                         {
    std::vector<uint32_t> t_dI(N_partitions, 0);
    for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
      auto idx = get_partition_linear_idx(sim_idx, t_idx, p_idx);
      t_dI[p_idx] = population_count[idx + 1].I - population_count[idx].I;
    }
    return t_dI;
  }
}
namespace SIR_SBM
{
  void Sim_Result::validate_step (uint32_t sim_idx, uint32_t t) const
                                                         {
    auto con_infs = merge_connection_infections(sim_idx, t);
    auto t_dI = get_t_dI(sim_idx, t);
    for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
      if (t_dI[p_idx] > con_infs[p_idx]) {
        std::string msg = "Inconsistent dI count for partition " +
                          std::to_string(p_idx) + " at time " +
                          std::to_string(t) + " for sim " +
                          std::to_string(sim_idx) + ": dI = " +
                          std::to_string(t_dI[p_idx]) + ", con_infs = " +
                          std::to_string(con_infs[p_idx]);
        throw std::runtime_error(msg);
      }
    }
  }
}
namespace SIR_SBM
{
  std::vector <uint32_t> Sim_Result::merge_connection_infections (uint32_t sim_idx, uint32_t t_idx) const
                                                                          {
    std::vector<uint32_t> connection_infections(2 * N_connections, 0);
    uint32_t con_idx = 0;
    for (auto comb : combinations_with_replacement(N_partitions, 2)) {
      //forward
      auto from = comb[0];
      auto to = comb[1];
      
      auto forward_idx = get_connection_linear_idx(sim_idx, t_idx, to);
      auto backward_idx = get_connection_linear_idx(sim_idx, t_idx, 2*from);
      connection_infections[to] += infected_count[forward_idx] + infected_count[backward_idx];

    }
    return connection_infections;
  }
}
#undef LZZ_INLINE
