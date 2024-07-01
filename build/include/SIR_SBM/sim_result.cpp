// sim_result.cpp
//

#include "sim_result.hpp"
#define LZZ_INLINE inline
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  Sim_Result::Sim_Result (Sim_Param const & p, SBM_Graph const & G)
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    : contact_events (p.N_sims, G.N_connections() * 2, p.Nt), population_count (p.N_sims, G.N_partitions(), p.Nt + 1), N_partitions (G.N_partitions()), N_connections (G.N_connections()), N_sims (p.N_sims), Nt (p.Nt)
#line 18 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                   {}
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::resize (Sim_Param const & p, SBM_Graph const & G)
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                      {
    contact_events = LinVec3D<uint32_t>(p.N_sims, G.N_connections() * 2, p.Nt);
    population_count =
        LinVec3D<Population_Count>(p.N_sims, G.N_partitions(), p.Nt + 1);
    N_partitions = G.N_partitions();
    N_connections = G.N_connections();
    N_sims = p.N_sims;
    Nt = p.Nt;
  }
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 33 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::write (std::filesystem::path const & dir)
#line 33 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                               {
    write_contact_events(dir);
    write_population_count(dir);
  }
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 38 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::write_contact_events (std::filesystem::path const & dir)
#line 38 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                              {
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
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::write_population_count (std::filesystem::path const & dir)
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                                {
    std::ofstream f;
    std::filesystem::create_directories(dir);
    size_t idx;
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
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 72 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::validate () const
#line 72 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
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
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 88 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::validate_partition_size (uint32_t sim_idx) const
#line 88 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                       {
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
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 107 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  std::vector <int> Sim_Result::get_t_dI (uint32_t sim_idx, uint32_t t_idx) const
#line 107 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                                    {
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
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 119 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  void Sim_Result::validate_step (uint32_t sim_idx, uint32_t t) const
#line 119 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                         {
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
}
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 134 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  std::vector <uint32_t> Sim_Result::merge_connection_infections (uint32_t sim_idx, uint32_t t_idx) const
#line 135 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
                                                                          {
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
}
#undef LZZ_INLINE
