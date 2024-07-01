// sim_result.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_sim_result_hpp
#define LZZ_SIR_SBM_LZZ_sim_result_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sim_param.hpp>
#include <SIR_SBM/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <execution>
#include <filesystem>
#include <fstream>
#define LZZ_INLINE inline
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
namespace SIR_SBM
{
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  struct Sim_Result
  {
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    explicit Sim_Result (Sim_Param const & p, SBM_Graph const & G);
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void resize (Sim_Param const & p, SBM_Graph const & G);
#line 28 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    LinVec3D <uint32_t> contact_events;
#line 29 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    LinVec3D <Population_Count> population_count;
#line 33 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void write (std::filesystem::path const & dir);
#line 38 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void write_contact_events (std::filesystem::path const & dir);
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void write_population_count (std::filesystem::path const & dir);
#line 72 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void validate () const;
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    size_t N_partitions;
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    size_t N_connections;
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    size_t N_sims;
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    size_t Nt;
#line 87 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
  private:
#line 88 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void validate_partition_size (uint32_t sim_idx) const;
#line 107 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    std::vector <int> get_t_dI (uint32_t sim_idx, uint32_t t_idx) const;
#line 119 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    void validate_step (uint32_t sim_idx, uint32_t t) const;
#line 134 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_result.hpp"
    std::vector <uint32_t> merge_connection_infections (uint32_t sim_idx, uint32_t t_idx) const;
  };
}
#undef LZZ_INLINE
#endif
