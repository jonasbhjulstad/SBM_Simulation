// sim_result.hpp
//

#ifndef LZZ_sim_result_hpp
#define LZZ_sim_result_hpp
#include <SIR_SBM/combination.hpp>
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sim_param.hpp>
#include <SIR_SBM/vector.hpp>
#include <filesystem>
#include <fstream>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  struct Sim_Result
  {
    explicit Sim_Result (Sim_Param const & p, SBM_Graph const & G);
    void resize (Sim_Param const & p, SBM_Graph const & G);
    std::vector <uint32_t> infected_count;
    std::vector <Population_Count> population_count;
    void write (std::filesystem::path const & dir);
    void write_infected_count (std::filesystem::path const & dir);
    void write_population_count (std::filesystem::path const & dir);
    void validate () const;
  private:
    uint32_t get_partition_linear_idx (uint32_t sim_idx, uint32_t t_idx, uint32_t p_idx) const;
    uint32_t get_connection_linear_idx (uint32_t sim_idx, uint32_t t_idx, uint32_t con_idx) const;
    void validate_partition_size (uint32_t sim_idx) const;
    std::vector <uint32_t> get_t_dI (uint32_t sim_idx, uint32_t t_idx) const;
    void validate_step (uint32_t sim_idx, uint32_t t) const;
    std::vector <uint32_t> merge_connection_infections (uint32_t sim_idx, uint32_t t_idx) const;
    size_t N_partitions;
    size_t N_connections;
    size_t N_sims;
    size_t Nt;
  };
}
#undef LZZ_INLINE
#endif
