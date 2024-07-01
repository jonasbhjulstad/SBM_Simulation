// infection_sampling.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_infection_sampling_hpp
#define LZZ_SIR_SBM_LZZ_infection_sampling_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/csv.hpp>
#include <SIR_SBM/eigen.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/sim_result.hpp>
#include <SIR_SBM/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#define LZZ_INLINE inline
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <int> get_connection_indices (int N_partitions, int p_idx);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 31 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <uint32_t> get_partition_connection_contacts (Vec2DView <uint32_t> const & contact_events, int N_partitions, int p_idx);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <uint32_t> sample_infections (Vec2DView <uint32_t> const & contact_events, Vec2DView <Population_Count> const & population_count, int p_idx, int t_idx, std::mt19937_64 & rng);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 60 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  LinVec2D <uint32_t> simulation_sample_infections (Vec2DView <uint32_t> const & contact_events, Vec2DView <Population_Count> const & population_count, std::mt19937_64 & rng);
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 88 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  Vec3D <uint32_t> sample_infections (Vec3D <uint32_t> const & contact_events, Vec3D <Population_Count> const & population_count, int seed);
}
#undef LZZ_INLINE
#endif
