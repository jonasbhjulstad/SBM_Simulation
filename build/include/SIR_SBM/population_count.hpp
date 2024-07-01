// population_count.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_population_count_hpp
#define LZZ_SIR_SBM_LZZ_population_count_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/vector.hpp>
#define LZZ_INLINE inline
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
namespace SIR_SBM
{
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
  void validate_population (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::range <3> range, sycl::range <3> offset);
}
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
namespace SIR_SBM
{
#line 28 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
  void validate_population (sycl::queue & q, sycl::buffer <SIR_State, 3> & state);
}
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
namespace SIR_SBM
{
#line 32 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
  sycl::event partition_population_count (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <Population_Count, 3> & count, sycl::buffer <uint32_t> & vpc, size_t t_offset, sycl::event dep_event = {});
}
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
namespace SIR_SBM
{
#line 86 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
  std::vector <Population_Count> partition_population_count (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <uint32_t> & vpc, size_t t_offset);
}
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
namespace SIR_SBM
{
#line 97 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//population_count.hpp"
  uint32_t get_new_infections (Vec2DView <Population_Count> const & pop_count, uint32_t p_idx, uint32_t t_idx);
}
#undef LZZ_INLINE
#endif
