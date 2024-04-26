// population_count.hpp
//

#ifndef LZZ_population_count_hpp
#define LZZ_population_count_hpp
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  void validate_population (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::range <3> range, sycl::range <3> offset);
}
namespace SIR_SBM
{
  void validate_population (sycl::queue & q, sycl::buffer <SIR_State, 3> & state);
}
namespace SIR_SBM
{
  sycl::event partition_population_count (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <Population_Count, 3> & count, sycl::buffer <uint32_t> & vpc, size_t t_offset, sycl::event dep_event = {});
}
namespace SIR_SBM
{
  std::vector <Population_Count> partition_population_count (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <uint32_t> & vpc, size_t t_offset);
}
#undef LZZ_INLINE
#endif
