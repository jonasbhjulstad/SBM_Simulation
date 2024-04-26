// epidemiological.hpp
//

#ifndef LZZ_epidemiological_hpp
#define LZZ_epidemiological_hpp
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <oneapi/dpl/random>
enum class SIR_State : char {
  Susceptible = 0,
  Infected = 1,
  Recovered = 2,
  Invalid = 3
};
#define LZZ_INLINE inline
namespace SIR_SBM
{
  struct Population_Count
  {
    uint32_t S;
    uint32_t I;
    uint32_t R;
    Population_Count ();
    Population_Count (uint32_t S, uint32_t I, uint32_t R);
    Population_Count (std::array <uint32_t, 3> const & arr);
    Population_Count operator + (Population_Count const & other) const;
    bool is_zero () const;
    uint32_t & operator [] (SIR_State s);
  };
}
namespace SIR_SBM
{
  Population_Count state_to_count (SIR_State s);
}
namespace SIR_SBM
{
  std::tuple <size_t, size_t, size_t> init_range (sycl::buffer <SIR_State, 3> & state);
}
namespace SIR_SBM
{
  sycl::event initialize (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_I0);
}
namespace SIR_SBM
{
  sycl::event state_copy (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, size_t t_src, size_t t_dest, sycl::event dep_event = {});
}
namespace SIR_SBM
{
  sycl::event recover (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_R, uint32_t t, sycl::event dep_event = {});
}
namespace SIR_SBM
{
  sycl::event infect (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <Edge_t> & edges, sycl::buffer <uint32_t> & ecc, sycl::buffer <uint32_t, 3> & infected_count, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_I, uint32_t t, uint32_t t_offset, sycl::event dep_event = {});
}
#undef LZZ_INLINE
#endif
