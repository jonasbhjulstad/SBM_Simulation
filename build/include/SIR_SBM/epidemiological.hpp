// epidemiological.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_epidemiological_hpp
#define LZZ_SIR_SBM_LZZ_epidemiological_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <oneapi/dpl/random>
#line 10 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
enum class SIR_State : char {
  Susceptible = 0,
  Infected = 1,
  Recovered = 2,
  Invalid = 3
};
#define LZZ_INLINE inline
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 18 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  struct Population_Count
  {
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    int S;
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    int I;
#line 19 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    int R;
#line 20 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    Population_Count ();
#line 21 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    Population_Count (int S, int I, int R);
#line 22 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    Population_Count (std::array <int, 3> const & arr);
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    Population_Count operator + (Population_Count const & other) const;
#line 27 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    bool is_zero () const;
#line 28 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
    int & operator [] (SIR_State s);
  };
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  Population_Count state_to_count (SIR_State s);
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 56 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  std::tuple <size_t, size_t, size_t> init_range (sycl::buffer <SIR_State, 3> & state);
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 60 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  sycl::event initialize (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_I0);
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 82 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  sycl::event state_copy (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, size_t t_src, size_t t_dest, sycl::event dep_event = {});
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 102 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  sycl::event recover (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_R, uint32_t t, sycl::event dep_event = {});
}
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
namespace SIR_SBM
{
#line 128 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//epidemiological.hpp"
  sycl::event infect (sycl::queue & q, sycl::buffer <SIR_State, 3> & state, sycl::buffer <Edge_t> & edges, sycl::buffer <uint32_t> & ecc, sycl::buffer <uint32_t, 3> & contact_events, sycl::buffer <oneapi::dpl::ranlux48> & rngs, float p_I, uint32_t t, uint32_t t_offset, sycl::event dep_event = {});
}
#undef LZZ_INLINE
#endif
