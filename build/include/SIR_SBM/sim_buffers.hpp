// sim_buffers.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_sim_buffers_hpp
#define LZZ_SIR_SBM_LZZ_sim_buffers_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/sim_param.hpp>
#include <SIR_SBM/sim_result.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/sycl_validate.hpp>
#line 49 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  template <typename... Ts>
  using Shared_Tup = std::tuple<std::shared_ptr<Ts>...>;
#define LZZ_INLINE inline
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  struct Sim_Buffers
  {
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    Sim_Buffers (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result);
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    static std::shared_ptr <Sim_Buffers> make (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result);
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void wait () const;
#line 57 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    std::vector <uint32_t> ecc_vec;
#line 58 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    std::vector <uint32_t> vpc_vec;
#line 59 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    std::vector <oneapi::dpl::ranlux48> rng_vec;
#line 62 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <uint32_t> ecc;
#line 63 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <uint32_t> vpc;
#line 64 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <Edge_t> edges;
#line 65 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <SIR_State, 3> state;
#line 66 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <uint32_t, 3> contact_events;
#line 67 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <Population_Count, 3> population_count;
#line 68 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    sycl::buffer <oneapi::dpl::ranlux48, 1> rngs;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_vertices;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_sims;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t Nt;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t Nt_alloc;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_edges;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_partitions;
#line 71 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_connections;
#line 72 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_con_largest;
#line 72 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    size_t N_part_largest;
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate (sycl::queue & q);
#line 82 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  private:
#line 84 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void buffer_copy_init (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p);
#line 87 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_vpc (sycl::queue & q);
#line 94 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_ecc (sycl::queue & q);
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_edges (sycl::queue & q);
#line 115 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_state (sycl::queue & q);
#line 130 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_infected_count (sycl::queue & q);
#line 138 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    void validate_population_count (sycl::queue & q);
#line 151 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    std::vector <sycl::event> events;
  };
}
#undef LZZ_INLINE
#endif
