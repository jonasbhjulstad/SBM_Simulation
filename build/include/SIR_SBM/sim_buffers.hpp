// sim_buffers.hpp
//

#ifndef LZZ_sim_buffers_hpp
#define LZZ_sim_buffers_hpp
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/sim_param.hpp>
#include <SIR_SBM/sim_result.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <SIR_SBM/sycl_validate.hpp>
  template <typename... Ts>
  using Shared_Tup = std::tuple<std::shared_ptr<Ts>...>;
#define LZZ_INLINE inline
namespace SIR_SBM
{
  struct Sim_Buffers
  {
    Sim_Buffers (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result);
    static std::shared_ptr <Sim_Buffers> make (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result);
    void wait () const;
    std::vector <uint32_t> ecc_vec;
    std::vector <uint32_t> vpc_vec;
    std::vector <oneapi::dpl::ranlux48> rng_vec;
    sycl::buffer <uint32_t> ecc;
    sycl::buffer <uint32_t> vpc;
    sycl::buffer <Edge_t> edges;
    sycl::buffer <SIR_State, 3> state;
    sycl::buffer <uint32_t, 3> infected_count;
    sycl::buffer <Population_Count, 3> population_count;
    sycl::buffer <oneapi::dpl::ranlux48, 1> rngs;
    size_t N_vertices;
    size_t N_sims;
    size_t Nt;
    size_t Nt_alloc;
    size_t N_edges;
    size_t N_partitions;
    size_t N_connections;
    size_t N_con_largest;
    size_t N_part_largest;
    void validate (sycl::queue & q);
  private:
    void buffer_copy_init (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p);
    void validate_vpc (sycl::queue & q);
    void validate_ecc (sycl::queue & q);
    void validate_edges (sycl::queue & q);
    void validate_state (sycl::queue & q);
    void validate_infected_count (sycl::queue & q);
    void validate_population_count (sycl::queue & q);
    std::vector <sycl::event> events;
  };
}
#undef LZZ_INLINE
#endif
