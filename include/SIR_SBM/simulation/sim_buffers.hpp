#pragma once
#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/simulation/sim_param.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <oneapi/dpl/random>
namespace SIR_SBM {
struct Sim_Buffers {
  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
              Sim_Result &result);
  static std::shared_ptr<Sim_Buffers> make(sycl::queue &q, const SBM_Graph &G,
                                           const Sim_Param &p,
                                           Sim_Result &result);

  template <typename... Ts>
  using Shared_Tup = std::tuple<std::shared_ptr<Ts>...>;

  void wait() const;

  // data
  std::vector<uint32_t> ecc_vec;
  std::vector<uint32_t> vpc_vec;
  std::vector<oneapi::dpl::ranlux48> rng_vec;

  // buffers
  sycl::buffer<uint32_t> ecc; // edge connection count
  sycl::buffer<uint32_t> vpc; // vertex partition count
  sycl::buffer<Edge_t> edges;
  sycl::buffer<SIR_State, 3> state;
  sycl::buffer<uint32_t, 3> contact_events;
  sycl::buffer<Population_Count, 3> population_count;
  sycl::buffer<oneapi::dpl::ranlux48, 1> rngs;

  // sizes
  uint32_t N_vertices, N_sims, Nt, Nt_alloc, N_edges, N_partitions,
      N_connections;
  uint32_t N_con_largest, N_part_largest;
  void validate(sycl::queue &q);

private:
  void buffer_copy_init(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p);

  void validate_vpc(sycl::queue &q);

  void validate_ecc(sycl::queue &q);

  void validate_edges(sycl::queue &q);

  void validate_state(sycl::queue &q);

  void validate_infected_count(sycl::queue &q);

  void validate_population_count(sycl::queue &q);
  std::vector<sycl::event> events;
};

} // namespace SIR_SBM