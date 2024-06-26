#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological/epidemiological.hpp>
#include <SIR_SBM/utils/exceptions.hpp>
#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/sim_param.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <SIR_SBM/sycl/sycl_routines.hpp>
#include <SIR_SBM/sycl/sycl_validate.hpp>
#end

#src
#include <SIR_SBM/vector/routines.hpp>
#end

namespace SIR_SBM {
struct Sim_Buffers {
  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
              Sim_Result &result)
      : // size initialization
        N_vertices(G.N_vertices()), N_sims(p.N_sims), Nt(p.Nt),
        Nt_alloc(p.Nt_alloc), N_edges(G.N_edges()),
        N_partitions(G.N_partitions()), N_connections(G.N_connections()),
        N_con_largest(G.largest_connection_size()),
        N_part_largest(G.largest_partition_size()),
        // data initialization
        ecc_vec(get_vector_sizes(G.edges)), vpc_vec(get_vector_sizes(G.vertices)),
        rng_vec(generate_rngs<oneapi::dpl::ranlux48>(p.seed, p.N_sims)),
        // buffer initialization
        ecc(ecc_vec.data(), G.N_connections()),
        vpc(vpc_vec.data(), G.N_partitions()), rngs(rng_vec.data(), p.N_sims),
        state(sycl::range<3>(p.N_sims, G.N_vertices(), p.Nt_alloc)),
        contact_events(
            result.contact_events.data(),
            sycl::range<3>(p.N_sims, G.N_connections() * 2, p.Nt)),
        population_count(result.population_count.data(),
                         sycl::range<3>(p.N_sims, G.N_partitions(), p.Nt + 1)),
        edges(make_buffer<Edge_t, 1>(q, G.flat_edges(),
                                     sycl::range<1>(G.N_edges()))) {
    events.push_back(buffer_fill(q, state, SIR_State::Susceptible));
    events.push_back(zero_fill(q, contact_events, contact_events.get_range(), sycl::range<3>(0,0,0)));
  }
  static std::shared_ptr<Sim_Buffers> make(sycl::queue &q, const SBM_Graph &G,
                                           const Sim_Param &p,
                                           Sim_Result &result) {
    result.resize(p, G);
    return std::make_shared<Sim_Buffers>(q, G, p, result);
  }

#hdr
  template <typename... Ts>
  using Shared_Tup = std::tuple<std::shared_ptr<Ts>...>;
#end


  void wait() const { sycl::event::wait(events); }

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
  uint32_t N_vertices, N_sims, Nt, Nt_alloc, N_edges, N_partitions, N_connections;
  uint32_t N_con_largest, N_part_largest;
  void validate(sycl::queue &q) {
    validate_vpc(q);
    validate_ecc(q);
    validate_edges(q);
    validate_state(q);
    validate_infected_count(q);
    validate_population_count(q);
  }

private:

  void buffer_copy_init(sycl::queue &q, const SBM_Graph &G,
                        const Sim_Param &p) {}

  void validate_vpc(sycl::queue &q) {
    validate_elements<uint32_t, 1>(
        q, vpc, [this](uint32_t elem) { return elem <= this->N_part_largest; },
        "vpc elements invalid");
    validate_range(sycl::range<1>(N_partitions), vpc.get_range());
  }

  void validate_ecc(sycl::queue &q) {
    validate_elements<uint32_t, 1>(
        q, ecc, [this](uint32_t elem) { return elem <= this->N_con_largest; },
        "ecc elements invalid");
    validate_range(sycl::range<1>(N_connections), ecc.get_range());
  }

  void validate_edges(sycl::queue &q) {
    auto str_f = [](std::pair<uint32_t, uint32_t> elem) {
      return std::to_string(elem.first) + " " + std::to_string(elem.second);
    };
    validate_elements<std::pair<uint32_t, uint32_t>, 1, str_f>(
        q, edges,
        [this](std::pair<uint32_t, uint32_t> elem) {
          return elem.first < this->N_vertices &&
                 elem.second < this->N_vertices;
        },
        "Edge vertex ids invalid");
    validate_range(sycl::range<1>(N_edges), edges.get_range());
  }

  void validate_state(sycl::queue &q) {
    auto str_f = [](SIR_State s) {
      return std::to_string(static_cast<int>(s));
    };
    validate_elements<SIR_State, 3, str_f>(
        q, state,
        [](SIR_State elem) {
          return elem == SIR_State::Susceptible ||
                 elem == SIR_State::Infected || elem == SIR_State::Recovered;
        },
        "Invalid state");
    validate_range(sycl::range<3>(N_sims, N_vertices, Nt_alloc),
                   state.get_range());
  }

  void validate_infected_count(sycl::queue &q) {
    validate_elements(
        q, contact_events, [](uint32_t elem) { return elem == 0; },
        "Invalid infected count");
    validate_range(sycl::range<3>(N_sims, N_partitions * 2, Nt),
                   contact_events.get_range());
  }

  void validate_population_count(sycl::queue &q) {
    auto str_f = [](Population_Count pc) {
      return std::to_string(pc.S) + " " + std::to_string(pc.I) + " " +
             std::to_string(pc.R);
    };
    validate_elements<Population_Count, 3, str_f>(
        q, population_count,
        [](Population_Count elem) { return elem.is_zero(); },
        "Invalid population count");
    validate_range(sycl::range<3>(N_sims, N_partitions, Nt + 1),
                   population_count.get_range());
  }

  std::vector<sycl::event> events;
};

} // namespace SIR_SBM