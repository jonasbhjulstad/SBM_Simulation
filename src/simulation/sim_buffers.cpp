#include <SIR_SBM/epidemiological/epidemiological.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/random/random.hpp>
#include <SIR_SBM/simulation/sim_buffers.hpp>
#include <SIR_SBM/sycl/sycl_routines.hpp>
#include <SIR_SBM/sycl/sycl_validate.hpp>
#include <SIR_SBM/utils/exceptions.hpp>

namespace SIR_SBM {
template <typename T>
static std::vector<uint32_t>
get_vector_sizes(const std::vector<std::vector<T>> vecs) {
  std::vector<uint32_t> sizes(vecs.size());
  std::transform(vecs.begin(), vecs.end(), sizes.begin(),
                 [](const std::vector<T> &vec) { return vec.size(); });
  return sizes;
}
Sim_Buffers::Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
                         Sim_Result &result)
    : // size initialization
      N_vertices(G.N_vertices()), N_sims(p.N_sims), Nt(p.Nt),
      Nt_alloc(p.Nt_alloc), N_edges(G.N_edges()),
      N_partitions(G.N_partitions()), N_connections(G.N_connections()),
      N_con_largest(G.largest_connection_size()),
      N_part_largest(G.largest_partition_size()),
      // data initialization
      ecc_vec(get_vector_sizes(G.edges)), vpc_vec(get_vector_sizes(G.vertices)),
      rng_vec(generate_rngs_dpl(p.seed, p.N_sims)),
      // buffer initialization
      ecc(ecc_vec.data(), G.N_connections()),
      vpc(vpc_vec.data(), G.N_partitions()), rngs(rng_vec.data(), p.N_sims),
      state(sycl::range<3>(p.N_sims, G.N_vertices(), p.Nt_alloc)),
      contact_events(result.contact_events.data(),
                     sycl::range<3>(p.N_sims, G.N_connections() * 2, p.Nt)),
      population_count(result.population_count.data(),
                       sycl::range<3>(p.N_sims, G.N_partitions(), p.Nt + 1)),
      edges(make_buffer<Edge_t, 1>(q, G.flat_edges(),
                                   sycl::range<1>(G.N_edges()))) {
  events.push_back(buffer_fill(q, state, SIR_State::Susceptible));
  events.push_back(zero_fill(q, contact_events, contact_events.get_range(),
                             sycl::range<3>(0, 0, 0)));
  events.push_back(buffer_fill(q, population_count, Population_Count()));
  assert(result.population_count.size() == p.N_sims*G.N_partitions()*(p.Nt+1) && "Inconsistent population count buffer size");
  assert(result.contact_events.size() == p.N_sims*G.N_connections()*2*p.Nt && "Inconsistent contact events buffer size");
}

void Sim_Buffers::wait() const { sycl::event::wait(events); }

void Sim_Buffers::validate(sycl::queue &q) {
  validate_vpc(q);
  validate_ecc(q);
  validate_edges(q);
  validate_state(q);
  validate_infected_count(q);
  validate_population_count(q);
}

void Sim_Buffers::buffer_copy_init(sycl::queue &q, const SBM_Graph &G,
                                   const Sim_Param &p) {}

void Sim_Buffers::validate_vpc(sycl::queue &q) {
  validate_elements(
      q, vpc, [this](uint32_t elem) { return elem <= this->N_part_largest; },
      [](uint32_t elem) { return std::to_string(elem); },
      "vpc elements invalid");
  validate_range(sycl::range<1>(N_partitions), vpc.get_range());
}

void Sim_Buffers::validate_ecc(sycl::queue &q) {
  validate_elements(
      q, ecc, [this](uint32_t elem) { return elem <= this->N_con_largest; },
      [](uint32_t elem) { return std::to_string(elem); },
      "ecc elements invalid");
  validate_range(sycl::range<1>(N_connections), ecc.get_range());
}

void Sim_Buffers::validate_edges(sycl::queue &q) {
  auto str_f = [](std::pair<uint32_t, uint32_t> elem) {
    return std::to_string(elem.first) + " " + std::to_string(elem.second);
  };
  validate_elements(
      q, edges,
      [this](std::pair<uint32_t, uint32_t> elem) {
        return elem.first < this->N_vertices && elem.second < this->N_vertices;
      },
      [](std::pair<uint32_t, uint32_t> elem) {
        return std::to_string(elem.first) + "," + std::to_string(elem.second);
      },

      "Edge vertex ids invalid");
  validate_range(sycl::range<1>(N_edges), edges.get_range());
}

void Sim_Buffers::validate_state(sycl::queue &q) {
  auto str_f = [](SIR_State s) { return std::to_string(static_cast<int>(s)); };
  validate_elements(
      q, state,
      [](SIR_State elem) {
        return elem == SIR_State::Susceptible || elem == SIR_State::Infected ||
               elem == SIR_State::Recovered;
      },
      [](SIR_State elem) {
        switch (elem) {
        case SIR_State::Susceptible:
          return "Susceptible";
        case SIR_State::Infected:
          return "Infected";
        case SIR_State::Recovered:
          return "Recovered";
        default:
          return "Invalid state";
        }
      },
      "Invalid state");
  validate_range(sycl::range<3>(N_sims, N_vertices, Nt_alloc),
                 state.get_range());
}

void Sim_Buffers::validate_infected_count(sycl::queue &q) {
  validate_elements(
      q, contact_events, [](uint32_t elem) { return elem == 0; },
      [](uint32_t elem) { return std::to_string(elem); },
      "Invalid infected count");
  validate_range(sycl::range<3>(N_sims, N_partitions * 2, Nt),
                 contact_events.get_range());
}

void Sim_Buffers::validate_population_count(sycl::queue &q) {
  auto str_f = [](Population_Count pc) {
    return std::to_string(pc.S) + " " + std::to_string(pc.I) + " " +
           std::to_string(pc.R);
  };
  validate_elements(
      q, population_count, [](Population_Count elem) { return elem.is_zero(); },
      [](Population_Count elem) {
        return std::to_string(elem.S) + "," + std::to_string(elem.I) + "," +
               std::to_string(elem.R);
      },
      "Invalid population count");
  validate_range(sycl::range<3>(N_sims, N_partitions, Nt + 1),
                 population_count.get_range());
}

} // namespace SIR_SBM