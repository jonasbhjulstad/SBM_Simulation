// sim_buffers.cpp
//

#include "sim_buffers.hpp"
#define LZZ_INLINE inline
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 16 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  Sim_Buffers::Sim_Buffers (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result)
#line 18 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
    : N_vertices (G.N_vertices()), N_sims (p.N_sims), Nt (p.Nt), Nt_alloc (p.Nt_alloc), N_edges (G.N_edges()), N_partitions (G.N_partitions()), N_connections (G.N_connections()), N_con_largest (G.largest_connection_size()), N_part_largest (G.largest_partition_size()), ecc_vec (subvector_sizes(G.edges)), vpc_vec (subvector_sizes(G.vertices)), rng_vec (generate_rngs<oneapi::dpl::ranlux48>(p.seed, p.N_sims)), ecc (ecc_vec.data(), G.N_connections()), vpc (vpc_vec.data(), G.N_partitions()), rngs (rng_vec.data(), p.N_sims), state (sycl::range<3>(p.N_sims, G.N_vertices(), p.Nt_alloc)), contact_events (result.contact_events.data(), sycl::range<3>(p.N_sims, G.N_connections() * 2, p.Nt)), population_count (result.population_count.data(), sycl::range<3>(p.N_sims, G.N_partitions(), p.Nt + 1)), edges (make_buffer<Edge_t, 1>(q, G.flat_edges(),
                                     sycl::range<1>(G.N_edges())))
#line 37 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                                                   {
    events.push_back(buffer_fill(q, state, SIR_State::Susceptible));
    events.push_back(zero_fill(q, contact_events, contact_events.get_range(), sycl::range<3>(0,0,0)));
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 41 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  std::shared_ptr <Sim_Buffers> Sim_Buffers::make (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p, Sim_Result & result)
#line 43 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                                               {
    result.resize(p, G);
    return std::make_shared<Sim_Buffers>(q, G, p, result);
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::wait () const
#line 54 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                    { sycl::event::wait(events); }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate (sycl::queue & q)
#line 73 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                {
    validate_vpc(q);
    validate_ecc(q);
    validate_edges(q);
    validate_state(q);
    validate_infected_count(q);
    validate_population_count(q);
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 84 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::buffer_copy_init (sycl::queue & q, SBM_Graph const & G, Sim_Param const & p)
#line 85 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                            {}
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 87 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_vpc (sycl::queue & q)
#line 87 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                    {
    validate_elements<uint32_t, 1>(
        q, vpc, [this](uint32_t elem) { return elem <= this->N_part_largest; },
        "vpc elements invalid");
    validate_range(sycl::range<1>(N_partitions), vpc.get_range());
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 94 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_ecc (sycl::queue & q)
#line 94 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                    {
    validate_elements<uint32_t, 1>(
        q, ecc, [this](uint32_t elem) { return elem <= this->N_con_largest; },
        "ecc elements invalid");
    validate_range(sycl::range<1>(N_connections), ecc.get_range());
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_edges (sycl::queue & q)
#line 101 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                      {
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
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 115 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_state (sycl::queue & q)
#line 115 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                      {
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
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 130 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_infected_count (sycl::queue & q)
#line 130 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                               {
    validate_elements(
        q, contact_events, [](uint32_t elem) { return elem == 0; },
        "Invalid infected count");
    validate_range(sycl::range<3>(N_sims, N_partitions * 2, Nt),
                   contact_events.get_range());
  }
}
#line 14 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
namespace SIR_SBM
{
#line 138 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
  void Sim_Buffers::validate_population_count (sycl::queue & q)
#line 138 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sim_buffers.hpp"
                                                 {
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
}
#undef LZZ_INLINE
