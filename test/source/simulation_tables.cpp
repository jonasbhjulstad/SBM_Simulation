#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/Random.hpp
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
TEST_CASE("Simulation_Buffer_Table_Write") {
  auto Np = 10;
  auto Ng = 100;
  auto N_sims = 100;
  auto N_connections = 10;
  auto Nt = 10;
  auto N_communities = 4;
  auto p_I_vec =
      Buffer_Routines::generate_floats(Np * Ng * N_sims * N_connections * Nt);
  auto p_Is = Buffer_Routines::make_shared_device_buffer<float, 3>(
      q, sycl::range<3>(N_sims, Nt, N_connections), p_I_vec);

  // template <typename T>
  // void buffer_to_table(sycl::queue &q, sycl::buffer<T, 3> &buf,
  //                      const std::string &table_name,
  //                      const std::vector<std::string> &column_id_names)

  buffer_to_table(q, p_Is, "p_Is", {"N_sims", "N_connections", "Nt"});

  std::vector<State_t> community_state_vec(N_sims * (Nt + 1) * N_communities,
                                           State_t{});
  auto community_state = Buffer_Routines::make_shared_device_buffer<State_t, 4>(
      q, sycl::range<3>(N_sims, Nt + 1, N_communities), community_state_vec);

  buffer_to_table(q, community_state, "community_state",
                  {"N_sims", "N_communities", "Np", "Ng"});

  std::vector<uint32_t> connection_events_vec(
      Np * Ng * N_sims * N_connections * Nt, 0);

  auto connection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, sycl::range<3>(N_sims, N_connections, Nt), connection_events_vec);

  buffer_to_table(q, connection_events, "connection_events",
                  {"N_sims", "N_connections", "Nt"});

  std::vector<uint32_t> infection_events_vec(
      Np * Ng * N_sims * N_connections * Nt, 0);
  auto infection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, sycl::range<3>(N_sims, N_connections, Nt), infection_events_vec);

  buffer_to_table(q, infection_events, "infection_events",
                  {"N_sims", "N_connections", "Nt"});

  uint32_t N_pop = 100;
  uint32_t p_out_id = 0;
  uint32_t graph_id = 0;
  uint32_t N_communities;
  uint32_t Nt_alloc = 4;
  uint32_t seed = 231;
  float p_in = 1.0f;
  float p_out = 0.5f;
  float p_I_min = 0.1f;
  float p_I_max = 0.2f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;

  Sim_Param p(N_pop, p_out_id, graph_id, N_communities, N_connections, N_sims,
              Nt, Nt_alloc, seed, p_in, p_out, p_I_min, p_I_max, p_R, p_I0,
              p_R0);

  sim_param_upsert(p.to_json());
}

TEST_CASE("Simulation_Buffer_Table_Read") {
  auto Np = 10;
  auto Ng = 100;
  auto N_sims = 100;
  auto N_connections = 10;
  auto Nt = 10;
  auto N_communities = 4;
  auto p_I_vec =
      Buffer_Routines::generate_floats(Np * Ng * N_sims * N_connections * Nt);
  auto p_Is = Buffer_Routines::make_shared_device_buffer<float, 3>(
      q, sycl::range<3>(N_sims, N_connections, Nt), p_I_vec);

  std::vector<State_t> community_state_vec(
      Np * Ng * N_sims * (Nt + 1) * N_communities, State_t{});
  auto community_state = Buffer_Routines::make_shared_device_buffer<State_t, 4>(
      q, sycl::range<4>(N_sims, N_communities, Np, Ng), community_state_vec);

  std::vector<uint32_t> connection_events_vec(
      Np * Ng * N_sims * N_connections * Nt, 0);
  auto connection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, sycl::range<3>(N_sims, N_connections, Nt), connection_events_vec);

  std::vector<uint32_t> infection_events_vec(
      Np * Ng * N_sims * N_connections * Nt, 0);
  auto infection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, sycl::range<3>(N_sims, N_connections, Nt), infection_events_vec);

  uint32_t N_pop = 100;
  uint32_t p_out_id = 0;
  uint32_t graph_id = 0;
  uint32_t N_communities;
  uint32_t Nt_alloc = 4;
  uint32_t seed = 231;
  float p_in = 1.0f;
  float p_out = 0.5f;
  float p_I_min = 0.1f;
  float p_I_max = 0.2f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;

  Sim_Param p(N_pop, p_out_id, graph_id, N_communities, N_connections, N_sims,
              Nt, Nt_alloc, seed, p_in, p_out, p_I_min, p_I_max, p_R, p_I0,
              p_R0);

  sim_param_upsert(p.to_json());
}