#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <doctest/doctest.h>
namespace Buffer_Routines {
template <>
void buffer_to_table<State_t>(sycl::queue &q, sycl::buffer<State_t, 3> &buf,
                              const std::string &table_name,
                              const std::vector<std::string> &column_id_names,
                              const std::string &value_colname,
                              const std::vector<std::string> &const_id_names,
                              const std::vector<uint32_t> &const_id_values) {
  auto N0 = buf.get_range()[0];
  auto N1 = buf.get_range()[1];
  auto N2 = buf.get_range()[2];
  // buffer_table<T>(table_name, column_id_names);

  std::vector<State_t> result(buf.size());
  sycl::event e = read_buffer<State_t, 3>(buf, q, result, {});
  e.wait();
  auto indices = indices_to_list(const_id_names, column_id_names);
  auto row = indices_to_map(const_id_names, const_id_values);

  row.insert(column_id_names[0].c_str(), 0);
  row.insert(column_id_names[1].c_str(), 0);
  row.insert(column_id_names[2].c_str(), 0);
  row.insert("S", result[0][0]);
  row.insert("I", result[0][0]);
  row.insert("R", result[0][0]);

  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      for (int k = 0; k < N2; k++) {
        auto row_ind = i * N1 * N2 + j * N2 + k;
      row[column_id_names[0].c_str()] = i;
      row[column_id_names[1].c_str()] = j;
      row[column_id_names[2].c_str()] = k;
      row["S"] = result[row_ind][0];
      row["I"] = result[row_ind][1];
      row["R"] = result[row_ind][2];

        Orm::DB::table(table_name.c_str())
        ->upsert({row}, indices, {"S", "I", "R"});
      }
    }

  }
}
} // namespace Buffer_Routines

TEST_CASE("Simulation_Buffer_Table_Write") {
  auto Np = 2;
  auto Ng = 2;
  auto N_sims = 2;
  auto N_connections = 2;
  auto Nt = 2;
  auto N_communities = 2;

  uint32_t N_pop = 100;
  uint32_t p_out_id = 0;
  uint32_t graph_id = 0;
  uint32_t Nt_alloc = 4;
  uint32_t seed = 231;
  float p_in = 1.0f;
  float p_out = 0.5f;
  float p_I_min = 0.1f;
  float p_I_max = 0.2f;
  float p_R = 0.1f;
  float p_I0 = 0.1f;
  float p_R0 = 0.0f;
  sycl::queue q(sycl::cpu_selector_v);
  std::vector<sycl::event> buf_events(4);
  auto p_I_vec = Buffer_Routines::generate_floats(
      N_sims * N_connections * Nt, p_I_min, p_I_max, seed);
  auto p_Is = Buffer_Routines::make_shared_device_buffer<float, 3>(
      q, p_I_vec, sycl::range<3>(N_sims, Nt, N_connections), buf_events[0]);

  // template <typename T>
  // void buffer_to_table(sycl::queue &q, sycl::buffer<T, 3> &buf,
  //                      const std::string &table_name,
  //                      const std::vector<std::string> &column_id_names)

  Buffer_Routines::buffer_to_table(q, *p_Is, "p_Is",
                                   {"simulation","t", "connection"}, {"value"}, {"p_out", "graph"}, {p_out_id, graph_id});

  std::vector<State_t> community_state_vec(N_sims * (Nt + 1) * N_communities,
                                           State_t{});
  auto community_state = Buffer_Routines::make_shared_device_buffer<State_t, 3>(
      q, community_state_vec, sycl::range<3>(N_sims, Nt + 1, N_communities),
      buf_events[1]);
  // void buffer_to_table(sycl::queue &q, sycl::buffer<T, 3> &buf,
  //  const std::string &table_name,
  //  const std::vector<std::string> &column_id_names)
  Buffer_Routines::buffer_to_table<State_t>(
      q, *community_state, "community_state", {"simulation", "t","community"}, "", {"p_out", "graph"}, {p_out_id, graph_id});

  std::vector<uint32_t> connection_events_vec(
      N_sims * N_connections * Nt, 0);

  auto connection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, connection_events_vec, sycl::range<3>(N_sims, N_connections, Nt),
          buf_events[2]);

  Buffer_Routines::buffer_to_table(q, *connection_events, "connection_events",
                                   {"simulation", "t", "connection"}, "value", {"p_out", "graph"}, {p_out_id, graph_id});

  std::vector<uint32_t> infection_events_vec(
      N_sims * N_connections * Nt, 0);
  auto infection_events =
      Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
          q, infection_events_vec, sycl::range<3>(N_sims, N_connections, Nt),
          buf_events[3]);

  Buffer_Routines::buffer_to_table(q, *infection_events, "infection_events",
                                   {"simulation", "t", "connection"}, "value", {"p_out", "graph"}, {p_out_id, graph_id});

  SBM_Simulation::Sim_Param p(N_pop, p_out_id, graph_id, N_communities,
                              N_connections, N_sims, Nt, Nt_alloc, seed, p_in,
                              p_out, p_I_min, p_I_max, p_R, p_I0, p_R0);

  SBM_Simulation::sim_param_upsert(p.to_json());
}

// TEST_CASE("Simulation_Buffer_Table_Read") {
//   auto Np = 10;
//   auto Ng = 100;
//   auto N_sims = 100;
//   auto N_connections = 10;
//   auto Nt = 10;
//   auto N_communities = 4;
//   auto p_I_vec =
//       Buffer_Routines::generate_floats(Np * Ng * N_sims * N_connections *
//       Nt);
//   auto p_Is = Buffer_Routines::make_shared_device_buffer<float, 3>(
//       q, sycl::range<3>(N_sims, N_connections, Nt), p_I_vec);

//   std::vector<State_t> community_state_vec(
//       Np * Ng * N_sims * (Nt + 1) * N_communities, State_t{});
//   auto community_state = Buffer_Routines::make_shared_device_buffer<State_t,
//   4>(
//       q, sycl::range<4>(N_sims, N_communities, Np, Ng), community_state_vec);

//   std::vector<uint32_t> connection_events_vec(
//       Np * Ng * N_sims * N_connections * Nt, 0);
//   auto connection_events =
//       Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
//           q, sycl::range<3>(N_sims, N_connections, Nt),
//           connection_events_vec);

//   std::vector<uint32_t> infection_events_vec(
//       Np * Ng * N_sims * N_connections * Nt, 0);
//   auto infection_events =
//       Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
//           q, sycl::range<3>(N_sims, N_connections, Nt),
//           infection_events_vec);

//   uint32_t N_pop = 100;
//   uint32_t p_out_id = 0;
//   uint32_t graph_id = 0;
//   uint32_t N_communities;
//   uint32_t Nt_alloc = 4;
//   uint32_t seed = 231;
//   float p_in = 1.0f;
//   float p_out = 0.5f;
//   float p_I_min = 0.1f;
//   float p_I_max = 0.2f;
//   float p_R = 0.1f;
//   float p_I0 = 0.1f;
//   float p_R0 = 0.0f;

//   SBM_Simulation::Sim_Param p(N_pop, p_out_id, graph_id, N_communities,
//   N_connections, N_sims,
//               Nt, Nt_alloc, seed, p_in, p_out, p_I_min, p_I_max, p_R, p_I0,
//               p_R0);

//   SBM_Simulation::sim_param_upsert(p.to_json());
// }