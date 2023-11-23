
#include <execution>
#include <Dataframe/Dataframe.hpp>
#include <SBM_Database/Graph/Graph_Tables.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
#include <Sycl_Buffer_Routines/Buffer_Utils.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <SBM_Simulation/Utils/Validation.hpp>
#include <Static_RNG/Static_RNG.hpp>
#include <iostream>
#include <orm/db.hpp>
#include <random>
#define INF_MAX_SAMPLE_LIMIT 10000

namespace SBM_Simulation {

const std::vector<uint32_t> get_delta_I(uint32_t p_out_id, uint32_t graph_id,
                                        uint32_t sim_id, uint32_t community,
                                        const QString &control_type) {

  auto Nt = Orm::DB::table("community_state_" + control_type)
                ->select("I")
                .where({{"p_out", p_out_id},
                        {"graph", graph_id},
                        {"simulation", sim_id},
                        {"community", community}})
                .count();
  auto get_community_state_traj = [&](auto state) {
    auto query = Orm::DB::table("community_state_" + control_type)
                     ->select(state)
                     .where({{"p_out", p_out_id},
                             {"graph", graph_id},
                             {"simulation", sim_id},
                             {"community", community}})
                     .orderBy("t", "asc")
                     .get();
    std::vector<uint32_t> res;
    res.reserve(Nt);
    while (query.next()) {
      res.push_back(query.value(0).toUInt());
    }
    return res;
  };
  auto Is = get_community_state_traj("I");
  auto Rs = get_community_state_traj("R");

  std::vector<uint32_t> delta_I(Nt - 1, 0);
  std::vector<uint32_t> delta_R(Nt - 1, 0);
  for (int t = 0; t < Nt - 1; t++) {
    delta_R[t] = Rs[t + 1] - Rs[t];
    delta_I[t] = Is[t + 1] - Is[t] + delta_R[t];
  }
  return delta_I;
}

Dataframe::Dataframe_t<uint32_t, 2>
get_connection_events(uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id,
                      uint32_t connection, const QString &control_type) {
  auto N_connections = Orm::DB::table("connection_events_" + control_type)
                           ->where({{"p_out", p_out_id},
                                    {"graph", graph_id},
                                    {"simulation", sim_id}})
                           .max("connection")
                           .toUInt() +
                       1;
  auto Nt = Orm::DB::table("connection_events_" + control_type)
                ->select("value")
                .where({{"p_out", p_out_id},
                        {"graph", graph_id},
                        {"simulation", sim_id},
                        {"connection", connection}})
                .count();
  auto query = Orm::DB::table("connection_events_" + control_type)
                   ->select({"value", "t", "connection"})
                   .where({{"p_out", p_out_id},
                           {"graph", graph_id},
                           {"simulation", sim_id},
                           {"connection", connection}})
                   .orderBy("t", "asc")
                   .get();
  Dataframe::Dataframe_t<uint32_t, 2> events(
      std::array<uint32_t, 2>({(uint32_t)Nt, (uint32_t)N_connections}));
  while (query.next()) {
    auto t = query.value("t").toUInt();
    auto value = query.value("value").toUInt();
    auto connection = query.value("connection").toUInt();
    events[t][connection] = value;
  }

  return events;
}



std::vector<uint32_t>
get_related_connections(size_t c_idx, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm) {
  std::vector<uint32_t> connection_indices;
  for (int i = 0; i < ccm.size(); i++) {
    if (ccm[i].to == c_idx) {
      connection_indices.push_back(i);
    }
  }
  return connection_indices;
}
std::vector<uint32_t>
get_related_events(size_t c_idx, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
                   const std::vector<uint32_t> &events) {
  auto r_con = get_related_connections(c_idx, ccm);
  std::vector<uint32_t> r_con_events(r_con.size(), 0);
  for (int i = 0; i < r_con_events.size(); i++) {
    r_con_events[i] = events[r_con[i]];
  }
  return r_con_events;
}
Dataframe::Dataframe_t<uint32_t, 2>
get_related_events(size_t c_idx, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm,
                   const Dataframe::Dataframe_t<uint32_t, 2> &events) {
  auto Nt = events.data.size();
  auto r_con = get_related_connections(c_idx, ccm);
  Dataframe::Dataframe_t<uint32_t, 2> result(
      std::array<uint32_t, 2>({(uint32_t)Nt, (uint32_t)r_con.size()}));
  for (int t = 0; t < Nt; t++) {
    for (int i = 0; i < r_con.size(); i++) {
      result[t][i] = events[t][r_con[i]];
    }
  }
  return result;
}

auto get_community_connections(size_t N_communities, const auto &ccm) {
  std::vector<uint32_t> community_indices(N_communities);
  std::iota(community_indices.begin(), community_indices.end(), 0);
  std::vector<std::vector<uint32_t>> indices(community_indices.size());
  std::vector<std::vector<uint32_t>> weights(community_indices.size());

  for (int i = 0; i < community_indices.size(); i++) {
    auto [rc, rw] = get_related_connections(community_indices[i], ccm);
    indices[i] = rc;
    weights[i] = rw;
  }
  return std::make_tuple(indices, weights);
}

std::vector<uint32_t> sample_community_dI(const auto &related_connections,
                                          const auto &related_weights,
                                          const auto &events, auto N_samples, uint32_t seed) {
  auto N_connections = events.size();
  std::vector<uint32_t> result(N_connections, 0);
  if (!N_samples)
    return result;
  std::vector<uint32_t> r_con_events(related_connections.size(), 0);
  for (int i = 0; i < related_connections.size(); i++) {
    r_con_events[i] = events[related_connections[i]];
  }

  auto sample_counts =
      Static_RNG::constrained_weight_sample<std::mt19937_64, float, uint32_t>(N_samples, related_weights, r_con_events, seed);
  for (int sample_idx = 0; sample_idx < sample_counts.size(); sample_idx++) {
    result[related_connections[sample_idx]] = sample_counts[sample_idx];
  }
  return result;
}

Dataframe::Dataframe_t<uint32_t, 2> sample_community_infections(
    uint32_t p_out_id, uint32_t graph_id, uint32_t sim_id, uint32_t community,
    const QString &control_type, const std::vector<SBM_Graph::Weighted_Edge_t> &ccm, uint32_t seed) {
  auto dIs = get_delta_I(p_out_id, graph_id, sim_id, community, control_type);
  auto related_connections = get_related_connections(community, ccm);
  auto connection_events = get_connection_events(p_out_id, graph_id, sim_id,
                                                 community, control_type);
  auto related_events = get_related_events(community, ccm, connection_events);
  auto seeds = Static_RNG::generate_seeds(dIs.size(), seed);

  Dataframe::Dataframe_t<uint32_t, 2> community_infections(
      std::array<uint32_t, 2>({(uint32_t)dIs.size(), (uint32_t)connection_events.size()}));
  for (int t = 0; t < dIs.size(); t++) {
    auto sample_counts = sample_community_dI(related_connections,
                                             SBM_Graph::Weighted_Edge_t::get_weights(ccm),
                                             related_events[t], dIs[t], seeds[t]);
    for (int i = 0; i < sample_counts.size(); i++) {
      community_infections[t][i] = sample_counts[i];
    }
  }
  return community_infections;
}

void sample_simulation_infections(uint32_t p_out_id, uint32_t graph_id,
                                  uint32_t sim_id, const QString &control_type,
                                  const std::vector<SBM_Graph::Weighted_Edge_t> &ccm, uint32_t seed) {
  auto N_communities = Orm::DB::table("community_state_" + control_type)
                           ->where({{"p_out", p_out_id},
                                    {"graph", graph_id},
                                    {"simulation", sim_id}})
                           .max("community")
                           .toUInt() +
                       1;
  auto Nt = Orm::DB::table("connection_events_" + control_type)
                ->where({{"p_out", p_out_id},
                         {"graph", graph_id},
                         {"simulation", sim_id}})
                .max("t")
                .toUInt() +
            1;
  Dataframe::Dataframe_t<uint32_t, 2> community_infections(
      std::array<uint32_t, 2>({(uint32_t)Nt, (uint32_t)N_communities}));
  auto seeds = Static_RNG::generate_seeds(N_communities, seed);
  for (int c_idx = 0; c_idx < N_communities; c_idx++) {
    auto community_infections = sample_community_infections(
        p_out_id, graph_id, sim_id, c_idx, control_type, ccm, seeds[c_idx]);
    for (int t = 0; t < Nt; t++) {
      for (int c_idx = 0; c_idx < N_communities; c_idx++) {
        community_infections[t][c_idx] += community_infections[t][c_idx];
      }
    }
  }
  SBM_Database::connection_upsert<uint32_t>(
      "infection_events", p_out_id, graph_id, sim_id, community_infections, 0,
      control_type, "sampling");
}

void sample_graph_infections(uint32_t p_out_id, uint32_t graph_id,
                             const QString &control_type, uint32_t seed) {
  auto ccm = SBM_Database::ccm_read(p_out_id, graph_id);
  auto N_sims = Orm::DB::table("community_state_" + control_type)
                    ->where({{"p_out", p_out_id}, {"graph", graph_id}})
                    .max("simulation")
                    .toUInt() +
                1;
  auto seeds = Static_RNG::generate_seeds(N_sims, seed);
  for (int sim_id = 0; sim_id < N_sims; sim_id++) {
    sample_simulation_infections(p_out_id, graph_id, sim_id, control_type, ccm, seeds[sim_id]);
  }
}

// void sample_graph_infections(uint32_t p_out_id, uint32_t graph_id,
//                              const QString &control_type) {
//   auto ccm = SBM_Database::ccm_read(p_out_id, graph_id);
//   auto N_sims = Orm::DB::table("community_state_" + control_type)
//                     ->where({{"p_out", p_out_id}, {"graph", graph_id}})
//                     .max("simulation")
//                     .toUInt() +
//                 1;
//   for (int sim_id = 0; sim_id < N_sims; sim_id++) {
//     sample_simulation_infections(p_out_id, graph_id, sim_id, control_type, ccm);
//   }
// }

void sample_all_infections(const QString &control_type, uint32_t seed) {
  auto N_graphs = Orm::DB::table("community_state_" + control_type)
                      ->select("graph")
                      .distinct()
                      .count();
  
  auto query = Orm::DB::table("connection_events_" + control_type)
                   ->select({"p_out", "graph"})
                   .distinct()
                   .get();
  auto seeds = Static_RNG::generate_seeds(N_graphs, seed);
  auto graph_id = 0;
  while (query.next()) {
    auto p_out_id = query.value("p_out").toUInt();
    auto graph_id = query.value("graph").toUInt();
    sample_graph_infections(p_out_id, graph_id, control_type, seeds[graph_id]);
    graph_id++;
  }
}

} // namespace SBM_Simulation