#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Sycl_Graph/SBM_types.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <algorithm>
#include <execution>
#include <iostream>
#include <itertools.hpp>
#include <memory>
#include <random>
namespace Sycl_Graph::SBM {

uint32_t get_susceptible_id_if_infected(const auto &v_acc, uint32_t id_from,
                                        uint32_t id_to) {
  if (((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
       (v_acc[id_from] == SIR_INDIVIDUAL_I))) {
    return id_to;
  } else if (((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
              (v_acc[id_to] == SIR_INDIVIDUAL_I))) {
    return id_from;
  } else {
    return std::numeric_limits<uint32_t>::max();
  }
}

SIR_SBM_Network::SIR_SBM_Network(const SBM_Graph_t &_G, float p_I0, float p_R,
                                 sycl::queue &q, uint32_t seed, float p_R0)
    : N_communities(_G.N_communities), N_vertices(_G.N_vertices),
      N_edges(_G.N_edges), N_connections(_G.N_connections), G(_G),
      ecm_buf(G.ecm.data(), sycl::range<1>(_G.ecm.size())),
      vcm_buf(G.vcm.data(), sycl::range<1>(_G.vcm.size())),
      edges(G.edge_list.data(), sycl::range<1>(_G.N_edges)), q(q), p_R(p_R),
      p_I0(p_I0), p_R0(p_R0), seeds(generate_seeds(_G.N_edges, seed)),
      seed_buf(seeds.data(), sycl::range<1>(_G.N_edges)),
      trajectory(sycl::range<2>(1, 1)),
      community_recoveries_buf(sycl::range<2>(1, 1)),
      connection_events_buf(sycl::range<2>(1, 1)),
      community_state_buf(sycl::range<2>(1, 1)), p_I_buf(sycl::range<2>(1, 1)) {
}

sycl::event
SIR_SBM_Network::initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                     sycl::buffer<SIR_State, 2> &buf) {
  q.submit([&](sycl::handler &h) {
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(buf.get_range(),
                   [=](sycl::id<2> id) { state_acc[id] = SIR_INDIVIDUAL_S; });
  });

  auto N_vertices = this->N_vertices;
  return q.submit([&](sycl::handler &h) {
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(N_vertices, [=](sycl::id<1> id) {
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
      Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);

      if (bernoulli_I(rng)) {
        state_acc[0][id] = SIR_INDIVIDUAL_I;
      } else if (bernoulli_R(rng)) {
        state_acc[0][id] = SIR_INDIVIDUAL_R;
      } else {
        state_acc[0][id] = SIR_INDIVIDUAL_S;
      }
    });
  });
}

std::vector<sycl::event>
SIR_SBM_Network::remap(const std::vector<uint32_t> &cmap) {
#ifdef DEBUG
  assert(cmap.size() == N_communities);
#endif

  auto ecm_idx_map_old = create_ecm_index_map(N_communities);
  auto cmap_buf =
      sycl::buffer<uint32_t>(cmap.data(), sycl::range<1>(cmap.size()));

  N_communities = *std::max_element(cmap.begin(), cmap.end()) + 1;
  auto vcm_remap_event = q.submit([&](sycl::handler &h) {
    auto cmap_acc =
        cmap_buf.template get_access<sycl::access::mode::read,
                                     sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read_write,
                                               sycl::access::target::device>(h);
    h.parallel_for(vcm_acc.size(),
                   [=](sycl::id<1> i) { vcm_acc[i] = cmap_acc[vcm_acc[i]]; });
  });
  auto ecm_idx_map_new = create_ecm_index_map(N_communities);
  std::vector<uint32_t> ecm_new;
  auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>();
  uint32_t n = 0;
  for (int i = 0; i < N_edges; i++) {
    auto map_elem = ecm_idx_map_old[ecm_acc[i]];
    auto old_to = map_elem.community_to;
    auto old_from = map_elem.community_from;

    auto new_to = cmap[old_to];
    auto new_from = cmap[old_from];

    auto it = std::find_if(
        ecm_idx_map_new.begin(), ecm_idx_map_new.end(), [&](auto &&el) {
          return (el.community_to == new_to && el.community_from == new_from) ||
                 (el.community_to == new_from && el.community_from == new_to);
        });

    assert(it != ecm_idx_map_new.end());
    ecm_new.push_back(std::distance(ecm_idx_map_new.begin(), it));
  }

  sycl::buffer<uint32_t> tmp2(ecm_new.data(), sycl::range<1>(ecm_new.size()));
  auto ecm_remap_event = q.submit([&](sycl::handler &h) {
    auto tmp_acc = tmp2.template get_access<sycl::access::mode::read,
                                            sycl::access::target::device>(h);
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::write,
                                               sycl::access::target::device>(h);
    h.parallel_for(ecm_acc.size(),
                   [=](sycl::id<1> i) { ecm_acc[i] = tmp_acc[i]; });
  });
  return {vcm_remap_event, ecm_remap_event};
}

sycl::event SIR_SBM_Network::infect(uint32_t t, sycl::event &dep_event) {
#ifdef DEBUG
  assert(state.size() == N_vertices);
  assert(p_I.size() == N_connections);
  assert(connection_events_buf.size() == N_connections);
#endif

  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>(h);
    auto p_I_acc = p_I_buf.template get_access<sycl::access::mode::read>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::read_write>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
    auto edge_acc = edges.template get_access<sycl::access::mode::read>(h);
    auto connection_events_acc =
        connection_events_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
      auto id_from = edge_acc[id].from;
      auto id_to = edge_acc[id].to;
      auto sus_id = get_susceptible_id_if_infected(v_acc[t], id_from, id_to);
      if (sus_id != std::numeric_limits<uint32_t>::max()) {
        Static_RNG::default_rng rng(seed_acc[id]);
        seed_acc[id]++;
        auto p_I = p_I_acc[t][ecm_acc[id]];
        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
        if (bernoulli_I(rng)) {
          v_acc[t + 1][sus_id] = SIR_INDIVIDUAL_I;
          if (sus_id == id_from) {
            connection_events_acc[t][ecm_acc[id]].from++;
          } else {
            connection_events_acc[t][ecm_acc[id]].to++;
          }
        }
      }
    });
  });
}
sycl::event SIR_SBM_Network::recover(uint32_t t,
                                     std::vector<sycl::event> &dep_event) {
  float p_R = this->p_R;
  auto N_vertices = this->N_vertices;
  sycl::buffer<bool> rec_buf(N_vertices);
  auto event = q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::read_write>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
    auto rec_acc = rec_buf.template get_access<sycl::access::mode::write>(h);
    //   sycl::stream out(1024, 256, h);
    h.parallel_for(N_vertices, [=](sycl::id<1> i) {
      auto state_prev = v_acc[t][i];
      rec_acc[i] = false;
      v_acc[t + 1][i] = state_prev;
      if (state_prev == SIR_INDIVIDUAL_I) {
        Static_RNG::default_rng rng(seed_acc[i]);
        seed_acc[i]++;
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
        if (bernoulli_R(rng)) {
          v_acc[t + 1][i] = SIR_INDIVIDUAL_R;
          rec_acc[i] = true;
        }
      }
    });
  });

  auto rec_count_event = q.submit([&](sycl::handler &h) {
    h.depends_on(event);
    auto rec_acc = rec_buf.template get_access<sycl::access::mode::read>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);
    auto community_rec_acc =
        community_recoveries_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(N_communities, [=](sycl::id<1> id) {
      community_rec_acc[t][id] = 0;
      for (int i = 0; i < N_vertices; i++) {
        if (vcm_acc[i] == id[0] && rec_acc[i]) {
          community_rec_acc[t][id] += 1;
        }
      }
    });
  });
  return rec_count_event;
}

sycl::event SIR_SBM_Network::advance(uint32_t t,
                                     std::vector<sycl::event> dep_events) {

  auto rec_event = recover(t, dep_events);

  // auto state_vec = read_buffer(state, state.size(), {rec_event});

  auto infect_event = infect(t, rec_event);
  // auto state_next_vec = read_buffer(state_next, state_next.size(),
  // {rec_event});
  return infect_event;
}

sycl::event SIR_SBM_Network::accumulate_community_state(
    std::vector<sycl::event> dep_event) {
  auto Nt = trajectory.get_range()[0] - 1;
  auto N_vertices = this->N_vertices;
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read,
                                       sycl::access::target::device>(h);
    auto result_acc = community_state_buf.template get_access<
        sycl::access::mode::read_write, sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read,
                                               sycl::access::target::device>(h);
    sycl::stream out(1024, 256, h);
    h.parallel_for(Nt + 1, [=](sycl::id<1> row_id) {
      for (int i = 0; i < N_vertices; i++) {
        result_acc[row_id][vcm_acc[i]][v_acc[row_id][i]]++;
      }
    });
  });
}

void SIR_SBM_Network::sim_init(const std::vector<std::vector<float>> &p_Is) {
  const uint32_t Nt = p_Is.size();

  p_I_buf = sycl::buffer<float, 2>(sycl::range<2>(Nt, N_communities));
  auto p_I_acc = p_I_buf.template get_access<sycl::access::mode::write>();
  for (int i = 0; i < Nt; i++) {
    for (int j = 0; j < N_communities; j++) {
      p_I_acc[i][j] = p_Is[i][j];
    }
  }

  connection_events_buf =
      sycl::buffer<Edge_t, 2>(sycl::range<2>(Nt, N_connections));

  buffer_fill(connection_events_buf, Edge_t{0, 0}, q);

  trajectory = sycl::buffer<SIR_State, 2>(sycl::range<2>(Nt + 1, N_vertices));
  community_recoveries_buf =
      sycl::buffer<uint32_t, 2>(sycl::range<2>(Nt, N_communities));
  community_state_buf =
      sycl::buffer<State_t, 2>(sycl::range<2>(Nt + 1, N_communities));

  buffer_fill(community_state_buf, State_t{0, 0, 0}, q);
}

std::vector<sycl::event>
SIR_SBM_Network::simulate(const SIR_SBM_Param_t &param) {

  static uint32_t N_sims = 0;
  auto buf_size = sizeof(sycl::buffer<SIR_State>);

  uint32_t Nt = param.p_I.size();

  sim_init(param.p_I);
  auto vertex_init_event = initialize_vertices(p_I0, p_R0, q, trajectory);
  std::vector<sycl::event> advance_event = {vertex_init_event};

  for (int i = 0; i < Nt; i++) {
    advance_event[0] = advance(i, advance_event);
    auto state = read_buffer(trajectory);
  }

  advance_event[0] = accumulate_community_state(advance_event);

  return advance_event;
}

std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<Edge_t>>,
           std::vector<std::vector<Edge_t>>>
SIR_SBM_Network::read_trajectory() {
  auto community_state = read_buffer(community_state_buf);
  auto connection_events = read_buffer(connection_events_buf);
  auto community_recoveries = read_buffer(community_recoveries_buf);
  auto connection_infections = sample_connection_infections(
      community_state, connection_events, community_recoveries, seeds.back());
  return {community_state, connection_events, connection_infections};
}

std::vector<Edge_t> SIR_SBM_Network::sample_connection_infections(
    uint32_t community_idx, uint32_t N_infected,
    const std::vector<Edge_t> &connection_events,
    uint32_t seed) {
  std::mt19937 rng(seed);

  std::vector<uint32_t> connection_sizes(N_connections);

  std::discrete_distribution<uint32_t> dist(connection_weights.begin(),
                                            connection_weights.end());
  std::vector<Edge_t> connection_infections(N_connections, Edge_t{0, 0});

  uint32_t N_inf_samples = 0;
  return connection_infections;
}

std::vector<std::vector<Edge_t>> SIR_SBM_Network::sample_connection_infections(
    const std::vector<std::vector<State_t>> &community_trajectory,
    const std::vector<std::vector<Edge_t>> &connection_events,
    const std::vector<std::vector<uint32_t>> &community_recoveries,
    uint32_t seed) {
  uint32_t Nt = community_trajectory.size() - 1;
  std::vector<std::vector<Edge_t>> connection_infections(Nt);
  std::vector<std::vector<int>> delta_Is(
      Nt, std::vector<int>(N_communities, 0));

  for (int i = 0; i < Nt; i++) {
    for (int j = 0; j < N_communities; j++) {
      delta_Is[i][j] =
          community_trajectory[i + 1][j][1] - community_trajectory[i][j][1] + community_recoveries[i][j];
      #ifdef DEBUG
      assert(community_trajectory[i + 1][j][1] <= N_vertices);
      assert(delta_Is[i][j] >= 0);
      #endif
    }
  }

  std::vector<uint32_t> community_idx(N_communities);
  std::iota(community_idx.begin(), community_idx.end(), 0);

  std::vector<uint32_t> rngs = generate_seeds(Nt, seed);

  for (int i = 0; i < Nt; i++) {
    auto seed = rngs[i];
    auto delta_I = delta_Is[i];
    auto c_events = connection_events[i];
    std::vector<std::vector<Edge_t>> infections_t(
        N_communities, std::vector<Edge_t>(N_connections, Edge_t{0, 0}));
    std::vector<uint32_t> seeds = generate_seeds(N_communities, seed);
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> zip(N_communities);
    for (int i = 0; i < N_communities; i++) {
      infections_t[i] = sample_connection_infections(i, delta_I[i], c_events, seeds[i]);
    }

    std::vector<Edge_t> accumulated_infections(N_connections, Edge_t{0, 0});
    for (int i = 0; i < N_communities; i++) {
      for (int j = 0; j < N_connections; j++) {
        accumulated_infections[j].to += infections_t[i][j].to;
        accumulated_infections[j].from += infections_t[i][j].from;
      }
    }
    #ifdef DEBUG
    uint32_t total_infections = std::accumulate(
        accumulated_infections.begin(), accumulated_infections.end(), 0,
        [](uint32_t a, Edge_t b) { return a + b.to + b.from; });
    assert(total_infections == std::accumulate(delta_I.begin(), delta_I.end(), 0));
    #endif
    connection_infections[i] = accumulated_infections;
  }
  return connection_infections;
}
std::vector<uint32_t> SIR_SBM_Network::generate_seeds(uint32_t N_rng,
                                                      uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(0, 1000000);
  std::vector<uint32_t> rngs(N_rng);
  std::generate(rngs.begin(), rngs.end(), [&]() { return dis(gen); });

  return rngs;
}

sycl::event SIR_SBM_Network::create_vcm(
    const std::vector<std::vector<uint32_t>> &node_lists) {
  std::vector<uint32_t> vcm;
  vcm.reserve(N_vertices);
  uint32_t n = 0;
  for (auto &&v_list : node_lists) {
    std::vector<uint32_t> vs(v_list.size(), n);
    vcm.insert(vcm.end(), vs.begin(), vs.end());
    n++;
  }
  sycl::buffer<uint32_t> tmp(vcm.data(), sycl::range<1>(vcm.size()));

  return q.submit([&](sycl::handler &h) {
    auto tmp_acc = tmp.template get_access<sycl::access::mode::read,
                                           sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::write,
                                               sycl::access::target::device>(h);
    h.parallel_for(vcm.size(), [=](sycl::id<1> i) { vcm_acc[i] = tmp_acc[i]; });
  });
}

std::vector<SIR_SBM_Network::ecm_map_elem_t>
SIR_SBM_Network::create_ecm_index_map(uint32_t N) {
  uint32_t n = 0;
  std::vector<uint32_t> vcm_indices(N);
  std::iota(vcm_indices.begin(), vcm_indices.end(), 0);
  std::vector<SIR_SBM_Network::ecm_map_elem_t> idx_map;
  for (auto &&comb : iter::combinations(vcm_indices, 2)) {
    idx_map.push_back({comb[0], comb[1]});
    n++;
  }
  for (uint32_t i = 0; i < N_communities; i++) {
    idx_map.push_back({i, i});
    n++;
  }
  return idx_map;
}

sycl::event
SIR_SBM_Network::create_state_buf(sycl::buffer<SIR_State> &state_buf) {
  return q.submit([&](sycl::handler &h) {
    auto state_acc =
        state_buf.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
    h.parallel_for(state_acc.size(),
                   [=](sycl::id<1> id) { state_acc[id] = SIR_INDIVIDUAL_S; });
  });
}

} // namespace Sycl_Graph::SBM
