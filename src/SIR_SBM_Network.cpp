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
namespace Sycl_Graph::SBM
{

  uint32_t get_susceptible_id_if_infected(const auto &v_acc, uint32_t id_from,
                                          uint32_t id_to)
  {
    if (((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
         (v_acc[id_from] == SIR_INDIVIDUAL_I)))
    {
      return id_to;
    }
    else if (((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
              (v_acc[id_to] == SIR_INDIVIDUAL_I)))
    {
      return id_from;
    }
    else
    {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  SIR_SBM_Network::SIR_SBM_Network(const SBM_Graph_t &_G, float p_I0, float p_R,
                                   sycl::queue &q, uint32_t seed, uint32_t N_wg, float p_R0)
      : N_communities(_G.N_communities), N_vertices(_G.N_vertices),
        N_edges(_G.N_edges), N_connections(_G.N_connections), G(_G),
        ecm_buf(G.ecm.data(), sycl::range<1>(_G.ecm.size())),
        vcm_buf(G.vcm.data(), sycl::range<1>(_G.vcm.size())),
        edges(G.edge_list.data(), sycl::range<1>(_G.N_edges)), q(q), p_R(p_R),
        p_I0(p_I0), p_R0(p_R0), seeds(generate_seeds(N_wg, seed)),
        N_wg(N_wg),
        seed_buf(seeds.data(), sycl::range<1>(_G.N_edges)),
        trajectory(sycl::range<2>(1, 1)),
        community_recoveries_buf(sycl::range<2>(1, 1)),
        connection_events_buf(sycl::range<2>(1, 1)),
        community_state_buf(sycl::range<2>(1, 1)), p_I_buf(sycl::range<2>(1, 1))
  {
  }

  sycl::event
  SIR_SBM_Network::initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                       sycl::buffer<SIR_State, 2> &buf)
  {
    q.submit([&](sycl::handler &h)
             {
    auto state_acc = buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(buf.get_range(),
                   [=](sycl::id<2> id) { state_acc[id] = SIR_INDIVIDUAL_S; }); });

    auto N_vertices = this->N_vertices;
    return q.submit([&](sycl::handler &h)
                    {
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
    }); });
  }

  std::vector<sycl::event>
  SIR_SBM_Network::remap(const std::vector<uint32_t> &cmap)
  {
#ifdef DEBUG
    assert(cmap.size() == N_communities);
#endif

    auto ecm_idx_map_old = create_ecm_index_map(N_communities);
    auto cmap_buf =
        sycl::buffer<uint32_t>(cmap.data(), sycl::range<1>(cmap.size()));

    N_communities = *std::max_element(cmap.begin(), cmap.end()) + 1;
    auto vcm_remap_event = q.submit([&](sycl::handler &h)
                                    {
    auto cmap_acc =
        cmap_buf.template get_access<sycl::access::mode::read,
                                     sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read_write,
                                               sycl::access::target::device>(h);
    h.parallel_for(vcm_acc.size(),
                   [=](sycl::id<1> i) { vcm_acc[i] = cmap_acc[vcm_acc[i]]; }); });
    auto ecm_idx_map_new = create_ecm_index_map(N_communities);
    std::vector<uint32_t> ecm_new;
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>();
    uint32_t n = 0;
    for (int i = 0; i < N_edges; i++)
    {
      auto map_elem = ecm_idx_map_old[ecm_acc[i]];
      auto old_to = map_elem.community_to;
      auto old_from = map_elem.community_from;

      auto new_to = cmap[old_to];
      auto new_from = cmap[old_from];

      auto it = std::find_if(
          ecm_idx_map_new.begin(), ecm_idx_map_new.end(), [&](auto &&el)
          { return (el.community_to == new_to && el.community_from == new_from) ||
                   (el.community_to == new_from && el.community_from == new_to); });

      assert(it != ecm_idx_map_new.end());
      ecm_new.push_back(std::distance(ecm_idx_map_new.begin(), it));
    }

    sycl::buffer<uint32_t> tmp2(ecm_new.data(), sycl::range<1>(ecm_new.size()));
    auto ecm_remap_event = q.submit([&](sycl::handler &h)
                                    {
    auto tmp_acc = tmp2.template get_access<sycl::access::mode::read,
                                            sycl::access::target::device>(h);
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::write,
                                               sycl::access::target::device>(h);
    h.parallel_for(ecm_acc.size(),
                   [=](sycl::id<1> i) { ecm_acc[i] = tmp_acc[i]; }); });
    return {vcm_remap_event, ecm_remap_event};
  }

  sycl::event SIR_SBM_Network::infect(uint32_t t, sycl::event &dep_event)
  {
#ifdef DEBUG
    assert(state.size() == N_vertices);
    assert(p_I.size() == N_connections);
    assert(connection_events_buf.size() == N_connections);
#endif

    uint32_t N_communities = this->N_communities;
    uint32_t Nt = this->Nt;
    uint32_t N_edges = G.N_edges;
    uint32_t N_wg = this->N_wg;

    assert(ecm_buf.size() == N_edges);
    return sycl::event{};
    return q.submit([&](sycl::handler &h)
                    {
    h.depends_on(dep_event);
    auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>(h);
    auto p_I_acc = p_I_buf.template get_access<sycl::access::mode::read>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::atomic>(h);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read_write>(h);
    auto edge_acc = edges.template get_access<sycl::access::mode::read>(h);
    auto connection_events_acc =
        connection_events_buf.template get_access<sycl::access::mode::write>(h);

    sycl::stream out(1024, 256, h);

    h.parallel_for(N_wg, [=](sycl::id<1> id) {
      auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
      uint32_t N_edge_per_wg = (N_edges / N_wg) + 1;
      for(int i = 0; i < N_edge_per_wg; i++)
      {
        auto edge_idx = id * N_edge_per_wg + i;
        if (edge_idx >= N_edges)
          break;
      auto id_from = edge_acc[edge_idx].from;
      auto id_to = edge_acc[edge_idx].to;
      auto sus_id = get_susceptible_id_if_infected(v_acc[t], id_from, id_to);
      if (sus_id != std::numeric_limits<uint32_t>::max()) {
        Static_RNG::default_rng rng(seed);
        auto p_I = p_I_acc[t][ecm_acc[edge_idx]];
        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
        if (bernoulli_I(rng)) {
          v_acc[t + 1][sus_id] = SIR_INDIVIDUAL_I;
          if (sus_id == id_from) {
            connection_events_acc[t][ecm_acc[edge_idx]].from++;
          } else {
            connection_events_acc[t][ecm_acc[edge_idx]].to++;
          }
        }
      }
      out << "edge: " << edge_idx << "complete" << sycl::endl;
      }
    }); });
  }
  sycl::event SIR_SBM_Network::recover(uint32_t t,
                                       sycl::event &dep_event)
  {
    float p_R = this->p_R;
    auto N_vertices = this->N_vertices;
    sycl::buffer<bool> rec_buf(N_vertices);
    auto event = q.submit([&](sycl::handler &h)
                          {
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
    }); });

    auto rec_count_event = q.submit([&](sycl::handler &h)
                                    {
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
    }); });
    return rec_count_event;
  }

  sycl::event SIR_SBM_Network::advance(uint32_t t,
                                       sycl::event dep_event)
  {

    auto rec_event = recover(t, dep_event);

    auto infect_event = infect(t, rec_event);
    return infect_event;
  }

  sycl::event SIR_SBM_Network::accumulate_community_state(
      sycl::event dep_event)
  {

    assert(std::all_of(G.vcm.begin(), G.vcm.end(), [this](auto id)
                       { return id < N_communities; }));
    auto N_vertices = this->N_vertices;
    return q.submit([&](sycl::handler &h)
                    {
    h.depends_on(dep_event);
    auto v_acc =
        trajectory.template get_access<sycl::access::mode::read>(h);
    auto result_acc = community_state_buf.template get_access<
        sycl::access::mode::read_write>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);
    sycl::stream out { 1024, 256, h };
    h.parallel_for(Nt + 1, [=](sycl::id<1> row_id) {
      for (int i = 0; i < N_vertices; i++) {
        result_acc[row_id][vcm_acc[i]][v_acc[row_id][i]]++;
      }
    }); });
  }

  void SIR_SBM_Network::sim_init(const std::vector<std::vector<float>> &p_Is)
  {
    Nt = p_Is.size();

    std::vector<float> p_I_flat(Nt * N_communities);
    for (int i = 0; i < Nt; i++)
    {
      for (int j = 0; j < N_communities; j++)
      {
        p_I_flat[i * N_communities + j] = p_Is[i][j];
      }
    }

    this->edges = sycl::buffer<Edge_t>(sycl::range<1>(G.edge_list.size()));

    sycl::buffer<Edge_t> tmp_edge_buf(G.edge_list.data(), G.edge_list.size());
    q.submit([&](sycl::handler &h)
             {
      auto tmp_acc = tmp_edge_buf.template get_access<sycl::access::mode::read>(h);
      auto edge_acc = edges.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> idx)
      {
        edge_acc[idx] = tmp_acc[idx];
      }); })
        .wait();
    std::vector<float> flattened_p_I(Nt * N_communities);
    for (int i = 0; i < Nt; i++)
    {
      for (int j = 0; j < N_communities; j++)
      {
        flattened_p_I[i * N_communities + j] = p_Is[i][j];
      }
    }
    auto tmp_buf = sycl::buffer<float, 2>(flattened_p_I.data(), sycl::range<2>(p_Is.size(), p_Is[0].size()));
    p_I_buf = sycl::buffer<float, 2>(sycl::range<2>(p_Is.size(), p_Is[0].size()));

    q.submit([&](sycl::handler &h)
             {
      auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
      auto p_I_acc = p_I_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(p_I_acc.get_range(), [=](sycl::id<2> idx)
      {
        p_I_acc[idx] = tmp_acc[idx];
      }); });

    connection_events_buf =
        sycl::buffer<Edge_t, 2>(sycl::range<2>(Nt, N_connections));

    buffer_fill(connection_events_buf, Edge_t{0, 0}, q).wait();

    trajectory = sycl::buffer<SIR_State, 2>(sycl::range<2>(Nt + 1, N_vertices));
    community_recoveries_buf =
        sycl::buffer<uint32_t, 2>(sycl::range<2>(Nt, N_communities));
    community_state_buf =
        sycl::buffer<State_t, 2>(sycl::range<2>(Nt + 1, N_communities));

    buffer_fill(community_state_buf, State_t{0, 0, 0}, q).wait();
  }

  sycl::event SIR_SBM_Network::initialize_vcm_buf()
  {
    sycl::buffer<uint32_t> vcm_tmp_buf(G.vcm.data(), G.N_vertices);
    return q.submit([&](sycl::handler &h)
                    {
              auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::write>(h);
              auto tmp_acc = vcm_tmp_buf.template get_access<sycl::access::mode::read>(h);
              h.parallel_for(sycl::range<1>(G.N_vertices), [=](sycl::id<1> idx)
              {
                vcm_acc[idx] = tmp_acc[idx];
              }); });
  }

  sycl::event SIR_SBM_Network::initialize_ecm_buf()
  {
    sycl::buffer<uint32_t> ecm_tmp_buf(G.ecm.data(), G.ecm.size());
    return q.submit([&](sycl::handler &h)
                    {
              auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::write>(h);
              auto tmp_acc = ecm_tmp_buf.template get_access<sycl::access::mode::read>(h);
              h.parallel_for(sycl::range<1>(G.N_vertices), [=](sycl::id<1> idx)
              {
                ecm_acc[idx] = tmp_acc[idx];
              }); });
  }

  sycl::event
  SIR_SBM_Network::simulate(const SIR_SBM_Param_t &param)
  {
    static uint32_t N_sims = 0;
    auto buf_size = sizeof(sycl::buffer<SIR_State>);

    uint32_t Nt = param.p_I.size();

    sim_init(param.p_I);
    initialize_vcm_buf().wait();
    initialize_ecm_buf().wait();
    auto vertex_init_event = initialize_vertices(p_I0, p_R0, q, trajectory);
    sycl::event advance_event = vertex_init_event;

    for (int i = 0; i < Nt; i++)
    {
      advance_event = advance(i, advance_event);
      // auto state = read_buffer(trajectory);
    }

    // advance_event = accumulate_community_state(advance_event);

    return advance_event;
  }

  std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<Edge_t>>,
             std::vector<std::vector<Edge_t>>>
  SIR_SBM_Network::read_trajectory()
  {
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
      uint32_t seed)
  {
    std::mt19937 rng(seed);

    std::vector<uint32_t> weights(G.connection_community_map.size() * 2);
    uint32_t idx = 0;
    for (const auto &connection_edge : G.connection_community_map)
    {
      weights[idx] = (connection_edge.to == community_idx) ? connection_edge.weight : 0;
      weights[idx + 1] = (connection_edge.from == community_idx) ? connection_edge.weight : 0;
      idx += 2;
    }
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());

    uint32_t N_sampled_infected = 0;
    std::vector<uint32_t> flattened_events(weights.size());
    std::vector<uint32_t> flattened_connection_infections(weights.size(), 0);
    for (int i = 0; i < connection_events.size(); i++)
    {
      flattened_events[i * 2] = connection_events[i].from;
      flattened_events[i * 2 + 1] = connection_events[i].to;
    }
    while (N_sampled_infected < N_infected)
    {
      auto idx = dist(rng);
      if (flattened_connection_infections[idx] < flattened_events[idx])
      {
        N_sampled_infected++;
        flattened_connection_infections[idx]++;
      }
    }
    std::vector<Edge_t> connection_infections(connection_events.size());
    for (int i = 0; i < connection_infections.size(); i++)
    {
      connection_infections[i].from = flattened_connection_infections[2 * i];
      connection_infections[i].to = flattened_connection_infections[2 * i + 1];
    }

    return connection_infections;
  }

  std::vector<std::vector<Edge_t>> SIR_SBM_Network::sample_connection_infections(
      const std::vector<std::vector<State_t>> &community_trajectory,
      const std::vector<std::vector<Edge_t>> &connection_events,
      const std::vector<std::vector<uint32_t>> &community_recoveries,
      uint32_t seed)
  {
    uint32_t Nt = community_trajectory.size() - 1;
    std::vector<std::vector<Edge_t>> connection_infections(Nt);
    std::vector<std::vector<int>> delta_Is(
        Nt, std::vector<int>(N_communities, 0));

    for (int i = 0; i < Nt; i++)
    {
      for (int j = 0; j < N_communities; j++)
      {
        delta_Is[i][j] =
            community_trajectory[i + 1][j][1] - community_trajectory[i][j][1] + community_recoveries[i][j];
      }
    }

    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);

    std::vector<uint32_t> rngs = generate_seeds(Nt, seed);

    for (int i = 0; i < Nt; i++)
    {
      auto seed = rngs[i];
      auto delta_I = delta_Is[i];
      auto c_events = connection_events[i];
      std::vector<std::vector<Edge_t>> infections_t(
          N_communities, std::vector<Edge_t>(N_connections, Edge_t{0, 0}));
      std::vector<uint32_t> seeds = generate_seeds(N_communities, seed);
      std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> zip(N_communities);
      for (int i = 0; i < N_communities; i++)
      {
        infections_t[i] = sample_connection_infections(i, delta_I[i], c_events, seeds[i]);
      }

      std::vector<Edge_t> accumulated_infections(N_connections, Edge_t{0, 0});
      for (int i = 0; i < N_communities; i++)
      {
        for (int j = 0; j < N_connections; j++)
        {
          accumulated_infections[j].to += infections_t[i][j].to;
          accumulated_infections[j].from += infections_t[i][j].from;
        }
      }
      connection_infections[i] = accumulated_infections;
    }
    return connection_infections;
  }
  std::vector<uint32_t> SIR_SBM_Network::generate_seeds(uint32_t N_rng,
                                                        uint32_t seed)
  {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });

    return rngs;
  }

  std::vector<SIR_SBM_Network::ecm_map_elem_t>
  SIR_SBM_Network::create_ecm_index_map(uint32_t N)
  {
    uint32_t n = 0;
    std::vector<uint32_t> vcm_indices(N);
    std::iota(vcm_indices.begin(), vcm_indices.end(), 0);
    std::vector<SIR_SBM_Network::ecm_map_elem_t> idx_map;
    for (auto &&comb : iter::combinations(vcm_indices, 2))
    {
      idx_map.push_back({comb[0], comb[1]});
      n++;
    }
    for (uint32_t i = 0; i < N_communities; i++)
    {
      idx_map.push_back({i, i});
      n++;
    }
    return idx_map;
  }

  sycl::event
  SIR_SBM_Network::create_state_buf(sycl::buffer<SIR_State> &state_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    auto state_acc =
        state_buf.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
    h.parallel_for(state_acc.size(),
                   [=](sycl::id<1> id) { state_acc[id] = SIR_INDIVIDUAL_S; }); });
  }

} // namespace Sycl_Graph::SBM
