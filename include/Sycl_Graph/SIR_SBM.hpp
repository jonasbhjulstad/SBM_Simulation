#ifndef SIR_SBM_HPP
#define SIR_SBM_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <execution>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stddef.h>
#include <tuple>
#include <Sycl_Graph/SBM_types.hpp>
namespace Sycl_Graph::SBM
{
  template <typename T>
  void print_buffer(sycl::buffer<T, 1> &buf)
  {
    auto acc = buf.get_host_access();
    for (int i = 0; i < buf.size(); i++)
    {
      std::cout << acc[i] << " ";
    }
    std::cout << std::endl;
  }

  std::vector<uint32_t> get_total_state(sycl::buffer<SIR_State> &v_buf)
  {
    auto acc = v_buf.get_host_access();
    uint32_t S = 0;
    uint32_t I = 0;
    uint32_t R = 0;
    for (int i = 0; i < v_buf.size(); i++)
    {
      if (acc[i] == SIR_INDIVIDUAL_S)
        S++;
      else if (acc[i] == SIR_INDIVIDUAL_I)
        I++;
      else if (acc[i] == SIR_INDIVIDUAL_R)
        R++;
    }
    return {S, I, R};
  }

  std::vector<std::vector<uint32_t>> get_community_state(sycl::buffer<SIR_State> &v_buf, sycl::buffer<uint32_t, 1> &vcm_buf, uint32_t N_communities)
  {
    auto v_acc = v_buf.get_host_access();
    auto vcm_acc = vcm_buf.get_host_access();
    std::vector<std::vector<uint32_t>> community_state(N_communities);
    std::for_each(community_state.begin(), community_state.end(), [](auto &v)
                  { v.resize(3); });

    for (int i = 0; i < v_buf.size(); i++)
    {
      assert(vcm_acc[i] < N_communities);
      if (v_acc[i] == SIR_INDIVIDUAL_S)
        community_state[vcm_acc[i]][0]++;
      else if (v_acc[i] == SIR_INDIVIDUAL_I)
        community_state[vcm_acc[i]][1]++;
      else if (v_acc[i] == SIR_INDIVIDUAL_R)
        community_state[vcm_acc[i]][2]++;
    }

    std::vector<uint32_t> total_state(3, 0);
    for (int i = 0; i < N_communities; i++)
    {
      total_state[0] += community_state[i][0];
      total_state[1] += community_state[i][1];
      total_state[2] += community_state[i][2];
    }

    return community_state;
  }

  // template <typename T>
  // sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
  //                            sycl::queue &q)
  // {
  //   sycl::buffer<T, 1> tmp(vec.data(), sycl::range<1>(vec.size()));
  //   return q.submit([&](sycl::handler &h)
  //                   {
  //   auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
  //   auto acc = buf.template get_access<sycl::access::mode::write>(h);
  //   h.copy(tmp_acc, acc); });
  // }

  static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
  sycl::buffer<uint32_t, 1> generate_seeds(uint32_t N_rng, sycl::queue &q,
                                           unsigned long seed = 42)
  {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });

    sycl::buffer<uint32_t, 1> seed_buf((sycl::range<1>(N_rng)));
    copy_to_buffer(seed_buf, rngs, q).wait();
    return seed_buf;
  }

  auto initialize(float p_I0, float p_R0, uint32_t N, sycl::queue &q,
                  sycl::buffer<uint32_t, 1> seed_buf)
  {

    sycl::buffer<SIR_State, 1> state((sycl::range<1>(N)));
    auto event = q.submit([&](sycl::handler &h)
                          {
    auto state_acc = state.template get_access<sycl::access::mode::write,
                                               sycl::access::target::device>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access::mode::read_write,
                                     sycl::access::target::device>(h);
    h.parallel_for(N, [=](sycl::id<1> id) {
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
      Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);

      if (bernoulli_I(rng)) {
        state_acc[id] = SIR_INDIVIDUAL_I;
      } else if (bernoulli_R(rng)) {
        state_acc[id] = SIR_INDIVIDUAL_R;
      } else {
        state_acc[id] = SIR_INDIVIDUAL_S;
      }
    }); });

    event.wait();
    // print_total_state(state);
    return std::make_tuple(state, event);
  }

  template <sycl::access_mode Mode, sycl::access::target Target = sycl::access::target::device>
  struct Edge_Accessor_t
  {
    Edge_Accessor_t(sycl::handler &h) : to(h), from(h), self(h) {}
    sycl::accessor<uint32_t, 1, Mode, Target> to;
    sycl::accessor<uint32_t, 1, Mode, Target> from;
    sycl::accessor<uint32_t, 1, Mode, Target> self;
  };

  struct Edge_Buffer_t
  {

    Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
        : to((sycl::range<1>(N_edges))),
          from((sycl::range<1>(N_edges))),
          self((sycl::range<1>(N_communities)))
    {
      auto to_acc = to.template get_access<sycl::access::mode::write>();
      auto from_acc = from.template get_access<sycl::access::mode::write>();
      auto self_acc = self.template get_access<sycl::access::mode::write>();
      for (uint32_t i = 0; i < N_edges; i++)
      {
        to_acc[i] = invalid_id;
        from_acc[i] = invalid_id;
      }
      for (uint32_t i = 0; i < N_communities; i++)
      {
        self_acc[i] = invalid_id;
      }
    }
    Edge_Buffer_t(uint32_t N_edges) : Edge_Buffer_t(N_edges, 1) {}
    sycl::buffer<uint32_t, 1> to;
    sycl::buffer<uint32_t, 1> from;
    sycl::buffer<uint32_t, 1> self;
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    template <sycl::access_mode Mode>
    auto get_access(sycl::handler &h)
    {
      return Edge_Accessor_t<Mode>(h);
    }

    void fill(uint32_t val)
    {
      auto to_acc = to.template get_access<sycl::access::mode::write>();
      auto from_acc = from.template get_access<sycl::access::mode::write>();
      auto self_acc = self.template get_access<sycl::access::mode::write>();
      for (uint32_t i = 0; i < to_acc.size(); i++)
      {
        to_acc[i] = val;
        from_acc[i] = val;
      }

      for (uint32_t i = 0; i < self_acc.size(); i++)
      {
        self_acc[i] = val;
      }

    }
  };
  auto create_edge_community_map(
      const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &SBM_ids, uint32_t N_communities)
  {
    std::vector<uint32_t> g0(SBM_ids.size());
    std::transform(SBM_ids.begin(), SBM_ids.end(), g0.begin(),
                   [](const auto &group)
                   { return group.size(); });
    std::vector<uint32_t> g1 = g0;
    g0.insert(g0.end(), g1.begin(), g1.end() - N_communities);
    assert(g0.size() == 2 * SBM_ids.size());
    std::vector<uint32_t> ecm_indices(2 * SBM_ids.size());
    std::iota(ecm_indices.begin(), ecm_indices.end(), 0);

    std::vector<std::vector<uint32_t>> ecm_components(2 * SBM_ids.size());
    std::transform(g0.begin(), g0.end(), ecm_indices.begin(), ecm_components.begin(),
                   [](const auto &g, const auto &i)
                   { return std::vector<uint32_t>(g, i); });

    // combine all ecm_components to one vector
    std::vector<uint32_t> edge_community_map;
    for (auto &v : ecm_components)
    {
      edge_community_map.insert(edge_community_map.end(), v.begin(), v.end());
    }

    Edge_Buffer_t ecm(edge_community_map.size(), N_communities);

    auto ecm_acc = ecm.get_access<sycl::access::mode::write>(q);
    for (uint32_t i = 0; i < edge_community_map.size(); i++)
    {
      if (i < SBM_ids.size() - N_communities)
      {
        ecm_acc.to[i] = edge_community_map[i];
      }
      else if ((i > SBM_ids.size() - N_communities) && (i < SBM_ids.size()))
      {
        ecm_acc.self[i] = edge_community_map[i];
      }
      else
      {
        ecm_acc.from[i] = edge_community_map[i];
      }
    }

    return edge_community_map;
  }

  bool is_susceptible_infected_edge(auto &v_acc, const uint32_t e0_to,
                                    const uint32_t e0_from)
  {
    if (e0_to == invalid_id || e0_from == invalid_id)
      return false;
    if (v_acc[e0_to] == SIR_INDIVIDUAL_S && v_acc[e0_from] == SIR_INDIVIDUAL_I)
      return true;
    if (v_acc[e0_to] == SIR_INDIVIDUAL_I && v_acc[e0_from] == SIR_INDIVIDUAL_S)
      return true;
    return false;
  }

  auto get_susceptible_neighbor(auto &v_acc, const uint32_t e_to,
                                const uint32_t e_from)
  {
    SIR_State v_data[2] = {v_acc[e_from], v_acc[e_to]};
    if (v_data[0] == SIR_INDIVIDUAL_S && v_data[1] == SIR_INDIVIDUAL_I)
      return e_from;
    else if (v_data[0] == SIR_INDIVIDUAL_I && v_data[1] == SIR_INDIVIDUAL_S)
      return e_to;
    else
      return invalid_id;
  }

  uint32_t get_susceptible_if_infected_to(auto &v_acc, uint32_t id_from,
                                          uint32_t id_to)
  {
    if ((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
        (v_acc[id_from] == SIR_INDIVIDUAL_I))
    {
      return id_to;
    }
    else
    {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  uint32_t get_susceptible_if_infected_from(auto &v_acc, uint32_t id_from,
                                            uint32_t id_to)
  {
    if ((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
        (v_acc[id_to] == SIR_INDIVIDUAL_I))
    {
      return id_to;
    }
    else
    {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  auto infection_event_spread(
      const std::vector<float> &p_I, sycl::buffer<SIR_State, 1> &v_buf,
      sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
      sycl::buffer<uint32_t, 1> seed_buf, sycl::buffer<uint32_t> &vcm_buf,
      auto &ecm_buf, sycl::queue &q, auto &dep_event)
  {

    const uint32_t N_edges = e_buf.size();
    const uint32_t N_vertices = v_buf.size();
    Edge_Buffer_t inf_event_idx_buf(N_edges, N_vertices);

    sycl::buffer<float, 1> p_I_buf((sycl::range<1>(p_I.size())));
    auto p_I_copy_event = copy_to_buffer(p_I_buf, p_I, q);
    auto event_spread = q.submit([&](sycl::handler &h)
                                 {
    h.depends_on({dep_event, p_I_copy_event});
    auto v_acc = v_buf.template get_access<sycl::access_mode::read_write,
                                           sycl::access::target::device>(h);
    auto e_acc = e_buf.template get_access<sycl::access_mode::read,
                                           sycl::access::target::device>(h);
    auto seed_acc =
        seed_buf.template get_access<sycl::access_mode::read_write,
                                     sycl::access::target::device>(h);
    auto p_I_acc = p_I_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    auto ecm_acc = ecm_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    auto inf_event_idx_acc =
        inf_event_idx_buf.get_access<sycl::access::mode::write,
                                     sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> id) {
      inf_event_idx_acc[id] = std::numeric_limits<uint32_t>::max();
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      Static_RNG::bernoulli_distribution<float> d_I(p_I_acc[ecm_acc[id]]);

      //to direction
      bool infected = false;
      uint32_t sus_id = get_susceptible_if_infected_to(v_acc, e_acc[id].first, e_acc[id].second);
        if ((sus_id != invalid_id) && d_I(rng)) {
          inf_event_idx_acc[id].to = ecm_acc[id];
          v_acc[sus_id] = SIR_INDIVIDUAL_I;
          return;
        }


      //from direction
      sus_id = get_susceptible_if_infected_from(v_acc, e_acc[id].first, e_acc[id].second);
          if ((sus_id != invalid_id) && d_I(rng)) {
            inf_event_idx_acc[id].from = ecm_acc[id];
            v_acc[sus_id] = SIR_INDIVIDUAL_I;
          return;
        }


    }); });

    return std::make_tuple(inf_event_idx_buf, edge_infs, event_spread);
  }

  auto infection_event_gather(auto &&inf_event_idx_buf,
                              uint32_t N_community_connections, sycl::queue &q,
                              sycl::event &dep_event)
  {
    Edge_Buffer_t infection_events(N_community_connections);
    infection_events.fill(0);
    // gather infection events
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto inf_event_idx_acc =
        inf_event_idx_buf
            .get_access<sycl::access::mode::read, sycl::access::target::device>(
                h);
    auto infection_events_acc =
        infection_events.get_access<sycl::access::mode::write,
                                    sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_community_connections),
                   [=](sycl::id<1> id) {
                     infection_events_acc[id] = 0;
                     for (int i = 0; i < inf_event_idx_acc.size(); i++) {
                       if (inf_event_idx_acc[i].to == id[0]) {
                         infection_events_acc[id].to++;
                       }
                      if (inf_event_idx_acc[i].from == id[0]) {
                         infection_events_acc[id].from++;
                       }
                     }
                   }); });
    return std::make_tuple(infection_events, event);
  }

  auto community_infection_count(sycl::buffer<SIR_State> &v_buf,
                                 sycl::buffer<uint32_t> &vcm_buf,
                                 uint32_t N_communities, sycl::queue &q,
                                 sycl::event &dep_event)
  {
    const uint32_t N_vertices = v_buf.size();
    sycl::buffer<uint32_t> community_infections((sycl::range<1>(N_communities)));
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto community_infections_acc = community_infections.template get_access<sycl::access::mode::write>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);
    auto v_acc = v_buf.template get_access<sycl::access::mode::read>(h);
    h.parallel_for(sycl::range<1>(N_communities), [=](sycl::id<1> id)
    {
      community_infections_acc[id] = 0;
      for(int i = 0; i < N_vertices; i++)
      {
        if(v_acc[i] == SIR_INDIVIDUAL_I && vcm_acc[i] == id[0])
          community_infections_acc[id]++;
      }
    }); });

    return std::make_tuple(community_infections, event);
  }

  std::vector<uint32_t> sample_connection_infections(auto &infection_events, std::vector<uint32_t> &community_infections, const std::vector<uint32_t> &ctm, uint32_t seed)
  {
    const uint32_t N_connections = infection_events.size();
    const uint32_t N_communities = community_infections.size();
    auto infection_events_acc = infection_events.template get_access<sycl::access::mode::read>();

    std::mt19937 rng(seed);
    std::vector<uint32_t> connection_infections(N_connections, 0);
    for (int i = 0; i < N_communities; i++)
    {
      std::vector<uint32_t> weights(N_connections, 0);

      for (int j = 0; j < N_connections; j++)
      {
        if (ctm[j] == i)
        {
          weights[j] = infection_events_acc[j];
        }
      }

      std::discrete_distribution<uint32_t> d(weights.begin(), weights.end());
      // sample community_infections[i] from weights
      for (int j = 0; j < community_infections[i]; j++)
      {
        connection_infections[d(rng)]++;
      }
    }

    return connection_infections;
  }

  auto buffer_delta(sycl::buffer<uint32_t> &prior, sycl::buffer<uint32_t> &posterior)
  {
    auto prior_acc = prior.template get_access<sycl::access::mode::read>();
    auto posterior_acc = posterior.template get_access<sycl::access::mode::read>();
    const uint32_t N_communities = prior.size();
    std::vector<uint32_t> deltas(N_communities);
    for (int i = 0; i < N_communities; i++)
    {
      deltas[i] = posterior_acc[i] - prior_acc[i];
    }
    return deltas;
  }

  auto infection_step(const std::vector<float> &p_I,
                      sycl::buffer<SIR_State, 1> &v_buf,
                      sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
                      sycl::buffer<uint32_t, 1> seed_buf,
                      sycl::buffer<uint32_t> &vcm_buf,
                      sycl::buffer<uint32_t, 1> &ecm_buf,
                      const std::vector<uint32_t> &ctm,
                      uint32_t N_community_connections, uint32_t N_communities,
                      sycl::queue &q, auto &dep_events, uint32_t seed = 47)
  {
    auto [community_infections_prior, community_inf_count_event_prior] = community_infection_count(
        v_buf, vcm_buf, N_communities, q, dep_events);
    auto [inf_event_idx_buf, node_infs, event_spread] = infection_event_spread(
        p_I, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, q, community_inf_count_event_prior);

    auto [infection_events, event_gather] = infection_event_gather(
        inf_event_idx_buf, N_community_connections, q, event_spread);

    auto [community_infections_posterior, community_inf_count_event_posterior] = community_infection_count(
        v_buf, vcm_buf, N_communities, q, event_gather);
    community_inf_count_event_posterior.wait();
    auto delta = buffer_delta(community_infections_prior, community_infections_posterior);
    auto connection_infs = sample_connection_infections(infection_events, delta, ctm, seed);

    return std::make_tuple(infection_events, community_infections_posterior,
                           connection_infs, community_inf_count_event_posterior);
  }

  auto recovery_count(sycl::buffer<SIR_State> &v_buf, sycl::buffer<uint32_t> &vcm_buf, uint32_t N_communities, sycl::queue &q, auto &dep_event)
  {
    sycl::buffer<uint32_t, 1> rec_count_buf((sycl::range<1>(N_communities)));
    const uint32_t N_vertices = v_buf.size();
    auto event = q.submit([&](sycl::handler &h)
                          {
    h.depends_on(dep_event);
    auto v_acc = v_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    auto rec_count_acc =
        rec_count_buf.get_access<sycl::access::mode::write,
                                 sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_communities), [=](sycl::id<1> id) {
      rec_count_acc[id] = 0;
      for (int i = 0; i < N_vertices; i++) {
        if ((vcm_acc[i] == id[0]) && v_acc[i] == SIR_INDIVIDUAL_R) {
          rec_count_acc[id]++;
        }
      }
    }); });
    return std::make_tuple(rec_count_buf, event);
  }

  sycl::event recovery_spread(sycl::buffer<SIR_State> &v_buf, sycl::buffer<uint32_t> &seed_buf, float p_R, sycl::queue &q, auto &dep_events)
  {
    const uint32_t N_vertices = v_buf.size();
    return q.submit([&](sycl::handler &h)
                    {
    h.depends_on(dep_events);
    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write,
                                        sycl::access::target::device>(h);
    auto v_acc = v_buf.template get_access<sycl::access::mode::write,
                                           sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      Static_RNG::bernoulli_distribution<float> d_R(p_R);
      if (v_acc[id] == SIR_INDIVIDUAL_I) {
        if (d_R(rng)) {
          v_acc[id] = SIR_INDIVIDUAL_R;
        }
      }
    }); });
  }

  auto recovery_step(float p_R, sycl::buffer<SIR_State, 1> &v_buf,
                     sycl::buffer<uint32_t, 1> &seed_buf,
                     sycl::buffer<uint32_t, 1> &vcm_buf, uint32_t N_communities,
                     sycl::queue &q, auto &dep_events)
  {

    auto [rec_count_prior, prior_event] = recovery_count(v_buf, vcm_buf, N_communities, q, dep_events);
    sycl::event spread_event = recovery_spread(v_buf, seed_buf, p_R, q, prior_event);
    auto [rec_count_posterior, posterior_event] = recovery_count(v_buf, vcm_buf, N_communities, q, spread_event);

    auto community_recovery_count = buffer_delta(rec_count_prior, rec_count_posterior);
    return std::make_pair(community_recovery_count, posterior_event);
  }

  Iteration_Buffers_t
  advance(const std::vector<float> &p_I, float p_R,
          sycl::buffer<SIR_State, 1> &v_buf,
          sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
          sycl::buffer<uint32_t, 1> seed_buf, sycl::buffer<uint32_t> &vcm_buf,
          sycl::buffer<uint32_t, 1> &ecm_buf, const std::vector<uint32_t> &ctm, uint32_t N_community_connections,
          uint32_t N_communities, sycl::queue &q, auto &dep_events, uint32_t seed = 47)
  {
    // auto state = get_total_state(v_buf);
    // std::cout << "State: " << state[0] << " " << state[1] << " " << state[2] << std::endl;
    auto [rec_counts, community_recovery_count_event] = recovery_step(
        p_R, v_buf, seed_buf, vcm_buf, N_communities, q, dep_events);
    auto [infection_events, community_infs, connection_infs, community_inf_count_event] =
        infection_step(p_I, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, ctm,
                       N_community_connections, N_communities, q,
                       community_recovery_count_event, seed);
    community_inf_count_event.wait();
    return std::make_tuple(infection_events, community_infs, rec_counts,
                           connection_infs, community_inf_count_event);
  }

  std::vector<uint32_t> vcm_from_node_list(const std::vector<std::vector<uint32_t>> &node_lists)
  {
    uint32_t N_nodes =
        std::accumulate(node_lists.begin(), node_lists.end(), 0,
                        [](auto acc, const auto &el)
                        { return acc + el.size(); });
    std::vector<uint32_t> vcm;
    vcm.reserve(N_nodes);
    uint32_t n = 0;
    for (auto &&v_list : node_lists)
    {
      std::vector<uint32_t> vs(v_list.size(), n);
      vcm.insert(vcm.end(), vs.begin(), vs.end());
      n++;
    }
    return vcm;
  }

  std::pair<std::vector<std::vector<uint32_t>>, std::vector<Iteration_Buffers_t>> SBM_simulate(
      const SBM_Graph_t &G,
      const SIR_SBM_Param_t &param,
      sycl::queue &q, uint32_t seed = 47)
  {
    uint32_t N_communities = G.node_list.size();
    auto ecm_buf = create_edge_community_map(G.edge_lists, N_communities);
    // flattened edge list
    uint32_t N_edges =
        std::accumulate(G.edge_lists.begin(), G.edge_lists.end(), 0,
                        [](auto acc, const auto &el)
                        { return acc + el.size(); });
    auto vcm = vcm_from_node_list(G.node_list);
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(N_edges);
    for (auto &&e_list : G.edge_lists)
    {
      edges.insert(edges.end(), e_list.begin(), e_list.end());
    }
    const uint32_t N_vertices = vcm.size();
    const uint32_t N_community_connections = G.connection_targets.size();
    auto seed_buf = generate_seeds(N_edges, q, seed);
    auto vcm_buf = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_vertices)));
    copy_to_buffer(vcm_buf, vcm, q);
    auto e_buf =
        sycl::buffer<std::pair<uint32_t, uint32_t>, 1>(sycl::range<1>(N_edges));
    copy_to_buffer(e_buf, edges, q);

    auto [v_buf, init_event] = initialize(param.p_I0, param.p_R0, N_vertices, q, seed_buf);

    q.wait();
    sycl::event sim_event;
    auto init_state = get_community_state(v_buf, vcm_buf, N_communities);

    std::vector<Iteration_Buffers_t> iteration_buffers;
    iteration_buffers.reserve(param.p_I.size());
    std::vector<uint32_t> iteration_seeds(param.p_I.size());
    // generate seeds
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < param.p_I.size(); i++)
    {
      iteration_seeds[i] = dis(gen);
    }
    sycl::event dep_event;
    for (int i = 0; i < param.p_I.size(); i++)
    {
      auto res = advance(param.p_I[i], param.p_R, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, G.connection_targets,
                         N_community_connections, N_communities, q, dep_event, iteration_seeds[i]);
      dep_event = std::get<4>(res);
      iteration_buffers.push_back(res);
    }

    return std::make_pair(init_state, iteration_buffers);
  }

} // namespace Sycl_Graph::SBM

#endif