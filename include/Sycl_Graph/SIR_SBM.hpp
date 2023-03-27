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
namespace Sycl_Graph::SBM {
enum SIR_State {
  SIR_INDIVIDUAL_S = 0,
  SIR_INDIVIDUAL_I = 1,
  SIR_INDIVIDUAL_R = 2
};

template <typename T> void print_buffer(sycl::buffer<T, 1> &buf) {
  auto acc = buf.get_host_access();
  for (int i = 0; i < buf.size(); i++) {
    std::cout << acc[i] << " ";
  }
  std::cout << std::endl;
}

std::vector<uint32_t> get_total_state(sycl::buffer<SIR_State> &v_buf) {
  auto acc = v_buf.get_host_access();
  uint32_t S = 0;
  uint32_t I = 0;
  uint32_t R = 0;
  for (int i = 0; i < v_buf.size(); i++) {
    if (acc[i] == SIR_INDIVIDUAL_S)
      S++;
    else if (acc[i] == SIR_INDIVIDUAL_I)
      I++;
    else if (acc[i] == SIR_INDIVIDUAL_R)
      R++;
  }
  return {S, I, R};
}

std::vector<std::vector<uint32_t>> get_community_state(sycl::buffer<SIR_State>& v_buf, sycl::buffer<uint32_t, 1>& vcm_buf, uint32_t N_communities)
{
  auto v_acc = v_buf.get_host_access();
  auto vcm_acc = vcm_buf.get_host_access();
  std::vector<std::vector<uint32_t>> community_state(N_communities);
  std::for_each(community_state.begin(), community_state.end(), [](auto &v) { v.resize(3); });

  for (int i = 0; i < v_buf.size(); i++) {
    assert(vcm_acc[i] < N_communities);
    if (v_acc[i] == SIR_INDIVIDUAL_S)
      community_state[vcm_acc[i]][0]++;
    else if (v_acc[i] == SIR_INDIVIDUAL_I)
      community_state[vcm_acc[i]][1]++;
    else if (v_acc[i] == SIR_INDIVIDUAL_R)
      community_state[vcm_acc[i]][2]++;
  }
  // print_buffer(vcm_buf);
  return community_state;
}

template <typename T>
sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
                           sycl::queue &q) {
  sycl::buffer<T, 1> tmp(vec.data(), sycl::range<1>(vec.size()));
  return q.submit([&](sycl::handler &h) {
    auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    h.copy(tmp_acc, acc);
  });
}

static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
sycl::buffer<uint32_t, 1> generate_seeds(uint32_t N_rng, sycl::queue &q,
                                         unsigned long seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(0, 1000000);
  std::vector<uint32_t> rngs(N_rng);
  std::generate(rngs.begin(), rngs.end(), [&]() { return dis(gen); });

  sycl::buffer<uint32_t, 1> seed_buf((sycl::range<1>(N_rng)));
  copy_to_buffer(seed_buf, rngs, q).wait();
  return seed_buf;
}

auto initialize(float p_I0, float p_R0, uint32_t N, sycl::queue &q,
                sycl::buffer<uint32_t, 1> seed_buf) {

  sycl::buffer<SIR_State, 1> state((sycl::range<1>(N)));
  auto event = q.submit([&](sycl::handler &h) {
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
    });
  });

  event.wait();
  // print_total_state(state);
  return std::make_tuple(state, event);
}

auto create_edge_community_map(
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &SBM_ids) {
  std::vector<uint32_t> group_sizes(SBM_ids.size());
  std::transform(SBM_ids.begin(), SBM_ids.end(), group_sizes.begin(),
                 [](const auto &group) { return group.size(); });

  auto N_edges = std::accumulate(group_sizes.begin(), group_sizes.end(), 0) - 1;
  std::vector<uint32_t> edge_community_map;
  edge_community_map.reserve(N_edges);

  // create a vector with sbm_ids[0].size() 0, sbm_ids[1].size() 1, ...
  std::vector<std::vector<uint32_t>> group_indices(SBM_ids.size());
  for (int i = 0; i < SBM_ids.size(); i++) {
    group_indices[i].resize(SBM_ids[i].size());
    std::fill(group_indices[i].begin(), group_indices[i].end(), i);
    edge_community_map.insert(edge_community_map.end(),
                              group_indices[i].begin(), group_indices[i].end());
  }

  // assert that all indices are lower than SBM_ids.size()
  assert(std::all_of(edge_community_map.begin(), edge_community_map.end(),
                     [&](const auto &e) { return e < SBM_ids.size(); }));

  return edge_community_map;
}

bool is_susceptible_infected_edge(auto &v_acc, const uint32_t e0_to,
                                  const uint32_t e0_from) {
  if (e0_to == invalid_id || e0_from == invalid_id)
    return false;
  if (v_acc[e0_to] == SIR_INDIVIDUAL_S && v_acc[e0_from] == SIR_INDIVIDUAL_I)
    return true;
  if (v_acc[e0_to] == SIR_INDIVIDUAL_I && v_acc[e0_from] == SIR_INDIVIDUAL_S)
    return true;
  return false;
}

auto get_susceptible_neighbor(auto &v_acc, const uint32_t e_to,
                              const uint32_t e_from) {
  SIR_State v_data[2] = {v_acc[e_from], v_acc[e_to]};
  if (v_data[0] == SIR_INDIVIDUAL_S && v_data[1] == SIR_INDIVIDUAL_I)
    return e_from;
  else if (v_data[0] == SIR_INDIVIDUAL_I && v_data[1] == SIR_INDIVIDUAL_S)
    return e_to;
  else
    return invalid_id;
}

uint32_t get_susceptible_id_if_infected_edge(auto &v_acc, uint32_t id_to,
                                             uint32_t id_from) {
  if ((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
      (v_acc[id_from] == SIR_INDIVIDUAL_I)) {
    return id_to;
  } else if ((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
             (v_acc[id_to] == SIR_INDIVIDUAL_I)) {
    return id_from;
  } else {
    return std::numeric_limits<uint32_t>::max();
  }
}

auto infection_event_spread(
    const std::vector<float> &p_I, sycl::buffer<SIR_State, 1> &v_buf,
    sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
    sycl::buffer<uint32_t, 1> seed_buf, sycl::buffer<uint32_t> &vcm_buf,
    sycl::buffer<uint32_t, 1> &ecm_buf, sycl::queue &q, auto &dep_event) {

  const uint32_t N_edges = e_buf.size();
  const uint32_t N_vertices = v_buf.size();
  sycl::buffer<uint32_t, 1> inf_event_idx_buf((sycl::range<1>(N_edges)));
  // initialize to false
  sycl::buffer<bool, 1> edge_infs((sycl::range<1>(N_edges)));

  sycl::buffer<float, 1> p_I_buf((sycl::range<1>(p_I.size())));
  auto p_I_copy_event = copy_to_buffer(p_I_buf, p_I, q);

  auto event_spread = q.submit([&](sycl::handler &h) {
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
    auto edge_infs_acc = edge_infs.get_access<sycl::access::mode::read_write,
                                              sycl::access::target::device>(h);
    auto inf_event_idx_acc =
        inf_event_idx_buf.get_access<sycl::access::mode::write,
                                     sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> id) {
      edge_infs_acc[id] = false;
      inf_event_idx_acc[id] = std::numeric_limits<uint32_t>::max();
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      Static_RNG::bernoulli_distribution<float> d_I(p_I_acc[ecm_acc[id]]);
      uint32_t sus_id = get_susceptible_id_if_infected_edge(
          v_acc, e_acc[id].first, e_acc[id].second);
      if (ecm_acc[id] > 0)
        // out << "Edge: " << e_acc[id].first << " " << e_acc[id].second << "
        // sus_id = " << sus_id<< sycl::endl;
        if ((sus_id != invalid_id) && d_I(rng)) {
          inf_event_idx_acc[id] = ecm_acc[id];
          edge_infs_acc[id] = true;
          v_acc[sus_id] = SIR_INDIVIDUAL_I;
        }
    });
  });

  event_spread.wait();
  return std::make_tuple(inf_event_idx_buf, edge_infs, event_spread);
}

auto infection_event_gather(sycl::buffer<uint32_t> &inf_event_idx_buf,
                            uint32_t N_community_connections, sycl::queue &q,
                            sycl::event &dep_event) {
  sycl::buffer<uint32_t> infection_events(N_community_connections);
  // gather infection events
  auto event = q.submit([&](sycl::handler &h) {
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
                       if (inf_event_idx_acc[i] == id[0]) {
                         infection_events_acc[id]++;
                       }
                     }
                   });
  });
  return std::make_tuple(infection_events, event);
}

auto community_infection_count(sycl::buffer<uint32_t> &infection_events,
                               sycl::buffer<bool> &v_buf,
                               sycl::buffer<uint32_t> &vcm_buf,
                               uint32_t N_communities, sycl::queue &q,
                               sycl::event &dep_event) {

  // const uint32_t N_edges = ecm_buf.size();
  // sycl::buffer<uint32_t> community_infections(
  //     (sycl::range<1>(N_community_connections)));
  // auto event = q.submit([&](sycl::handler &h) {
  //   h.depends_on(dep_event);
  //   auto inf_event_acc =
  //       infection_events.template get_access<sycl::access::mode::read,
  //                                            sycl::access::target::device>(h);
  //   auto edge_infs_acc =
  //       edge_infections.template get_access<sycl::access::mode::read,
  //                                           sycl::access::target::device>(h);
  //   auto community_infs_acc = community_infections.template get_access<
  //       sycl::access::mode::write, sycl::access::target::device>(h);
  //   auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read,
  //                                              sycl::access::target::device>(h);
  //   h.parallel_for(sycl::range<1>(N_community_connections),
  //                  [=](sycl::id<1> id) {
  //                    uint32_t N_infs = 0;
  //                    for (int i = 0; i < N_edges; i++) {
  //                      if ((edge_infs_acc[i]) && (ecm_acc[i] == id[0])) {
  //                        N_infs++;
  //                      }
  //                    }
  //                    community_infs_acc[id[0]] = N_infs;
  //                  });
  // });

  sycl::buffer<uint32_t> community_infections(N_communities);
  const uint32_t N_vertices = v_buf.size();
  q.submit([](sycl::handler& h)
  {
    auto community_infections_acc = community_infections.template get_access<sycl::access::mode::write>(h);
    auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read>(h);
    h.parallel_for(sycl::range<1>(N_communities), [=](sycl::id<1> id)
    {
      community_infections_acc[id] = 0;
      for(int i = 0; i < N_vertices; i++)
      {
        if(v_acc[i] == SIR_INDIVIDUAL_I && vcm_acc[i] == id[0])
          community_infections_acc[id]++;
      }
    });
  })

  return std::make_tuple(community_infections, event);
}

auto sample_connection_infections(sycl::buffer<uint32_t, 1>& infection_events, const std::vector<uint32_t>& ctm, uint32_t N_communities, uint32_t seed)
{
  auto infection_events_acc = infection_events.host_access();
  std::vector<uint32_t> inf_events_tot(N_communities);
  for(int i = 0; i < N_communities)
  { 
    inf_events_tot[i] = 0;
    std::for_each(ctm.begin(), ctm.end(), [&](uint32_t ctm_val)
    {
      inf_events_tot[i] += (ctm_val == i) ? infection_events_acc[ctm_val] : 0;
    });
  }
  std::vector<uint32_t> connection_infections(ctm.size());
  std::mt19937 gen(seed);
  for(int i = 0; i < N_communities; i++)
  {
    // create discrete distribution with infection_evens_acc as weights
    std::vector<uint32_t> weights;
    std::for_each(ctm.begin(), ctm.end(), [&](uint32_t ctm_val)
    {
      weights.push_back((ctm_val == i) ? infection_events_acc[ctm_val] : 0);
    });
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    // sample from distribution
    connection_infections[i] = dist(gen);
  }
  return connection_infections;

}


auto infection_step(const std::vector<float> &p_I,
                    sycl::buffer<SIR_State, 1> &v_buf,
                    sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
                    sycl::buffer<uint32_t, 1> seed_buf,
                    sycl::buffer<uint32_t> &vcm_buf,
                    sycl::buffer<uint32_t, 1> &ecm_buf,
                    sycl::buffer<uint32_t> &ctm_buf,
                    uint32_t N_community_connections, uint32_t N_communities,
                    sycl::queue &q, auto &dep_events, uint32_t seed = 47) {
  // std::cout << "Infection step" << std::endl;
  auto [inf_event_idx_buf, node_infs, event_spread] = infection_event_spread(
      p_I, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, q, dep_events);

  // std::cout << "Infection event spread" << std::endl;
  auto [infection_events, event_gather] = infection_event_gather(
      inf_event_idx_buf, N_community_connections, q, event_spread);
  // std::cout << "Infection event gather" << std::endl;
  auto [community_infs, community_inf_count_event] = community_infection_count(
      infection_events, node_infs, ecm_buf, N_communities,
      N_community_connections, q, event_gather);

  auto connection_infs = sample_connection_infections(infection_events, ctm_buf, N_communities, seed);
  return std::make_tuple(infection_events, community_infs,
                         community_inf_count_event, connection_infs);
}

auto recovery_step(float p_R, sycl::buffer<SIR_State, 1> &v_buf,
                   sycl::buffer<uint32_t, 1> &seed_buf,
                   sycl::buffer<uint32_t, 1> &vcm_buf, uint32_t N_communities,
                   sycl::queue &q, auto &dep_events) {
  const uint32_t N_vertices = v_buf.size();
  sycl::buffer<bool, 1> rec_buf((sycl::range<1>(N_vertices)));
  auto recovery_event = q.submit([&](sycl::handler &h) {
    h.depends_on(dep_events);
    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write,
                                        sycl::access::target::device>(h);
    auto v_acc = v_buf.template get_access<sycl::access::mode::write,
                                           sycl::access::target::device>(h);
    auto rec_acc = rec_buf.get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
    h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
      Static_RNG::default_rng rng(seed_acc[id]);
      seed_acc[id]++;
      rec_acc[id] = false;
      Static_RNG::bernoulli_distribution<float> d_R(p_R);
      if (v_acc[id] == SIR_INDIVIDUAL_I) {
        if (d_R(rng)) {
          rec_acc[id] = true;
          v_acc[id] = SIR_INDIVIDUAL_R;
        }
      }
    });
  });
  sycl::buffer<uint32_t, 1> rec_count_buf((sycl::range<1>(N_communities)));

  auto community_recovery_count = q.submit([&](sycl::handler &h) {
    h.depends_on(recovery_event);
    auto rec_acc = rec_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    auto rec_count_acc =
        rec_count_buf.get_access<sycl::access::mode::write,
                                 sycl::access::target::device>(h);
    auto vcm_acc = vcm_buf.get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
    sycl::stream out(1024, 256, h);
    h.parallel_for(sycl::range<1>(N_communities), [=](sycl::id<1> id) {
      rec_count_acc[id[0]] = 0;
      for (int i = 0; i < N_vertices; i++) {
        if ((vcm_acc[i] == id[0]) && rec_acc[i]) {
          rec_count_acc[id[0]]++;
        }
      }
    });
  });
  return std::make_pair(rec_count_buf, community_recovery_count);
}

typedef std::tuple<sycl::buffer<uint32_t, 1>, sycl::buffer<uint32_t, 1>,
                   sycl::buffer<uint32_t, 1>, sycl::event>
    Iteration_Buffers_t;

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>
read_iteration_buffer(Iteration_Buffers_t &buffers) {
  auto inf_event_buf = std::get<0>(buffers);
  auto community_infs_buf = std::get<1>(buffers);
  auto community_rec_buf = std::get<2>(buffers);
  auto connection_infs = std::get<3>(buffers);
  auto event = std::get<3>(buffers);
  event.wait();
  auto inf_event_acc = inf_event_buf.get_access<sycl::access::mode::read>();
  auto community_infs_acc =
      community_infs_buf.get_access<sycl::access::mode::read>();
  auto community_rec_acc =
      community_rec_buf.get_access<sycl::access::mode::read>();
  std::vector<uint32_t> inf_event(inf_event_acc.size());
  std::vector<uint32_t> community_infs(community_infs_acc.size());
  std::vector<uint32_t> community_rec(community_rec_acc.size());
  for (int i = 0; i < inf_event_acc.size(); i++) {
    inf_event[i] = inf_event_acc[i];
  }
  for (int i = 0; i < community_infs_acc.size(); i++) {
    community_infs[i] = community_infs_acc[i];
  }
  for (int i = 0; i < community_rec_acc.size(); i++) {
    community_rec[i] = community_rec_acc[i];
  }
  return std::make_tuple(inf_event, community_infs, community_rec, connection_infs);
}

Iteration_Buffers_t
advance(const std::vector<float> &p_I, float p_R,
        sycl::buffer<SIR_State, 1> &v_buf,
        sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
        sycl::buffer<uint32_t, 1> seed_buf, sycl::buffer<uint32_t> &vcm_buf, sycl::buffer<uint32_t>& ctm_buf,
        sycl::buffer<uint32_t, 1> &ecm_buf, uint32_t N_community_connections,
        uint32_t N_communities, sycl::queue &q, auto &dep_events, uint32_t seed = 47) {
  auto [rec_count_buf, community_recovery_count_event] = recovery_step(
      p_R, v_buf, seed_buf, vcm_buf, N_communities, q, dep_events);
  community_recovery_count_event.wait();
  // auto state = get_total_state(v_buf);
  // std::cout << state[0] << ", " << state[1] << ", " << state[2] << std::endl;
  auto [infection_events, community_infs, community_inf_count_event] =
      infection_step(p_I, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, ctm_buf,
                     N_community_connections, N_communities, q,
                     community_recovery_count_event, seed);
  community_inf_count_event.wait();
  return std::make_tuple(infection_events, community_infs, rec_count_buf,
                         community_inf_count_event);
}

std::vector<Iteration_Buffers_t> SBM_simulate(
    const std::vector<std::vector<float>> &p_Is, float p_R,
    sycl::buffer<SIR_State, 1> &v_buf,
    sycl::buffer<std::pair<uint32_t, uint32_t>, 1> &e_buf,
    sycl::buffer<uint32_t, 1> seed_buf, sycl::buffer<uint32_t> &vcm_buf,
    sycl::buffer<uint32_t, 1> &ecm_buf, sycl::buffer<uint32_t>& ctm_buf, uint32_t N_community_connections,
    uint32_t N_communities, sycl::queue &q, auto dep_event, uint32_t seed = 47) {
  std::vector<Iteration_Buffers_t> iteration_buffers;
  iteration_buffers.reserve(p_Is.size());
  std::vector<uint32_t> seeds(p_Is.size());
  //generate seeds
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max()); 
  for (int i = 0; i < p_Is.size(); i++) {
    seeds[i] = dis(gen);
  }


  for (int i = 0; i < p_Is.size(); i++) {
    auto res = advance(p_Is[i], p_R, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, ctm_buf,
                       N_community_connections, N_communities, q, dep_event, seeds[i]);
    dep_event = std::get<3>(res);
    iteration_buffers.push_back(res);
  }
  return iteration_buffers;
}
auto SBM_simulate(const std::vector<std::vector<float>> &p_Is,
             float p_R, float p_I0, float p_R0,
             const std::vector<uint32_t> &vertices,
             std::vector<std::pair<uint32_t, uint32_t>> &edges,
             std::vector<uint32_t> &vcm, std::vector<uint32_t> &ecm, const auto& connection_targets,
             sycl::queue &q, uint32_t seed = 47) {

  const uint32_t N_vertices = vcm.size();
  const uint32_t N_edges = ecm.size();
  const uint32_t N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
  const uint32_t N_community_connections =
      *std::max_element(ecm.begin(), ecm.end()) + 1;

  auto seed_buf = generate_seeds(N_edges, q, seed);
  auto vcm_buf = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_vertices)));
  auto ecm_buf = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_edges)));
  auto ctm_buf = sycl::buffer<uint32_t, 1>(sycl::range<1>(connection_targets.size()));

  copy_to_buffer(vcm_buf, vcm, q);
  copy_to_buffer(ecm_buf, ecm, q);
  copy_to_buffer(ctm_buf, connection_targets, q);
  auto e_buf =
      sycl::buffer<std::pair<uint32_t, uint32_t>, 1>(sycl::range<1>(N_edges));
  copy_to_buffer(e_buf, edges, q);
  q.wait();

  auto [v_buf, init_event] = initialize(p_I0, p_R0, N_vertices, q, seed_buf);
  init_event.wait();
  sycl::event event;
        auto iteration_buffers = SBM_simulate(p_Is, p_R, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, ctm_buf,
                     N_community_connections, N_communities, q, event, seed);
  return iteration_buffers;
}

auto SBM_simulate(const std::vector<std::vector<float>> &p, float p_I0,
                  float p_R0, float p_R,
                  std::vector<std::pair<uint32_t, uint32_t>> &edges,
                  std::vector<uint32_t> &vcm, std::vector<uint32_t> &ecm,
                  const auto& connection_targets,
                  sycl::queue &q, unsigned long seed = 42) {
  const uint32_t N_vertices = vcm.size();
  const uint32_t N_edges = ecm.size();
  const uint32_t N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
  const uint32_t N_community_connections =
      *std::max_element(ecm.begin(), ecm.end()) + 1;
  auto seed_buf = generate_seeds(N_edges, q, seed);
  auto vcm_buf = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_vertices)));
  auto ecm_buf = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_edges)));
  auto ctm_buf = sycl::buffer<uint32_t, 1>(sycl::range<1>(connection_targets.size()));
  copy_to_buffer(vcm_buf, vcm, q);
  copy_to_buffer(ecm_buf, ecm, q);
  copy_to_buffer(ctm_buf, connection_targets, q);
  auto e_buf =
      sycl::buffer<std::pair<uint32_t, uint32_t>, 1>(sycl::range<1>(N_edges));
  copy_to_buffer(e_buf, edges, q);

  auto [v_buf, init_event] = initialize(p_I0, p_R0, N_vertices, q, seed_buf);

  q.wait();
  sycl::event sim_event;
  auto init_state = get_community_state(v_buf, vcm_buf, N_communities);
  auto iteration_buffers =
      SBM_simulate(p, p_R, v_buf, e_buf, seed_buf, vcm_buf, ecm_buf, ctm_buf,
                   N_community_connections, N_communities, q, sim_event, seed);
  return std::make_pair(init_state, iteration_buffers);
}

auto vcm_from_node_list(const std::vector<std::vector<uint32_t>> &node_lists) {
  uint32_t N_nodes =
      std::accumulate(node_lists.begin(), node_lists.end(), 0,
                      [](auto acc, const auto &el) { return acc + el.size(); });
  std::vector<uint32_t> vcm;
  vcm.reserve(N_nodes);
  uint32_t n = 0;
  for (auto &&v_list : node_lists) {
    std::vector<uint32_t> vs(v_list.size(), n);
    vcm.insert(vcm.end(), vs.begin(), vs.end());
    n++;
  }
  return vcm;
}

auto SBM_simulate(
    const std::vector<std::vector<float>> &p, float p_R,
    float p_I0, float p_R0,
    const std::vector<std::vector<uint32_t>> &node_lists,
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_lists,
    const auto& connection_targets,
    sycl::queue &q, uint32_t seed = 47) {
  auto ecm = create_edge_community_map(edge_lists);
  // flattened edge list
  uint32_t N_edges =
      std::accumulate(edge_lists.begin(), edge_lists.end(), 0,
                      [](auto acc, const auto &el) { return acc + el.size(); });
  auto vcm = vcm_from_node_list(node_lists);
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  edges.reserve(N_edges);
  for (auto &&e_list : edge_lists) {
    edges.insert(edges.end(), e_list.begin(), e_list.end());
  }
  return SBM_simulate(p, p_I0, p_R0, p_R, edges, vcm, ecm, q, seed);
}

} // namespace Sycl_Graph::SBM

#endif