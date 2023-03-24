#ifndef SIR_Bernoulli_SBM_SYCL_IMPL_HPP
#define SIR_Bernoulli_SBM_SYCL_IMPL_HPP
#include "SIR_Bernoulli_SBM_Types.hpp"
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <stddef.h>
#include <sycl/sycl.hpp>
#ifdef SYCL_GRAPH_USE_ONEAPI
#include <oneapi/dpl/algorithm>
#endif
#include <atomic>
#include <type_traits>
#include <utility>
template <>
struct sycl::is_device_copyable<Sycl_Graph::Network_Models::SIR_Edge>
    : std::true_type {};

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Individual_State> : std::true_type {};
namespace Sycl_Graph {
namespace Sycl::Network_Models {
// sycl::is_device_copyable_v<SIR_Individual_State>
// is_copyable_SIR_Invidual_State;
using namespace Sycl_Graph::Network_Models;
template <typename T> using SIR_vector_t = std::vector<T, std::allocator<T>>;
using SIR_Graph =
    Sycl_Graph::Sycl::Graph<SIR_Individual_State, SIR_Edge, uint32_t>;

struct SIR_Bernoulli_SBM_Network
    : public Network<SIR_Bernoulli_SBM_Network, std::vector<uint32_t>,
                     SIR_Bernoulli_SBM_Temporal_Param<>> {
  using Graph_t = SIR_Graph;
  using Vertex_t = typename Graph_t::Vertex_t;
  using Edge_t = typename Graph_t::Edge_t;
  using Base_t = Network<SIR_Bernoulli_SBM_Network, std::vector<uint32_t>,
                         SIR_Bernoulli_SBM_Temporal_Param<>>;
  const float p_I0;
  const float p_R0;
  sycl::buffer<int, 1> seed_buf;
  bool record_group_infections = true;
  uint32_t counter = 0;

  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> SBM_ids;
  std::vector<std::vector<uint32_t>> SBM_community_ids;
  std::vector<uint32_t> vertex_community_map;
  std::vector<uint32_t> edge_community_map;
  // SIR_Bernoulli_SBM_Network(): p_R0(0), p_I0(0),seed_buf(sycl::range<1>(0)),
  // G(q, 0, 0) {}

  SIR_Bernoulli_SBM_Network(const Graph_t &G, float p_I0, float p_R0,
                            const auto &SBM_ids, const auto &SBM_community_ids,
                            int seed = 777)
      : q(G.q), G(G), p_I0(p_I0), p_R0(p_R0),
        seed_buf(sycl::range<1>(G.N_vertices())), SBM_ids(SBM_ids),
        SBM_community_ids(SBM_community_ids) {
    counter = 0;
    assert(G.N_vertices() > 0 && "Graph must have at least one vertex");
    // generate G.N_vertices() random numbers
    // create rng
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1000000);
    std::vector<int> seeds(G.N_vertices());
    std::generate(seeds.begin(), seeds.end(), [&]() { return dist(rng); });
    // copy seeds to buffer
    sycl::buffer<int, 1> seeds_buf(seeds.data(),
                                   sycl::range<1>(G.N_vertices()));
    q.submit([&](sycl::handler &h) {
      auto seeds = seeds_buf.get_access<sycl::access::mode::read>(h);
      auto seed = seed_buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()),
                     [=](sycl::id<1> i) { seed[i] = seeds[i]; });
    });
  }

  uint32_t N_communities() const {
    // inverse of n(n−1)/2
    return SBM_community_ids.size()
    // return SBM_ids.size();
  }
  void initialize() {

    const float p_I0 = this->p_I0;
    const float p_R0 = this->p_R0;

    // generate seeds for
    sycl::buffer<float, 1> p_I_buf(sycl::range<1>(G.N_vertices()));
    sycl::buffer<float, 1> p_R_buf(sycl::range<1>(G.N_vertices()));
    std::vector<SIR_Individual_State> v_host(G.N_vertices());
    sycl::buffer<SIR_Individual_State, 1> v_buf(v_host.data(),
                                                sycl::range<1>(G.N_vertices()));

    q.submit([&](sycl::handler &h) {
      auto seed = seed_buf.get_access<sycl::access::mode::read>(h);
      auto p_I = p_I_buf.get_access<sycl::access::mode::write>(h);
      auto p_R = p_R_buf.get_access<sycl::access::mode::write>(h);
      auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> i) {
        Static_RNG::uniform_real_distribution<float> d_I;
        Static_RNG::uniform_real_distribution<float> d_R;
        Static_RNG::default_rng rng(seed[i]);

        if (d_I(rng) < p_I0) {
          v_acc.data[i] = SIR_INDIVIDUAL_I;
        } else if (d_R(rng) < p_R0) {
          v_acc.data[i] = SIR_INDIVIDUAL_R;
        } else {
          v_acc.data[i] = SIR_INDIVIDUAL_S;
        }
      });
    });
    create_vertex_community_map();
  }

  void create_vertex_community_map()
  {
    vertex_community_map.resize(G.N_vertices());
    std::generate(vertex_community_map.begin(), vertex_community_map.end(), [n = 0]() mutable { 
    //find n in SBM_community_ids
    std::find_if(SBM_community_ids.begin(), SBM_community_ids.end(), [&](const auto& node_vec)
    {
      return std::find(node_vec.begin(), node_vec.end(), n) != node_vec.end();
    });
    n++;
    return std::distance(SBM_community_ids.begin(), it);
  });
  }

  void create_edge_community_map()
  {
    edge_community_map.resize(G.N_edges());
    //get edges
    auto edges = G.edge_buf.get_edges();
    std::transform(edges.begin(), edges.end(), edge_community_map.begin(), [&](const auto& edge)
    {
      std::pair<uint32_t, uint32_t> edge_pair = {edge.to, edge.from};
      auto it = std::find_if(SBM_community_ids.begin(), SBM_community_ids.end(), [&](const auto& node_vec)
      {
        return std::find(node_vec.begin(), node_vec.end(), edge_pair) != node_vec.end();
      });
      return std::distance(SBM_community_ids.begin(), it);
    });
  }

  std::vector<uint32_t>
  read_state(const SIR_Bernoulli_SBM_Temporal_Param<> &tp) {
    counter++;
    std::vector<uint32_t> count(3, 0);
    sycl::buffer<uint32_t, 1> count_buf(count.data(), sycl::range<1>(3));
    const uint32_t N_vertices = G.N_vertices();

    q.submit([&](sycl::handler &h) {
      auto count_acc = count_buf.get_access<sycl::access::mode::write>(h);
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);

      h.single_task([=] {
        for (int i = 0; i < N_vertices; i++) {
          auto v_i = v_acc.data[i];
          if (v_i == SIR_INDIVIDUAL_S) {
            count_acc[0]++;
          } else if (v_i == SIR_INDIVIDUAL_I) {
            count_acc[1]++;
          } else if (v_i == SIR_INDIVIDUAL_R) {
            count_acc[2]++;
          }
        }
      });
    });
    q.wait();
    sycl::host_accessor count_acc(count_buf, sycl::read_only);
    // read into vector
    std::vector<uint32_t> res(3);
    for (int i = 0; i < 3; i++) {
      res[i] = count_acc[i];
    }

    return res;
  }

  bool is_susceptible_infected_edge(auto &v_acc, const uint32_t e0_to,
                                    const uint32_t e0_from){
    if (e0_to == Graph_t::invalid_id || e0_from == Graph_t::invalid_id)
      return false;
    if (v_acc[e0_to] == SIR_INDIVIDUAL_S && v_acc[e0_from] == SIR_INDIVIDUAL_I)
      return true;
    if (v_acc[e0_to] == SIR_INDIVIDUAL_I && v_acc[e0_from] == SIR_INDIVIDUAL_S)
      return true;
    return false;
  }

  auto get_susceptible_neighbor(auto &v_acc, const uint32_t e_to,
                                const uint32_t e_from) {
    SIR_Individual_State v_data[2] = {v_acc[e_from], v_acc[e_to]};
    if (v_data[0] == SIR_INDIVIDUAL_S && v_data[1] == SIR_INDIVIDUAL_I)
      return e_from;
    else if (v_data[0] == SIR_INDIVIDUAL_I && v_data[1] == SIR_INDIVIDUAL_S)
      return e_to;
    else
      return Graph_t::invalid_id;
  }

auto get_susceptible_neighbors(sycl::buffer<std::pair<uint32_t, uint32_t>>& SBM_connection_id_buf)
{
    sycl::buffer<bool> sn_buf(sycl::range<1>(G.N_vertices()));
      auto sn_mark_event = q.submit([&](sycl::handler &h) {
        auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
        auto e_SBM_acc = SBM_connection_id_buf.get_access<sycl::access::mode::read>(h);
        auto sn_acc = sn_buf.get_access<sycl::access::mode::write>(h);
        // parallel for

        h.parallel_for<class edge_validity>(
            sycl::range<1>(e_SBM_acc.size()), [=](sycl::id<1> index) {
              // get the index of the element to sort
              int i = index[0];
              uint32_t sn = get_susceptible_neighbor(
                  v_acc.data, e_SBM_acc[i].first, e_SBM_acc[i].second);
              if (sn != Graph_t::invalid_id) {
                sn_acc[sn] = true;
              }
            });
      });
    return std::make_pair(sn_buf, sn_mark_event)
}

auto count_susceptible_neighbors(sycl::buffer<bool>& sn_buf, auto dep_event)
{
      uint32_t count = 0;
      sycl::buffer<uint32_t> sn_count_buf(&count, sycl::range<1>(1));

    auto count_event = q.submit([&](sycl::handler &h) {
        h.depends_on(dep_event);
        auto sn_count_acc =
            sn_count_buf.get_access<sycl::access::mode::write>(h);
        auto sn_acc = sn_buf.get_access<sycl::access::mode::read>(h);
        h.single_task([=]() {
          for (int i = 0; i < sn_acc.size(); i++) {
            if (sn_acc[i]) {
              sn_count_acc[0]++;
            }
          }
        });
      }).wait();

      return std::make_pair(sn_count_buf, count_event);
}

    auto create_susceptible_neighbor_id_buf(sycl::buffer<bool>& sn_buf, uint32_t count, auto dep_event){
        sycl::buffer<uint32_t> sn_ids_buf((sycl::range<1>(count)));

        auto sn_count_event = q.submit([&](sycl::handler &h) {
          h.depends_on(dep_event);
          auto sn_ids_acc = sn_ids_buf.get_access<sycl::access::mode::write>(h);
          auto sn_acc = sn_buf.get_access<sycl::access::mode::read>(h);
          h.single_task([=]() {
            int j = 0;
            for (int i = 0; i < sn_acc.size(); i++) {
              if (sn_acc[i]) {
                sn_ids_acc[j] = i;
                j++;
              }
            }
          });
        });
        return std::make_pair(sn_ids_buf, sn_count_event);
        }

    auto sample_infection_events(sycl::buffer<uint32_t>& sn_ids_buf, auto sn_count_event)
    {
        sycl::buffer<bool, 1> sn_inf_event_buf((sycl::range<1>(sn_ids_buf.size())));
        auto infection_sample_event = q.submit([&](sycl::handler &h) {
          h.depends_on(sn_count_event);
          auto sn_ids_acc = sn_ids_buf.get_access<sycl::access::mode::read>(h);
          auto v_acc = G.get_vertex_access<sycl::access_mode::read_write>(h);
          auto seed_acc =
              seed_buf.get_access<sycl::access::mode::read_write>(h);
          auto sn_inf_acc = sn_inf_event_buf.get_access<sycl::access::mode::write>(h);
          h.parallel_for(
              sycl::range<1>(sn_ids_acc.size()), [=](sycl::id<1> id) {
                Static_RNG::default_rng rng(seed_acc[id]);
                Static_RNG::uniform_real_distribution<float> d_R(0, 1);
                if (v_acc.data[sn_ids_acc[id[0]]] == SIR_INDIVIDUAL_S) {
                  if (d_R(rng) < p_I) {
                    v_acc.data[sn_ids_acc[id[0]]] = SIR_INDIVIDUAL_I;
                    sn_inf_acc[id] = true;
                  }
                }
                seed_acc[id] += 1;
              });
        });
        return std::make_pair(sn_inf_event_buf, infection_sample_event)
    }

    auto count_infection_events(sycl::buffer<bool, 1>& sn_inf_event_buf, auto dep_event)
    {
        sycl::buffer<uint32_t, 1> inf_count((sycl::range<1>(1)));
        q.submit([&](sycl::handler &h) {
           h.depends_on(dep_event);
           auto sn_inf_acc = sn_inf_buf.get_access<sycl::access::mode::read>(h);
           auto inf_count_acc =
               inf_count.get_access<sycl::access::mode::write>(h);
           // accumulate the number of infected neighbors
           h.single_task([=]() {
             inf_count_acc[0] = 0;
             for (int i = 0; i < sn_inf_acc.size(); i++) {
               if (sn_inf_acc[i]) {
                 inf_count_acc[0]++;
               }
             }
           });
         });
         return std::make_pair(inf_count, inf_count_event);
    }

  auto get_susceptible_id_if_infected_edge(auto& v_acc, uint32_t id_to, uint32_t id_from)
  {
    if (v_acc.data[id_to] == SIR_INDIVIDUAL_S && v_acc.data[id_from] == SIR_INDIVIDUAL_I) {
      return id_to;
    } else if (v_acc.data[id_from] == SIR_INDIVIDUAL_S && v_acc.data[id_to] == SIR_INDIVIDUAL_I) {
      return id_from;
    } else {
      return std::numeric_limits<uint32_t>::max();
    }
  }

  auto get_SBM_connection_idx(auto& SBM_Connection_ID_acc, std::pair<uint32_t, uint32_t> node_id_pair)
  {
    for(int i = 0; i < SBM_ids.size(); i++){
      if(SBM_Connection_ID_acc[i] == node_id_pair)
        return i;
      }
  }

  bool invalid_id(uint32_t id)
  {
    return id == std::numeric_limits<uint32_t>::max();
  }

  auto sample_community_infection_events(
      std::vector<float> p_Is,
      const std::vector<std::pair<uint32_t, uint32_t>> &SBM_connection) {

    using Sycl_Graph::Network_Models::SIR_Individual_State;
    if (SBM_connection.size() == 0)
      return 0;

    sycl::buffer<float> p_I_buf(p_Is.data(), sycl::range<1>(p_Is.size()));
    sycl::buffer<bool> infection_events((sycl::range<1>(G.N_edges())));
    sycl::buffer<uint32_t> ecm_buf(edge_community_map.data(), sycl::range<1>(edge_community_map.size()));
    sycl::buffer<uint32_t> sus_ids(G.N_edges());
    sycl::buffer<bool> node_infs((sycl::range<1>(G.N_vertices)));
    q.submit([&](sycl::handler& h)
    {
      auto v_acc = G.get_vertex_access<sycl::access_mode::read>(h);
      auto e_acc = G.get_edge_access<sycl::access_mode::read>(h);
      auto seed_acc =
          seed_buf.get_access<sycl::access::mode::read_write>(h);
      auto p_I_acc = p_I_buf.get_access<sycl::access::mode::read>(h);
      auto ecm_acc = ecm_buf.get_access<sycl::access::mode::read>(h);
      auto sus_id_acc = sus_ids.get_access<sycl::access::mode::write>(h);
      auto inf_event_acc = infection_events.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()), [&](sycl::id<1> id)
      {
          Static_RNG::default_rng rng(seed_acc[id]);
          Static_RNG::bernoulli_distribution<float> d_I(p_I_acc[ecm_acc[id]]);
          auto sus_id_acc[id] = get_susceptible_id_if_infected_edge(v_acc, e_acc.data[id[0]].to, e_acc.data[id[0]].from);
          if(!invalid_id(sus_id_acc[id]) && d_I(rng)){
            inf_event_acc[ecm_acc[id]]++;
            node_infs(sus_id_acc[id]) = true;
          }
      });

    });

    sycl::buffer<uint32_t> vcm_buf(vertex_community_map.data(), sycl::range<1>(vertex_community_map.size()));
    sycl::buffer<uint32_t> community_infs((sycl::range<1>(N_communities())));
    q.submit([&](sycl::handler& h)
    {
      auto inf_event_acc = infection_events.get_access<sycl::access::mode::read>(h);
      auto node_infs_acc = node_infs.get_access<sycl::access::mode::read>(h);
      auto community_infs_acc = community_infs.get_access<sycl::access::mode::write>(h);
      auto vcm_acc = vcm_buf.get_access<sycl::access::mode::read>(h);
      h.parallel_for(sycl::range<1>(N_communities()), [&](sycl::id<1> id)
      {
        for(int i = 0; i < G.N_edges(); i++){
          if(vcm_acc[i] == id[0] && node_infs_acc[sus_ids[i]])
          {
            community_infs_acc[id[0]]++;
          }
        }
      });
    });

    sycl::buffer<uint32_t> connection_infs((sycl::range<1>(SBM_connection.size())));
    q.submit([&](sycl::handler& h)
    {
      auto inf_event_acc = infection_events.get_access<sycl::access::mode::read>(h);
      auto connection_infs_acc = connection_infs.get_access<sycl::access::mode::write>(h);
      auto ecm_acc = ecm_buf.get_access<sycl::access::mode::read>(h);
      h.parallel_for(sycl::range<1>(SBM_connection.size()), [&](sycl::id<1> id)
      {
        connection_infs_acc[id[0]] = inf_event_acc[ecm_acc[id[0]]];
      });
    });

  }

  auto get_community_infected_counts(auto sn_inf_event_pairs)
  {
    //create a vector mapping vertex indices to community indices
    sycl::buffer<uint32_t, 1> vcm_buf(vertex_community_map.data(), sycl::range<1>(vertex_community_map.size()));
    sycl::buffer<bool, 1> vertex_infs(sycl::range<1>(G.N_vertices()), false);
    std::vector<sycl::event> events(sn_inf_event_pairs.size());

    std::transform(sn_inf_event_pairs.begin(), sn_inf_event_pairs.end(), events.begin(), [&](auto& p) {
      auto sn_inf_buf = p.first;
      auto sn_inf_event = p.second;
      sycl::buffer<uint32_t> inf_count((sycl::range<1>(vertex_community_map.size())));
        return q.submit([&](sycl::handler &h) {
       h.depends_on(sn_inf_event);
        auto sn_inf_acc = sn_inf_buf.get_access<sycl::access::mode::read>(h);
        auto vertex_inf_acc = vertex_infs.get_access<sycl::access::mode::write>(h);
        h.parallel_for(
            sycl::range<1>(sn_inf_acc.size()), [=](sycl::id<1> id) {
            vertex_inf_acc[sn_inf_acc[id]] = true;
       });
     });
     });

    //map vertex_infs to community counts
    auto count_event = q.submit([&](sycl::handler &h) {
       h.depends_on(events);
       auto vertex_inf_acc = vertex_infs.get_access<sycl::access::mode::read>(h);
       auto vcm_acc = vcm_buf.get_access<sycl::access::mode::read>(h);
       auto inf_count_acc =
           inf_count.get_access<sycl::access::mode::read_write>(h);
       h.single_task(
           sycl::range<1>(vertex_inf_acc.size()), [&](){
            for (int i = 0; i < vertex_inf_acc.size(); i++) {
             if (vertex_inf_acc[i]) {
               inf_count_acc[vcm_acc[i]]++;
             }
            }
           });
    });

     return std::make_pair(vertex_infs, count_event);
  }

  typedef std::pair<sycl::buffer<uint32_t>, sycl::event> Infected_Count_t;

  std::vector<uint32_t> infection_step(const SIR_Bernoulli_SBM_Temporal_Param<> &p)
  {
    std::vector<Infected_Count_t> infection_events(p.p_Is.size());
    std::transform(std::execution::par_unseq,
        SBM_ids.begin(), SBM_ids.end(), p.p_Is.begin(), infection_events.begin(),
        [&](const auto &v, float p_I) {
            return sample_community_infection_events(p_I, );
        });
    auto N_community_infected = get_community_infected_counts(infection_events);
  }


  std::vector<uint32_t> recovery_step(float p_R) {

    sycl::buffer<bool, 1> rec_buf((sycl::range<1>(G.N_vertices())));
    q.submit([&](sycl::handler &h) {
       auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write>(h);
       auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);
       auto rec_acc = rec_buf.get_access<sycl::access::mode::write>(h);
       //  auto nv =
       //  neighbors_buf.get_access<sycl::access::mode::read_write>(h);
       h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
         Static_RNG::default_rng rng(seed_acc[id]);
         seed_acc[id]++;
         Static_RNG::uniform_real_distribution<float> d_R(0, 1);
         if (v_acc.data[id] == SIR_INDIVIDUAL_I) {
           if (d_R(rng) < p_R) {
             rec_acc[id] = true;
             v_acc.data[id] = SIR_INDIVIDUAL_R;
           }
         }
       });
     }).wait();
    auto rec_acc = rec_buf.get_access<sycl::access::mode::read>();
    std::vector<uint32_t> N_recovered(SBM_community_ids.size(), 0);
    for (int i = 0; i < rec_acc.size(); i++) {
      // get index of i in SBM_community_ids
      auto idx =
          std::find_if(SBM_community_ids.begin(), SBM_community_ids.end(),
                       [i](const std::vector<uint32_t> &v) {
                         return std::find(v.begin(), v.end(), i) != v.end();
                       });
      if (idx != SBM_community_ids.end()) {
        N_recovered[std::distance(SBM_community_ids.begin(), idx)] +=
            rec_acc[i];
      }
    }
    return N_recovered;
  }

  void advance(const SIR_Bernoulli_SBM_Temporal_Param<> &p,
               std::vector<uint32_t> &N_infected,
               std::vector<uint32_t> &N_recovered) {

    std::transform(
        SBM_ids.begin(), SBM_ids.end(), p.p_Is.begin(), N_infected.begin(),
        [&](const auto &v, float p_I) { 
            
            
            return infection_step(p_I, v); });
    N_recovered = recovery_step(p.p_R);
  }

  bool terminate(const std::vector<uint32_t> &x,
                 const SIR_Bernoulli_SBM_Temporal_Param<> &p) {
    static int t = 0;
    bool early_termination = ((t > p.Nt_min) && (x[1] < p.N_I_min));
    return early_termination;
  }

  void reset() {
    q.submit([&](sycl::handler &h) {
      auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        v_acc.data[id[0]] = SIR_INDIVIDUAL_S;
      });
    });
  }

  SIR_Bernoulli_SBM_Network &operator=(SIR_Bernoulli_SBM_Network other) {
    q = other.q;
    std::swap(G, other.G);
    std::swap(SBM_ids, other.SBM_ids);
    std::swap(seed_buf, other.seed_buf);
    return *this;
  }

  using Base_t::simulate;

  typedef std::vector<std::vector<uint32_t>> Trajectory_t;
  typedef std::pair<Trajectory_t, Trajectory_t> Trajectory_pair_t;

  auto
  simulate_groups(const std::vector<SIR_Bernoulli_SBM_Temporal_Param<>> tp) {

    assert(tp[0].p_Is.size() == SBM_ids.size() &&
           "Must have one p_I for each SBM_id_group");
    auto Nt = tp.size() - 1;
    std::vector<std::vector<uint32_t>> delta_Is(Nt);
    std::vector<std::vector<uint32_t>> delta_Rs(Nt);
    // resize delta_Is
    for (int i = 0; i < Nt; i++) {
      delta_Is[i].resize(SBM_ids.size(), 0);
      delta_Rs[i].resize(SBM_community_ids.size(), 0);
    }
    std::vector<std::vector<uint32_t>> trajectory(Nt + 1);
    uint32_t t = 0;
    auto tp_i = tp[0];
    trajectory[0] = read_state(tp[0]);
    for (int i = 0; i < Nt; i++) {
      auto tp_i = tp[i + 1];
      advance(tp_i, delta_Is[i], delta_Rs[i]);
      uint32_t Delta_I_sum =
          std::accumulate(delta_Is[i].begin(), delta_Is[i].end(), 0);
      uint32_t Delta_R_sum =
          std::accumulate(delta_Rs[i].begin(), delta_Rs[i].end(), 0);
      trajectory[i + 1] = {trajectory[i][0] - Delta_I_sum,
                           trajectory[i][1] + Delta_I_sum - Delta_R_sum,
                           trajectory[i][2] + Delta_R_sum};
      if (terminate(trajectory[i + 1], tp_i)) {
        break;
      }
    }
    return std::make_tuple(trajectory, delta_Is, delta_Rs);
  }
  auto byte_size() const { return G.byte_size(); }

  Graph_t G;

private:
  sycl::queue &q;
};
} // namespace Sycl::Network_Models
} // namespace Sycl_Graph
#endif