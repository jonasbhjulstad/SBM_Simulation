#ifndef SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#define SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#include <oneapi/dpl/internal/random_impl/uniform_real_distribution.h>
#include <random>
#ifdef SYCL_GRAPH_USE_SYCL
#include "SIR_Metapopulation_Types.hpp"
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/random.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/random>
#include <stddef.h>
#include <sycl/CL/sycl.hpp>
#include <type_traits>
#include <utility>
template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_Node_Param>
    : std::true_type
{
};

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_Param> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_State> : std::true_type
{
};
namespace Sycl_Graph
{

  namespace Sycl::Network_Models
  {

    float compute_infection_probability(float beta, float N_I, float N, float dt, float c = 1.f)
    {
      return 1 - std::exp(-beta * N_I * dt / N);
    }

    // sycl::is_device_copyable_v<SIR_Metapopulation_State>
    // is_copyable_SIR_Invidual_State;
    using namespace Sycl_Graph::Network_Models;
    using namespace Sycl_Graph::random;
    template <typename T>
    using SIR_vector_t = std::vector<T, std::allocator<T>>;
    struct SIR_Metapopulation_Node
    {
      SIR_Metapopulation_State state;
      SIR_Metapopulation_Param param;
    };
    struct SIR_Metapopulation_Temporal_Param
    {
      uint32_t Nt_max = 50;
      float dt = 0.1f;
    };

    using SIR_Metapopulation_Graph =
        Sycl_Graph::Sycl::Graph<SIR_Metapopulation_Node, SIR_Metapopulation_Param,
                                uint32_t>;
    template <typename RNG = Sycl_Graph::random::default_rng>
    struct SIR_Metapopulation_Network
        : public Network<SIR_Metapopulation_Network<RNG>,
                         SIR_Metapopulation_State, SIR_Metapopulation_Temporal_Param>
    {
      using Graph_t = SIR_Metapopulation_Graph;
      using Vertex_t = typename Graph_t::Vertex_t;
      using Edge_t = typename Graph_t::Edge_t;
      using Base_t =
          Network<SIR_Metapopulation_Network<RNG>, SIR_Metapopulation_Temporal_Param,
                         SIR_Metapopulation_State>;
      const uint32_t t = 0;

      sycl::buffer<RNG, 1> rng_buf;
      const std::vector<uint32_t> N_pop;
      sycl::buffer<normal_distribution<float>, 1> I0_dist;
      sycl::buffer<normal_distribution<float>, 1> R0_dist;
      const std::vector<float> alpha_0;
      const std::vector<float> node_beta_0;
      const std::vector<float> edge_beta_0;
      SIR_Metapopulation_Network(Graph_t &G,
                                 const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> &I0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777) : SIR_Metapopulation_Network(G, N_pop, I0, std::vector<normal_distribution<float>>(I0.size()), alpha, node_beta, edge_beta, seed) {}

      SIR_Metapopulation_Network(Graph_t &G,
                                 const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> &I0,
                                 const std::vector<normal_distribution<float>> &R0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : q(G.q), G(G), N_pop(N_pop), I0_dist(I0),
            R0_dist(R0), rng_buf(sycl::range<1>(G.NE)), alpha_0(alpha), node_beta_0(node_beta), edge_beta_0(edge_beta)
      {
        generate_seeds(seed);
      }
      void initialize()
      {
        sycl::buffer<uint32_t, 1> N_pop_buf(N_pop);
        q.submit([&](sycl::handler &h)
                 {
      auto N_pop_acc = N_pop_buf.get_access<sycl::access::mode::read>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::read_write>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      auto I0_dist_acc = I0_dist.template get_access<sycl::access::mode::read_write>(h);
      auto R0_dist_acc = R0_dist.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        // total population stored in susceptible state
        float N_pop = v.data[id].state.S;
        SIR_Metapopulation_State v_i;
        // v_i.I = I0_dist_acc[id](rng_acc[id]) * N_pop;
        // v_i.R = R0_dist_acc[id](rng_acc[id]) * N_pop;
        v_i.I = 100;
        v_i.R = 0;
        v_i.S = std::max({N_pop_acc[id] - v_i.I - v_i.R, 0.f});
        v.data[id].state = v_i;
      }); });
        set_alpha(alpha_0);
        set_node_beta(node_beta_0);
        set_edge_beta(edge_beta_0);
      }

      void set_alpha(const std::vector<float> &alpha)
      {
        set_alpha(alpha, Sycl_Graph::range(0, alpha.size()));
      }

      void set_alpha(const std::vector<float> &alpha, const std::vector<uint32_t> &idx)
      {
        sycl::buffer<float, 1> alpha_buf(alpha);
        sycl::buffer<uint32_t, 1> idx_buf(idx);
        q.submit([&](sycl::handler &h)
                 {
      auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
      auto idx_acc = idx_buf.get_access<sycl::access::mode::read>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(idx_acc.size()), [=](sycl::id<1> id) {
        v.data[idx_acc[id]].param.alpha = alpha_acc[id];
      }); });
      }

      void set_node_beta(const std::vector<float> &beta)
      {
        set_node_beta(beta, Sycl_Graph::range(0, beta.size()));
      }

      void set_node_beta(const std::vector<float> &beta, const std::vector<uint32_t> &idx)
      {
        sycl::buffer<float, 1> beta_buf(beta);
        sycl::buffer<uint32_t, 1> idx_buf(idx);
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto idx_acc = idx_buf.get_access<sycl::access::mode::read>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(idx.size()), [=](sycl::id<1> id) {
        v.data[idx_acc[id]].param.beta = beta_acc[id];
      }); });
      }

      void set_edge_beta(const std::vector<float> &beta)
      {
        sycl::buffer<float, 1> beta_buf(beta);
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto e_acc = G.edge_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(beta_acc.size()), [=](sycl::id<1> id) {
        e_acc.data[id].beta = beta_acc[id];
      }); });
      }

      void set_edge_beta(const std::vector<float> &beta, const std::vector<uint32_t> &to_idx, const std::vector<uint32_t> &from_idx)
      {
        sycl::buffer<float, 1> beta_buf(beta);
        sycl::buffer<uint32_t, 1> to_idx_buf(to_idx);
        sycl::buffer<uint32_t, 1> from_idx_buf(from_idx);
        uint32_t N_vertices = G.N_vertices();
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto to_idx_acc = to_idx_buf.get_access<sycl::access::mode::read>(h);
      auto from_idx_acc = from_idx_buf.get_access<sycl::access::mode::read>(h);
      auto e_acc = G.edge_buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(to_idx_acc.size()), [=](sycl::id<1> id) {
        for (int i = 0; i < N_vertices; i++)
        {
          if (e_acc.to[i] == to_idx_acc[id] && e_acc.from[i] == from_idx_acc[id])
            e_acc.data[i].beta = beta_acc[id];
        }
      }); });
      }

      SIR_Metapopulation_State read_state(SIR_Metapopulation_Temporal_Param tp)
      {
        SIR_Metapopulation_State state;
        sycl::buffer<SIR_Metapopulation_State, 1> state_buf(&state, sycl::range<1>(1));
        const uint32_t N_vertices = G.N_vertices();

        q.submit([&](sycl::handler &h)
                 {
      auto v = G.get_vertex_access<sycl::access::mode::read>(h);
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);
      h.single_task([=] {
        for (int i = 0; i < N_vertices; i++) {
          state_acc[0] = v.data[i].state;
        }
      }); });
        q.wait();

        return state;
      }

    void advance(SIR_Metapopulation_Temporal_Param tp)
    {
      infection_scatter(tp.dt);
      recovery_scatter(tp.dt);
      q.wait();

    }

    bool terminate(SIR_Metapopulation_State x, const SIR_Metapopulation_Temporal_Param tp)
    {
      return false;
    }
    private:


      // function for infection step
      std::pair<sycl::buffer<float, 1>, sycl::buffer<float, 1>> infection_scatter(float dt)
      {
        using Sycl_Graph::Network_Models::SIR_Metapopulation_State;

        // buffer for unmerged infections generated by vertices
        sycl::buffer<float, 1> v_inf_buf(sycl::range<1>(G.N_vertices()));

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_inf_acc = v_inf_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()), [=](sycl::id<1> id) {
        

        auto beta = v_acc.data[id].param.beta;
        auto S = v_acc.data[id].state.S;
        auto I = v_acc.data[id].state.I;
        auto R = v_acc.data[id].state.R;
        auto N_pop = S + I + R;

        auto p_I = compute_infection_probability(beta, I, N_pop, dt);
        Sycl_Graph::random::binomial_distribution<float, RNG> d_I(v_acc.data[id].state.S, p_I);
        auto d_S = d_I(rng_acc[id]);
        v_inf_acc[id] = d_S;
      }); });
        uint32_t N_edges = G.N_edges();
        uint32_t N_vertices = G.N_vertices();
        // buffer for unmerged infections generated by edges
        sycl::buffer<float, 1> e_inf_buf(0, sycl::range<1>(N_vertices));
        // buffer for index positions of infections generated by edges

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
      auto e_acc = G.get_edge_access<sycl::access::mode::read>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::read_write>(h);
      auto e_inf_acc = e_inf_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> id) {
        
        auto beta = v_acc.data[id].param.beta;
        auto S = v_acc.data[id].state.S;
        auto I = v_acc.data[id].state.I;
        auto R = v_acc.data[id].state.R;
        auto N_pop = S + I + R;

        //Delta_I = bin(S, p_I)
        //p_I = 1 - exp(-beta*c*I/N_pop*dt)
        auto p_I = compute_infection_probability(beta, I, N_pop, dt);
        Sycl_Graph::random::binomial_distribution d_I(v_acc.data[id].state.S,
                                                      p_I);
        
        auto N_infected = d_I(rng_acc[id]);
        e_inf_acc[e_acc.to[id]] += N_infected;
      }); });

      return std::make_pair(v_inf_buf, e_inf_buf);
      }

      sycl::buffer<float, 1> recovery_scatter(float dt)
      {

        sycl::buffer<float, 1> v_rec_buf(sycl::range<1>(G.N_vertices()));

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_rec_acc = v_rec_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()), [=](sycl::id<1> id) {
        

        auto alpha = v_acc.data[id].param.alpha;
        auto p_R = 1 - exp(-alpha * dt);
        auto I = v_acc.data[id].state.I;

        // Delta_R = bin(I, p_R)
        Sycl_Graph::random::binomial_distribution<float, RNG> d_R(v_acc.data[id].state.I, p_R);
        auto N_recovered = d_R(rng_acc[id]);
        v_rec_acc[id] = N_recovered;
      }); });

        return v_rec_buf;
      }


      void gather(sycl::buffer<float, 1>& v_inf_buf, sycl::buffer<float, 1>& e_inf_buf, sycl::buffer<float, 1>& v_rec_buf)
      {
        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
      auto v_inf_acc = v_inf_buf.template get_access<sycl::access::mode::read>(h);
      auto e_inf_acc = e_inf_buf.template get_access<sycl::access::mode::read>(h);
      auto v_rec_acc = v_rec_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        
        v_acc.data[id].state.S -= v_inf_acc[id] + e_inf_acc[id];
        v_acc.data[id].state.I += v_inf_acc[id] + e_inf_acc[id] - v_rec_acc[id];
        v_acc.data[id].state.R += v_rec_acc[id];
      }); });
      }



      void reset()
      {
        initialize();
      }

      void generate_seeds(int seed)
      {
        // generate seeds
        std::vector<int> seeds(G.NE);
        // random device
        // mt19937 generator
        std::mt19937_64 gen(seed);
        std::generate(seeds.begin(), seeds.end(), gen);
        sycl::buffer<int, 1> seed_buf(seeds.data(), sycl::range<1>(G.NE));
        q.submit([&](sycl::handler &h)
                 {
      auto seed_acc = seed_buf.template get_access<sycl::access::mode::read>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.NE),
                     [=](sycl::id<1> id) { rng_acc[id].seed(seed_acc[id]); }); });
      }

      Graph_t &G;
      sycl::queue &q;
    };
  } // namespace Sycl::Network_Models
} // namespace Sycl_Graph
#endif
#endif