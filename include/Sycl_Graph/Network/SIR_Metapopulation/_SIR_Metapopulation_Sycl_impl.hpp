#ifndef SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#define SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#include <oneapi/dpl/internal/random_impl/uniform_real_distribution.h>
#include <random>
#ifdef SYCL_GRAPH_USE_SYCL
#include "SIR_Metapopulation_Types.hpp"
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/statistical_typedefs.hpp>
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
    };

    using SIR_Metapopulation_Graph =
        Sycl_Graph::Sycl::Graph<SIR_Metapopulation_Node, SIR_Metapopulation_Param,
                                uint32_t>;
    template <typename RNG = Sycl_Graph::random::default_rng>
    struct SIR_Metapopulation_Network
        : public Network<SIR_Metapopulation_Temporal_Param,
                         SIR_Metapopulation_Network<RNG>>
    {
      using Graph_t = SIR_Metapopulation_Graph;
      using Vertex_t = typename Graph_t::Vertex_t;
      using Edge_t = typename Graph_t::Edge_t;
      using Base_t =
          Network<SIR_Metapopulation_Temporal_Param, SIR_Metapopulation_Network>;
      const uint32_t t = 0;

      sycl::buffer<int, 1> seed_buf;
      sycl::buffer<RNG, 1> rng_buf;
      const std::vector<uint32_t> N_pop;
      std::vector<Normal_Distribution<float>> I0_dist;
      std::vector<Normal_Distribution<float>> R0_dist;

      SIR_Metapopulation_Network(Graph_t &G,
                                 const std::vector<uint32_t> &N_pop,
                                 const std::vector<Normal_Distribution<>> &I0,
                                 const std::vector<Normal_Distribution<>> &R0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : Base_t(3), q(G.q), G(G), N_pop(N_pop), I0_dist(I0),
            R0_dist(R0), rng_buf(sycl::range<1>(G.NE))
      {

        generate_seeds(seed);
      }
      void initialize()
      {
        sycl::buffer<uint32_t, 1> N_pop_buf(N_pop);

        q.submit([&](sycl::handler &h)
                 {
        auto N_pop_acc = N_pop_buf.get_access<sycl::access::mode::read>(h);
      auto seed = seed_buf.get_access<sycl::access::mode::read>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        // total population stored in susceptible state
        float N_pop = v.data[id].state.S;
        Sycl_Graph::random::default_rng rng(seed[id]);
        SIR_Metapopulation_State v_i;
        v_i.I = I0_dist[id](rng) * N_pop;
        v_i.R = R0_dist[id](rng) * N_pop;
        v_i.S = std::max({N_pop_acc[id] - v_i.I - v_i.R, 0.f});
        v.data[id].state = v_i;
      }); });
      }

      std::vector<uint32_t> population_count()
      {
        std::vector<uint32_t> count(3, 0);
        sycl::buffer<uint32_t, 1> count_buf(count.data(), sycl::range<1>(3));
        const uint32_t N_vertices = G.N_vertices();

        q.submit([&](sycl::handler &h)
                 {
      auto count_acc = count_buf.get_access<sycl::access::mode::write>(h);
      auto v = G.get_vertex_access<sycl::access::mode::read>(h);

      h.single_task([=] {
        for (int i = 0; i < N_vertices; i++) {
          count_acc[0] += v.data[i].state.S;
          count_acc[1] += v.data[i].state.I;
          count_acc[2] += v.data[i].state.R;
        }
      }); });
        q.wait();
        sycl::host_accessor count_acc(count_buf, sycl::read_only);
        // read into vector
        std::vector<uint32_t> res(3);
        for (int i = 0; i < 3; i++)
        {
          res[i] = count_acc[i];
        }

        return res;
      }

      // function for infection step
      void infection_scatter(float dt)
      {
        using Sycl_Graph::Network_Models::SIR_Metapopulation_State;

        // buffer for unmerged infections generated by vertices
        sycl::buffer<float, 1> v_inf_buf(sycl::range<1>(G.N_vertices()));

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_inf_acc = v_inf_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
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
      auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
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
      }

      void recovery_scatter(float dt)
      {

        sycl::buffer<float, 1> v_rec_buf(sycl::range<1>(G.N_vertices()));

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_rec_acc = v_rec_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()), [=](sycl::id<1> id) {
        

        auto alpha = v_acc.data[id].param.alpha;
        auto p_R = 1 - exp(-alpha * dt);
        auto I = v_acc.data[id].state.I;

        // Delta_R = bin(I, p_R)
        Sycl_Graph::random::binomial_distribution<float, RNG> d_R(v_acc.data[id].state.I, p_R);
        auto N_recovered = d_I(rng_acc[id]);
        v_rec_acc[id] = N_recovered;
      }); });
      }

      bool terminate()
      {
        return false;
      }

      void reset()
      {
        initialize();
      }

    private:
      void generate_seeds(int seed)
      {
        // generate seeds
        std::vector<int> seeds(G.NE);
        // random device
        // mt19937 generator
        std::mt19937_64 gen(seed);
        std::generate(seeds.begin(), seeds.end(), gen);
        q.submit([&](sycl::handler &h)
                 {
      auto rng_acc = rng_buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.NE),
                     [=](sycl::id<1> id) { rng_acc[id].seed(seeds[id]); }); });
      }

      Graph_t &G;
      sycl::queue &q;
    };
  } // namespace Sycl::Network_Models
} // namespace Sycl_Graph
#endif
#endif