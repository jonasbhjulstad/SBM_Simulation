#ifndef SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#define SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#include "Sycl_Graph/path_config.hpp"
#include <Sycl_Graph/Tracy_Config.hpp>
#define SYCL_FLOAT_PRECISION 32
#include "SIR_Metapopulation_Types.hpp"
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Static_RNG/distributions.hpp>
#ifdef SYCL_GRAPH_USE_ONEAPI
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/internal/random_impl/uniform_real_distribution.h>
#include <oneapi/dpl/random>
#endif
#include <CL/sycl.hpp>
#include <fmt/format.h>
#include <random>
#include <stddef.h>
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

    float compute_infection_probability(float beta, float N_I, float N, float dt,
                                        float c = 1.f)
    {
      return 1 - std::exp(-beta * N_I * dt / N);
    }

    // sycl::is_device_copyable_v<SIR_Metapopulation_State>
    // is_copyable_SIR_Invidual_State;
    using namespace Sycl_Graph::Network_Models;
    using namespace Static_RNG;
    template <typename T>
    using SIR_vector_t = std::vector<T, std::allocator<T>>;
    struct SIR_Metapopulation_Node
    {
      SIR_Metapopulation_State state;
      SIR_Metapopulation_Param param;
    };
    struct SIR_Metapopulation_Temporal_Param
    {
      SIR_Metapopulation_Temporal_Param() = default;
      uint32_t Nt_max = 50;
      float dt = 5.f;
    };

    using SIR_Metapopulation_Graph =
        Sycl_Graph::Sycl::Graph<SIR_Metapopulation_Node, SIR_Metapopulation_Param,
                                uint32_t>;
    template <Static_RNG::rng_type RNG = Static_RNG::default_rng>
    struct SIR_Metapopulation_Network
        : public Network<SIR_Metapopulation_Network<RNG>, SIR_Metapopulation_State,
                         SIR_Metapopulation_Temporal_Param>
    {
      using Graph_t = SIR_Metapopulation_Graph;
      using Vertex_t = typename Graph_t::Vertex_t;
      using Edge_t = typename Graph_t::Edge_t;
      using Base_t =
          Network<SIR_Metapopulation_Network<RNG>,
                  SIR_Metapopulation_Temporal_Param, SIR_Metapopulation_State>;

#ifdef SYCL_GRAPH_DEBUG
      static uint32_t Instance_Count = 0;
      uint32_t debug_instance_ID = Instance_Count++;
#endif
      sycl::queue &q;
      Graph_t &G;

      const uint32_t t = 0;

      sycl::buffer<RNG, 1> rng_buf;
      const std::vector<uint32_t> N_pop;
      sycl::buffer<normal_distribution<float>, 1> I0_dist;
      sycl::buffer<normal_distribution<float>, 1> R0_dist;
      const std::vector<float> alpha_0;
      const std::vector<float> node_beta_0;
      const std::vector<float> edge_beta_0;
      SIR_Metapopulation_Network(Graph_t &G, const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> &I0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : SIR_Metapopulation_Network(
                G, N_pop, I0, std::vector<normal_distribution<float>>(I0.size()),
                alpha, node_beta, edge_beta, seed) {}

      SIR_Metapopulation_Network(Graph_t &G, const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> &I0,
                                 const std::vector<normal_distribution<float>> &R0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : q(G.q), G(G), N_pop(N_pop), I0_dist(I0), R0_dist(R0),
            rng_buf(sycl::range<1>(std::max({G.N_vertices(), G.N_edges()}))), alpha_0(alpha),
            node_beta_0(node_beta), edge_beta_0(edge_beta)
      {
        if (G.N_edges() > 0)
          generate_seeds(seed);
#ifdef SYCL_GRAPH_DEBUG
        construction_debug_report();
#endif
      }

      void initialize()
      {
        ZoneScoped;
        sycl::buffer<uint32_t, 1> N_pop_buf(N_pop);
        q.submit([&](sycl::handler &h)
                 {
      auto N_pop_acc = N_pop_buf.get_access<sycl::access::mode::read>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      auto I0_dist_acc =
          I0_dist.template get_access<sycl::access::mode::read_write>(h);
      auto R0_dist_acc =
          R0_dist.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        // total population stored in susceptible state
        auto &N_pop = N_pop_acc[id];
        auto &rng = rng_acc[id];
        auto& I0_dist = I0_dist_acc[id];
        auto& R0_dist = R0_dist_acc[id];

        auto I0 = I0_dist(rng);
        auto R0 = R0_dist(rng);

        auto &state = v.data[id].state;
        state.S = N_pop - I0 - R0;
        state.I = I0;
        state.R = R0;

      }); });
        set_edge_beta(edge_beta_0);
        set_node_beta(node_beta_0);
        set_alpha(alpha_0);
      }

      void set_alpha(const std::vector<float> &alpha)
      {
        set_alpha(alpha, Sycl_Graph::range(0, alpha.size()));
      }

      void set_alpha(const std::vector<float> &alpha,
                     const std::vector<uint32_t> &idx)
      {
        ZoneScoped;
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

      void set_node_beta(const std::vector<float> &beta,
                         const std::vector<uint32_t> &idx)
      {
        ZoneScoped;

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
        ZoneScoped;

        if (G.N_edges() == 0)
        {
          std::cout << "Warning: Unable to set edge beta, graph has no edges."
                    << std::endl;
          return;
        }
        sycl::buffer<float, 1> beta_buf(beta);
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto e_acc = G.edge_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(beta_acc.size()), [=](sycl::id<1> id) {
        e_acc.data[id].beta = beta_acc[id];
      }); });
      }

      void set_edge_beta(const std::vector<float> &beta,
                         const std::vector<uint32_t> &to_idx,
                         const std::vector<uint32_t> &from_idx)
      {
        ZoneScoped;

        if (G.N_edges() == 0)
        {
          std::cout << "Warning: Unable to set edge beta, graph has no edges."
                    << std::endl;
          return;
        }
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
        for (int i = 0; i < N_vertices; i++) {
          if (e_acc.to[i] == to_idx_acc[id] &&
              e_acc.from[i] == from_idx_acc[id])
            e_acc.data[i].beta = beta_acc[id];
        }
      }); });
      }

      SIR_Metapopulation_State read_state(SIR_Metapopulation_Temporal_Param tp)
      {
        ZoneScoped;

        SIR_Metapopulation_State state;
        sycl::buffer<SIR_Metapopulation_State, 1> state_buf(&state,
                                                            sycl::range<1>(1));
        // set state_buf to 0
        q.submit([&](sycl::handler &h)
                 {
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);
      h.single_task([=] { state_acc[0] = SIR_Metapopulation_State(); }); });
        const uint32_t N_vertices = G.N_vertices();

        q.submit([&](sycl::handler &h)
                 {
      auto v = G.get_vertex_access<sycl::access::mode::read>(h);
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);

      h.single_task([=] {
        for (int i = 0; i < N_vertices; i++) {
          state_acc[0] += v.data[i].state;
        }
      }); });
        q.wait();

        return state;
      }

      void advance(SIR_Metapopulation_Temporal_Param tp)
      {
        ZoneScoped;

        auto inf_bufs = infection_scatter(tp.dt);
        auto rec_buf = recovery_scatter(tp.dt);
        q.wait();
        gather(inf_bufs.first, inf_bufs.second, rec_buf);
      }

      bool terminate(SIR_Metapopulation_State x,
                     const SIR_Metapopulation_Temporal_Param tp)
      {
        return false;
      }

    private:
      // function for infection step
      std::pair<sycl::buffer<float, 1>, sycl::buffer<float, 1>>
      infection_scatter(float dt)
      {
        ZoneScoped;

        using Sycl_Graph::Network_Models::SIR_Metapopulation_State;

        // buffer for unmerged infections generated by vertices
        // sycl::buffer<float, 1> v_inf_buf(sycl::range<1>(G.N_vertices()));
        // ensure that buffer is 0-initialized
        sycl::buffer<float, 1> v_inf_buf(sycl::range<1>(G.N_vertices()));
        FrameMarkStart("Vertex Infections");
        auto event_v_inf = q.submit([&](sycl::handler &h)
                                    {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_inf_acc =
          v_inf_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        auto beta = v_acc.data[id].param.beta;
        auto S = v_acc.data[id].state.S;
        auto I = v_acc.data[id].state.I;
        auto R = v_acc.data[id].state.R;
        auto N_pop = S + I + R;

        auto p_I = compute_infection_probability(beta, I, N_pop, dt);
        Static_RNG::binomial_distribution<float> d_I(
            v_acc.data[id].state.S, p_I);
        v_inf_acc[id] = d_I(rng_acc[id]);
      }); });
        FrameMarkEnd("Vertex Infections");
        uint32_t N_edges = G.N_edges();

        uint32_t N_vertices = G.N_vertices();
        auto v_inf_acc = v_inf_buf.template get_access<sycl::access::mode::read>();
        sycl::buffer<float, 1> e_inf_buf(0, sycl::range<1>(N_vertices));

        if (N_edges > 0)
        {

          FrameMarkStart("Edge Infections");
          q.submit([&](sycl::handler &h)
                   {
        auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
        auto e_acc = G.get_edge_access<sycl::access::mode::read>(h);
        auto rng_acc =
            rng_buf.template get_access<sycl::access::mode::read_write>(h);
        auto e_inf_acc =
            e_inf_buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> id) {
          auto beta = v_acc.data[id].param.beta;
          auto S = v_acc.data[id].state.S;
          auto I = v_acc.data[id].state.I;
          auto R = v_acc.data[id].state.R;
          auto N_pop = S + I + R;

          // Delta_I = bin(S, p_I)
          // p_I = 1 - exp(-beta*c*I/N_pop*dt)
          auto p_I = compute_infection_probability(beta, I, N_pop, dt);
          // float p_I = 0.1f;
          Static_RNG::binomial_distribution d_I(v_acc.data[id].state.S,
                                                        p_I);

          auto N_infected = d_I(rng_acc[id]);
          e_inf_acc[e_acc.to[id]] += N_infected;
        }); });
          FrameMarkEnd("Edge Infections");
        }

        return std::make_pair(v_inf_buf, e_inf_buf);
      }
      sycl::buffer<float, 1> recovery_scatter(float dt)
      {
        ZoneScoped;

        sycl::buffer<float, 1> v_rec_buf(sycl::range<1>(G.N_vertices()));

        q.submit([&](sycl::handler &h)
                 {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_rec_acc =
          v_rec_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()), [=](sycl::id<1> id) {
        auto alpha = v_acc.data[id].param.alpha;
        auto p_R = 1 - sycl::exp(-alpha * dt);
        auto I = v_acc.data[id].state.I;

        // Delta_R = bin(I, p_R)
        Static_RNG::binomial_distribution<float> d_R(
            v_acc.data[id].state.I, p_R);
        auto N_recovered = d_R(rng_acc[id]);
        v_rec_acc[id] = N_recovered;
      }); });

        return v_rec_buf;
      }
      void gather(sycl::buffer<float, 1> &v_inf_buf,
                  sycl::buffer<float, 1> &e_inf_buf,
                  sycl::buffer<float, 1> &v_rec_buf)
      {
        ZoneScoped;

        q.submit([&](sycl::handler &h)
                 {
  
      auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
      auto v_inf_acc =
          v_inf_buf.template get_access<sycl::access::mode::read>(h);
      auto v_rec_acc =
          v_rec_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
        // Vertex infections and recoveries are updated first
        auto Delta_x = SIR_Metapopulation_State{
            -v_inf_acc[id], v_inf_acc[id] - v_rec_acc[id], v_rec_acc[id]};

        v_acc.data[id].state += Delta_x;
      }); });
        q.wait();
        if (G.N_edges() > 0)
        {
          q.submit([&](sycl::handler &h)
                   {
        auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
        auto e_acc = G.get_edge_access<sycl::access::mode::read>(h);
        auto v_inf_acc =
            v_inf_buf.template get_access<sycl::access::mode::read>(h);
        auto e_inf_acc =
            e_inf_buf.template get_access<sycl::access::mode::read>(h);
        auto rng_acc =
            rng_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(G.N_edges()), [=](sycl::id<1> id) {
          // Probability of new infection is determined by the population
          // fraction of infected

          auto S = v_acc.data[id].state.S;
          auto I = v_acc.data[id].state.I;
          if (S == 0) return;
          float I_edge = e_inf_acc[id];
          float susceptible_frac = S / (S + I);
          binomial_distribution<> d_edge((uint32_t)I_edge, susceptible_frac);
          uint32_t N_edge_infected = d_edge(rng_acc[id]);

          v_acc.data[id].state.S -= (float)N_edge_infected;
          v_acc.data[id].state.I += (float)N_edge_infected;
        }); });
        }
        q.wait();
      }

      void reset() { initialize(); }

      void generate_seeds(int seed)
      {
        ZoneScoped;
        // generate seeds
        std::vector<int> seeds(G.N_edges());
        // random device
        // mt19937 generator
        std::mt19937_64 gen(seed);
        std::generate(seeds.begin(), seeds.end(), gen);
        sycl::buffer<int, 1> seed_buf(seeds.data(), sycl::range<1>(G.N_edges()));
        q.submit([&](sycl::handler &h)
                 {
      auto seed_acc = seed_buf.template get_access<sycl::access::mode::read>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(G.N_edges()),
                     [=](sycl::id<1> id) { rng_acc[id].seed(seed_acc[id]); }); });
      }

#ifdef SYCL_GRAPH_DEBUG
      void construction_debug_report()
      {
        // fmt open file
        const std::string filename =
            SYCL_GRAPH_LOG_DIR +
            std::string("SIR_Metapopulation/Network" +
                        std::to_string(debug_instance_ID) + ".txt");
        // create directory if it doesn't exist
        std::filesystem::create_directories(SYCL_GRAPH_LOG_DIR +
                                            std::string("SIR_Metapopulation/"));
        std::ofstream file(filename);
        if (!file.is_open())
        {
          std::cout << "Error opening file: " << filename << std::endl;
          return;
        }

        file << "SIR_Metapopulation Network Report for instance "
             << debug_instance_ID << std::endl;
        file << "---------------------------------" << std::endl;
        file << "N_vertices: " << G.N_vertices() << std::endl;
        file << "N_edges: " << G.N_edges() << std::endl;
        file << "---------------------------------" << std::endl;
        file << "Buffer sizes" << std::endl;
        file << "---------------------------------" << std::endl;
        file << "Vertex buffer size: " << G.vertex_buf.byte_size() << std::endl;
        file << "Edge buffer size: " << G.edge_buf.byte_size() << std::endl;
        file << "RNG buffer size: " << rng_buf.byte_size() << std::endl;
        file << "---------------------------------" << std::endl;
        size_t total_passive_size =
            G.vertex_buf.byte_size() + G.edge_buf.byte_size() + rng_buf.byte_size();
        size_t max_active_size =
            total_passive_size + 3 * G.N_vertices() * sizeof(float);
        file << "Total Buffer space requirement: " << max_active_size << std::endl;
        file.close();
      }
#endif
    };
  } // namespace Sycl::Network_Models
} // namespace Sycl_Graph
#endif