#ifndef SIR_BERNOULLI_NETWORK_SYCL_IMPL_HPP
#define SIR_BERNOULLI_NETWORK_SYCL_IMPL_HPP
#ifdef SYCL_GRAPH_USE_SYCL
#include "SIR_Bernoulli_Types.hpp"
#include <stddef.h>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Network/Network.hpp>

namespace Sycl_Graph
{
    namespace Sycl::Network_Models
    {
        using namespace Sycl_Graph::Network_Models;
        template <typename T>
        using SIR_vector_t = std::vector<T, std::allocator<T>>;
        template <uint32_t NV, uint32_t NE>
        using SIR_Graph =
            Sycl_Graph::Sycl::Graph<SIR_Individual_State, SIR_Edge, uint32_t, NV, NE>;

        template <typename RNG, uint32_t Nt, uint32_t NV, uint32_t NE, typename dType>
        struct SIR_Bernoulli_Network
            : public Network<SIR_Bernoulli_Param<>,3,  Nt, SIR_Bernoulli_Network<RNG, Nt, NV, NE, dType>>
        {
            using Graph_t = SIR_Graph<NV, NE>;
            using Vertex_t = typename Graph_t::Vertex_t;
            using Edge_t = typename Graph_t::Edge_t;
            using Edge_Prop_t = typename Graph_t::Edge_Prop_t;
            using Vertex_Prop_t = typename Graph_t::Vertex_Prop_t;
            const dType p_I0;
            const dType p_R0;
            const uint32_t t = 0;

            SIR_Bernoulli_Network(Graph_t &G, dType p_I0, dType p_R0,
                                  RNG rng)
                : G(G), rng(rng), p_I0(p_I0), p_R0(p_R0) {}

            SIR_Bernoulli_Network
            operator=(const SIR_Bernoulli_Network &other)
            {
                return SIR_Bernoulli_Network(other.G, other.p_I0, other.p_R0,
                                             other.rng);
            }

            void initialize()
            {

                Sycl_Graph::random::uniform_real_distribution<dType> d_I;
                Sycl_Graph::random::uniform_real_distribution<dType> d_R;
                std::for_each(Sycl_Graph::execution::par_unseq, G.begin(), G.end(), [&](auto &v)
                              {
      if (d_I(rng) < p_I0) {
        v.data = SIR_I;
      } else if (d_R(rng) < p_R0) {
        v.data = SIR_R;
      } else {
        v.data = SIR_S;
      } });
            }

            std::vector<uint32_t> population_count()
            {
                std::vector<uint32_t> count = {0, 0, 0};
                std::for_each(G.begin(), G.end(),
                              [&count](const Vertex_t &v)
                              { count[v.data]++; });
                return count;
            }

            // function for infection step
            void infection_step(dType p_I)
            {

                q.submit([&](sycl::handler & h)
                {
                    auto acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
                });
                    Sycl_Graph::random::uniform_real_distribution<dType>
                        d_I;

                // print distance between G.begin() and G.end/()
                //  std::cout << std::distance(G.begin(), G.end()) << std::endl;
                std::for_each(Sycl_Graph::execution::par_unseq, G.begin(), G.end(), [&](auto v0)
                              {
      // print id of thread
      if (v0.data == SIR_I) {
        for (const auto v : G.neighbors(v0.id)) {
          if (!v)
            continue;
          bool trigger = d_I(rng) < p_I;
          if (v->data == SIR_S && trigger) {
            G.assign(v->id,
                     ((v->data == SIR_S) && (trigger)) ? SIR_I : v->data);
          }
        };
      } });
            }

            void recovery_step(dType p_R)
            {

                Sycl_Graph::random::uniform_real_distribution<dType> d_R;
                std::for_each(Sycl_Graph::execution::par_unseq, G.begin(), G.end(),
                              [&](const auto &v)
                              {
                                  bool recover_trigger = (v.data == SIR_I) && d_R(rng) < p_R;
                                  G.assign(v.id, (recover_trigger) ? SIR_R : v.data);
                              });
            }

            bool terminate(const SIR_Bernoulli_Param<> &p, const std::vector<uint32_t> &x)
            {
                bool early_termination = ((t > p.Nt_min) && (x[1] < p.N_I_min));
                return early_termination;
            }

            void advance(const SIR_Bernoulli_Param<> &p)
            {
                infection_step(p.p_I);
                recovery_step(p.p_R);
            }

            void reset()
            {
                std::for_each(G.begin(), G.end(), [&](auto v)
                              { G.assign(v.id, SIR_S); });
            }

        private:
            Graph_t &G;
            sycl::queue &q;
            RNG rng;
        };
    } // namespace Sycl
} // namespace Network_Models
#endif
#endif