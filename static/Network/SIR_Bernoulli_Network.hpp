#ifndef SYCL_GRAPH_SIR_BERNOULLI_NETWORK_HPP
#define SYCL_GRAPH_SIR_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include "Network.hpp"
#include <Graph_Math.hpp>
#include <Sycl/Sycl_Graph_Execution.hpp>
#include <Sycl/Sycl_Graph_Random.hpp>
#include <Sycl_Graph.hpp>
#include <ranges>
#include <stddef.h>
#include <thread>
#include <utility>
#include <vector>

namespace Network_Models {
enum SIR_State { SIR_S = 0, SIR_I = 1, SIR_R = 2 };
struct SIR_Edge {};

template <typename dType = float> struct SIR_Param {
  dType p_I = 0;
  dType p_R = 0;
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
};

namespace Fixed {
template <typename V, typename E, uint32_t NV, uint32_t NE>
using SIR_Graph = Sycl::Graph::Fixed::Graph<SIR_State, SIR_Edge, uint32_t, NV,
                                            NE, size_t, std::array>;
}
namespace Dynamic {
template <typename T> using SIR_vector_t = std::vector<T, std::allocator<T>>;
using SIR_Graph =
    Sycl::Graph::Dynamic::Graph<SIR_State, SIR_Edge, uint32_t, SIR_vector_t>;

template <typename RNG, typename dType>
struct SIR_Bernoulli_Network
    : public Network<SIR_Param<>, SIR_Bernoulli_Network<RNG, dType>> {
  using Vertex_t = typename SIR_Graph::Vertex_t;
  using Edge_t = typename SIR_Graph::Edge_t;
  using Edge_Prop_t = typename SIR_Graph::Edge_Prop_t;
  using Vertex_Prop_t = typename SIR_Graph::Vertex_Prop_t;
  const dType p_I0;
  const dType p_R0;
  const uint32_t t = 0;

  SIR_Bernoulli_Network(SIR_Graph &G, dType p_I0, dType p_R0,
                               RNG rng)
      : G(G), rng(rng), p_I0(p_I0), p_R0(p_R0) {}

  SIR_Bernoulli_Network
  operator=(const SIR_Bernoulli_Network &other) {
    return SIR_Bernoulli_Network(other.G, other.p_I0, other.p_R0,
                                        other.rng);
  }

  void initialize() {

    Sycl::Graph::random::uniform_real_distribution<dType> d_I;
    Sycl::Graph::random::uniform_real_distribution<dType> d_R;
    std::for_each(Sycl::Graph::execution::par_unseq, G.begin(), G.end(), [&](auto& v) {
      if (d_I(rng) < p_I0) {
        v.data = SIR_I;
      } else if (d_R(rng) < p_R0) {
        v.data = SIR_R;
      } else {
        v.data = SIR_S;
      }
    });
  }

  std::vector<uint32_t> population_count() {
    std::vector<uint32_t> count = {0, 0, 0};
    std::for_each(G.begin(), G.end(),
                  [&count](const Vertex_t &v) { count[v.data]++; });
    return count;
  }

  // function for infection step
  void infection_step(dType p_I) {

    Sycl::Graph::random::uniform_real_distribution<dType> d_I;

    // print distance between G.begin() and G.end/()
    //  std::cout << std::distance(G.begin(), G.end()) << std::endl;
    std::for_each(Sycl::Graph::execution::par_unseq, G.begin(), G.end(), [&](auto v0) {
      // print id of thread
      if (v0.data == SIR_I) {
        for (const auto v : G.neighbors(v0.id)) {
          if (!v)
            break;
          bool trigger = d_I(rng) < p_I;
          if (v->data == SIR_S && trigger) {
            G.assign(v->id,
                     ((v->data == SIR_S) && (trigger)) ? SIR_I : v->data);
          }
        };
      }
    });
  }

  void recovery_step(dType p_R) {

    Sycl::Graph::random::uniform_real_distribution<dType> d_R;
    std::for_each(std::execution::par_unseq, G.begin(), G.end(),
                  [&](const auto &v) {
                    bool recover_trigger = (v.data == SIR_I) && d_R(rng) < p_R;
                    G.assign(v.id, (recover_trigger) ? SIR_R : v.data);
                  });
  }

  bool terminate(const SIR_Param<> &p, const std::vector<uint32_t> &x) {
    bool early_termination = ((t > p.Nt_min) || (x[1] < p.N_I_min));
    return early_termination;
  }

  void advance(const SIR_Param<> &p) {
    infection_step(p.p_I);
    recovery_step(p.p_R);
  }

  void reset() {
    std::for_each(G.begin(), G.end(), [&](auto v) { G.assign(v.id, SIR_S); });
  }

private:
  SIR_Graph &G;
  RNG rng;
};
} // namespace Dynamic

} // namespace Network_Models
#endif