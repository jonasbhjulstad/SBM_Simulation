#ifndef SIR_BERNOULLI_NETWORK_DYNAMIC_IMPL_HPP
#define SIR_BERNOULLI_NETWORK_DYNAMIC_IMPL_HPP
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/random.hpp>

namespace Sycl_Graph::Dynamic {
namespace Network_Models {
using namespace Sycl_Graph::Network_Models;

template <typename T> using SIR_vector_t = std::vector<T, std::allocator<T>>;
using SIR_Graph =
    Sycl_Graph::Dynamic::Graph<SIR_Individual_State, SIR_Edge, uint32_t, SIR_vector_t>;

template <typename RNG, typename dType>
struct SIR_Bernoulli_Network
    : public Network<SIR_Bernoulli_Param<>, SIR_Bernoulli_Network<RNG, dType>> {
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

    Sycl_Graph::random::uniform_real_distribution<dType> d_I;
    Sycl_Graph::random::uniform_real_distribution<dType> d_R;
    std::for_each(Sycl_Graph::execution::par_unseq,G.begin(), G.end(), [&](auto& v) {
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

    Sycl_Graph::random::uniform_real_distribution<dType> d_I;

    // print distance between G.begin() and G.end/()
    //  std::cout << std::distance(G.begin(), G.end()) << std::endl;
    std::for_each(Sycl_Graph::execution::seq,G.begin(), G.end(), [&](auto v0) {
      if (v0.data == SIR_I) {
        for (const auto v : G.neighbors(v0.id)) {
          if (v.id == SIR_Graph::Vertex_t::invalid_id)
            continue;
          bool trigger = d_I(rng) < p_I;
          if (v.data == SIR_S && trigger) {
            auto&& I = SIR_I;
            G.assign(v.id,
                     ((v.data == SIR_S) && (trigger)) ? I : v.data);
          }
        };
      }
    });
  }

  void recovery_step(dType p_R) {

    Sycl_Graph::random::uniform_real_distribution<dType> d_R;
    std::for_each(Sycl_Graph::execution::par_unseq,G.begin(), G.end(),
                  [&](const auto &v) {
                    bool recover_trigger = (v.data == SIR_I) && d_R(rng) < p_R;
                    auto && R = SIR_R;
                    G.assign(v.id, (recover_trigger) ? R : v.data);
                  });
  }

  bool terminate(const SIR_Bernoulli_Param<> &p, const std::vector<uint32_t> &x) {
    bool early_termination = ((t > p.Nt_min) && (x[1] < p.N_I_min));
    return early_termination;
  }

  void advance(const SIR_Bernoulli_Param<> &p) {
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
} // namespace Sycl_Graph::Dynamic
} // namespace Network_Models
#endif