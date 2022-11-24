#ifndef SYCL_GRAPH_NETWORK_HPP
#define SYCL_GRAPH_NETWORK_HPP

#include "Graph_Generation.hpp"
#include <Graph_Math.hpp>
#include <Sycl/Sycl_Graph_Execution.hpp>
#include <random>
#include <stddef.h>
#include <vector>

namespace Network_Models {

namespace Fixed {
template <typename Param, uint32_t Nx, uint32_t Nt, class Derived>
struct Network {

  using Trajectory = std::array<uint32_t, Nx>;
  Trajectory population_count() {
    return static_cast<Derived *>(this)->population_count();
  }
  void advance(const Param &p) { static_cast<Derived *>(this)->advance(p); }
  void reset() { static_cast<Derived *>(this)->reset(); }
  bool terminate(const Param &p, const std::array<uint32_t, Nx> &x) {
    return static_cast<Derived *>(this)->terminate(p, x);
  }

  std::array<std::array<uint32_t, Nt + 1>, Nx>
  simulate(const std::array<Param, Nt> &p_vec, uint32_t Nt_min = 15) {

    std::array<Trajectory, Nt + 1> trajectory;
    uint32_t t = 0;
    trajectory[0] = population_count();
    for (int i = 0; i < Nt; i++) {
      advance(p_vec[i]);
      trajectory[i + 1] = population_count();
      if (terminate(p_vec[i], trajectory[i + 1])) {
        break;
      }
    }
    return Sycl::Graph::transpose(trajectory);
  }
};
} // namespace Fixed

namespace Dynamic {
template <typename Param, class Derived> struct Network {

  using Trajectory = std::vector<uint32_t>;
  Trajectory population_count() {
    return static_cast<Derived *>(this)->population_count();
  }
  void advance(const Param &p) { static_cast<Derived *>(this)->advance(p); }
  void reset() { static_cast<Derived *>(this)->reset(); }
  bool terminate(const Param &p, const std::vector<uint32_t> &x) {
    return static_cast<Derived *>(this)->terminate(p, x);
  }

  void initialize() { return static_cast<Derived *>(this)->initialize(); }

  std::vector<std::vector<uint32_t>>
  simulate(const std::vector<Param> &p_vec, uint32_t Nt, uint32_t Nt_min = 15) {

    std::vector<Trajectory> trajectory(Nt + 1);
    uint32_t t = 0;
    trajectory[0] = population_count();
    for (int i = 0; i < Nt; i++) {
      advance(p_vec[i]);
      trajectory[i + 1] = population_count();
      if (terminate(p_vec[i], trajectory[i + 1])) {
        break;
      }
    }
    return Sycl::Graph::transpose(trajectory);
  }
};
} // namespace Dynamic
} // namespace Network_Models
#endif