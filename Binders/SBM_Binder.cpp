#include "Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM_Types.hpp"
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utility>
using Sycl_Graph::Dynamic::Network_Models::generate_SBM;
using namespace Sycl_Graph::Sycl::Network_Models;
typedef typename SIR_Bernoulli_SBM_Network::Base_t SBM_Base_t;
typedef SIR_Bernoulli_SBM_Network Network_t;
typedef typename Network_t::Graph_t Graph_t;
// create pybind11 module
// auto create_SIR_Bernoulli_SBM(sycl::queue& q, const std::vector<size_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true)
// {
//     // create profiling queue
//     //reshape vector to vector<vector> of size N_pop
//     std::vector<std::vector<float>> p_SBM_reshaped(N_pop.size());
//     for (size_t i = 0; i < N_pop.size(); i++)
//     {
//         p_SBM_reshaped[i] = std::vector<float>(N_pop.size());
//         for (size_t j = 0; j < N_pop.size(); j++)
//         {
//             p_SBM_reshaped[i][j] = p_SBM[i * N_pop.size() + j];
//         }
//     }

//     auto [G, edge_ids_SBM] = generate_SBM<Graph_t, Static_RNG::default_rng>(q, N_pop, p_SBM_reshaped, undirected);
//     return std::make_tuple(G, Network_t(G, p_I0, p_R0, edge_ids_SBM), edge_ids_SBM);
// }

std::pair<Network_t, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>>
create_SIR_Bernoulli_SBM(const std::vector<size_t>& N_pop,
                         const std::vector<float>& p_SBM, const float p_I0,
                         const float p_R0 = 0.0, bool undirected = true) {
  // create profiling queue
  // reshape vector to vector<vector> of size N_pop
  std::vector<std::vector<float>> p_SBM_reshaped(N_pop.size());
  for (size_t i = 0; i < N_pop.size(); i++) {
    p_SBM_reshaped[i] = std::vector<float>(N_pop.size());
    for (size_t j = 0; j < N_pop.size(); j++) {
      p_SBM_reshaped[i][j] = p_SBM[i * N_pop.size() + j];
    }
  }

  static sycl::queue q(sycl::gpu_selector_v);
  auto [G, edge_ids_SBM] = generate_SBM<Graph_t, Static_RNG::default_rng>(
      q, N_pop, p_SBM_reshaped, undirected);
  return std::make_pair(Network_t(G, p_I0, p_R0, edge_ids_SBM), edge_ids_SBM);
}

PYBIND11_MODULE(SBM_Binder, m) {
  typedef typename Network_t::Trajectory_t Traj_t;
  typedef typename Network_t::Trajectory_pair_t TrajPair_t;
  // class for temporal_param
  pybind11::class_<SIR_Bernoulli_SBM_Temporal_Param<float>>(m, "Temporal_Param")
      .def(pybind11::init<>())
      .def_readwrite("p_Is", &SIR_Bernoulli_SBM_Temporal_Param<float>::p_Is)
      .def_readwrite("p_R", &SIR_Bernoulli_SBM_Temporal_Param<float>::p_R)
      .def_readwrite("Nt_min", &SIR_Bernoulli_SBM_Temporal_Param<float>::Nt_min)
      .def_readwrite("NI_min",
                     &SIR_Bernoulli_SBM_Temporal_Param<float>::N_I_min);

  pybind11::class_<SBM_Base_t>(m, "Network")
      .def("simulate",
           static_cast<Traj_t (SBM_Base_t::*)(
               std::vector<SIR_Bernoulli_SBM_Temporal_Param<>>)>(
               &SBM_Base_t::simulate));

  // define class
  pybind11::class_<SIR_Bernoulli_SBM_Network, SBM_Base_t>(m, "SBM")
      .def("initialize", &SIR_Bernoulli_SBM_Network::initialize)
      // .def("simulate_groups", &SIR_Bernoulli_SBM_Network::simulate_groups);
      .def("simulate_groups", &Network_t::simulate_groups);

  m.def("create_SIR_Bernoulli_SBM", &create_SIR_Bernoulli_SBM);
}

// int main() {

//   double p_I0 = 0.1;
//   double p_R0 = 0.1;
//   static constexpr size_t N_clusters = 2;
//   const std::vector<uint32_t> N_pop = {100, 100};

//   std::vector<std::vector<float>> p_SBM = {{0.8, 0.1}, {0.1, 0.8}};
//   using namespace Sycl_Graph::Sycl::Network_Models;
//   typedef SIR_Bernoulli_SBM_Network Network_t;
//   typedef typename Network_t::Graph_t Graph_t;
//   // create profiling queue
//   sycl::queue q(sycl::gpu_selector_v,
//                 sycl::property::queue::enable_profiling{});
//   auto [G, edge_ids_SBM] = generate_SBM <Graph_t, Static_RNG::default_rng>(q,
//   N_pop, p_SBM, true); SIR_Bernoulli_SBM_Network sir(G, p_I0, p_R0,
//   edge_ids_SBM);
//   // generate sir_param
//   size_t Nt = 100;
//   std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>> sir_param;
//   for (int i = 0; i < Nt; i++)
//   {
//     SIR_Bernoulli_SBM_Temporal_Param<> param;
//     param.p_Is = {0.1, 0.1, 0.05};
//     sir_param.push_back(param);
//   }

//   sir.initialize();

//   auto traj = sir.simulate(Nt, sir_param);
//   // print traj
//   for (auto &x : traj) {
//     std::cout << x[0] << ", " << x[1] << ", " << x[2] << std::endl;
//   }

//   // write to file
//   std::filesystem::create_directory(
//       std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
//   std::ofstream file(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) +
//                      "/SIR_sim/SBM_traj.csv");

//   for (auto &x : traj) {
//     file << x[0] << ", " << x[1] << ", " << x[2] << "\n";
//   }
//   file.close();
// }