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
#include <execution>
using Sycl_Graph::Dynamic::Network_Models::generate_SBM;
using namespace Sycl_Graph::Sycl::Network_Models;
typedef typename SIR_Bernoulli_SBM_Network::Base_t SBM_Base_t;
typedef SIR_Bernoulli_SBM_Network Network_t;
typedef typename Network_t::Graph_t Graph_t;


auto create_SIR_Bernoulli_SBM(const std::vector<uint32_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true)
{
    static sycl::queue q;
    // create profiling queue
    //reshape vector to vector<vector> of size N_pop
    std::vector<std::vector<float>> p_SBM_reshaped(N_pop.size());
    for (uint32_t i = 0; i < N_pop.size(); i++)
    {
        p_SBM_reshaped[i] = std::vector<float>(N_pop.size());
        for (uint32_t j = 0; j < N_pop.size(); j++)
        {
            p_SBM_reshaped[i][j] = p_SBM[i * N_pop.size() + j];
        }
    }

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_ids_SBM = generate_SBM(N_pop, p_SBM_reshaped, undirected);
    


    std::vector<uint32_t> vertex_ids = Sycl_Graph::range(0, std::accumulate(N_pop.begin(), N_pop.end(), 0));

    
    //flatten edge_ids_SBM to vector of pairs
    std::vector<std::pair<uint32_t, uint32_t>> edge_ids;
    for (uint32_t i = 0; i < edge_ids_SBM.size(); i++)
    {
        for (uint32_t j = 0; j < edge_ids_SBM[i].size(); j++)
        {
            edge_ids.push_back(edge_ids_SBM[i][j]);
        }
    }


    std::cout << "N_vertices: " << vertex_ids.size() << std::endl;
    std::cout << "N_edges: " << edge_ids.size() << std::endl;

    Graph_t G(q, 0, 0);
    G.add_vertex(vertex_ids);
    G.add_edge(edge_ids);

    //create network
    Network_t sir(G, p_I0, p_R0, edge_ids_SBM);

    
    
    
    
    return std::make_pair(G, sir);
}


std::vector<Network_t> create_SIR_Bernoulli_SBMs(
    const std::vector<std::vector<uint32_t>>& N_pops,
    const std::vector<std::vector<float>>& p_SBMs,
    const float p_I0,
    const float p_R0 = 0.0, bool undirected = true) {
  std::vector<std::vector<Network_t>> networks(N_pops.size());


  std::vector<std::tuple<std::vector<uint32_t>, std::vector<float>>> N_pops_p_SBMs;
  //make tuple zip of N_pops and p_SBMs
  std::transform(N_pops.begin(), N_pops.end(), p_SBMs.begin(), std::back_inserter(N_pops_p_SBMs), [](auto N_pop, auto p_SBM){
    return std::make_tuple(N_pop, p_SBM);
  });

  //OpenMP parallel for loop
  #pragma omp parallel for
  for (uint32_t i = 0; i < N_pops_p_SBMs.size(); i++)
  {
    auto [N_pop, p_SBM] = N_pops_p_SBMs[i];
    auto [G, sir] = create_SIR_Bernoulli_SBM(N_pop, p_SBM, p_I0, p_R0, undirected);
    networks[i].push_back(sir);
  }

  //combine vectors
  std::vector<Network_t> return_networks;
  for (uint32_t i = 0; i < networks.size(); i++)
  {
    return_networks.insert(return_networks.end(), networks[i].begin(), networks[i].end());
  }
  return return_networks;
}

auto simulate_parallel(std::vector<Network_t>& networks, const std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>& tp_list)
{
  typedef typename Network_t::Trajectory_pair_t Traj_t;
  std::for_each(networks.begin(), networks.end(), [](auto& network){
    network.initialize();
  });
  std::vector<Traj_t> trajectories(networks.size());
  std::transform(std::execution::par_unseq, networks.begin(), networks.end(), tp_list.begin(), trajectories.begin(), [](auto& network, auto& tp_list){
    return network.simulate_groups(tp_list);
  });
  return trajectories;
}

auto simulate_N_parallel(std::vector<Network_t>& networks, const std::vector<std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>>& tp_list)
{
  typedef typename Network_t::Trajectory_pair_t Traj_t;
  uint32_t N = tp_list.size();
  std::vector<std::vector<Traj_t>> trajectories(N);
  //reserve
  for (uint32_t i = 0; i < N; i++)
  {
    trajectories[i].reserve(networks.size());
  }
  for (uint32_t i = 0; i < N; i++)
  {
    std::cout << i << " of " << N << std::endl;
    auto traj_i = simulate_parallel(networks, tp_list[i]);
    for(uint32_t j = 0; j < traj_i.size(); j++)
    {
      trajectories[j].push_back(traj_i[j]);
    }
  }
  return trajectories;
}

auto simulate_N_parallel_copied(std::vector<Network_t>& networks, const std::vector<std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>>& tp_list)
{
  typedef typename Network_t::Trajectory_pair_t Traj_t;

  std::vector<std::vector<Network_t>> networks_list(tp_list.size());
  for (uint32_t i = 0; i < tp_list.size(); i++)
  {
    networks_list[i] = networks;
  }
  
  //print total byte size
  uint32_t total_size = 0;
  for (uint32_t i = 0; i < networks_list.size(); i++)
  {
    for (uint32_t j = 0; j < networks_list[i].size(); j++)
    {
      total_size += networks_list[i][j].byte_size();
    }
  }
  std::cout << "Total byte size: " << total_size << std::endl;

  uint32_t N = tp_list.size();
  auto N_networks = networks.size();
  std::vector<std::vector<Traj_t>> trajectories(N_networks);
    //reserve
    for (uint32_t i = 0; i < N_networks; i++)
    {
      trajectories[i].reserve(N);
    }

    #pragma omp parallel for
    for (uint32_t i = 0; i < N; i++)
    {
      std::cout << i << " of " << N << std::endl;
      auto traj_i = simulate_parallel(networks_list[i], tp_list[i]);
      for(uint32_t j = 0; j < traj_i.size(); j++)
      {
        trajectories[j].push_back(traj_i[j]);
      }
    }
  return trajectories;
}

void simulate_N_parallel_to_file(std::vector<Network_t>& networks, const std::vector<std::vector<std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>>>>& tp_list, const std::string& file_path)
{
  typedef typename Network_t::Trajectory_pair_t Traj_t;

  std::vector<std::vector<Network_t>> networks_list(tp_list.size());
  for (uint32_t i = 0; i < tp_list.size(); i++)
  {
    networks_list[i] = networks;
  }
  
  //print total byte size
  uint32_t total_size = 0;
  for (uint32_t i = 0; i < networks_list.size(); i++)
  {
    for (uint32_t j = 0; j < networks_list[i].size(); j++)
    {
      total_size += networks_list[i][j].byte_size();
    }
  }
  std::cout << "Total byte size: " << total_size << std::endl;

  uint32_t N = tp_list.size();
  auto N_networks = networks.size();
  std::vector<std::vector<Traj_t>> trajectories(N_networks);
    //reserve
    for (uint32_t i = 0; i < N_networks; i++)
    {
      trajectories[i].reserve(N);
    }

    #pragma omp parallel for
    for (uint32_t i = 0; i < N; i++)
    {
      std::cout << i << " of " << N << std::endl;
      auto traj_i = simulate_parallel(networks_list[i], tp_list[i]);
      for(uint32_t j = 0; j < traj_i.size(); j++)
      {
        std::ofstream f_traj(file_path + std::string("SBM_") + std::to_string(j) + "_traj_" + std::to_string(i) + ".csv");
        std::ofstream f_community(file_path + std::string("SBM_") + std::to_string(j) + "_Community_Infected_" + std::to_string(i) + ".csv");
        f_traj << "S,I,R,p_I_00,p_I_01,p_I_10,p_I_11";
        uint32_t t = 0;
        for (auto&& traj: traj_i[j].first)
        {
            for(auto&& state: traj)
              f_traj << state << ",";
            for(auto&& p: tp_list[i][j][t].p_Is)
              f_traj << p << ",";
            f_traj << "\n";
            t++;
        }

        for(auto&& t: traj_i[j].second)
        {
            for(auto&& state: t)
              f_community << state << ",";
            f_community << "\n";
        }
      }
    }
}


PYBIND11_MODULE(SBM_Binder, m) {
  typedef typename Network_t::Trajectory_t Traj_t;
  typedef typename Network_t::Trajectory_pair_t TrajPair_t;
  // class for temporal_param
  pybind11::class_<SIR_Bernoulli_SBM_Temporal_Param<float>>(m, "Temporal_Param")
      .def(pybind11::init<>())
      .def(pybind11::init<const std::vector<float>&, const float>())
      .def_readwrite("p_Is", &SIR_Bernoulli_SBM_Temporal_Param<float>::p_Is)
      .def_readwrite("p_R", &SIR_Bernoulli_SBM_Temporal_Param<float>::p_R)
      .def_readwrite("Nt_min", &SIR_Bernoulli_SBM_Temporal_Param<float>::Nt_min)
      .def_readwrite("NI_min",
                     &SIR_Bernoulli_SBM_Temporal_Param<float>::N_I_min);

  // .def("add_vertex", &Graph_t::add_vertex)
  // .def("add_edge", &Graph_t::add_edge)
  // .def("remove_vertex", &Graph_t::remove_vertex)
  // .def("remove_edge", &Graph_t::remove_edge)
  // .def("assign_vertex", &Graph_t::assign_vertex)
  // .def("assign_edge", &Graph_t::assign_edge)
  // .def("get_vertex", &Graph_t::get_vertex)
  // .def("get_edge", &Graph_t::get_edge)
  // .def("get_vertex_data", &Graph_t::get_vertex_data)
  // .def("get_edge_data", &Graph_t::get_edge_data)
  // .def("get_vertex_ids", &Graph_t::get_vertex_ids)
  // .def("resize", &Graph_t::resize);

  pybind11::class_<SBM_Base_t>(m, "SIR_SBM_Network_Base")
      .def("simulate",
           static_cast<Traj_t (SBM_Base_t::*)(
               std::vector<SIR_Bernoulli_SBM_Temporal_Param<>>)>(
               &SBM_Base_t::simulate));

  // pybind11::class_<Graph_t>(m, "SBM_Graph")
  //     .def("byte_size", &Graph_t::byte_size);

  // define class
  pybind11::class_<SIR_Bernoulli_SBM_Network, SBM_Base_t>(m, "SBM")
      .def("initialize", &SIR_Bernoulli_SBM_Network::initialize)
      // .def("simulate_groups", &SIR_Bernoulli_SBM_Network::simulate_groups);
      .def("simulate_groups", &Network_t::simulate_groups)
      .def_readwrite("SBM_ids", &SIR_Bernoulli_SBM_Network::SBM_ids)
      .def("byte_size", &SIR_Bernoulli_SBM_Network::byte_size);


  m.def("create_SIR_Bernoulli_SBM", &create_SIR_Bernoulli_SBM);
  m.def("create_SIR_Bernoulli_SBMs", &create_SIR_Bernoulli_SBMs);
  m.def("simulate_parallel", &simulate_parallel);
  m.def("simulate_N_parallel", &simulate_N_parallel);
  m.def("simulate_N_parallel_copied", &simulate_N_parallel_copied);
  m.def("simulate_N_parallel_to_file", &simulate_N_parallel_to_file);
}
