#include <Sycl_Graph/Regression.hpp>
#include "Sycl_Graph/SBM_types.hpp"
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SBM_write.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
// #include <Sycl_Graph/SBM_write.hpp>
#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utility>
using namespace Sycl_Graph;
using namespace Sycl_Graph::SBM;

namespace py = pybind11;

typedef std::vector<std::vector<uint32_t>> Nodelist_t;
#ifdef ENABLE_GRAPH_TOOL

#endif
PYBIND11_MODULE(SIR_SBM, m) {

  py::class_<sycl::gpu_selector>(m, "gpu_selector").def(py::init<>());
  py::class_<sycl::cpu_selector>(m, "cpu_selector").def(py::init<>());

  py::class_<sycl::queue>(m, "sycl_queue")
      .def(py::init<>())
      .def(py::init<const sycl::gpu_selector &>())
      .def(py::init<const sycl::cpu_selector &>());
  py::class_<sycl::event>(m, "sycl_event").def(py::init<>());

  py::class_<Edge_t>(m, "Edge_t")
      .def(py::init<>())
      .def(py::init<uint32_t, uint32_t>())
      .def_readwrite("_from", &Edge_t::from)
      .def_readwrite("_to", &Edge_t::to);
  py::class_<SIR_SBM_Param_t>(m, "SIR_SBM_Param_t")
      .def(py::init<>())
      .def_readwrite("p_I", &SIR_SBM_Param_t::p_I)
      .def_readwrite("p_R", &SIR_SBM_Param_t::p_R)
      .def_readwrite("p_I0", &SIR_SBM_Param_t::p_I0)
      .def_readwrite("p_R0", &SIR_SBM_Param_t::p_R0);
  py::class_<SBM_Graph_t>(m, "SBM_Graph_t")
      .def(py::init<>())
      .def(py::init<const std::vector<Node_List_t> &,
                    const std::vector<Edge_List_t> &>())
      .def_readwrite("node_list", &SBM_Graph_t::node_list)
      .def_readwrite("edge_list", &SBM_Graph_t::edge_list)
      .def_readwrite("ecm", &SBM_Graph_t::ecm)
      .def_readwrite("vcm", &SBM_Graph_t::vcm)
      .def_readwrite("N_vertices", &SBM_Graph_t::N_vertices)
      .def_readwrite("N_edges", &SBM_Graph_t::N_edges)
      .def_readwrite("N_connections", &SBM_Graph_t::N_connections)
      .def_readwrite("N_communities", &SBM_Graph_t::N_communities)
      .def_readwrite("connection_community_map", &SBM_Graph_t::connection_community_map)
      .def("ccmap_write", &SBM_Graph_t::ccmap_write)
      .def("remap", &SBM_Graph_t::remap);
  m.def("n_choose_k", &n_choose_k);
  m.def("create_SBM", &create_SBM);
  m.def("create_planted_SBM", &create_planted_SBM);
  m.def("create_planted_SBMs", &create_planted_SBMs);
  m.def("generate_p_Is",
        static_cast<std::vector<std::vector<float>> (*)(
            uint32_t, float, float, uint32_t, uint32_t)>(&generate_p_Is));
  m.def("generate_p_Is",
        static_cast<std::vector<std::vector<std::vector<float>>> (*)(
            uint32_t, uint32_t, float, float, uint32_t, uint32_t)>(
            &generate_p_Is));
  m.def(
      "generate_p_Is",
      static_cast<std::vector<std::vector<std::vector<std::vector<float>>>> (*)(
          uint32_t, uint32_t, uint32_t, float, float, uint32_t, uint32_t)>(
          &generate_p_Is));

  py::class_<SIR_SBM_Network>(m, "SIR_SBM_Network")
      .def(py::init<const SBM_Graph_t &, float, float, sycl::queue &,
                    uint32_t>())
      .def_readwrite("N_communities", &SIR_SBM_Network::N_communities)
      .def_readwrite("N_connections", &SIR_SBM_Network::N_connections)
      .def_readwrite("N_vertices", &SIR_SBM_Network::N_vertices)
      .def_readwrite("N_edges", &SIR_SBM_Network::N_edges)
      .def_readonly("p_R", &SIR_SBM_Network::p_R)
      .def_readonly("p_I0", &SIR_SBM_Network::p_I0)
      .def_readonly("p_R0", &SIR_SBM_Network::p_R0)
      .def_readwrite("G", &SIR_SBM_Network::G)
      .def_readwrite("seeds", &SIR_SBM_Network::seeds)
      .def("initialize_vertices", &SIR_SBM_Network::initialize_vertices)
    //   .def("remap", &SIR_SBM_Network::remap)
      .def("recover", &SIR_SBM_Network::recover)
      .def("infect", &SIR_SBM_Network::infect)
      .def("advance", &SIR_SBM_Network::advance);

  //   .def("wait", &sycl::event::wait);

  m.def("simulate_to_file", &simulate_to_file);
  m.def(
      "parallel_simulate_to_file",
      static_cast<void (*)(
          const SBM_Graph_t &, const std::vector<SIR_SBM_Param_t> &,
          std::vector<sycl::queue> &, const std::string &, uint32_t, uint32_t)>(
          &parallel_simulate_to_file));
  m.def("parallel_simulate_to_file",
        static_cast<void (*)(const std::vector<SBM_Graph_t> &,
                             const std::vector<std::vector<SIR_SBM_Param_t>> &,
                             std::vector<std::vector<sycl::queue>> &,
                             const std::vector<std::string> &, uint32_t)>(
            &parallel_simulate_to_file));


    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::string &, uint32_t, float, uint32_t)>(&regression_on_datasets));
    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::vector<std::string> &, uint32_t, float, uint32_t)>(&regression_on_datasets));
    m.def("load_N_datasets", &load_N_datasets);
}
