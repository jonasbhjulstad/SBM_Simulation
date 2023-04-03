#include <Sycl_Graph/Regression.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
// #include <Sycl_Graph/SBM_write.hpp>
#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <utility>
#include <execution>
// using Sycl_Graph::Dynamic::Network_Models::generate_SBM;
using namespace Sycl_Graph;
using namespace Sycl_Graph::SBM;

namespace py = pybind11;

typedef std::vector<std::vector<uint32_t>> Nodelist_t;

PYBIND11_MODULE(SIR_SBM, m) {

    // //define class
    // py::class_<sycl::cpu_selector>(m, "cpu_selector").def(py::init<>());
    // py::class_<sycl::gpu_selector>(m, "gpu_selector").def(py::init<>());
    // py::class_<sycl::queue>(m, "sycl_queue").def(py::init<>()).def(py::init<sycl::cpu_selector>()).def(py::init<sycl::gpu_selector>());
    // py::class_<sycl::range<1>>(m, "sycl_range").def(py::init<uint32_t>());
    // py::class_<sycl::buffer<uint32_t>>(m, "sycl_buffer_uint32").def(py::init<sycl::range<1>>());
    // py::class_<sycl::buffer<float>>(m, "sycl_buffer_float").def(py::init<sycl::range<1>>());


    // py::class_<SIR_SBM_Param_t>(m, "SIR_SBM_Param")
    //     .def(py::init<>())
    //     .def_readwrite("p_I", &SIR_SBM_Param_t::p_I)
    //     .def_readwrite("p_R", &SIR_SBM_Param_t::p_R)
    //     .def_readwrite("p_I0", &SIR_SBM_Param_t::p_I0)
    //     .def_readwrite("p_R0", &SIR_SBM_Param_t::p_R0);

    // py::class_<SBM_Graph_t>(m, "SBM_Graph")
    //     .def(py::init<>())
    //     .def_readwrite("node_list", &SBM_Graph_t::node_list)
    //     .def_readwrite("edge_lists", &SBM_Graph_t::edge_lists)
    //     .def_readwrite("connection_targets", &SBM_Graph_t::connection_targets)
    //     .def_readwrite("connection_sources", &SBM_Graph_t::connection_sources);

    // m.def("create_planted_SBM", &create_planted_SBM, "Create a planted SBM graph");
    // m.def("create_planted_SBMs", &create_planted_SBMs, "Create a multiple SBM graphs");
    
    
    // m.def("generate_p_Is", static_cast<std::vector<std::vector<float>> (*)(uint32_t, float, float, uint32_t, uint32_t)>(&generate_p_Is), "Generate p_Is");
    // m.def("generate_p_Is", static_cast<std::vector<std::vector<std::vector<float>>> (*)(uint32_t, uint32_t, float, float, uint32_t, uint32_t)>(&generate_p_Is), "Generate p_Is");
    // m.def("generate_p_Is", static_cast<std::vector<std::vector<std::vector<std::vector<float>>>> (*)(uint32_t, uint32_t, uint32_t, float, float, uint32_t, uint32_t)>(&generate_p_Is), "Generate p_Is");

    // m.def("SBM_simulate", &SBM_simulate, "Simulate SBM");
    // m.def("nodelist_to_vertex_community_map", &vcm_from_node_list, "Convert node list to vertex community map");
    // m.def("edgelist_to_edge_community_map", &create_edge_community_map, "Convert edge list to edge community map");
    
    // m.def("get_total_state", &get_total_state, "Get total state");
    // m.def("get_community_state", &get_community_state, "Get community state");

    // m.def("read_iteration_buffer", &read_iteration_buffer, "Read iteration buffer");
    // m.def("iteration_lists_to_community_state", &iteration_lists_to_community_state, "Convert iteration lists to community state");

    // m.def("load_N_datasets", &load_N_datasets, "Load N datasets");
    // m.def("beta_regression", &beta_regression, "Beta regression");
    // m.def("alpha_regression", &alpha_regression, "Alpha regression");

    // m.def("parallel_simulate_to_file", static_cast<void (*)(const SBM_Graph_t&, const std::vector<SIR_SBM_Param_t>&, std::vector<sycl::queue>&, const std::string&, uint32_t, uint32_t)>(&parallel_simulate_to_file), "Parallel simulate to file");
    // m.def("parallel_simulate_to_file", static_cast<void (*)(const std::vector<SBM_Graph_t>&, const std::vector<std::vector<SIR_SBM_Param_t>>&, std::vector<std::vector<sycl::queue>>&, const std::vector<std::string>&, uint32_t)>(&parallel_simulate_to_file), "Parallel simulate to file");
 
    // m.def("regression_on_datasets", static_cast<std::tuple<float, std::vector<float>, std::vector<float>> (*)(const std::string&, uint32_t, float, uint32_t)>(&regression_on_datasets), "Regression on datasets");
    // m.def("regression_on_datasets", static_cast<std::tuple<float, std::vector<float>, std::vector<float>> (*)(const std::vector<std::string>&, uint32_t, float, uint32_t)>(&regression_on_datasets), "Regression on datasets");

    // m.def("rearrange_SBM_with_cmap", &rearrange_SBM_with_cmap, "Rearrange SBM with cmap");
}

