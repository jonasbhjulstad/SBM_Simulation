#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Regression.hpp>
#include <Sycl_Graph/Simulation.hpp>

namespace py = pybind11;

PYBIND11_MODULE(SIR_SBM, m)
{

    py::class_<sycl::gpu_selector>(m, "gpu_selector").def(py::init<>());
    py::class_<sycl::cpu_selector>(m, "cpu_selector").def(py::init<>());

    py::class_<sycl::queue>(m, "sycl_queue")
        .def(py::init<>())
        .def(py::init<const sycl::gpu_selector &>())
        .def(py::init<const sycl::cpu_selector &>());
    py::class_<sycl::event>(m, "sycl_event").def(py::init<>());
    m.def("generate_planted_SBM_edges", &generate_planted_SBM_edges, "generate_planted_SBM_edges");
    py::class_<Sim_Param>(m, "Sim_Param").def(py::init<>())
    .def_readwrite("N_pop", &Sim_Param::N_pop)
    .def_readwrite("N_clusters", &Sim_Param::N_clusters)
    .def_readwrite("p_in", &Sim_Param::p_in)
    .def_readwrite("p_out", &Sim_Param::p_out)
    .def_readwrite("Nt", &Sim_Param::Nt) //p_R0, p_I0, sim_idx, seed
    .def_readwrite("p_R0", &Sim_Param::p_R0)
    .def_readwrite("p_I0", &Sim_Param::p_I0)
    .def_readwrite("sim_idx", &Sim_Param::sim_idx)
    .def_readwrite("seed", &Sim_Param::seed)
    .def_readwrite("p_R", &Sim_Param::p_R);
    m.def("excite_simulate", &excite_simulate, "excite_simulate");


}
