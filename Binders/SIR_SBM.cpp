#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Regression.hpp>
#include <algorithm>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <utility>

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
}
