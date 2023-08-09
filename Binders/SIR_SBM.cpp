#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Regression.hpp>
#include <Sycl_Graph/Simulation.hpp>
#include <Sycl_Graph/Profiling.hpp>
namespace py = pybind11;

sycl::queue create_sycl_device_queue(std::string device_type, uint32_t index = 0)
{
    if (device_type == "gpu")
    {
        auto GPU_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (index >= GPU_devices.size())
            std::cout << "Warning: GPU index out of range, using GPU 0" << std::endl;
        return sycl::queue(GPU_devices[index]);
    }
    else if (device_type == "cpu")
    {
        auto CPU_devices = sycl::device::get_devices(sycl::info::device_type::cpu);
        if (index >= CPU_devices.size())
            std::cout << "Warning: CPU index out of range, using CPU 0" << std::endl;
        return sycl::queue(CPU_devices[index]);
    }
    else
    {
        throw std::runtime_error("Invalid device type");
    }
}

std::vector<sycl::queue> get_sycl_gpus()
{
    auto GPU_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::vector<sycl::queue> queues;
    for (auto &device : GPU_devices)
    {
        queues.push_back(sycl::queue(device));
    }
    return queues;
}

PYBIND11_MODULE(SIR_SBM, m)
{


    py::class_<sycl::gpu_selector>(m, "gpu_selector").def(py::init<>());
    py::class_<sycl::cpu_selector>(m, "cpu_selector").def(py::init<>());

    py::class_<sycl::queue>(m, "sycl_queue")
        .def(py::init<>())
        .def(py::init<const sycl::gpu_selector &>())
        .def(py::init<const sycl::cpu_selector &>());
    py::class_<sycl::event>(m, "sycl_event").def(py::init<>());
    py::class_<Device_Info>(m, "Device_Info").def(py::init<>())
    .def("print", &Device_Info::print);
    m.def("create_sycl_device_queue", &create_sycl_device_queue, "create_sycl_device_queue");
    m.def("get_device_info", &get_device_info, "get_device_info");
    m.def("determine_device_workload", &determine_device_workload, "determine_device_workload");
    m.def("get_sycl_gpus", &get_sycl_gpus, "get_sycl_gpus");
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
    .def_readwrite("p_R", &Sim_Param::p_R)
    .def_readwrite("max_infection_samples", &Sim_Param::max_infection_samples);

    m.def("simulate", &simulate, "simulate");
    m.def("excite_simulate", &excite_simulate, "excite_simulate");
    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::string &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");
    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::vector<std::string> &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");


}
