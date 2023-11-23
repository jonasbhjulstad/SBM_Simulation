#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <SBM_Simulation/SBM_Simulation.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>

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

PYBIND11_MODULE(SBM_Simulation, m)
{

    py::class_<sycl::gpu_selector>(m, "gpu_selector").def(py::init<>());
    py::class_<sycl::cpu_selector>(m, "cpu_selector").def(py::init<>());

    py::class_<sycl::queue>(m, "sycl_queue")
        .def(py::init<>())
        .def(py::init<const sycl::gpu_selector &>())
        .def(py::init<const sycl::cpu_selector &>());
    py::class_<sycl::event>(m, "sycl_event").def(py::init<>());
    py::class_<Buffer_Routines::Device_Info>(m, "Device_Info").def(py::init<>()).def("print", &Buffer_Routines::Device_Info::print);
    m.def("create_sycl_device_queue", &create_sycl_device_queue, "create_sycl_device_queue");
    m.def("get_device_info", static_cast<Buffer_Routines::Device_Info (*)(sycl::queue &)>(&Buffer_Routines::get_device_info), "get_device_info");
    m.def("get_sycl_gpus", &get_sycl_gpus, "get_sycl_gpus");
    // m.def("generate_planted_SBM_edges", &generate_planted_SBM_edges, "generate_planted_SBM_edges");
    // m.def("generate_N_SBM_graphs", &generate_N_SBM_graphs, "generate_N_SBM_graphs");


    // m.def("sim_param_upsert", static_cast<void (*)(const Sim_Param &)>(&SBM_Database::sim_param_upsert), "sim_param_upsert");

}
