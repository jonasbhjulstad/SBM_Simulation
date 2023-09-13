#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Regression.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
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
    py::class_<Device_Info>(m, "Device_Info").def(py::init<>()).def("print", &Device_Info::print);
    m.def("create_sycl_device_queue", &create_sycl_device_queue, "create_sycl_device_queue");
    // static_cast
    m.def("get_device_info", static_cast<Device_Info (*)(sycl::queue &)>(&get_device_info), "get_device_info");
    // m.def("get_device_info", &get_device_info, "get_device_info");
    m.def("determine_device_workload", &determine_device_workload, "determine_device_workload");
    m.def("get_sycl_gpus", &get_sycl_gpus, "get_sycl_gpus");
    m.def("generate_planted_SBM_edges", &generate_planted_SBM_edges, "generate_planted_SBM_edges");
    m.def("generate_planted_SBM_flat", &generate_planted_SBM_flat, "generate_planted_SBM_flat");
    m.def("generate_N_SBM_graphs_flat", &generate_N_SBM_graphs_flat, "generate_N_SBM_graphs_flat");
    m.def("generate_N_SBM_graphs", &generate_N_SBM_graphs, "generate_N_SBM_graphs");
    m.def("run", static_cast<void (*)(sycl::queue &, Sim_Param, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &, const std::vector<std::vector<uint32_t>> &, const std::vector<std::vector<uint32_t>> &)>(&run), "run");
    m.def("p_I_run", static_cast<void (*)(sycl::queue &, Sim_Param, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &, const std::vector<std::vector<uint32_t>> &, const std::vector<std::vector<uint32_t>> &, const std::vector<std::vector<std::vector<float>>>&)>(&p_I_run), "p_I_run");
    m.def("read_edgelist", &read_edgelist, "read_edgelist");
    m.def("write_vector", &write_vector, "write_vector");
    m.def("ecm_from_vcm", &ecm_from_vcm, "ecm_from_vcm");
    m.def("project_on_connection", &project_on_connection, "project_on_connection");
    py::class_<sycl::range<1>>(m, "sycl_range_1").def(py::init<std::size_t>());

    py::class_<Sim_Param>(m, "Sim_Param").def(py::init<>()).def_readwrite("N_pop", &Sim_Param::N_pop).def_readwrite("N_communities", &Sim_Param::N_communities).def_readwrite("p_in", &Sim_Param::p_in).def_readwrite("p_out", &Sim_Param::p_out).def_readwrite("Nt", &Sim_Param::Nt) // p_R0, p_I0, sim_idx, seed
        .def_readwrite("p_R0", &Sim_Param::p_R0)
        .def_readwrite("N_pop", &Sim_Param::N_pop)
        .def_readwrite("p_I0", &Sim_Param::p_I0)
        .def_readwrite("p_R", &Sim_Param::p_R)
        .def_readwrite("max_infection_samples", &Sim_Param::max_infection_samples)
        .def_readwrite("N_sims", &Sim_Param::N_sims)
        .def_readwrite("Nt_alloc", &Sim_Param::Nt_alloc)
        .def_readwrite("seed", &Sim_Param::seed)
        .def_readwrite("N_graphs", &Sim_Param::N_graphs)
        .def_readwrite("output_dir", &Sim_Param::output_dir)
        .def_readwrite("compute_range", &Sim_Param::compute_range)
        .def_readwrite("wg_range", &Sim_Param::wg_range)
        .def_readwrite("p_I_min", &Sim_Param::p_I_min)
        .def_readwrite("p_I_max", &Sim_Param::p_I_max)
        .def_readwrite("tau", &Sim_Param::tau);

    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::string &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");
    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>> (*)(const std::vector<std::string> &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");
}
