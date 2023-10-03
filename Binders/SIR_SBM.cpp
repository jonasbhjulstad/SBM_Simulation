#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Sycl_Graph/Sycl_Graph.hpp>
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

    m.def("run", static_cast<void (*)(sycl::queue &, Sim_Param, Sim_Buffers &)>(&run), "run");

    m.def("run", static_cast<void (*)(sycl::queue &, Sim_Param, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &, const std::vector<std::vector<uint32_t>> &)>(&run), "run");
    // m.def("p_I_run",p_I_run, "p_I_run");

    m.def("p_I_run", static_cast<void (*)(sycl::queue &, Sim_Param, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &, const std::vector<std::vector<uint32_t>> &, const std::vector<std::vector<std::vector<float>>> &)>(&p_I_run), "p_I_run");

    m.def("read_edgelist", static_cast<void (*)(const std::string &, std::vector<std::pair<uint32_t, uint32_t>> &)>(&read_edgelist), "read_edgelist");
    m.def("write_vector", &write_vector, "write_vector");
    m.def("ecm_from_vcm", &ecm_from_vcm, "ecm_from_vcm");
    m.def("project_on_connection", &project_on_connection, "project_on_connection");
    m.def("complete_ccm", &complete_ccm, "complete_ccm");
    py::class_<sycl::range<1>>(m, "sycl_range_1").def(py::init<std::size_t>());

    py::class_<Sim_Param>(m, "Sim_Param").def(py::init<>())
        .def(py::init<const std::string &>())
        .def("dump", &Sim_Param::dump)
        .def_readwrite("N_pop", &Sim_Param::N_pop)
        .def_readwrite("N_communities", &Sim_Param::N_communities)
        .def_readwrite("p_in", &Sim_Param::p_in)
        .def_readwrite("p_out", &Sim_Param::p_out)
        .def_readwrite("Nt", &Sim_Param::Nt) // p_R0, p_I0, sim_idx, seed
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
        .def_readwrite("tau", &Sim_Param::tau)
        .def_readwrite("N_connections", &Sim_Param::N_connections);


    py::class_<Multiple_Sim_Param_t>(m, "Multiple_Sim_Param").def(py::init<sycl::queue &>())
        .def(py::init<const std::string &>())
        .def("dump", &Multiple_Sim_Param_t::dump)
        .def_readwrite("N_pop", &Multiple_Sim_Param_t::N_pop)
        .def_readwrite("N_communities", &Multiple_Sim_Param_t::N_communities)
        .def_readwrite("p_in", &Multiple_Sim_Param_t::p_in)
        .def_readwrite("p_out", &Multiple_Sim_Param_t::p_out)
        .def_readwrite("Nt", &Multiple_Sim_Param_t::Nt) // p_R0, p_I0, sim_idx, seed
        .def_readwrite("p_R0", &Multiple_Sim_Param_t::p_R0)
        .def_readwrite("N_pop", &Multiple_Sim_Param_t::N_pop)
        .def_readwrite("p_I0", &Multiple_Sim_Param_t::p_I0)
        .def_readwrite("p_R", &Multiple_Sim_Param_t::p_R)
        .def_readwrite("max_infection_samples", &Multiple_Sim_Param_t::max_infection_samples)
        .def_readwrite("N_sims", &Multiple_Sim_Param_t::N_sims)
        .def_readwrite("Nt_alloc", &Multiple_Sim_Param_t::Nt_alloc)
        .def_readwrite("seed", &Multiple_Sim_Param_t::seed)
        .def_readwrite("N_graphs", &Multiple_Sim_Param_t::N_graphs)
        .def_readwrite("output_dir", &Multiple_Sim_Param_t::output_dir)
        .def_readwrite("compute_range", &Multiple_Sim_Param_t::compute_range)
        .def_readwrite("wg_range", &Multiple_Sim_Param_t::wg_range)
        .def_readwrite("p_I_min", &Multiple_Sim_Param_t::p_I_min)
        .def_readwrite("p_I_max", &Multiple_Sim_Param_t::p_I_max)
        .def_readwrite("tau", &Multiple_Sim_Param_t::tau)
        .def("to_sim_param", &Multiple_Sim_Param_t::to_sim_param);


// struct Multiple_Sim_Param_t
// {
//     Multiple_Sim_Param_t(sycl::queue &q) : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1))
//     {
//         auto device = q.get_device();
//         global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
//         local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
//     }

//     Multiple_Sim_Param_t() : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1)) {}
//     Multiple_Sim_Param_t(const std::string& fname);

//     uint32_t N_pop = 100;
//     float p_in = 1.0f;
//     uint32_t Nt = 30;
//     float p_R0 = .0f;
//     float p_I0;
//     float p_R;
//     uint32_t N_sims = 2;
//     std::vector<float> p_out;
//     std::vector<float> p_I_min;
//     std::vector<float> p_I_max;
//     std::vector<float> p_I_min_vec;
//     std::vector<float> p_I_max_vec;
//     float tau = .9f;
//     uint32_t Nt_alloc = 2;
//     uint32_t seed = 238;
//     uint32_t N_communities = 4;
//     uint32_t max_infection_samples = 1000;
//     uint32_t N_graphs = 1;
//     std::size_t local_mem_size = 0;
//     std::size_t global_mem_size = 0;
//     sycl::range<1> compute_range;
//     sycl::range<1> wg_range;
//     std::string output_dir;
//     std::size_t N_vertices() const { return N_communities * N_pop; }
//     void print() const;
//     void dump(const std::string &fname) const;
//     static Multiple_Sim_Param_t parse_json(const std::string& fname);
//     static void generate_default_json(const std::string& fname);
//     Sim_Param to_sim_param(size_t idx = 0) const;


// };


    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>,std::vector<float>, std::vector<float>> (*)(const std::string &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");
    m.def("regression_on_datasets", static_cast<std::tuple<std::vector<float>, std::vector<float>,std::vector<float>, std::vector<float>> (*)(const std::vector<std::string> &, uint32_t, float, uint32_t)>(&regression_on_datasets), "regression_on_datasets");
}
