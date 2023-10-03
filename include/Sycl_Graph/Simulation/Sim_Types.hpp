#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
#include <Sycl_Graph/Utils/Common.hpp>
#include <nlohmann/json.hpp>

struct Edge_t
{
    uint32_t from;
    uint32_t to;
    uint32_t weight;
    friend std::fstream &operator<<(std::fstream &os, const Edge_t &e)
    {
        os << e.from << "," << e.to << "," << e.weight;
        return os;
    }
    friend std::ofstream &operator<<(std::ofstream &os, const Edge_t &e)
    {
        os << e.from << "," << e.to << "," << e.weight;
        return os;
    }
};

std::vector<uint32_t> get_weights(const std::vector<Edge_t>& edges);
std::vector<uint32_t> get_from(const std::vector<Edge_t>& edges);
std::vector<uint32_t> get_to(const std::vector<Edge_t>& edges);

struct Sim_Data
{
    Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections);
    std::vector<std::vector<std::vector<uint32_t>>> events_to_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> events_from_timeseries;
    std::vector<std::vector<std::vector<State_t>>> state_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> connection_infections;
};

struct Sim_Param
{

    Sim_Param(sycl::queue &q) : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1))
    {
        auto device = q.get_device();
        global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
        local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    }

    Sim_Param() : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1)) {}
    Sim_Param(const std::vector<Sim_Param>& ps);
    Sim_Param(const std::string& fname);
    Sim_Param(sycl::queue& q, const std::vector<Sim_Param>& ps);

    uint32_t N_communities = 4;
    uint32_t N_pop = 100;
    uint32_t N_sims = 2;
    float p_in = 1.0f;
    float p_out = 0.f;
    std::vector<float> p_out_vec;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    float p_I_min = 0.0f;
    float p_I_max = 0.0f;
    std::vector<float> p_I_min_vec;
    std::vector<float> p_I_max_vec;
    float tau = .9f;
    uint32_t Nt_alloc = 2;
    uint32_t seed = 238;
    uint32_t max_infection_samples = 1000;
    uint32_t N_graphs = 1;
    uint32_t N_connections = 0;
    std::size_t local_mem_size = 0;
    std::size_t global_mem_size = 0;
    sycl::range<1> compute_range;
    sycl::range<1> wg_range;
    std::string output_dir;
    std::size_t N_vertices() const { return N_communities * N_pop; }
    void print() const;
    void dump(const std::string &fname) const;
    static Sim_Param get_settings();
    static Sim_Param parse_json(const std::string& fname);
    static void generate_default_json(const std::string& fname);
};

struct Multiple_Sim_Param_t
{
    Multiple_Sim_Param_t(sycl::queue &q) : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1))
    {
        auto device = q.get_device();
        global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
        local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    }

    Multiple_Sim_Param_t() : compute_range(sycl::range<1>(1)), wg_range(sycl::range<1>(1)) {}
    Multiple_Sim_Param_t(const std::string& fname);

    uint32_t N_pop = 100;
    float p_in = 1.0f;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    uint32_t N_sims = 2;
    std::vector<float> p_out;
    std::vector<float> p_I_min;
    std::vector<float> p_I_max;
    std::vector<float> p_I_min_vec;
    std::vector<float> p_I_max_vec;
    float tau = .9f;
    uint32_t Nt_alloc = 2;
    uint32_t seed = 238;
    uint32_t N_communities = 4;
    uint32_t max_infection_samples = 1000;
    uint32_t N_graphs = 1;
    std::size_t local_mem_size = 0;
    std::size_t global_mem_size = 0;
    sycl::range<1> compute_range;
    sycl::range<1> wg_range;
    std::string output_dir;
    std::size_t N_vertices() const { return N_communities * N_pop; }
    void print() const;
    void dump(const std::string &fname) const;
    static Multiple_Sim_Param_t parse_json(const std::string& fname);
    static void generate_default_json(const std::string& fname);
    Sim_Param to_sim_param(size_t idx = 0) const;


};
std::size_t get_sim_data_byte_size(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections);




#endif
