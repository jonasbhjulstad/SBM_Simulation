#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
Sim_Data::Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections) : events_to_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0))),
                                                                                                   events_from_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0))),
                                                                                                   state_timeseries(N_sims, std::vector<std::vector<State_t>>(Nt+1, std::vector<State_t>(N_communities, {0, 0, 0}))),
                                                                                                   connection_infections(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0)))
{
}

// static Sim_Data assert_size(sycl::queue& q, uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections)
// {

// }
std::vector<uint32_t> get_weights(const std::vector<Edge_t>& edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
    {
        return edge.weight;
    });
    return result;
}
std::vector<uint32_t> get_from(const std::vector<Edge_t>& edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
    {
        return edge.from;
    });
    return result;
}
std::vector<uint32_t> get_to(const std::vector<Edge_t>& edges)
{
    std::vector<uint32_t> result(edges.size());
    std::transform(edges.begin(), edges.end(), result.begin(), [](auto edge)
    {
        return edge.to;
    });
    return result;
}


std::size_t get_sim_data_byte_size(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections)
{
    return sizeof(uint32_t) * Nt * N_sims * N_connections * 2 + sizeof(State_t) * Nt * N_sims * N_communities + sizeof(uint32_t) * Nt * N_sims * N_connections;
}

sycl::range<1> get_wg_range(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

sycl::range<1> get_compute_range(sycl::queue &q, uint32_t N_sims)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    float d_sims = N_sims;
    float d_max_wg_size = max_wg_size;
    auto N_compute_units = static_cast<uint32_t>(std::ceil(d_sims / d_max_wg_size));
    return N_compute_units;
}


void Sim_Param::print() const
{
    std::cout << "SBM:\t" << N_communities << " communities, with " << N_pop << " individuals" << "\n";
    std::cout << "N_sims:\t" << N_sims << "\n";
    std::cout << "p_in:\t" << p_in << "\n";
    std::cout << "p_out:\t" << p_out << "\n";
    std::cout << "Nt:\t" << Nt << "\n";
    std::cout << "Initial probabilities:\t " << p_I0 << ", " << p_R0 << "\n";
    std::cout << "p_I in:\t [" << p_I_min << "," << p_I_max << "]" << "\n";
    std::cout << "compute_range:\t" << compute_range[0] << "\n";
    std::cout << "work_group_range:\t" << wg_range[0] << "\n";
    std::cout << "output directory:\t" << output_dir << std::endl;

}

void Sim_Param::dump(const std::string& fname) const
{

    nlohmann::json j;
    j["N_pop"] = N_pop;
    j["N_communities"] = N_communities;
    j["p_in"] = p_in;
    j["p_out"] = p_out;
    j["p_R0"] = p_R0;
    j["p_I0"] = p_I0;
    j["p_R"] = p_R;
    j["Nt"] = Nt;
    j["Nt_alloc"] = Nt_alloc;
    j["p_I_max"] = p_I_max;
    j["p_I_min"] = p_I_min;
    j["seed"] = seed;
    j["N_graphs"] = N_graphs;
    j["N_sims"] = N_sims;
    j["output_dir"] = output_dir;
    j["tau"] = tau;
    j["simulation_subdir"] = simulation_subdir;
    std::ofstream o(fname);
    o << j.dump();
    o.close();
}
