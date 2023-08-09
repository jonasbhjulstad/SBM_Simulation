#ifndef SIR_SIMULATION_HPP
#define SIR_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
struct Common_Buffers;

struct Common_Buffers
{
    Common_Buffers(sycl::queue &q, const std::vector<uint32_t> &edge_from_init, const std::vector<uint32_t> &edge_to_init, const std::vector<uint32_t> &ecm_init, const std::vector<uint32_t> &vcm_init, uint32_t N_clusters, uint32_t N_connections, uint32_t Nt, uint32_t seed);
    Common_Buffers(const auto p_edge_from, const auto p_edge_to, const auto p_ecm, const auto p_vcm, const auto &ccm, const auto &ccm_weights, const std::vector<sycl::event> &events);
    Common_Buffers(const Common_Buffers &other);

    std::string get_sizes();

    uint32_t N_connections() const;
    std::vector<sycl::event> events = std::vector<sycl::event>(4);
    std::shared_ptr<sycl::buffer<uint32_t>> edge_from;
    std::shared_ptr<sycl::buffer<uint32_t>> edge_to;
    std::shared_ptr<sycl::buffer<uint32_t>> ecm;
    std::shared_ptr<sycl::buffer<uint32_t>> vcm;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
};
void simulate(sycl::queue &q, const Sim_Param &p, const Common_Buffers &cb, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<std::vector<float>> &p_Is, const std::string output_dir, uint32_t N_simulations, uint32_t seed);
void excite_simulate(sycl::queue &q, const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, uint32_t seed);

#endif
