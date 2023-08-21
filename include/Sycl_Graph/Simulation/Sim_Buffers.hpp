#ifndef SIM_BUFFERS_HPP
#define SIM_BUFFERS_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
struct Sim_Buffers
{
    cl::sycl::buffer<Static_RNG::default_rng> rngs;
    cl::sycl::buffer<SIR_State, 3> vertex_state;
    cl::sycl::buffer<uint32_t, 3> events_from;
    cl::sycl::buffer<uint32_t, 3> events_to;
    cl::sycl::buffer<float, 3> p_Is;
    cl::sycl::buffer<uint32_t> edge_from;
    cl::sycl::buffer<uint32_t> edge_to;
    cl::sycl::buffer<uint32_t> ecm;
    cl::sycl::buffer<uint32_t> vcm;
    cl::sycl::buffer<State_t, 3> community_state;
    const std::vector<std::pair<uint32_t, uint32_t>> ccm;
    const std::vector<uint32_t> ccm_weights;
    Sim_Buffers(cl::sycl::buffer<Static_RNG::default_rng> rngs,
                         cl::sycl::buffer<SIR_State, 3> vertex_state,
                         cl::sycl::buffer<uint32_t, 3> events_from,
                         cl::sycl::buffer<uint32_t, 3> events_to,
                         cl::sycl::buffer<float, 3> p_Is,
                         cl::sycl::buffer<uint32_t> edge_from,
                         cl::sycl::buffer<uint32_t> edge_to,
                         cl::sycl::buffer<uint32_t> ecm,
                         cl::sycl::buffer<uint32_t> vcm,
                         cl::sycl::buffer<State_t, 3> community_state,
                         const std::vector<std::pair<uint32_t, uint32_t>> &ccm,
                         const std::vector<uint32_t> &ccm_weights);
    std::size_t byte_size() const;
    static Sim_Buffers make(sycl::queue &q, const Sim_Param &p, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &vcm_init, std::vector<float> p_Is_init = {});
};

#endif
