#ifndef SIM_BUFFERS_HPP
#define SIM_BUFFERS_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
struct Sim_Buffers
{
    Sim_Buffers() = default;
    Static_RNG::default_rng *rngs;
    SIR_State *vertex_state;
    uint32_t *events_from;
    uint32_t *events_to;
    float *p_Is;
    uint32_t *edge_from;
    uint32_t *edge_to;
    uint32_t *edge_offsets;
    uint32_t *ecm;
    uint32_t *vcm;
    uint32_t *edge_counts;
    State_t *community_state;

    std::size_t N_connections;
    std::size_t N_edges;
    std::size_t b_size;

    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccm;
    const std::vector<std::vector<uint32_t>> ccm_weights;
    std::size_t byte_size() const;
    static Sim_Buffers make(sycl::queue &q, const Sim_Param &p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm_init, std::vector<float> p_Is_init);
};

template <>
struct sycl::is_device_copyable<Sim_Buffers> : std::true_type
{
};


#endif
