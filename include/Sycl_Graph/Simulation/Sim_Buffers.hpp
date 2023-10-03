#ifndef SIM_BUFFERS_HPP
#define SIM_BUFFERS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Common.hpp>
#include <Sycl_Graph/Utils/Dataframe.hpp>

struct Sim_Buffers
{
    sycl::buffer<Static_RNG::default_rng> rngs;
    sycl::buffer<SIR_State, 3> vertex_state;
    sycl::buffer<uint32_t, 3> accumulated_events;
    // sycl::buffer<uint8_t, 3> edge_events;
    sycl::buffer<float, 3> p_Is;
    sycl::buffer<uint32_t> edge_from;
    sycl::buffer<uint32_t> edge_to;
    sycl::buffer<uint32_t> ecm;
    sycl::buffer<uint32_t, 2> vcm;
    sycl::buffer<uint32_t> edge_counts;
    sycl::buffer<uint32_t> edge_offsets;
    sycl::buffer<State_t, 3> community_state;
    sycl::buffer<uint32_t> N_connections;
    const std::vector<uint32_t> N_connections_vec;
    const std::vector<uint32_t> N_communities_vec;
    const uint32_t N_connections_max;
    const uint32_t N_communities_max;
    const std::vector<uint32_t> vcm_vec;
    const Dataframe_t<Edge_t, 2> ccm;
    std::size_t N_edges_tot;
    Sim_Buffers(sycl::buffer<Static_RNG::default_rng> &rngs,
                sycl::buffer<SIR_State, 3> &vertex_state,
                sycl::buffer<uint32_t, 3> &accumulated_events,
                //  sycl::buffer<uint8_t, 3> &edge_events,
                sycl::buffer<float, 3> &p_Is,
                sycl::buffer<uint32_t> &edge_from,
                sycl::buffer<uint32_t> &edge_to,
                sycl::buffer<uint32_t> &ecm,
                sycl::buffer<uint32_t, 2> &vcm,
                const std::vector<uint32_t> &vcm_vec,
                sycl::buffer<uint32_t> &edge_counts,
                sycl::buffer<uint32_t> &edge_offsets,
                sycl::buffer<uint32_t> &N_connections,
                sycl::buffer<State_t, 3> &community_state,
                const Dataframe_t<Edge_t, 2> &ccm,
                const std::vector<uint32_t> &N_connections_vec,
                const std::vector<uint32_t> &N_communities);
    void validate_sizes(const Sim_Param &p) const;

    std::size_t byte_size() const;
    static Sim_Buffers make(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe_t<uint32_t, 2> &vcms, std::vector<float> p_Is_init);

};

Dataframe_t<float, 1> generate_duplicated_p_Is(auto Nt, auto N_sims_tot, auto N_connections, float p_I_min, float p_I_max, uint32_t seed);
#endif
