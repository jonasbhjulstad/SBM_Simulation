#ifndef SIR_SBM_NETWORK_HPP
#define SIR_SBM_NETWORK_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SBM_types.hpp>
#include <limits>
#include <numeric>
#include <stddef.h>
#include <tuple>
namespace Sycl_Graph::SBM
{

    uint32_t get_susceptible_id_if_infected(sycl::buffer<SIR_State> &v_acc, uint32_t id_from,
                                            uint32_t id_to);

    struct SIR_SBM_Network
    {

        SIR_SBM_Network(const SBM_Graph_t &G, float p_I0, float p_R, sycl::queue &q,
                        uint32_t seed = 52, float p_R0 = .0f);
        uint32_t N_communities;
        uint32_t N_connections;
        uint32_t N_vertices;
        uint32_t N_edges;
        const float p_R;
        const float p_I0;
        const float p_R0;
        sycl::queue &q;
        sycl::buffer<Edge_t> edges;
        std::vector<sycl::buffer<SIR_State>> trajectory;
        std::vector<sycl::buffer<Edge_t>> connection_events;
        sycl::buffer<uint32_t> seed_buf;
        sycl::buffer<uint32_t> ecm_buf;
        sycl::buffer<uint32_t> vcm_buf;
        std::vector<sycl::event> init_events;

        sycl::event initialize_vertices(float p_I0, float p_R0, uint32_t N,
                                        sycl::queue &q,
                                        sycl::buffer<uint32_t, 1> &seed_buf,
                                        sycl::buffer<SIR_State> &buf);

        std::vector<sycl::event> remap(const std::vector<uint32_t> &cmap);
        sycl::event recover(sycl::buffer<SIR_State> &state,
                            sycl::buffer<SIR_State> &state_next, auto &dep_event);

        sycl::event advance(sycl::buffer<SIR_State> &state,
                            sycl::buffer<SIR_State> &state_next,
                            sycl::buffer<Edge_t> &connection_events_buf,
                            sycl::buffer<float> &p_I, auto &dep_event);

        typedef std::vector<State_t> Community_States_t;
        typedef std::vector<Community_States_t> Community_Trajectory_t;

        std::vector<sycl::event>
        accumulate_community_state(std::vector<sycl::buffer<State_t>> &result,
                                   std::vector<sycl::buffer<SIR_State>> &v_bufs,
                                   auto &dep_event);

        typedef std::tuple < std::vector<sycl::buffer<float>>, std::vector<sycl::buffer<uint32_t>>, std::vector<sycl::buffer<SIR_State>>, std::vector<sycl::buffer<State_t>>, std::vector<sycl::event>> init_t;

        init_t sim_init(const std::vector<std::vector<float>> &p_Is);

        typedef std::tuple<std::vector<sycl::buffer<SIR_State>>, std::vector<sycl::buffer<Edge_t>>, std::vector<sycl::event>> trajectory_t;

        trajectory_t simulate(const SIR_SBM_Param_t &param);

    private:
        sycl::buffer<uint32_t, 1> generate_seeds(uint32_t N_rng, sycl::queue &q,
                                                 unsigned long seed = 42);
        sycl::event create_ecm(const SBM_Graph_t &G);

        struct ecm_map_elem_t
        {
            uint32_t community_from = std::numeric_limits<uint32_t>::max();
            uint32_t community_to = std::numeric_limits<uint32_t>::max();
        };

        std::vector<ecm_map_elem_t> create_ecm_index_map(uint32_t N);

        sycl::event create_state_buf(sycl::buffer<SIR_State> &state_buf);
    };

} // namespace Sycl_Graph::SBM

// make sycl device copyable
template <>
struct sycl::is_device_copyable<Sycl_Graph::SBM::State_t> : std::true_type
{
};

#endif