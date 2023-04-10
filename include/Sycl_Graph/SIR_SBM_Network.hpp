#ifndef SIR_SBM_NETWORK_HPP
#define SIR_SBM_NETWORK_HPP
#include <Sycl_Graph/SBM_types.hpp>
namespace Sycl_Graph::SBM
{

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
        SBM_Graph_t G;
        sycl::buffer<Edge_t> edges;
        std::vector<uint32_t> seeds;
        sycl::buffer<uint32_t> seed_buf;
        sycl::buffer<uint32_t> ecm_buf;
        sycl::buffer<uint32_t> vcm_buf;

        sycl::buffer<SIR_State, 2> trajectory;
        sycl::buffer<float, 2> p_I_buf;
        sycl::buffer<Edge_t, 2> connection_events_buf;
        sycl::buffer<State_t, 2> community_state_buf;
        std::vector<sycl::event> init_events;

        sycl::event initialize_vertices(float p_I0, float p_R0,
                                        sycl::queue &q,
                                        sycl::buffer<SIR_State, 2> &buf);

        std::vector<sycl::event> remap(const std::vector<uint32_t> &cmap);
        sycl::event recover(uint32_t t, std::vector<sycl::event> &dep_event);
        sycl::event infect(uint32_t t, sycl::event &dep_event);

        sycl::event advance(uint32_t t, std::vector<sycl::event> dep_events = {});

        typedef std::vector<State_t> Community_States_t;
        typedef std::vector<Community_States_t> Community_Trajectory_t;

        sycl::event
        accumulate_community_state(
            std::vector<sycl::event> dep_event = {});

        // typedef std::tuple<std::vector<sycl::buffer<float>>, std::vector<sycl::buffer<Edge_t>>, std::vector<sycl::buffer<SIR_State>>, std::vector<sycl::buffer<State_t>>> init_t;

        void sim_init(const std::vector<std::vector<float>> &p_Is);

        typedef std::tuple<std::vector<sycl::buffer<State_t>>, std::vector<sycl::buffer<Edge_t>>, std::vector<sycl::event>> trajectory_t;

        std::vector<sycl::event> simulate(const SIR_SBM_Param_t &param);
        std::tuple<std::vector<std::vector<State_t>>, std::vector<std::vector<Edge_t>>> read_trajectory();

    private:
        std::vector<uint32_t> generate_seeds(uint32_t N_rng,
                                             uint32_t seed = 42);
        sycl::event create_ecm(const SBM_Graph_t &G);
        sycl::event create_vcm(const std::vector<std::vector<uint32_t>> &node_lists);

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