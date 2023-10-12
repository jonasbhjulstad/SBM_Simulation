#ifndef SYCL_GRAPH_SIMULATION_SIM_BUFFERS_HPP
#define SYCL_GRAPH_SIMULATION_SIM_BUFFERS_HPP
#include <CL/sycl.hpp>
#include <pqxx/pqxx>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
struct Sim_Buffers
{
    std::shared_ptr<sycl::buffer<Static_RNG::default_rng>> rngs;   // = std::make_shared(sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<SIR_State, 3>> vertex_state;      // = std::make_shared(sycl::buffer<SIR_State, 3>(sycl::range<3>(1,1,1)));
    std::shared_ptr<sycl::buffer<uint32_t, 3>> accumulated_events; // = std::make_shared(sycl::buffer<uint32_t, 3>(sycl::range<3>(1,1,1)));
    std::shared_ptr<sycl::buffer<float, 3>> p_Is;                  // = std::make_shared(sycl::buffer<float, 3>(sycl::range<3>(1,1,1)));
    std::shared_ptr<sycl::buffer<uint32_t>> edge_from;             //= std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<uint32_t>> edge_to;               // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<uint32_t>> ecm;                   // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<uint32_t, 2>> vcm;                // = std::make_shared(sycl::buffer<uint32_t, 2>(sycl::range<2>(1,1)));
    std::shared_ptr<sycl::buffer<uint32_t>> edge_counts;           // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<uint32_t>> edge_offsets;          // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::shared_ptr<sycl::buffer<State_t, 3>> community_state;     //= std::make_shared(sycl::buffer<State_t, 3>(sycl::range<3>(1,1,1)));
    std::shared_ptr<sycl::buffer<uint32_t>> N_connections;         // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
    std::vector<sycl::event> construction_events;
    const Dataframe_t<Edge_t, 2> ccm;
    Sim_Buffers(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe_t<uint32_t, 2> &vcms, Dataframe_t<float, 3> p_Is_init = {});

    void validate_sizes(const Sim_Param &p) const;

    std::size_t byte_size() const;

private:
    Dataframe_t<float, 3> generate_random_p_Is(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe_t<uint32_t, 2> &vcms);

    void construct_buffers(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcms, Dataframe_t<float, 3> p_Is_init);

    static Sim_Buffers make_impl(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcms, Dataframe_t<float, 3> p_Is_init);
};
void validate_buffer_init_sizes(Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<uint32_t>> &ecms, std::vector<float> p_Is_init);

Dataframe_t<float, 3> generate_duplicated_p_Is(uint32_t Nt, uint32_t N_sims_tot, const uint32_t N_connections_tot, float p_I_min, float p_I_max, uint32_t seed);
#endif
