#ifndef SYCL_GRAPH_SIMULATION_HPP
#define SYCL_GRAPH_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Dynamics.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Utils/Compute_Range.hpp>

struct Simulation_t
{

    Simulation_t(sycl::queue &q, pqxx::connection &con, const Sim_Param &sim_param, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm, sycl::range<1> compute_range = 0, sycl::range<1> wg_range = 0, const Dataframe_t<float, 3> &p_Is = {});

    Simulation_t(sycl::queue &q, pqxx::connection &con, const Sim_Param &sim_param, const Sim_Buffers &sim_buffers, sycl::range<1> compute_range, sycl::range<1> wg_range);

    Sim_Buffers b;
    Sim_Param p;
    sycl::queue &q;
    pqxx::connection &con;
    sycl::range<1> compute_range = sycl::range<1>(1);
    sycl::range<1> wg_range = sycl::range<1>(1);

    void run();

private:
    // sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims);
    void write_initial_steps(sycl::queue &q, const Sim_Param &p, pqxx::connection& con, Sim_Buffers &b, std::vector<sycl::event> &dep_events);

    void write_allocated_steps(uint32_t t, std::vector<sycl::event> &dep_events, uint32_t N_max_steps = 0);

    std::tuple<sycl::range<1>, sycl::range<1>> default_compute_range(sycl::queue& q);
};

#endif
