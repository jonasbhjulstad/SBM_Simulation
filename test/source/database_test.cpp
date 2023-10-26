#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>
#include <array>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>


void sim_param_test()
{

    uint32_t N_pop = 100;
    uint32_t N_graphs = 10;
    const std::vector<int32_t> &N_communities = 1;
    uint32_t p_out_idx = 2;
    float p_in = .1;
    float p_out = .1;
    uint32_t N_sims = 10;
    uint32_t Nt = 100;
    uint32_t Nt_alloc = 1;
    uint32_t seed = 23;
    float p_I_min = 0.1;
    float p_I_max = .2;
    float p_R = 0.1f;
    float p_I0 = 0.1f;
    float p_R0 = 0.0f;

    auto sim_param = create_sim_parameters(
        N_pop,
        N_graphs,
        N_communities,
        p_out_idx,
        p_in,
        p_out,
        N_sims,
        Nt,
        Nt_alloc,
        seed,
        p_I_min,
        p_I_max,
        p_R,
        p_I0,
        p_R0);
    sim_param_insert(sim_param);
    auto sim_param_2 = sim_param_read(p_out_idx);
    CHECK(sim_param_2["N_pop"] == N_pop);
    CHECK(sim_param_2["N_graphs"] == N_graphs);
    CHECK(sim_param_2["p_in"] == p_in);
    CHECK(sim_param_2["p_out"] == p_out);
    CHECK(sim_param_2["N_sims"] == N_sims);
    CHECK(sim_param_2["Nt"] == Nt);
    CHECK(sim_param_2["Nt_alloc"] == Nt_alloc);
    CHECK(sim_param_2["seed"] == seed);
    CHECK(sim_param_2["p_I_min"] == p_I_min);
    CHECK(sim_param_2["p_I_max"] == p_I_max);
    CHECK(sim_param_2["p_R"] == p_R);
    CHECK(sim_param_2["p_I0"] == p_I0);
    CHECK(sim_param_2["p_R0"] == p_R0);


}

TEST_CASE("Sim_Param_test")
{
    drop_simulation_tables();
    create_simulation_tables();
    sim_param_test();
}

TEST_CASE("Community_State_table_test")
{
    drop_simulation_tables();
    create_simulation_tables();
    std::array<State_t, 4> size({10, 10, 10, 10});
    Dataframe_t df(size);
    community_state_insert(1, 1, df);

}

TEST_CASE("Connection_tables_test")
{
    drop_simulation_tables();
    create_simulation_tables();
    Dataframe_t<uint32_t, 4> connection_events({10, 10, 10, 10});

    connection_insert("connection_events", 1, 1, df);

    Dataframe_t<uint32_t, 4> infection_events({10, 10, 10, 10});
    connection_insert("infection_events", 1, 1, df);

    Dataframe_t<float, 4> p_Is({10, 10, 10, 10});
    connection_insert("p_Is", 1, 1, df);
}
