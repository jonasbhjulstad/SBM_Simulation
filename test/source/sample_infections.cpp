#include <doctest/doctest.h>
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <SBM_Graph/Community_Mappings.hpp>

TEST_CASE("Related Connections")
{
    using namespace SBM_Simulation;
    auto N_communities = 2;
    auto ccm = SBM_Graph::complete_ccm(N_communities, true);

    auto r_con_0 = get_related_connections(0, ccm);
    CHECK(r_con_0[0] == 0);
    CHECK(r_con_0[1] == 2);
    auto r_con_1 = get_related_connections(1, ccm);
    CHECK(r_con_1[0] == 1);
    CHECK(r_con_1[1] == 3);
}

TEST_CASE("Weighted Sampling")
{
    
}
