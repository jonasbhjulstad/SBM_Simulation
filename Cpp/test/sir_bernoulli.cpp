#include <gtest/gtest.h>
#include <Models/SIR_Bernoulli_Network.hpp>
#include <fstream>



TEST(SIR_Bernoulli_Network_Test, run_simulation_test)
{
    size_t N_pop = 100;
    double p_ER = .4;
    double p_I0 = .1;
    double p_infect = .1;
    double p_recover = .1;
    size_t N_steps = 100;

    igraph_t G;
    
    auto state = generate_SIR_ER_model(G, N_pop, p_ER, p_I0);

    auto traj = run_SIR_simulation(G, state, N_steps, p_infect, p_recover);


    std::ofstream outfile("test_traj.txt");
    for (size_t i = 0; i < traj.size(); i++) {
        for (auto val: traj[i]) {
            outfile << val << ",";
        }
        outfile << "\n";
    }
    outfile.close();



}