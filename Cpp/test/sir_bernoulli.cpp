#include <gtest/gtest.h>
#include <SIR_Bernoulli_Network.hpp>
#include <fstream>
#include <random>
#include <graph_lite.h>


TEST(SIR_Bernoulli_Network_Test, run_simulation_test)
{
    uint16_t N_pop = 100;
    double p_ER = .4;
    double p_I0 = .1;
    uint16_t N_steps = 100;

    std::random_device rd;
    std::mt19937 generator(rd());

    double p_R0 = 0.05;

    Network_Models::SIR_Bernoulli_Network G(N_pop, p_ER, generator);
    G.generate_initial_infections(p_I0, p_R0);
    uint16_t Nt = 100;
    double p_R = 0.01;
    double p_I = 0.1;
    auto traj = G.simulate(Nt, p_I, p_R);

    std::ofstream outfile("test_traj.txt");
    for (uint16_t i = 0; i < traj.size(); i++) {
        for (auto val: traj[i]) {
            outfile << val << ",";
        }
        outfile << "\n";
    }
    outfile.close();



}