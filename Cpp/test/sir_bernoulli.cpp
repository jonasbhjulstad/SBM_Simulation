#include <gtest/gtest.h>
#include <Models/SIR_Bernoulli_Network.hpp>
#include <fstream>
#include <random>


TEST(SIR_Bernoulli_Network_Test, run_simulation_test)
{
    size_t N_pop = 100;
    double p_ER = .4;
    double p_I0 = .1;
    std::vector<double> p_I, p_R;
    std::fill(p_R.begin(), p_R.end(), 1./9);
    std::fill(p_I.begin(), p_I.end(), 1.2/9);
    size_t N_steps = 100;

    igraph_t G;
    
    std::random_device rd;
    std::mt19937 generator(rd());

    double p_R0 = 0.05;

    generate_SIR_ER_model(G, N_pop, p_ER, generator);
    auto state = generate_initial_infections(N_pop, p_I0, p_R0, generator);

    auto traj = run_SIR_simulation(G, state, N_steps, p_I, p_R, generator);


    std::ofstream outfile("test_traj.txt");
    for (size_t i = 0; i < traj.size(); i++) {
        for (auto val: traj[i]) {
            outfile << val << ",";
        }
        outfile << "\n";
    }
    outfile.close();



}