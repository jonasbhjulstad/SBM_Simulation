#include <Models/SIR_Bernoulli_Network.hpp>
#include <random>
#include <fstream>
#include <iostream>
#include <omp.h>
void MC_SIR_to_file(const std::string& fPath)
{
    int thread_id = omp_get_thread_num();
    std::random_device rd;
    std::mt19937 generator(rd());
    double alpha_mean = 1./9;
    double beta_mean = 1.05*alpha_mean;
    double N_pop = 1000;
    size_t Nt = 100;
    std::vector<double> p_I(Nt), p_R(Nt);
    size_t N_sim = 100;
    //Sample alphas and betas from a uniform distribution
    std::uniform_real_distribution<double> alpha_dist(0.5*alpha_mean, 1.5*alpha_mean);
    std::uniform_real_distribution<double> beta_dist(0.5*beta_mean, 1.5*beta_mean);
    for (size_t i = 0; i < Nt; i++) {
        p_I[i] = alpha_dist(generator);
        p_R[i] = beta_dist(generator);
    }

    //Run SIR_simulations
    std::vector<std::vector<std::array<size_t, 3>>> traj(N_sim);
    igraph_t G;
    auto state = generate_SIR_ER_model(G, N_pop, 0.6, 0.05, generator);
    for (size_t i = 0; i < N_sim; i++) {
        auto traj_i = run_SIR_simulation(G, state, Nt, p_I, p_R, generator);
        traj [i] = traj_i;
    }

    //Write each trajectory to a file with parameters and time
    for (size_t i = 0; i < N_sim; i++) {
        std::ofstream outfile(fPath + "Bernoulli_SIR_MC_" + std::to_string(N_sim*thread_id + i) + ".csv");
        //add header to csv
        outfile << "S,I,R,p_I,p_R,t\n";

        for (size_t j = 0; j < Nt; j++) {
            for (auto val: traj[i][j]) {
                outfile << val << ",";
            }
            outfile << p_I[j] << "," << p_R[j] << "," << j << "\n";
        }
    }

    std::cout << "Thread " << thread_id << " finished" << std::endl;

}


int main()
{
    #pragma omp parallel
    {
        MC_SIR_to_file("C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\");
    }

}