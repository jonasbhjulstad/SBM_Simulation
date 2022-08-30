#include <Models/SIR_Bernoulli_Network.hpp>
#include <FROLS_Path_Config.hpp>
#include <random>
#include <fstream>
#include <iostream>
#include <omp.h>

static_assert(IGRAPH_THREAD_SAFE);
struct MC_SIR_Params
{
    double N_pop = 60;
    double p_ER = 1.0;
    double p_I0 = 0.1;
    double p_I_min = .005;
    double p_I_max = .03;
    size_t N_sim = 1000;
    size_t Nt = 100;
    double p_R = 0.01;
};

void MC_SIR_to_file(const std::string fPath, 
const MC_SIR_Params& p)
{
    int thread_id = omp_get_thread_num();
    std::random_device rd;
    std::mt19937 generator(rd());
    std::vector<double> p_I(p.Nt), p_R(p.Nt);
    std::fill(p_R.begin(), p_R.end(), p.p_R);
    //Sample alphas and betas from a uniform distribution
    // std::uniform_real_distribution<double> alpha_dist();
    std::uniform_real_distribution<double> beta_dist(p.p_I_min, p.p_I_max);
    for (size_t i = 0; i < p.Nt; i++) {
        p_I[i] = beta_dist(generator);
    }
    

    //Run SIR_simulations
    std::vector<std::vector<std::array<size_t, 3>>> traj(p.N_sim);
    igraph_t G;
    auto state = generate_SIR_ER_model(G, p.N_pop, p.p_ER, p.p_I0, generator);
    for (size_t i = 0; i < p.N_sim; i++) {
        auto traj_i = run_SIR_simulation(G, state, p.Nt, p_I, p_R, generator);
        traj[i] = traj_i;
    }

    //Write each trajectory to a file with parameters and time
    for (size_t i = 0; i < p.N_sim; i++) {
        std::ofstream outfile(fPath + "/Bernoulli_SIR_MC_" + std::to_string(p.N_sim*thread_id + i) + ".csv");
        //add header to csv
        outfile << "S,I,R,p_I,p_R,t\n";

        for (size_t j = 0; j < p.Nt; j++) {
            for (auto val: traj[i][j]) {
                outfile << val << ",";
            }
            outfile << p_I[j] << "," << p_R[j] << "," << j << "\n";
        }
    }
    igraph_destroy(&G);

    std::cout << "Thread " << thread_id << " finished" << std::endl;

}


int main()
{
    MC_SIR_Params P;
    #pragma omp parallel
    {
        MC_SIR_to_file(FROLS_DATA_DIR, P);
    }

}