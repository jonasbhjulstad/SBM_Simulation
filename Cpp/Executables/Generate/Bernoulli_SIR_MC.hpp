#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <DataFrame.hpp>
#include <quantiles.hpp>
#include <Math.hpp>
#include <Models/SIR_Bernoulli_Network.hpp>
#include <bits/stdc++.h>
#include <igraph_threading.h>
#include <omp.h>

namespace FROLS {
    struct MC_SIR_Params {
        double N_pop = 60;
        double p_ER = 1.0;
        double p_I0 = 0.05;
        double p_R0 = 0.05;
        double p_I_min = 1e-8;
        double p_I_max = 1e-1;
        size_t N_sim_tot = 10000;
        size_t Nt = 100;
        double p_R = 0.01;
        size_t seed;
    };


    void MC_SIR_to_file(const std::string &fPath, const MC_SIR_Params &p) {
        int thread_id = omp_get_thread_num();
        int N_threads = omp_get_num_threads();
        size_t N_thread_sims = p.N_sim_tot / N_threads;
        std::random_device rd;
        // std::cout << "Thread " << thread_id << " initialized with seed: " << seed
        // << std::endl;
        thread_local std::mt19937 generator(p.seed);
        std::vector<double> p_I(p.Nt), p_R(p.Nt);
        std::fill(p_R.begin(), p_R.end(), p.p_R);
        // Sample alphas and betas from a uniform distribution
        //  std::uniform_real_distribution<double> alpha_dist();
        std::uniform_real_distribution<double> beta_dist(p.p_I_min, p.p_I_max);
        static_assert(IGRAPH_THREAD_SAFE);
        // Run SIR_simulations
        thread_local igraph_t G;
        std::vector<size_t> idx(p.Nt + 1);
        std::vector<size_t> t(p.Nt + 1);
        std::generate(t.begin(), t.end(), [n = 0]() mutable { return n++; });
        FROLS::DataFrame df(p.Nt);
        std::vector<std::string> colnames = {"S", "I", "R"};
        df.assign("t", t);
        size_t p_I_duration = 100;
        generate_SIR_ER_model(G, p.N_pop, p.p_ER, generator);


        df.assign("p_R", p_R);
        for (size_t i = 0; i < N_thread_sims; i++) {
            auto state = generate_initial_infections(p.N_pop, p.p_I0, p.p_R0, generator);
            // std::fill(p_I.begin(), p_I.end(), p_I_const);
            //fill p_I with random generated value;
            for (int j = 0; j < p_I.size(); j++) {
                static double p_I_const;
                if (!(j % p_I_duration)) {
                    p_I_const = beta_dist(generator);
                }
                p_I[j] = p_I_const;
            }
            auto traj = run_SIR_simulation(G, state, p.Nt, p_I, p_R, generator);
            df.assign(colnames, traj);
            df.assign("p_I", p_I);

            df.write_csv(MC_sim_filename(p.N_pop, p.p_ER, N_thread_sims * thread_id + i),
                         ",");
        }

        igraph_destroy(&G);

        std::cout << "Thread " << thread_id << " finished" << std::endl;
    }

    void compute_SIR_quantiles(size_t N_simulations, size_t N_tau, size_t N_pop,
                               double p_ER) {
        std::vector<std::string> filenames(N_simulations);
        {
            size_t N_threads = omp_get_num_threads();
            size_t thread_id = omp_get_thread_num();
            size_t sims_per_thread = N_simulations / N_threads;
#pragma omp parallel for
            for (int i = 0; i < N_simulations; i++) {
                filenames[i] = MC_sim_filename(N_pop, p_ER, i);
            }
            using namespace FROLS;
            DataFrameStack dfs(filenames);
            size_t N_rows = dfs[0].get_N_rows();
            std::vector<double> t = (*dfs[0]["t"]);
            std::vector<double> xk(N_simulations);

            std::vector<double> quantiles = arange(0.05, 0.95, 0.05);

            std::vector<SIR_Trajectory> q_trajectories(quantiles.size());
            for (auto &traj: q_trajectories) {
                traj.resize(N_rows);
            }
            std::cout << "Computing SIR-Quantiles..." << std::endl;
            std::vector<std::string> q_names(quantiles.size());
#pragma omp parallel for
            for (int i = 0; i < quantiles.size(); i++) {
                q_names[i] = quantile_filename(N_pop, p_ER, quantiles[i]);
                q_trajectories[i].S = dataframe_quantiles(dfs, "S", quantiles[i]);
                q_trajectories[i].I = dataframe_quantiles(dfs, "I", quantiles[i]);
                q_trajectories[i].R = dataframe_quantiles(dfs, "R", quantiles[i]);
                q_trajectories[i].p_I = dataframe_quantiles(dfs, "p_I", quantiles[i]);
            }

            std::cout << "Writing SIR_Quantiles.." << std::endl;
#pragma omp parallel for
            for (int i = 0; i < q_trajectories.size(); i++) {
                DataFrame df;
                df.assign("t", t);
                df.assign("S", q_trajectories[i].S);
                df.assign("I", q_trajectories[i].I);
                df.assign("R", q_trajectories[i].R);
                df.assign("p_I", q_trajectories[i].p_I);
                df.write_csv(q_names[i], ",");
            }
        }
    }
}

#endif