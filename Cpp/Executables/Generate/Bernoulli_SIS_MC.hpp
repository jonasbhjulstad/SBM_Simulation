#ifndef FROLS_BERNOULLI_SIS_MC_HPP
#define FROLS_BERNOULLI_SIS_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <DataFrame.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <SIS_Bernoulli_Network.hpp>
#include <graph_lite.h>
#include <omp.h>

namespace FROLS {
    struct MC_SIS_Params {
        double N_pop = 60;
        double p_ER = 1.0;
        double p_I0 = 0.2;
        double p_I_max = 0.2;
        double p_I_min = 0.;
        size_t N_sim_tot = 500;
        size_t Nt_max = 100;
        size_t Nt_min = 15;
        double p_S = 0.4;
        size_t seed;
        size_t infection_count_tolerance = N_pop/15;
    };
    template <typename RNG>
    std::vector<std::pair<double, double>> generate_interaction_probabilities(const MC_SIS_Params& p, RNG& rng)
    {
        double omega_bounds[] = {(2*M_PI)/1000, (2*M_PI)/100};
        std::uniform_real_distribution<double> d_omega(omega_bounds[0], omega_bounds[1]);
        std::uniform_real_distribution<double> d_offset(0, 2*M_PI);
        std::vector<std::pair<double, double>> res(p.Nt_max);
        double offset = d_offset(rng);
        double p_I_mean = (p.p_I_max - p.p_I_min)/2 + p.p_I_min;
        double p_I_std = p_I_mean - p.p_I_min;
        double omega = d_omega(rng);
        for (int i = 0; i < p.Nt_max; i++)
        {
            double p_I = std::max({p_I_mean + p_I_std*sin(omega*i + offset), 0.});
            res[i] = std::make_pair(p_I, p.p_S);
        }
        return res;
    }


    void MC_SIS_to_file(const std::string &fPath, const MC_SIS_Params &p) {
        int thread_id = omp_get_thread_num();
        int N_threads = omp_get_num_threads();
        size_t N_thread_sims = (thread_id == N_threads-1) ? p.N_sim_tot / N_threads : p.N_sim_tot / N_threads + p.N_sim_tot % N_threads;

        std::random_device rd;
        // std::cout << "Thread " << thread_id << " initialized with seed: " << seed
        // << std::endl;
        thread_local std::mt19937 generator(p.seed);
        FROLS::DataFrame df;
        std::vector<std::string> colnames = {"S", "I"};

        thread_local Network_Models::SIS_Bernoulli_Network G(p.N_pop, p.p_ER, generator);

        for (size_t i = 0; i < N_thread_sims; i++) {
            G.reset();
            while(G.population_count()[1] == 0) {
                G.generate_initial_infections(p.p_I0);
            }
            auto p_vec = generate_interaction_probabilities(p, generator);
            auto traj = G.simulate(p_vec, p.infection_count_tolerance, p.Nt_min);
            auto [p_I_vec, p_S_vec] = FROLS::unzip(p_vec);
            size_t Nt = traj[0].size();
            p_I_vec.resize(Nt);
            p_S_vec.resize(Nt);
            df.assign("S", traj[0]);
            df.assign("I", traj[1]);
            df.assign("p_I", p_I_vec);
            df.assign("p_S", p_S_vec);
            df.assign("t", FROLS::range(0, Nt));
            if ((thread_id == 0) && (i % (N_thread_sims/10) == 0))
            {
                std::cout << i << " of " << N_thread_sims << " complete" << std::endl;
            }
            df.write_csv(MC_sim_filename(p.N_pop, p.p_ER, N_thread_sims * thread_id + i, std::string("SIS")),
                         ",");
        }


        std::cout << "Thread " << thread_id << " finished" << std::endl;
    }

    void compute_SIS_quantiles(size_t N_simulations, size_t N_tau, size_t N_pop,
                               double p_ER) {
        std::vector<std::string> filenames(N_simulations);
        {
            size_t N_threads = omp_get_num_threads();
            size_t thread_id = omp_get_thread_num();
            size_t sims_per_thread = N_simulations / N_threads;
#pragma omp parallel for
            for (int i = 0; i < N_simulations; i++) {
                filenames[i] = MC_sim_filename(N_pop, p_ER, i, "SIS");
            }
            using namespace FROLS;
            DataFrameStack dfs(filenames);
            size_t N_rows = dfs[0].get_N_rows();
            std::vector<double> t = (*dfs[0]["t"]);
            std::vector<double> xk(N_simulations);

            std::vector<double> quantiles = arange(0.05, 0.95, 0.05);

            std::vector<std::vector<size_t>> q_trajectories(quantiles.size());
            for (auto &traj: q_trajectories) {
                traj.resize(N_rows);
            }
            std::cout << "Computing SIS-Quantiles..." << std::endl;
#pragma omp parallel for
            for (int i = 0; i < q_trajectories.size(); i++) {
                DataFrame df;
                df.assign("t", t);
                df.assign("S", dataframe_quantiles(dfs, "S", quantiles[i]));
                df.assign("I", dataframe_quantiles(dfs, "I", quantiles[i]));
                df.assign("p_I", dataframe_quantiles(dfs, "p_I", quantiles[i]));
                df.write_csv(FROLS_DATA_DIR + std::string("/Bernoulli_SIS_MC_Quantiles_") + std::to_string(N_pop) + "_" + std::to_string(p_ER) + "_" + std::to_string(quantiles[i]) +".csv", ",");
            }
        }
    }
}

#endif