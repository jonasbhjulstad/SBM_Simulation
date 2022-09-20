#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <DataFrame.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Eigen.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <graph_lite.h>
#include <omp.h>

namespace FROLS {
    struct MC_SIR_Params {
        double N_pop = 80;
        double p_ER = 1.0;
        double p_I0 = 0.2;
        double p_R0 = 0.1;
        double p_I_max = .2;
        double p_I_min = .0;
        size_t N_sim = 500;
        size_t Nt_max = 20;
        size_t Nt_min = 15;
        double p_R = 0.01;
        size_t seed;
        size_t infection_count_tolerance = N_pop/15;
        size_t iter_offset = 0;
    };
    template <typename RNG>
    std::vector<std::pair<double, double>> generate_interaction_probabilities(const MC_SIR_Params& p, RNG& rng)
    {
        double omega_bounds[] = {(2*M_PI)/100, (2*M_PI)/10};
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
            res[i] = std::make_pair(p_I, p.p_R);
        }
        return res;
    }


    void MC_SIR_to_file(const std::string &fPath, const MC_SIR_Params &p) {

        static std::thread::id thread_0 = std::this_thread::get_id();
        std::random_device rd;
        thread_local std::mt19937 generator(p.seed);
        FROLS::DataFrame df;
        std::vector<std::string> colnames = {"S", "I", "R"};

        thread_local Network_Models::SIR_Bernoulli_Network G(p.N_pop, p.p_ER, generator);

        for (size_t i = 0; i < p.N_sim; i++) {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim/10)))
            {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while(G.population_count()[1] == 0) {
                G.generate_initial_infections(p.p_I0, p.p_R0);
            }
            auto p_vec = generate_interaction_probabilities(p, generator);
            auto traj = G.simulate(p_vec, p.infection_count_tolerance, p.Nt_min);
            auto [p_I_vec, p_R_vec] = FROLS::unzip(p_vec);
            size_t Nt = traj[0].size();
            p_I_vec.resize(Nt);
            p_R_vec.resize(Nt);
            df.assign("S", traj[0]);
            df.assign("I", traj[1]);
            df.assign("R", traj[2]);
            df.assign("p_I", p_I_vec);
            df.assign("p_R", p_R_vec);
            df.assign("t", FROLS::range(0, Nt));
            df.write_csv(MC_sim_filename(p.N_pop, p.p_ER, p.iter_offset + i),
                         ",");
        }
    }

    Mat MC_SIR_to_Mat(const MC_SIR_Params &p) {

        static std::thread::id thread_0 = std::this_thread::get_id();
        std::random_device rd;
        thread_local std::mt19937 generator(p.seed);
        std::vector<Mat> Xi_vec;
        thread_local Network_Models::SIR_Bernoulli_Network G(p.N_pop, p.p_ER, generator);
        for (size_t i = 0; i < p.N_sim; i++) {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim/10)))
            {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while(G.population_count()[1] == 0) {
                G.generate_initial_infections(p.p_I0, p.p_R0);
            }
            auto p_vec = generate_interaction_probabilities(p, generator);
            auto traj = G.simulate(p_vec, p.infection_count_tolerance, p.Nt_min);
            auto [p_I_vec, p_R_vec] = FROLS::unzip(p_vec);
            size_t Nt = traj[0].size();
            p_I_vec.resize(Nt);
            Mat Xi(Nt, 4);
            Xi.leftCols(3) = vecs_to_mat(traj);
            Xi.col(4) = Eigen::Map<Vec>(p_I_vec.data(), p_I_vec.size());
            Xi_vec.emplace_back(Xi);
        }
        size_t N_rows = 0;
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto& xi){N_rows += xi.rows();});
        Mat X(N_rows, 4);
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto& xi){X << xi;});
        return X;
    }

    void compute_SIR_quantiles(size_t N_simulations, size_t N_tau, size_t N_pop,
                               double p_ER) {
        std::vector<std::string> filenames(N_simulations);
        {
            for (int i = 0; i < N_simulations; i++) {
                filenames[i] = MC_sim_filename(N_pop, p_ER, i);
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
            for (int i = 0; i < q_trajectories.size(); i++) {
                DataFrame df;
                df.assign("t", t);
                df.assign("S", dataframe_quantiles(dfs, "S", quantiles[i]));
                df.assign("I", dataframe_quantiles(dfs, "I", quantiles[i]));
                df.assign("R", dataframe_quantiles(dfs, "R", quantiles[i]));
                df.assign("p_I", dataframe_quantiles(dfs, "p_I", quantiles[i]));
                df.write_csv(FROLS_DATA_DIR + std::string("/Bernoulli_SIR_MC_Quantiles_") + std::to_string(N_pop) + "_" + std::to_string(p_ER) + "_" + std::to_string(quantiles[i]) +".csv", ",");
            }
        }
    }
}

#endif