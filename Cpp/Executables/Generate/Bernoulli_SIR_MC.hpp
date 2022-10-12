#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <DataFrame.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Eigen.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <graph_lite.h>


namespace FROLS {
    struct MC_SIR_Params {
        size_t N_pop = 100;
        double p_ER = 1.0;
        double p_I0 = 0.2;
        double p_R0 = 0.0;
        double R0_max = 1.6;
        double R0_min = 0.0;
        double alpha = 0.1;
        double p_I = 0;
        size_t N_sim = 10;
        size_t Nt_min = 15;
        double p_R = 0.1;
        size_t seed;
        size_t N_I_min = N_pop / 15;
        size_t iter_offset = 0;
        double csv_termination_tol = 0.;
    };

    template<typename RNG, size_t Nt>
    std::array<Network_Models::SIR_Param, Nt> generate_interaction_probabilities(const MC_SIR_Params &p, RNG &rng) {
        std::array<Network_Models::SIR_Param, Nt> param_vec;
        double omega_bounds[] = {(2 * M_PI) / 5, (2 * M_PI) / 100};
#ifdef FROLS_USE_DPCPP
        using oneapi::dpl::uniform_real_distribution;
#else
        using std::uniform_real_distribution;
#endif
        uniform_real_distribution<double> d_omega(omega_bounds[0], omega_bounds[1]);
        uniform_real_distribution<double> d_offset(0, 2 * M_PI);
        double offset = d_offset(rng);
        double R0_mean = (p.R0_max - p.R0_min) / 2 + p.R0_min;
        double R0_std = R0_mean - p.R0_min;
        double omega = d_omega(rng);
        std::array<double, Nt> beta;
        std::for_each(param_vec.begin(), param_vec.end(), [&, t = 0](auto &p_SIR) mutable {
            double R0 = R0_mean + R0_std * sin(omega * t + offset);
//            p_SIR.p_I = 1 - exp(-R0*p.alpha/p.N_pop);
            p_SIR.p_I = R0 / p.N_pop;
            p_SIR.p_R = 1 - exp(-p.alpha);
            t++;
        });
        return param_vec;
    }

    template<size_t Nt>
    struct MC_SIR_SimData {
        std::array<std::array<size_t, Nt + 1>, 3> traj;
        std::array<Network_Models::SIR_Param, Nt> p_vec;
    };

    template<size_t Nt, size_t NV, size_t NE>
    MC_SIR_SimData<Nt>
    MC_SIR_simulation(const Network_Models::SIR_Graph<NV, NE> &G_structure, const MC_SIR_Params &p, size_t seed) {
#ifdef FROLS_USE_DPCPP
        oneapi::dpl::ranlux48 generator(seed);
#else
        std::mt19937 generator(seed);
#endif
        Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(G_structure, p.p_I0, p.p_R0,
                                                                                 generator);
        MC_SIR_SimData<Nt> data;
        G.reset();
        while (G.population_count()[1] == 0) {
            G.initialize();
        }

        data.p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
        data.traj = G.simulate(data.p_vec);
        return data;
    }

    template<size_t Nt, size_t NV, size_t NE>
    void MC_SIR_to_file(const MC_SIR_Params &p, size_t thread_id) {

        static std::thread::id thread_0 = std::this_thread::get_id();
        thread_local std::mt19937 generator(p.seed);
        FROLS::DataFrame df;
        FROLS::DataFrame delta_df;
        std::vector<std::string> colnames = {"S", "I", "R"};

        thread_local Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(p.N_pop, p.p_ER, p.p_I0,
                                                                                              p.p_R0, generator);

        for (size_t i = 0; i < p.N_sim; i++) {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim / 10))) {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while (G.population_count()[1] == 0) {
                G.initialize();
            }
            auto p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
            auto traj = G.simulate(p_vec);
            std::array<double, Nt> p_Is;
            std::array<double, Nt> p_Rs;
            std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
            std::transform(p_vec.begin(), p_vec.end(), p_Is.begin(), [](const auto &pv) { return pv.p_I; });
            df.assign("S", traj[0]);
            df.assign("I", traj[1]);
            df.assign("R", traj[2]);
            df.assign("p_I", p_Is);
            df.assign("p_R", p_Rs);
            df.assign("t", FROLS::range(0, Nt + 1));
            df.write_csv(MC_filename(p.N_pop, p.p_ER, i + p.iter_offset, "SIR"),
                         ",", p.csv_termination_tol);


            delta_df.assign("S", diff<Nt + 1, size_t, int>(traj[0]));
            delta_df.assign("I", diff<Nt + 1, size_t, int>(traj[1]));
            delta_df.assign("R", diff<Nt + 1, size_t, int>(traj[2]));
            delta_df.assign("p_I", p_Is);
            delta_df.assign("p_R", p_Rs);
            delta_df.assign("t", FROLS::range(0, Nt));
            delta_df.write_csv(MC_filename(p.N_pop, p.p_ER, i + p.iter_offset, "SIR_Delta"),
                               ",", p.csv_termination_tol);

        }
    }

    template<size_t Nt, size_t NV, size_t NE>
    Mat MC_SIR_to_Mat(const MC_SIR_Params &p) {

        static std::thread::id thread_0 = std::this_thread::get_id();
        thread_local std::mt19937 generator(p.seed);
        std::vector<Mat> Xi_vec;
        thread_local Network_Models::SIR_Bernoulli_Network<decltype(generator), Nt, NV, NE> G(p.N_pop, p.p_ER, p.p_I0,
                                                                                              p.p_R0, generator);
        for (size_t i = 0; i < p.N_sim; i++) {
            if ((std::this_thread::get_id() == thread_0) && !(i % (p.N_sim / 10))) {
                std::cout << "Simulation " << i << " of " << p.N_sim << std::endl;
            }
            G.reset();
            while (G.population_count()[1] == 0) {
                G.initialize(p.p_I0, p.p_R0);
            }
            auto p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
            auto traj = G.simulate(p_vec, p.N_I_min, p.Nt_min);
            auto [p_I_vec, p_R_vec] = FROLS::unzip(p_vec);
            std::array<double, Nt> p_Is;
            std::transform(p_vec.begin(), p_vec.end(), p_Is.begin(), [](const auto &pv) { return pv.p_I; });
            Mat Xi(Nt, 4);
            Xi.leftCols(3) = vecs_to_mat(traj);
            Xi.col(4) = Eigen::Map<Vec>(p_I_vec.data(), p_I_vec.size());
            Xi_vec.emplace_back(Xi);
        }
        size_t N_rows = 0;
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto &xi) { N_rows += xi.rows(); });
        Mat X(N_rows, 4);
        std::for_each(Xi_vec.begin(), Xi_vec.end(), [&](auto &xi) { X << xi; });
        return X;
    }

}

#endif