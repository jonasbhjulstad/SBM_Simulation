#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <FROLS_Random.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Eigen.hpp>
#include <Regressor.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <utility>
#include <FROLS_Eigen.hpp>
namespace FROLS
{
    template <typename dType = float>
    struct MC_SIR_Params
    {
        uint32_t N_pop = 100;
        dType p_ER = 1.0f;
        dType p_I0 = 0.1f;
        dType p_R0 = 0.0f;
        dType R0_max = 1.6f;
        dType R0_min = 0.0f;
        dType alpha = 0.1f;
        dType p_I = 0.f;
        uint32_t N_sim = 1000;
        uint32_t Nt_min = 15;
        dType p_R = 0.1f;
        uint32_t seed;
        uint32_t N_I_min = N_pop / 10;
        uint32_t iter_offset = 0;
        dType csv_termination_tol = 1e-6f;
    };
}
#include "Bernoulli_SIR_File.hpp"

namespace FROLS
{
    // using Vector_SIR_Bernoulli_Network=  Network_Models::Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float>;

    template <typename RNG, typename dType = float>
    std::vector<dType> generate_infection_probabilities(const MC_SIR_Params<> &p, RNG &rng, uint32_t Nt)
    {
        std::vector<dType> p_Is(Nt);
        dType omega_bounds[] = {(2.f * M_PIf) / 2000.f, (2.f * M_PIf) / 100.f};
        random::uniform_real_distribution<dType> d_omega(omega_bounds[0], omega_bounds[1]);
        random::uniform_real_distribution<dType> d_offset(0.f, 2.f * M_PIf);
        random::uniform_real_distribution<dType> d_white(0.f, p.R0_max);
        dType offset = d_offset(rng);
        dType R0_mean = (p.R0_max - p.R0_min) / 2.f + p.R0_min;
        dType R0_std = (p.R0_max - p.R0_min)/2;
        dType omega = d_omega(rng);
        // omega = 0;
        std::vector<dType> beta(Nt);
        R0_mean = 1.5;
        std::exponential_distribution<dType> d_exp(1);
        dType p_I_val = R0_mean*d_exp(rng)/p.N_pop;
        std::for_each(p_Is.begin(), p_Is.end(), [&, t = 0](auto &p_I) mutable
                      {
            dType R0 = R0_mean + R0_std * std::sin(omega * t + offset);
            // p_I = std::max({R0 / p.N_pop, (dType) 0});//1 - std::exp(-R0/ p.N_pop);
            p_I = p_I_val;
            if ((t % 7) == 0)
            {
                p_I_val = R0_mean*d_exp(rng)/p.N_pop;
            }
            t++; });
        return p_Is;
    }

    template <typename RNG, typename dType = float>
    std::vector<Network_Models::SIR_Param<>> generate_interaction_probabilities(const MC_SIR_Params<> &p_gen, RNG &rng, uint32_t Nt)
    {
        std::vector<Network_Models::SIR_Param<>> param_vec(Nt);
        auto p_Is = generate_infection_probabilities(p_gen, rng, Nt);
        std::transform(p_Is.begin(), p_Is.end(), param_vec.begin(), [&](auto p_I)
                       {
            Network_Models::SIR_Param<> p;
            p.p_R = 1 - std::exp(-p_gen.alpha);
            p.p_I = p_I; 
            return p; });
        return param_vec;
    }

    template <typename dType = float>
    std::vector<Network_Models::SIR_Param<>> fixed_interaction_probabilities(const MC_SIR_Params<> &p, const std::vector<float> &p_Is, uint32_t Nt)
    {
        std::vector<Network_Models::SIR_Param<>> param_vec(Nt);
        std::for_each(std::execution::par_unseq, param_vec.begin(), param_vec.end(), [&, t = 0](auto &p_SIR) mutable
                      {
            p_SIR.p_I = p_Is[t];
            p_SIR.p_R = 1 - std::exp(-p.alpha);
            t++; });
        return param_vec;
    }
    struct MC_SIR_VectorData
    {
        MC_SIR_VectorData() {}
        MC_SIR_VectorData(const std::vector<Network_Models::SIR_Param<>>& p_vec,const std::vector<std::vector<uint32_t>>& t)
        {
            traj.resize(t[0].size(), t.size());
            for (int i = 0;i < t.size(); i++)
            {
                for (int j = 0; j < t[i].size(); j++)
                {
                    traj(j,i) = t[i][j];
                }
            }
            //put p_vec into params
            p_I.resize(p_vec.size(), 1);
            for (int i = 0; i < p_vec.size(); i++)
                p_I(i) = p_vec[i].p_I;
        }
        Mat traj;
        Vec p_I;
    };

    Regression::Regression_Data MC_SIR_to_regression_data(const std::vector<MC_SIR_VectorData>& data)
    {
        Regression::Regression_Data reg_data(data.size());
        for (int i = 0; i < data.size(); i++)
        {
            reg_data.U[i] = data[i].p_I;
            reg_data.X[i] = data[i].traj.topRows(data[i].traj.rows() - 1);
            reg_data.Y[i] = data[i].traj.bottomRows(data[i].traj.rows() - 1);
        }
        return reg_data;
    }

    // void dataframe_to_regression_data(DataFrameStack& dfs, const std::vector<std::string>& colnames_x, const std::vector<std::string> colnames_u, Regression_Data& reg_data)
    // {
    //     uint32_t N_dfs = dfs.dataframes.size();
    //     for (int i = 0; i < N_dfs; i++)
    //     {
    //         reg_data.U[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_u, 0, -2);
    //         reg_data.X[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_x, 0, -2);
    //         reg_data.Y[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_x, 1, -1);
    //     }
    //     return reg_data;
    // }

    Network_Models::SIR_VectorGraph generate_SIR_ER_graph(uint32_t N_pop, float p_ER, uint32_t seed)
    {
        using namespace Network_Models;
        uint32_t NV = N_pop;
        size_t nk = FROLS::n_choose_k(NV, 2);
        uint32_t NE = 1.5 * nk;

        std::mt19937_64 rng(seed);
        std::vector<std::shared_ptr<std::mutex>> v_mx(NV + 1);
        // create mutexes
        for (auto &mx : v_mx)
        {
            mx = std::make_shared<std::mutex>();
        }
        std::vector<std::shared_ptr<std::mutex>> e_mx(NE + 1);
        // create mutexes
        for (auto &mx : e_mx)
        {
            mx = std::make_shared<std::mutex>();
        }

        SIR_VectorGraph G(v_mx, e_mx);
        generate_erdos_renyi<SIR_VectorGraph, decltype(rng)>(G, N_pop, p_ER, SIR_S, rng);
        return G;
    }

    Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float>
    generate_Bernoulli_SIR_Network(Network_Models::SIR_VectorGraph &G_structure, float p_I0, uint32_t seed, float p_R0 = 0.f)
    {
        random::default_rng generator(seed);
        Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> G(G_structure, p_I0, p_R0,
                                                                                   generator);
        G.reset();
        G.initialize();
        return G;
    }

    MC_SIR_VectorData
    MC_SIR_simulation(Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> &G, uint32_t seed, const MC_SIR_Params<float> &p, uint32_t Nt)
    {
        random::default_rng generator(seed);
        auto p_vec = generate_interaction_probabilities(p, generator, Nt);
        auto traj = G.simulate(p_vec, Nt, p.N_I_min, p.Nt_min);
        
        return MC_SIR_VectorData(p_vec, traj);
    }

    MC_SIR_VectorData
    MC_SIR_simulation(Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> &G, const MC_SIR_Params<> &p, uint32_t seed, const std::vector<float> &p_Is)
    {
        uint32_t Nt = p_Is.size();
        random::default_rng generator(seed);
        auto p_vec = fixed_interaction_probabilities<decltype(generator)>(p, p_Is, Nt);
        auto traj = G.simulate(p_vec, Nt, p.N_I_min, p.Nt_min);
        return MC_SIR_VectorData(p_vec, traj);
    }
    std::vector<MC_SIR_VectorData>
    MC_SIR_simulations(Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> &G, const MC_SIR_Params<> &p_gen, const std::vector<uint32_t> &seeds, uint32_t Nt)
    {
        uint32_t N_sims = seeds.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        random::default_rng rng(seeds[0]);
        std::transform(seeds.begin(), seeds.end(), data_vec.begin(), [&](const auto &seed)
                       { return MC_SIR_simulation(G, seed, p_gen, Nt); });
        return data_vec;
    }

    std::vector<MC_SIR_VectorData>
    MC_SIR_simulations(Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> &G, const MC_SIR_Params<> &p, const std::vector<uint32_t> &seeds, const std::vector<float> &p_Is, uint32_t N_sims)
    {
        uint32_t Nt = p_Is.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        std::transform(seeds.begin(), seeds.end(), data_vec.begin(), [&](const auto &seed)
                       { return MC_SIR_simulation(G, p, seed, p_Is); });
        return data_vec;
    }

    MC_SIR_VectorData
    MC_SIR_simulation(Network_Models::SIR_VectorGraph &G_structure, const MC_SIR_Params<> &p, uint32_t seed, const std::vector<float> &p_Is)
    {
        uint32_t Nt = p_Is.size();
        random::default_rng generator(seed);
        auto G = generate_Bernoulli_SIR_Network(G_structure, p.p_I0, seed, p.p_R0);
        return MC_SIR_simulation(G, p, seed, p_Is);
    }

    std::vector<MC_SIR_VectorData>
    MC_SIR_simulations(Network_Models::SIR_VectorGraph &G_structure, const MC_SIR_Params<> &p, const std::vector<uint32_t> &seeds, uint32_t Nt)
    {
        uint32_t N_sims = seeds.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        std::for_each(data_vec.begin(), data_vec.end(), [&, n = 0](auto &data) mutable
                      {
            Network_Models::SIR_VectorGraph G_copy = G_structure;
            auto G = generate_Bernoulli_SIR_Network(G_structure, p.p_I0, seeds[n], p.p_R0);
            data = MC_SIR_simulation(G, seeds[n], p, Nt);
            n++; });
        return data_vec;
    }


    Regression::Regression_Data
    MC_SIR_simulations_to_regression(Network_Models::SIR_VectorGraph &G_structure, const MC_SIR_Params<> &p, const std::vector<Network_Models::SIR_Param<>>& p_vec, const std::vector<uint32_t> &seeds, uint32_t Nt)
    {
        uint32_t N_sims = seeds.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        std::transform(seeds.begin(), seeds.end(), data_vec.begin(), [&](const auto &seed) mutable
                      {
            Network_Models::SIR_VectorGraph G_copy = G_structure;
            Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> G(G_copy, p.p_I0, p.p_R0, random::default_rng(seed));
            
            auto traj = G.simulate(p_vec, Nt, p.N_I_min, p.Nt_min);
            return MC_SIR_VectorData(p_vec, traj);
            ; });


        return MC_SIR_to_regression_data(data_vec);
    }

    Regression::Regression_Data
    MC_SIR_simulations_to_regression(Network_Models::SIR_VectorGraph &G_structure, const MC_SIR_Params<> &p, const std::vector<uint32_t> &seeds, uint32_t Nt)
    {
        uint32_t N_sims = seeds.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        std::for_each(data_vec.begin(), data_vec.end(), [&, n = 0](auto &data) mutable
                      {
            Network_Models::SIR_VectorGraph G_copy = G_structure;
            Network_Models::Vector_SIR_Bernoulli_Network<random::default_rng, float> G(G_copy, p.p_I0, p.p_R0, random::default_rng(seeds[n]));
            
            data = MC_SIR_simulation(G, seeds[n], p, Nt);
            n++; });
        return MC_SIR_to_regression_data(data_vec);
    }

    Regression::Regression_Data
    MC_SIR_simulations_to_regression(const MC_SIR_Params<> &p, const std::vector<uint32_t> &seeds, uint32_t Nt)
    {
        uint32_t N_sims = seeds.size();
        std::vector<MC_SIR_VectorData> data_vec(N_sims);
        auto G_structure = generate_SIR_ER_graph(p.N_pop, p.p_ER, seeds[0]);
        std::for_each(data_vec.begin(), data_vec.end(), [&, n = 0](auto &data) mutable
                      {
            auto G = generate_Bernoulli_SIR_Network(G_structure,  p.p_I0, seeds[n], p.p_R0);
            data = MC_SIR_simulation(G, seeds[n], p, Nt);
            n++; });



        
        
        return MC_SIR_to_regression_data(data_vec);
    }


    template <typename dType = float>
    void traj_to_file(const FROLS::MC_SIR_Params<> &p, const FROLS::MC_SIR_VectorData &d, uint32_t iter, uint32_t Nt)
    {
        // print iter
        FROLS::DataFrame df;
        std::vector<dType> p_Is(Nt);
        //put d.p_I into p_Is
        std::transform(d.p_I.begin(), d.p_I.end(), p_Is.begin(), [](const auto &p_I)
                       { return p_I; });
        std::vector<dType> p_Rs(Nt + 1);
        std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
        p_Is.back() = 0.;
        df.assign("S", d.traj.col(0));
        df.assign("I", d.traj.col(1));
        df.assign("R", d.traj.col(2));
        df.assign("p_I", p_Is);
        df.assign("p_R", p_Rs);
        auto t = FROLS::range(0, Nt + 1);
        df.assign("t", t);
        df.resize(Nt + 1);
        df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                     ",", p.csv_termination_tol);
    }

    std::vector<MC_SIR_VectorData> MC_SIR_simulations(uint32_t N_pop, float p_ER, float p_I0, const std::vector<uint32_t> &seeds, std::vector<float> p_Is, uint32_t Nt, uint32_t N_sims)
    {
        using namespace FROLS;
        using namespace Network_Models;
        auto G = generate_SIR_ER_graph(N_pop, p_ER, seeds[0]);
        auto enum_seeds = enumerate(seeds);
        std::vector<MC_SIR_VectorData> simdatas(seeds.size());
        MC_SIR_Params<> p;
        p.N_pop = N_pop;
        p.p_ER = p_ER;
        p.p_I0 = p_I0;
        p.p_R0 = 0.f;
        p.p_R = 0.f;
        p.N_sim = N_sims;
        std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto &es)
                       {
            uint32_t iter = es.first;
            uint32_t seed = es.second;
            if ((iter % (p.N_sim / 10)) == 0)
            {
                std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
            }
            return MC_SIR_simulation(G, p, seed, p_Is); });
        return simdatas;
    }

    std::vector<MC_SIR_VectorData> MC_SIR_simulations(uint32_t N_pop, float p_ER, float p_I0, const std::vector<uint32_t> &seeds, uint32_t Nt, uint32_t N_sims)
    {
        using namespace FROLS;
        using namespace Network_Models;

        FROLS::random::default_rng rng(seeds[0]);

        auto G = generate_SIR_ER_graph(N_pop, p_ER, seeds[0]);
        std::vector<MC_SIR_VectorData> simdatas(seeds.size());
        MC_SIR_Params<> p;
        p.N_pop = N_pop;
        p.p_ER = p_ER;
        p.p_I0 = p_I0;
        p.p_R0 = 0.f;
        p.p_R = 0.f;
        p.N_sim = N_sims;

        auto MC_params = generate_interaction_probabilities(p, rng, Nt);
        std::vector<float> p_Is(Nt);
        std::transform(MC_params.begin(), MC_params.end(), p_Is.begin(), [](auto &p)
                       { return p.p_I; });
        std::transform(seeds.begin(), seeds.end(), simdatas.begin(), [&, iter = 0](auto &seed) mutable
                       {
            // if ((iter % (p.N_sim / 10)) == 0)
            // {
            //     std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
            // }
            iter++;
            return MC_SIR_simulation(G, p, seed, p_Is); });
        return simdatas;
    }

} // namespace FROLS

#endif
