#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Eigen.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <array>
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
    template <uint32_t NV, uint32_t NE, uint32_t Nt>
    using Array_SIR_Bernoulli_Network = Network_Models::Array_SIR_Bernoulli_Network<random::default_rng, NV, NE, Nt>;

    template <typename RNG, uint32_t Nt, typename dType = float>
    std::array<Network_Models::SIR_Param<>, Nt> generate_interaction_probabilities(const MC_SIR_Params<> &p, RNG &rng)
    {
        std::array<Network_Models::SIR_Param<>, Nt> param_vec;
        dType omega_bounds[] = {(2.f * M_PIf) / 5.f, (2.f * M_PIf) / 10.f};
        random::uniform_real_distribution<dType> d_omega(omega_bounds[0], omega_bounds[1]);
        random::uniform_real_distribution<dType> d_offset(0.f, 2.f * M_PIf);
        dType offset = d_offset(rng);
        dType R0_mean = (p.R0_max - p.R0_min) / 2.f + p.R0_min;
        dType R0_std = R0_mean - p.R0_min;
        dType omega = d_omega(rng);
        std::array<dType, Nt> beta;
        std::for_each(std::execution::par_unseq, param_vec.begin(), param_vec.end(), [&, t = 0](auto &p_SIR) mutable
                      {
            dType R0 = R0_mean + R0_std * std::sin(omega * t + offset);
//            p_SIR.p_I = 1 - exp(-R0*p.alpha/p.N_pop);
            p_SIR.p_I = R0/ p.N_pop;
            p_SIR.p_R = 1 - std::exp(-p.alpha);
            t++; });
        return param_vec;
    }

    template <typename RNG, uint32_t Nt, typename dType = float>
    std::array<Network_Models::SIR_Param<>, Nt> fixed_interaction_probabilities(const MC_SIR_Params<> &p, const std::array<float, Nt> &p_Is)
    {
        std::array<Network_Models::SIR_Param<>, Nt> param_vec;
        std::for_each(std::execution::par_unseq, param_vec.begin(), param_vec.end(), [&, t = 0](auto &p_SIR) mutable
                      {
//            p_SIR.p_I = 1 - exp(-R0*p.alpha/p.N_pop);
            p_SIR.p_I = p_Is[t];
            p_SIR.p_R = 1 - std::exp(-p.alpha);
            t++; });
        return param_vec;
    }
    template <uint32_t Nt>
    struct MC_SIR_ArrayData
    {
        std::array<std::array<uint32_t, Nt + 1>, 3> traj;
        std::array<Network_Models::SIR_Param<>, Nt> p_vec;
    };


    template <uint32_t NV, uint32_t NE, uint32_t Nt>
    MC_SIR_ArrayData<Nt>
    MC_SIR_simulation(Network_Models::SIR_ArrayGraph<NV, NE> &G_structure, const MC_SIR_Params<> &p, uint32_t seed)
    {

        random::default_rng generator(seed);
        Array_SIR_Bernoulli_Network<NV, NE, Nt> G(G_structure, p.p_I0, p.p_R0,
                                                                                    generator);
        MC_SIR_ArrayData<Nt> data;
        G.reset();
        while (G.population_count()[1] == 0)
        {
            G.initialize();
        }

        data.p_vec = generate_interaction_probabilities<decltype(generator), Nt>(p, generator);
        data.traj = G.simulate(data.p_vec);
        return data;
    }

    template <uint32_t NV, uint32_t NE, uint32_t Nt>
    MC_SIR_ArrayData<Nt>
    MC_SIR_simulation(Network_Models::SIR_ArrayGraph<NV, NE> &G_structure, const MC_SIR_Params<> &p, uint32_t seed, const std::array<float, Nt> &p_Is)
    {

        random::default_rng generator(seed);
        Array_SIR_Bernoulli_Network<NV, NE, Nt> G(G_structure, p.p_I0, p.p_R0,
                                                                                    generator);
        MC_SIR_ArrayData<Nt> data;
        G.reset();
        while (G.population_count()[1] == 0)
        {
            G.initialize();
        }

        data.p_vec = fixed_interaction_probabilities<decltype(generator), Nt>(p, p_Is);
        data.traj = G.simulate(data.p_vec);
        return data;
    }

    template <uint32_t Nt, typename dType = float>
    void traj_to_file(const FROLS::MC_SIR_Params<> &p, const FROLS::MC_SIR_ArrayData<Nt> &d, uint32_t iter)
    {
        // print iter
        FROLS::DataFrame df;
        std::array<dType, Nt + 1> p_Is;
        std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](auto &p)
                       { return p.p_I; });
        std::array<dType, Nt + 1> p_Rs;
        std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
        p_Is.back() = 0.;
        df.assign("S", d.traj[0]);
        df.assign("I", d.traj[1]);
        df.assign("R", d.traj[2]);
        df.assign("p_I", p_Is);
        df.assign("p_R", p_Rs);
        auto t = FROLS::range(0, Nt + 1);
        df.assign("t", t);
        df.resize(Nt + 1);
        df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                     ",", p.csv_termination_tol);
    }

    template <uint32_t NV, uint32_t NE>
    Network_Models::SIR_ArrayGraph<NV, NE> generate_SIR_ER_graph(uint32_t N_pop, float p_ER, uint32_t seed)
    {
        using namespace Network_Models;
        random::default_rng rng(seed);
        std::array<std::shared_ptr<std::mutex>, NV+1> v_mx;
        // create mutexes
        for (auto &mx : v_mx)
        {
            mx = std::make_shared<std::mutex>();
        }
        std::array<std::shared_ptr<std::mutex>, NE+1> e_mx;
        // create mutexes
        for (auto &mx : e_mx)
        {
            mx = std::make_shared<std::mutex>();
        }

        SIR_ArrayGraph<NV, NE> G(v_mx, e_mx);
        generate_erdos_renyi<decltype(G), decltype(rng)>(G, N_pop, p_ER, SIR_S, rng);
        return G;
    }

    template <uint32_t NV, uint32_t NE, uint32_t Nt>
    Array_SIR_Bernoulli_Network<NV, NE, Nt>
    generate_Bernoulli_SIR_Network(const Network_Models::SIR_ArrayGraph<NV, NE>& G_structure, uint32_t seed, float p_I0, float p_R0 = 0.f)
    {

        random::default_rng generator(seed);
        Array_SIR_Bernoulli_Network<NV, NE, Nt> G(G_structure, p_I0, p_R0,
                                                            generator);
        G.reset();
        while (G.population_count()[1] == 0)
        {
            G.initialize();
        }
        return G;
    }

    template <uint32_t NV, uint32_t NE, uint32_t Nt>
    Array_SIR_Bernoulli_Network<NV, NE, Nt>
    generate_Bernoulli_SIR_Network(uint32_t N_pop, float p_ER, float p_I0, uint32_t seed, float p_R0 = 0.f)
    {

        random::default_rng generator(seed);
        auto G_structure = generate_SIR_ER_graph<NV, NE>(N_pop, p_ER, seed);
        return generate_Bernoulli_SIR_Network(G_structure,p_I0, seed, p_R0);
    }

    template <uint32_t NV, uint32_t NE, uint32_t Nt, uint32_t N_sims>
    std::vector<MC_SIR_ArrayData<Nt>> Bernoulli_SIR_MC_Simulations(uint32_t N_pop, float p_ER, float p_I0, const std::array<uint32_t, N_sims> &seeds, std::array<float, Nt> p_Is)
    {
        using namespace FROLS;
        using namespace Network_Models;
        auto G = generate_Bernoulli_SIR_Network<NV, NE, Nt>(N_pop, p_ER, p_I0);
        auto enum_seeds = enumerate(seeds);
        std::vector<MC_SIR_ArrayData<Nt>> simdatas(seeds.size());
        std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto &es)
                       {
            uint32_t iter = es.first;
            uint32_t seed = es.second;
            if ((iter % (N_sims / 10)) == 0)
            {
                std::cout << "Simulation " << iter << " of " << N_sims << std::endl;
            }
            return MC_SIR_simulation<NV, NE, Nt>(G, MC_SIR_Params{}, seed, p_Is).traj; });
        return simdatas;
    }

}

#endif