#ifndef FROLS_BERNOULLI_SIR_MC_HPP
#define FROLS_BERNOULLI_SIR_MC_HPP

#include <FROLS_Path_Config.hpp>
#include <FROLS_Random.hpp>
#include <quantiles.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Eigen.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <graph_lite.h>
namespace FROLS {
    template<typename dType=float>
    struct MC_SIR_Params{
        uint16_t N_pop = 100;
        dType p_ER = 1.0f;
        dType p_I0 = 0.01f;
        dType p_R0 = 0.0f;
        dType R0_max = 1.6f;
        dType R0_min = 0.0f;
        dType alpha = 0.1f;
        dType p_I = 0.f;
        uint16_t N_sim = 10;
        uint16_t Nt_min = 15;
        dType p_R = 0.8f;
        uint16_t seed;
        uint16_t N_I_min = N_pop / 15;
        uint16_t iter_offset = 0;
        dType csv_termination_tol = 0.f;
    };
}
#include "Bernoulli_SIR_File.hpp"

namespace FROLS {
    template<typename RNG, uint16_t Nt,typename dType=float>
    std::array<Network_Models::SIR_Param<>, Nt> generate_interaction_probabilities(const MC_SIR_Params<> &p, RNG &rng) {
        std::array<Network_Models::SIR_Param<>, Nt> param_vec;
        dType omega_bounds[] = {(2.f * M_PIf) / 5.f, (2.f * M_PIf) / 100.f};
        random::uniform_real_distribution<dType> d_omega(omega_bounds[0], omega_bounds[1]);
        random::uniform_real_distribution<dType> d_offset(0.f, 2.f * M_PIf);
        dType offset = d_offset(rng);
        dType R0_mean = (p.R0_max - p.R0_min) / 2.f + p.R0_min;
        dType R0_std = R0_mean - p.R0_min;
        dType omega = d_omega(rng);
        std::array<dType, Nt> beta;
        std::for_each(param_vec.begin(), param_vec.end(), [&, t = 0](auto &p_SIR) mutable {
            dType R0 = R0_mean + R0_std * std::sin(omega * t + offset);
//            p_SIR.p_I = 1 - exp(-R0*p.alpha/p.N_pop);
            p_SIR.p_I = R0 / p.N_pop;
            p_SIR.p_R = 1 - std::exp(-p.alpha);
            t++;
        });
        return param_vec;
    }

    template<uint16_t Nt>
    struct MC_SIR_SimData {
        std::array<std::array<uint16_t, Nt + 1>, 3> traj;
        std::array<Network_Models::SIR_Param<>, Nt> p_vec;
    };

    template<uint16_t Nt, uint16_t NV, uint16_t NE>
    MC_SIR_SimData<Nt>
    MC_SIR_simulation(const Network_Models::SIR_Graph<NV, NE> &G_structure, const MC_SIR_Params<>&p, uint16_t seed) {

        random::default_rng generator(seed);
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


}

#endif