//
// Created by arch on 9/17/22.
//
#include <Typedefs.hpp>
#include <Integrators/SIR_Integrators.hpp>
#include <random>
#include <DataFrame.hpp>

#include <FROLS_Path_Config.hpp>
#include <FROLS_Math.hpp>
#include <Typedefs.hpp>

constexpr uint32_t
        Nt = 100;

std::string SIR_Sine_Discrete_filename(uint32_t
                              sim_iter) {
    return FROLS::FROLS_DATA_DIR + std::string("/SIR_Sine_Trajectory_Discrete_") +
           std::to_string(sim_iter)
           + ".csv";
}

FROLS::Integrators::SIR_Sine_Param
sineparam_gen(uint32_t
              seed) {
    std::mt19937 rng(seed);
    FROLS::Integrators::SIR_Sine_Param p;
    float R0 = 2;
    p.
            alpha = .11;
    p.N_pop = 100000;

    p.
            beta = R0*p.alpha/p.N_pop;
    std::uniform_real_distribution<float> d_u((2 * M_PI) / 100, (2 * M_PI) / 1000);
    p.
            omega = d_u(rng);
    std::uniform_real_distribution<float> d_pi(0, 2 * M_PI);
    p.
            offset = d_pi(rng);

    std::uniform_real_distribution<float> d_amp(p.beta / 5, p.beta / 1);
    p.
            amplitude = d_amp(rng);
    return
            p;
}


int main() {
    using namespace FROLS;
    using namespace FROLS::Integrators;
    using Trajectory = SIR_Sine<Nt>::Trajectory;
    float dt = 1.0;

    uint32_t N_sim = 10000;
    std::vector<uint32_t> seeds(N_sim);
    std::random_device rd;
    std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
    std::vector<std::string> colnames = {"S", "I", "R"};
    typedef std::vector<std::vector<float>> vec_Trajectory;
    std::for_each(seeds.begin(), seeds.end(), [&](const uint32_t seed) {
        static int iter = 0;
        SIR_Sine_Param p = sineparam_gen(seed);
        const std::array<float, 3> x0 = {p.N_pop*9.f/10.f, p.N_pop/10.f, 0.f};
        Integrators::SIR_Stochastic<Nt> model(x0, p, dt);
        auto [traj, t] = model.simulate(x0);
        std::vector<std::vector<float>> result(Nt + 1);
        DataFrame df;
        auto p_I = t;
        std::transform(t.begin(), t.end(), p_I.begin(),
                       [&](auto ti) { return FROLS::Integrators::SIR_sine_param_gen((void *) &p, ti).beta; });
        df.assign("p_I", p_I);
        df.assign("t", std::vector<float>(t.data(), t.end()));
        df.assign(colnames, traj);
        df.write_csv(SIR_Sine_Discrete_filename(iter), ",");
        iter++;
    });
    std::vector<std::string> fnames(N_sim);
    auto range = FROLS::range(0, N_sim);
    std::transform(range.begin(), range.end(), fnames.begin(), [&](const uint32_t &i) {
        return SIR_Sine_Discrete_filename(i);
    });

    return EXIT_SUCCESS;
}