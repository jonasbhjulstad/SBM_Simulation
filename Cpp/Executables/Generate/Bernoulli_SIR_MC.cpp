
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <functional>
#include <FROLS_Thread.hpp>
constexpr size_t N_configurations = 4;
constexpr size_t Nt = 20;

int main() {
    using namespace FROLS;
    std::vector<MC_SIR_Params> params(N_configurations);
//    params[1].N_pop = 20;
//    params[2].p_ER = 0.1;
//    params[3].p_ER = 0.1;
//    params[3].N_pop = 20;
    for (int i = 0; i < (N_configurations-1); i++)
    {
        params[i+1].iter_offset = params[i].iter_offset + params[i].N_sim;
    }
    std::random_device rd;
    for (int i = 0; i < N_configurations; i++) {
        params[i].seed = rd();
    }
    using namespace std::placeholders;
    std::atomic<size_t> n = 0;
    std::for_each(std::execution::par_unseq, params.begin(), params.end(),
                  [&n] (const auto &p) mutable { MC_SIR_to_file<Nt>(p, n++); });
    const std::vector<std::string> colnames = {"S", "I", "R"};

    std::for_each(std::execution::par_unseq, params.begin(), params.end(),
                  [&](const auto &p) {
                        auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
                        auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
                      quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);
                  });

    return 0;
}