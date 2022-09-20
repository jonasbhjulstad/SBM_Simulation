
#include "Bernoulli_SIR_MC.hpp"

constexpr size_t N_configurations = 4;

int main() {
    using namespace FROLS;
    std::vector<MC_SIR_Params> params(N_configurations);
    params[1].N_pop = 20;
    params[2].p_ER = 0.1;
    params[3].p_ER = 0.1;
    params[3].N_pop = 20;
    std::random_device rd;
    for (int i = 0; i < N_configurations; i++) {
        params[i].Nt_max = 100;
        params[i].seed = rd();
    }

//    std::vector<size_t> iter_offset = FROLS::range(0, N_configurations);
//    for (int i = 0; i < N_configurations; i++)
//    {
//        iter_offset[i] *= params[0].N_sim;
//        params[i].iter_offset = iter_offset[i];
//    }

    std::for_each(std::execution::par_unseq, params.begin(), params.end(),
                  [](const auto &p) { MC_SIR_to_file(FROLS_DATA_DIR, p); });
    std::for_each(std::execution::par_unseq, params.begin(), params.end(),
                  [](const auto &p) {
                      compute_SIR_quantiles(p.N_sim, 20, p.N_pop,
                                            p.p_ER);
                  });


    return 0;
}