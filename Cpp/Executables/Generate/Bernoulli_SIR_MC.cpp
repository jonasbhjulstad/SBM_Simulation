
#include "Bernoulli_SIR_MC.hpp"
constexpr size_t N_configurations = 4;
int main() {
  using namespace FROLS;
  size_t N_threads = omp_get_max_threads();
  MC_SIR_Params params[N_configurations];
  params[1].N_pop = 20;
  params[2].p_ER = 0.1;
  params[3].p_ER = 0.1;
  params[3].N_pop = 20;
  for (int i = 0; i < N_configurations; i++)
  {
    params[i].Nt = 20;
  }
  omp_set_num_threads(N_threads);

  for (int i = 0; i < N_configurations; i++) {
    #pragma omp parallel
    { MC_SIR_to_file(DATA_DIR, params[i]); }
    compute_SIR_quantiles(params[i].N_sim_tot, 20, params[i].N_pop,
                          params[i].p_ER);
  }

  return 0;
}