
#include "Bernoulli_SIS_MC.hpp"
constexpr size_t N_configurations = 4;
int main() {
  using namespace FROLS;
  size_t N_threads = omp_get_max_threads();
  std::cout << "Running on " << N_threads << " threads" << std::endl;
  MC_SIS_Params params[N_configurations];
  params[1].N_pop = 20;
  params[2].p_ER = 0.1;
  params[3].p_ER = 0.1;
  params[3].N_pop = 20;
  std::random_device rd;
  for (int i = 0; i < N_configurations; i++)
  {
    params[i].Nt_max = 100;
    params[i].seed = rd();
  }
  omp_set_num_threads(N_threads);

  for (int i = 0; i < 1; i++) {
    #pragma omp parallel
    { MC_SIS_to_file(FROLS_DATA_DIR, params[i]); }
    compute_SIS_quantiles(params[i].N_sim_tot, 20, params[i].N_pop,
                          params[i].p_ER);
  }

  return 0;
}