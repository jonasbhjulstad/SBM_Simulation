#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <iostream>
int main() {
  size_t N_sims = 100; // 10000;
  size_t N_pop = 60;
  double p_ER = 1.0;
  using namespace FROLS;
  std::vector<std::string> df_names(N_sims);

  for (int i = 0; i < N_sims; i++) {
    df_names[i] = MC_sim_filename(N_pop, p_ER, i);
  }

  DataFrameStack dfs(df_names);
  Mat X = dataframe_to_matrix(dfs, {"S", "I", "R"},
                               0, -2);
  Mat Y = dataframe_to_matrix(dfs, {"S", "I", "R"}, 1, -1);
  Mat U = dataframe_to_matrix(dfs, {"p_I"}, 0, -2);

  std::cout << X << std::endl;

  size_t d_max = 1;
  size_t N_output_features = 16;
  using namespace FROLS::Features;
  size_t Nx = X.cols();
  size_t Nu = U.cols();

  FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
  double ERR_tol = 1e-1;
  Regression::ERR_Regressor regressor(ERR_tol);
  regressor.transform_fit(X, U, Y, model);

  Vec x0 = X.row(0);
  double u0 = U(0, 0);
  size_t Nt = 30;
  Vec u = Vec::Ones(Nt) * u0;
  // print x0, u
  std::cout << "x0 = " << x0.transpose() << std::endl;
  std::cout << "u0 = " << u0 << std::endl;
  model.feature_summary();
  model.simulate(x0, u, Nt);

  return 0;
}