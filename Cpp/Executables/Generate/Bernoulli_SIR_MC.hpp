#ifndef BERNOULLI_SIR_MC_HPP
#define BERNOULLI_SIR_MC_HPP
#include <FROLS_DataFrame.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_quantiles.hpp>
#include <Models/SIR_Bernoulli_Network.hpp>
#include <bits/stdc++.h>
#include <omp.h>
namespace FROLS
{
struct MC_SIR_Params {
  double N_pop = 60;
  double p_ER = 1.0;
  double p_I0 = 0.2;
  double p_I_min = 1e-8;
  double p_I_max = .1;
  size_t N_sim = 10000;
  size_t Nt = 100;
  double p_R = 0.1;
};

std::vector<double> linspace(double min, double max, int N)
{
  std::vector<double> res(N);
  for (int i = 0; i < N; i++)
  {
    res[i] = min + (max - min) * i / (N - 1);
  }
  return res;
}

std::vector<double> arange(double min, double max, double step)
{
  double s = min;
  std::vector<double> res;
  while(s <= max)
  {
    res.push_back(s);
    s += step;
  }
  return res;
}

static_assert(IGRAPH_THREAD_SAFE);


inline std::string MC_sim_filename(size_t N_pop, double p_ER, size_t idx) {
  return FROLS_DATA_DIR + std::string("/Bernoulli_SIR_MC_") +
         std::to_string(idx) + "_" + std::to_string(N_pop) + "_" +
         std::to_string(p_ER) + ".csv";
}

inline std::string quantile_filename(size_t N_pop, double p_ER, double tau) {
  return FROLS_DATA_DIR + std::string("/Bernoulli_SIR_MC_Quantiles_") +
         std::to_string(N_pop) + "_" + std::to_string(p_ER) + "_" +
         std::to_string(tau) + ".csv";
}

void MC_SIR_to_file(const std::string fPath, const MC_SIR_Params &p) {
  int thread_id = omp_get_thread_num();
  std::random_device rd;
  auto seed = rd();
  // std::cout << "Thread " << thread_id << " initialized with seed: " << seed
  // << std::endl;
  std::mt19937 generator(seed);
  std::vector<double> p_I(p.Nt), p_R(p.Nt);
  std::fill(p_R.begin(), p_R.end(), p.p_R);
  // Sample alphas and betas from a uniform distribution
  //  std::uniform_real_distribution<double> alpha_dist();
  std::uniform_real_distribution<double> beta_dist(p.p_I_min, p.p_I_max);

  // Run SIR_simulations
  igraph_t G;
  std::vector<size_t> idx(p.Nt + 1);
  std::vector<size_t> t(p.Nt + 1);
  std::generate(t.begin(), t.end(), [n = 0]() mutable { return n++; });
  auto state = generate_SIR_ER_model(G, p.N_pop, p.p_ER, p.p_I0, generator);
  FROLS::DataFrame df(p.Nt);
  std::vector<std::string> colnames = {"S", "I", "R"};
  df.assign("t", t);

  df.assign("p_R", p_R);
  for (size_t i = 0; i < p.N_sim; i++) {
    double p_I_const = beta_dist(generator);
    std::fill(p_I.begin(), p_I.end(), p_I_const);
    auto traj = run_SIR_simulation(G, state, p.Nt, p_I, p_R, generator);
    df.assign(colnames, traj);
    df.assign("p_I", p_I);

    df.write_csv(MC_sim_filename(p.N_pop, p.p_ER, p.N_sim * thread_id + i),
                 ",");
  }

  igraph_destroy(&G);

  std::cout << "Thread " << thread_id << " finished" << std::endl;
}

void compute_SIR_quantiles(size_t N_simulations, size_t N_tau, size_t N_pop,
                           double p_ER) {
  std::vector<std::string> filenames(N_simulations);
#pragma omp parallel for
  for (int i = 0; i < N_simulations; i++) {
    filenames[i] = MC_sim_filename(N_pop, p_ER, i);
  }
  using namespace FROLS;
  DataFrameStack dfs(filenames);
  size_t N_rows = dfs[0].get_N_rows();
  std::vector<double> t = (*dfs[0]["t"]);
  std::vector<double> xk(N_simulations);

  std::vector<double> quantiles = arange(0.05, 0.95, 0.05);

  std::vector<SIR_Trajectory> q_trajectories(quantiles.size());
  for (auto &traj : q_trajectories) {
    traj.resize(N_rows);
  }
  std::cout << "Computing SIR-Quantiles..." << std::endl;
  std::vector<std::string> q_names(quantiles.size());
#pragma omp parallel for
  for (int i = 0; i < quantiles.size(); i++) {
    q_names[i] = quantile_filename(N_pop, p_ER, quantiles[i]);
    q_trajectories[i].S = dataframe_quantiles(dfs, "S", quantiles[i]);
    q_trajectories[i].I = dataframe_quantiles(dfs, "I", quantiles[i]);
    q_trajectories[i].R = dataframe_quantiles(dfs, "R", quantiles[i]);
    q_trajectories[i].p_I = dataframe_quantiles(dfs, "p_I", quantiles[i]);
  }

  std::cout << "Writing SIR_Quantiles.." << std::endl;
  for (int i = 0; i < q_trajectories.size(); i++) {
    DataFrame df;
    df.assign("t", t);
    df.assign("S", q_trajectories[i].S);
    df.assign("I", q_trajectories[i].I);
    df.assign("R", q_trajectories[i].R);
    df.assign("p_I", q_trajectories[i].p_I);
    df.write_csv(q_names[i], ",");
  }
}
}


#endif