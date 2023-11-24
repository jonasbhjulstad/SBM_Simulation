#define QT_FATAL_WARNINGS 1
#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/SBM_Simulation.hpp>
// #include <execution>
// #include <algorithm>
#include <thread>
#include <tom/tom_config.hpp>

auto get_N_p_out() {
  return Orm::DB::table("simulation_parameters")
      ->select("p_out_id")
      .whereEq("graph_id", 0)
      .count();
}
auto get_N_parameters() {
  return Orm::DB::table("simulation_parameters")->count();
}
auto get_Ng_out(auto p_out_id) {
  return Orm::DB::table("simulation_parameters")
      ->select("graph_id")
      .whereEq("p_out_id", p_out_id)
      .count();
}

auto read_sim_params() {
  std::vector<SBM_Database::Sim_Param> params;
  auto N_p_out = get_N_p_out();
  for (int i = 0; i < N_p_out; i++) {
    auto Ng = get_Ng_out(i);
    for (int j = 0; j < Ng; j++) {
      auto p = SBM_Database::sim_param_read(i, j);
      params.push_back(p);
    }
  }
  return params;
}

struct run_sim {
  std::shared_ptr<Orm::DatabaseManager> manager;
  SBM_Simulation::Simulation_t sim;
  run_sim(auto& manager, auto& default_connection, auto& sim) : manager(manager), sim(sim) {
    manager->setDefaultConnection(default_connection);
  }
  void run() { 
    sim.run(); 
    }
};
int main() {
  using namespace SBM_Database;
  using namespace SBM_Simulation;
  auto manager = tom_config::default_db_connection_postgres();
  auto default_connection = manager->getDefaultConnection();
  

  // project root
  std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
  std::string output_dir = root_dir + "data/";
  std::chrono::high_resolution_clock::time_point t1, t2;
  auto selector = sycl::gpu_selector_v;

  auto ps = read_sim_params();
  auto Np = ps.size();
  Np = 2;
  std::vector<sycl::queue> qs(Np, sycl::queue(selector));
  uint32_t seed = 283;
  std::vector<Simulation_t> sims;
  sims.reserve(Np);
  t1 = std::chrono::high_resolution_clock::now();
  // std::vector<uint32_t> dummy(Np);
  // std::transform(std::execution::par_unseq, sims.begin(), sims.end(),
  // dummy.begin(),
  //                [&](auto &sim) {
  //                  sim.run();
  //                  return 0;
  //                });

  std::vector<std::thread> threads;
  for (int i = 0; i < Np; i++) {
    sims.push_back(Simulation_t(qs[i], ps[i], "Community"));
    run_sim rs(manager, default_connection, sims[i]);
    threads.push_back(std::thread(&run_sim::run, &rs));
  }

  std::for_each(threads.begin(), threads.end(), [](auto &t) { t.join(); });

  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Execution time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;

  return 0;
}