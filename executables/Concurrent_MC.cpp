#define QT_FATAL_WARNINGS 1
#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/SBM_Simulation.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
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
  // auto N_p_out = get_N_p_out();
  auto N_p_out = 1;
  for (int i = 0; i < N_p_out; i++) {
    auto Ng = 2;
    for (int j = 0; j < Ng; j++) {
      auto p = SBM_Database::sim_param_read(i, j);
      params.push_back(p);
    }
  }
  return params;
}

auto allocate_buffers(const std::vector<SBM_Database::Sim_Param> &ps,
                      sycl::queue &q) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<SBM_Simulation::Sim_Buffers> bs;
  auto tot_byte_size = std::accumulate(ps.begin(), ps.end(), 0, [](auto acc, auto p){return acc + p.buffer_byte_size();});
  Buffer_Routines::validate_memory_size(q, tot_byte_size);

  bs.reserve(ps.size());
  std::transform(ps.begin(), ps.end(), std::back_inserter(bs), [&q](auto &p) {
    return SBM_Simulation::Sim_Buffers(q, p, "Community");
  });


  auto buffer_byte_size =
      std::accumulate(bs.begin(), bs.end(), 0, [](auto acc, const auto &b) {
        return acc + b.byte_size();
      });
  std::cout << "Total buffer size: " << buffer_byte_size << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Buffer allocation time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
  return bs;
}

int main() {
  using namespace SBM_Database;
  using namespace SBM_Simulation;
  auto manager = tom_config::default_db_connection_postgres();
  auto default_connection = manager->getDefaultConnection();

  // project root
  std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
  std::string output_dir = root_dir + "data/";
  sycl::queue q{sycl::gpu_selector_v};

  auto ps = read_sim_params();
  auto bs = allocate_buffers(ps, q);
  uint32_t seed = 283;
  SBM_Database::drop_simulation_tables("excitation");
  auto t1 = std::chrono::high_resolution_clock::now();
  run_simulations(q, ps, bs, "Community");
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Execution time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;

  return 0;
}