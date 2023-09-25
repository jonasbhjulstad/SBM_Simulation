#include <Eigen/Dense>
#include <Sycl_Graph/Regression.hpp>
#include <Sycl_Graph/Utils/json_settings.hpp>
#include <Sycl_Graph/Utils/path_config.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;
using namespace std;
static constexpr size_t MAXBUFSIZE = 100000;

using namespace Eigen;
using namespace Sycl_Graph;
void linewrite(std::ofstream &file, const std::vector<float> &theta)
{
  for (const auto &t_i_i : theta)
  {
    file << t_i_i;
    if (&t_i_i != &theta.back())
      file << ",";
    else
      file << "\n";
  }
}

std::vector<std::string> get_subdirs(const std::string &dir, const std::string &search_str)
{
  std::vector<std::string> subdirs;
  auto N_letters = search_str.size();
  for (const auto &entry : std::filesystem::directory_iterator(dir))
  {
    // and if dirname starts with 'Graph_'
    if (entry.is_directory() && entry.path().filename().string().substr(0, N_letters) == search_str)
    {
      subdirs.push_back(entry.path().string() + "/");
    }
  }
  return subdirs;
}

int main()
{
  auto regression_routine = [](const auto &dir, auto N_sims, auto tau)
  {
    auto [theta_LS, theta_QR, MSE, MAE] = regression_on_datasets(dir, N_sims, tau, 0);
    std::ofstream LS_f(dir + "theta_LS.csv");
    std::ofstream QR_f(dir + "theta_QR.csv");
    linewrite(LS_f, theta_LS);
    linewrite(QR_f, theta_QR);

    std::ofstream MSE_f(dir + "MSE.csv");
    std::ofstream MAE_f(dir + "MAE.csv");
    linewrite(MSE_f, MSE);
    linewrite(MAE_f, MAE);
  };

  auto str_append = [](const auto &dirs, const auto &str)
  {
    std::vector<std::string> new_dirs(dirs.size());
    std::transform(dirs.begin(), dirs.end(), new_dirs.begin(), [&str](const auto &s)
                   { return s + str; });
    return new_dirs;
  };

  float tau = .8f;
  std::string path = Sim_Datapath;
  std::vector<std::string> subdirs = get_subdirs(path, "p_out_");
  std::for_each(subdirs.begin(), subdirs.end(), [&](const std::string &s)
                {
      auto graphdirs = get_subdirs(s, "Graph_");
      auto true_dirs = str_append(graphdirs, "True_Communities/");
      auto detected_dirs = str_append(graphdirs, "Detected_Communities/");
      Sim_Param p = parse_json(s + "/Sim_Param.json");
      for(auto&& td: true_dirs)
      {
        regression_routine(td, p.N_sims, tau);
      }
      for(auto&& dd: detected_dirs)
      {
        regression_routine(dd, p.N_sims, tau);
      } });

  return 0;
}
