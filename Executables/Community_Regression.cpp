#include <Eigen/Dense>
#include <Sycl_Graph/Regression/Regression.hpp>
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
  float tau = .8f;
  std::string path = Sim_Datapath;
  std::vector<std::string> subdirs = get_subdirs(path, "p_out_");
  std::for_each(subdirs.begin(), subdirs.end(), [tau](const std::string &s)
                {
      auto graphdirs = get_subdirs(s, "Graph_");
      Sim_Param p = parse_json(s + "/Sim_Param.json");
      for(auto&& gdir: graphdirs)
      {
      auto [theta_LS, theta_QR] = regression_on_datasets(gdir, p.N_sims, tau, 0);
      std::ofstream LS_f(gdir + "theta_LS.csv");
      std::ofstream QR_f(gdir + "theta_QR.csv");
      linewrite(LS_f, theta_LS);
      linewrite(QR_f, theta_QR);
      } });

  return 0;
}
