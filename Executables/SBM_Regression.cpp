#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/Regression.hpp>
#include <string>
#include <sstream>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;
using namespace std;
static constexpr size_t MAXBUFSIZE = 100000;

using namespace Eigen;
using namespace Sycl_Graph;
void linewrite(std::ofstream &file, const std::vector<float> &theta) {
  for (const auto &t_i_i : theta) {
    file << t_i_i;
    if (&t_i_i != &theta.back())
      file << ",";
    else
      file << "\n";
  }
}
int main()
{
    float tau = .9f;
    uint32_t N_sims = 2;
    std::string path = Sim_Datapath + "/Graph_0/";
    auto [theta_LS, theta_QR] = regression_on_datasets(path, N_sims, tau, 0);
    std::ofstream LS_f(path + "theta_LS.csv");
    std::ofstream QR_f(path + "theta_QR.csv");
    linewrite(LS_f, theta_LS);
    linewrite(QR_f, theta_QR);

    return 0;
}
