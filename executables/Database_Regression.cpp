#include <Eigen/Dense>
#include <SBM_Simulation/Regression/Regression.hpp>
// #include <SBM_Simulation/Utils/path_config.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tom/tom_config.hpp>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;
using namespace std;
static constexpr size_t MAXBUFSIZE = 100000;

using namespace Eigen;

int main()
{
  float tau = .8f;
  auto manager = tom_config::default_db_connection_postgres();
  using namespace SBM_Regression;
  regression_on_database(tau, 0, "Community");
  return 0;
}
