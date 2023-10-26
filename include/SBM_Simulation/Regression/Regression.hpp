#ifndef SBM_SIMULATION_REGRESSION_HPP
#define SBM_SIMULATION_REGRESSION_HPP
#include <Eigen/Dense>
#include <string>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>

Eigen::MatrixXf openData(const std::string& fileToOpen);

float quantile_regression(const Eigen::VectorXf& x, const Eigen::VectorXf& y, float tau, float y_tol = 1e-6f, float x_tol=1e-6f);
std::pair<Eigen::MatrixXf, Eigen::MatrixXf> load_beta_regression(const std::string& datapath, uint32_t idx, bool truncate=true);

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> load_N_datasets(const std::string &datapath, uint32_t N, uint32_t offset = 0);

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> beta_regression(const Eigen::MatrixXf &F_beta_rs_mat, const Eigen::MatrixXf &connection_infs, float tau);

float alpha_regression(const Eigen::VectorXf &x, const Eigen::VectorXf &y);


std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> regression_on_datasets(const std::string &datapath, uint32_t N, float tau, uint32_t offset);

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> regression_on_datasets(const std::vector<std::string> &datapaths, uint32_t N, float tau, uint32_t offset);



#endif
