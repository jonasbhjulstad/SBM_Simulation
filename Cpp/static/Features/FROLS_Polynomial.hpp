#ifndef FROLS_POLYNOMIAL_HPP
#define FROLS_POLYNOMIAL_HPP
#include <FROLS_Typedefs.hpp>

namespace FROLS::Features::Polynomial {

class Polynomial_Model {
  const size_t N_output_features, d_max;
  std::vector<std::vector<Feature>> features;
  size_t Nx = -1, Nu = -1;

public:
  Polynomial_Model(size_t N_output_features, size_t d_max)
      : N_output_features(N_output_features), d_max(d_max) {}
  Mat transform(Eigen::Ref<const Mat> X_raw) const;
  void print() const;
  const std::string get_equations() const;
  void multiple_response_regression(Eigen::Ref<const Mat> X_raw,
                                    Eigen::Ref<const Mat> U_raw,
                                    Eigen::Ref<const Mat> Y_raw,
                                    double ERR_tol);
  const std::vector<std::vector<Feature>> get_features() const;
  Vec step(Eigen::Ref<const Vec> x, Eigen::Ref<const Vec> u);
  Mat simulate(Eigen::Ref<const Vec> x0, Eigen::Ref<const Mat> U, size_t Nt);
  void feature_summary() const;
  void write_csv(const std::string &) const;
};

Mat feature_transform(const Eigen::Ref<const Mat> X_raw, size_t d_max,
                      size_t N_output_features);
double single_feature_transform(Eigen::Ref<const Vec> x_raw, size_t d_max,
                                size_t target_index);

Vec single_feature_transform(const Eigen::Ref<const Mat> X_raw, size_t d_max,
                             size_t target_index);
std::vector<double>
single_feature_transform(std::vector<std::vector<double>> X_raw, size_t d_max,
                         size_t target_index);
const std::string feature_name(size_t d_max, size_t Nx, size_t Nu,
                               size_t N_output_features, size_t target_index);

const std::vector<std::string> feature_names(size_t d_max, size_t Nx, size_t Nu,
                                             size_t N_output_features);

std::string response_print(const std::vector<Feature> &rd, size_t d_max,
                           size_t Nx, size_t Nu, size_t N_output_features);

std::string model_print(const std::vector<std::vector<Feature>> &rds,
                        size_t d_max, size_t Nx, size_t Nu,
                        size_t N_output_features);

} // namespace FROLS::Features::Polynomial
#endif