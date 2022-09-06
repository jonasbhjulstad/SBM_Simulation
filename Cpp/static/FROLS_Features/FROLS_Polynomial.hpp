#ifndef FROLS_POLYNOMIAL_HPP
#define FROLS_POLYNOMIAL_HPP
#include <FROLS_Typedefs.hpp>

namespace FROLS::Features::Polynomial {

struct Polynomial_Model : public Regression_Model {
  size_t d_max;
  size_t N_ouput_features;
  Polynomial_Model(size_t N_states, size_t N_control_inputs,
                   size_t N_output_features, size_t d_max)
      : Regression_Model(N_states, N_control_inputs), d_max(d_max) {}

    Vec transform(const Vec &x) const;
};

Mat feature_transform(const Eigen::Ref<const Mat> X_raw, size_t d_max, size_t N_output_features);

Vec single_feature_transform(const Eigen::Ref<const Mat> x_raw, size_t d_max,
                             size_t target_index);

const std::string feature_name(size_t d_max, size_t N_input_features,
                               size_t N_output_features, size_t target_index);

const std::vector<std::string>
feature_names(size_t d_max, size_t N_input_features, size_t N_output_features);

const std::string model_print(const std::vector<Feature> &rd, size_t d_max,
                              size_t N_input_features,
                              size_t N_output_features);

const std::string model_print(const std::vector<std::vector<Feature>> &rds,
                              size_t d_max, size_t N_input_features,
                              size_t N_output_features);
} // namespace FROLS::Features::Polynomial
#endif