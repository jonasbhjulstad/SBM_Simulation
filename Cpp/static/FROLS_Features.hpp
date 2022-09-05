#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP
#include "combinations_with_replacement.hpp"
#include <FROLS_Typedefs.hpp>
#include <FROLS_math.hpp>
#include <iostream>
#include <itertools.hpp>
#include <vector>
namespace FROLS::Features::Polynomial {

Mat feature_transform(const Mat &X_raw, size_t d_max,
                      size_t N_output_features) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = X_raw.cols();
  size_t N_rows = X_raw.rows();
  Mat X_poly(N_rows, N_output_features);
  size_t feature_index = 0;
  // generate all combinations of powers of the original features up to d_max
  for (auto&& powers : iter::combinations_with_replacement(
           range(0, d_max + 1), N_input_features)) {
    // if the sum of the powers is greater than d_max, skip this combination
    if (std::any_of(powers.begin(), powers.end(),
                    [d_max](size_t p) { return p > d_max; }) ||
        feature_index >= N_output_features) {
      continue;
    }
    // otherwise, compute the polynomial combination of the original features
    X_poly.col(feature_index) = X_raw.col(0).array().pow(powers[0]);
    for (size_t i = 1; i < N_input_features; i++) {
      X_poly.col(feature_index) = X_poly.col(feature_index).array() *
                                  X_raw.col(i).array().pow(powers[i]);
    }
    feature_index++;
  }
  return X_poly;
}

const std::vector<std::string>
feature_names(size_t d_max, size_t N_input_features, size_t N_output_features) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  std::vector<std::string> feature_names;
  size_t feature_index = 0;
  for (auto &&powers : iter::combinations_with_replacement(
           range(0, d_max + 1), N_input_features)) {
    std::string feature_name = "";
    size_t x_idx = 0;
    for (auto&& pow: powers) {
        feature_name += (pow > 0) ? "x" + std::to_string(x_idx) : "";
        feature_name += (pow > 1) ? "^" + std::to_string(powers[x_idx]) : "";
        x_idx++;
      }
    if (!feature_name.empty())
      feature_names.push_back(feature_name);
  }
  return feature_names;
}

void feature_display(const Mat &X, size_t d_max, size_t N_input_features) {
  auto f_names = feature_names(d_max, N_input_features, X.cols());
  for (auto name : f_names) {
    std::cout << name << "\t";
  }
  std::cout << std::endl;
}

} // namespace FROLS::Features::Polynomial

#endif