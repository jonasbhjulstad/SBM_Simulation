#include "FROLS_Features.hpp"
#include <FROLS_Math.hpp>
#include <iostream>
#include <itertools.hpp>

namespace FROLS::Features::Polynomial {

Mat Polynomial_Model::transform(Eigen::Ref<const Mat> &x) const {
  Mat x_transformed(x.rows(), x.cols());

  size_t y_idx = 0;
  for (int i = 0; i < features.size(); i++) {
    for (int j = 0; j < features[i].size(); j++) {
      x_transformed. =
          features[i][j].g *
          single_feature_transform(x, d_max, features[i][j].index);
      y_idx++;
    }
  }

  return x_transformed;
}

Vec single_feature_transform(Eigen::Ref<const Mat> X_raw, size_t d_max,
                             size_t target_index) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = X_raw.cols();
  size_t N_rows = X_raw.rows();
  Vec X_poly(N_rows);
  size_t feature_index = 0;
  // for (int d = 1; d < d_max; d++) {
  for (auto &&comb : iter::combinations_with_replacement(range(0, d_max + 1),
                                                         N_input_features)) {
    // generate all combinations of powers of the original features up to d_max
    for (auto &&powers : iter::permutations(comb)) {
      // if the sum of the powers is greater than d_max, skip this combination
      if (feature_index == target_index) {
        // otherwise, compute the polynomial combination of the original
        // features
        X_poly = X_raw.col(0).array().pow(powers[0]);
        for (size_t i = 1; i < N_input_features; i++) {
          X_poly = X_poly.col(feature_index).array() *
                   X_raw.col(i).array().pow(powers[i]);
        }
        return X_poly;
      }

      feature_index++;
    }
  }
  return X_poly;
}

Mat feature_transform(Eigen::Ref<const Mat> X_raw, size_t d_max,
                      size_t N_output_features) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = X_raw.cols();
  size_t N_rows = X_raw.rows();
  Mat X_poly(N_rows, N_output_features);
  size_t feature_index = 0;
  // for (int d = 1; d < d_max; d++) {
  for (auto &&comb : iter::combinations_with_replacement(range(0, d_max + 1),
                                                         N_input_features)) {
    // generate all combinations of powers of the original features up to d_max
    for (auto &&powers : iter::permutations(comb)) {
      // if the sum of the powers is greater than d_max, skip this combination
      if ((feature_index >= N_output_features) ||
          std::any_of(powers.begin(), powers.end(),
                      [d_max](size_t p) { return p > d_max; })) {
        return X_poly;
      }

      // otherwise, compute the polynomial combination of the original features
      X_poly.col(feature_index) = X_raw.col(0).array().pow(powers[0]);
      for (size_t i = 1; i < N_input_features; i++) {
        X_poly.col(feature_index) = X_poly.col(feature_index).array() *
                                    X_raw.col(i).array().pow(powers[i]);
      }
      feature_index++;
    }
  }
  return X_poly;
}

const std::string feature_name(size_t d_max, size_t N_input_features,
                               size_t N_output_features, size_t target_index) {
  std::string feature_name;
  size_t feature_index = 0;
  // for (int d = 1; d < d_max; d++) {
  for (auto &&comb : iter::combinations_with_replacement(range(0, d_max + 1),
                                                         N_input_features)) {
    for (auto &&powers : iter::permutations(comb)) {
      if (feature_index == target_index) {
        size_t x_idx = 0;
        for (auto &&pow : powers) {
          feature_name += (pow > 0) ? "x" + std::to_string(x_idx) : "";
          feature_name += (pow > 1) ? "^" + std::to_string(powers[x_idx]) : "";
          feature_name += (pow > 0) ? " " : "";
          x_idx++;
        }
        return feature_name;
      }
      feature_index++;
    }
  }
  return "";
}

const std::vector<std::string>
feature_names(size_t d_max, size_t N_input_features, size_t N_output_features) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  std::vector<std::string> feature_names;
  for (auto &&comb : iter::combinations_with_replacement(range(0, d_max + 1),
                                                         N_input_features)) {
    for (auto &&powers : iter::permutations(comb)) {
      std::string feature_name = "";
      size_t x_idx = 0;
      for (auto &&pow : powers) {
        feature_name += (pow > 0) ? "x" + std::to_string(x_idx) : "";
        feature_name += (pow > 1) ? "^" + std::to_string(powers[x_idx]) : "";
        feature_name += (pow > 0) ? " " : "";
        x_idx++;
      }
      if (!feature_name.empty())
        feature_names.push_back(feature_name);
    }
  }
  return feature_names;
}

// void feature_display(const Mat &X, size_t d_max, size_t N_input_features) {
//   auto f_names = feature_names(d_max, N_input_features, X.cols());
//   for (auto name : f_names) {
//     std::cout << name << "\t";
//   }
//   std::cout << std::endl;
// }

const std::string model_print(const std::vector<Feature> &rd, size_t d_max,
                              size_t N_input_features,
                              size_t N_output_features) {
  std::string model;
  for (int i = 0; i < rd.size(); i++) {
    model += std::to_string(rd[i].coeff);
    model +=
        feature_name(d_max, N_input_features, N_output_features, rd[i].index);
    if (i != rd.size() - 1) {
      model += " + ";
    }
  }
  model += "\n";
  return model;
}

const std::string model_print(const std::vector<std::vector<Feature>> &rds,
                              size_t d_max, size_t N_input_features,
                              size_t N_output_features) {
  std::string model;
  size_t response_idx = 0;
  for (auto &rd : rds) {
    model += "y" + std::to_string(response_idx) + "=\t";
    model += model_print(rd, d_max, N_input_features, N_output_features);
    response_idx++;
  }
  return model;
}
} // namespace FROLS::Features::Polynomial