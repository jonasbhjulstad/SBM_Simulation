#include "FROLS_Polynomial.hpp"
#include "FROLS_Features.hpp"
#include <Eigen/src/Core/Array.h>
#include <FROLS_Math.hpp>
#include <FROLS_Algorithm.hpp>
#include <iostream>
#include <iomanip>
#include <itertools.hpp>

namespace FROLS::Features::Polynomial {

Mat Polynomial_Model::transform(Eigen::Ref<const Mat> X_raw) const {
  return feature_transform(X_raw, d_max, N_output_features);
}

Vec Polynomial_Model::step(Eigen::Ref<const Vec> x, Eigen::Ref<const Vec> u)
{
  Vec x_next(x.rows());
  x_next.setZero();
  for (int i = 0; i < features.size(); i++)
  {
    for (int j = 0; j < features.size(); j++)
    {
      x_next(i) += single_feature_transform(x, d_max, features[i][j].index);
    }
  }
  return x_next;
}

Mat Polynomial_Model::simulate(Eigen::Ref<const Vec> x0, Eigen::Ref<const Mat> U, size_t Nt)
{
  Mat X(Nt+1, x0.rows());
  X.row(0) = x0;
  for (int i = 0; i < Nt; i++)
  {
    X.row(i+1) = step(X.row(i), U.row(i));
  }
  return X;
}

void Polynomial_Model::print() const {
  if (Nx == -1 || Nu == -1) {
    std::cout << "Model not yet trained" << std::endl;
    return;
  }
  std::cout << model_print(features, d_max, Nx, Nu, N_output_features)
            << std::endl;
}

const std::string Polynomial_Model::get_equations() const{
  if (Nx == -1 || Nu == -1) {
    return "Model not yet trained";
  }
  return model_print(features, d_max, Nx, Nu, N_output_features);
}

void Polynomial_Model::multiple_response_regression(Eigen::Ref<const Mat> X_raw,
                                                    Eigen::Ref<const Mat> U_raw,
                                                    Eigen::Ref<const Mat> Y_raw,
                                                    double ERR_tol) {
  Nx = X_raw.cols();
  Nu = U_raw.cols();
  if (X_raw.rows() != U_raw.rows() || X_raw.rows() != Y_raw.rows()) {
    throw std::runtime_error("X, U, and Y must have the same number of rows");
  }
  
  Mat XU(X_raw.rows(), Nx + Nu);
  XU << X_raw, U_raw;
  Mat X_poly = feature_transform(XU, d_max, N_output_features);
  features = FROLS::Regression::multiple_response_regression(X_poly, Y_raw, ERR_tol);
}

void Polynomial_Model::feature_summary() const
{
    std::cout << "y\tFeature\t\tg\tTheta\tERR\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < features.size(); i++)
    {
      for (auto &feature : features[i])
      {
        std::string name = feature_name(d_max, Nx, Nu, N_output_features, feature.index);
        name = (name == "") ? "1" : name;
        std::cout << i << "\t" << name << "\t\t" << feature.g << "\t" << feature.coeff << "\t" << feature.ERR << std::endl;
      }
    }
}

void Polynomial_Model::write_csv(const std::string& filename) const
{
  std::ofstream f(filename);
  f << "Response,Feature,Index,g,ERR" << std::endl;
  for (int i = 0; i < features.size(); i++)
  {
    for (auto &feature : features[i])
    {
      f << i << "," << feature_name(d_max, Nx, Nu, N_output_features, feature.index) << "," << feature.index << "," << feature.g << "," << feature.ERR << std::endl;
    }
  }
}

const std::vector<std::vector<Feature>> Polynomial_Model::get_features() const {
  if (Nx == -1 || Nu == -1) {
    std::cout << "Model not yet trained" << std::endl;
    return {};
  }
  return features;
}

double single_feature_transform(Eigen::Ref<const Vec> x_raw, size_t d_max, size_t target_index)
{
    // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = x_raw.cols();
  double result;
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
        // x_poly = power(X_raw, powers);
        for (int i = 0; i < powers.size(); i++) {
          result += pow(x_raw(i), powers[i]);
        }
        return result;
      }

      feature_index++;
    }
  }
  return result;
}

Vec single_feature_transform(Eigen::Ref<const Mat> X_raw, size_t d_max,
                             size_t target_index) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = X_raw.cols();
  size_t N_rows = X_raw.size();
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
        // x_poly = power(X_raw, powers);
        Eigen::ArrayXd x_poly = X_raw.col(0).array().pow(powers[0]);
        for (int i = 1; i < powers.size(); i++) {
          x_poly += X_raw.col(i).array().pow(powers[i]);
        }
        X_poly = Vec(x_poly);
        return X_poly;
      }

      feature_index++;
    }
  }
  return X_poly;
}

std::vector<double>
single_feature_transform(std::vector<std::vector<double>> X_raw, size_t d_max,
                         size_t target_index) {
  // get feature names for polynomial combinations with powers between d_min,
  // d_max of the original features
  size_t N_input_features = X_raw[0].size();
  size_t N_rows = X_raw.size();
  std::vector<double> X_poly(N_rows);
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
        // x_poly = power(X_raw, powers);
        for (int i = 0; i < N_rows; i++) {
          X_poly[i] = 1;
          for (int j = 0; j < N_input_features; j++) {
            X_poly[i] *= pow(X_raw[i][j], powers[j]);
          }
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

const std::string feature_name(size_t d_max, size_t Nx, size_t Nu,
                               size_t N_output_features, size_t target_index) {
  std::string feature_name;
  size_t feature_index = 0;
  size_t N_input_features = Nx + Nu;
  // for (int d = 1; d < d_max; d++) {
  for (auto &&comb : iter::combinations_with_replacement(range(0, d_max + 1),
                                                         N_input_features)) {
    for (auto &&powers : iter::permutations(comb)) {
      if (feature_index == target_index) {
        size_t x_idx = 0;
        for (auto &&pow : powers) {
          std::string x_or_u = x_idx < Nx ? "x" : "u";
          size_t idx_offset = x_idx < Nx ? 0 : Nx;
          feature_name += (pow > 0) ? x_or_u + std::to_string(x_idx - idx_offset) : "";
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

std::string response_print(const std::vector<Feature> &rd, size_t d_max, size_t Nx,
                           size_t Nu, size_t N_output_features) {
  std::string model;
  for (int i = 0; i < rd.size(); i++) {
    model += std::to_string(rd[i].coeff);
    model += feature_name(d_max, Nx, Nu, N_output_features, rd[i].index);
    if (i != rd.size() - 1) {
      model += " + ";
    }
  }
  model += "\n";
  return model;
}

std::string model_print(const std::vector<std::vector<Feature>> &rds, size_t d_max,
                        size_t Nx, size_t Nu, size_t N_output_features) {
  std::string model;
  size_t response_idx = 0;
  for (auto &rd : rds) {
    model += "y" + std::to_string(response_idx) + "=\t";
    model += response_print(rd, d_max, Nx, Nu, N_output_features);
    response_idx++;
  }
  return model;
}
} // namespace FROLS::Features::Polynomial