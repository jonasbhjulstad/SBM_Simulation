#ifndef FROLS_HPP
#define FROLS_HPP
#include <FROLS_Math.hpp>
#include <FROLS_Features.hpp>
#include <algorithm>
#include <fstream>
namespace FROLS {
// Orthogonalizes x with respect to previously selected features in Q
Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                               const iVec &used_indices,
                               size_t current_feature_idx) {
  size_t N_features = X.cols();
  size_t N_samples = X.rows();
  Mat Q_current = Mat::Zero(N_samples, N_features);
  for (int k = 0; k < N_features; k++) {
    if (!used_indices.head(current_feature_idx).cwiseEqual(k).any()) {
      Q_current.col(k) = vec_orthogonalize(X.col(k), Q.leftCols(k));
    }
  }
  return Q_current;
}
bool feature_tolerance_check(const std::vector<Feature> &features,
                             double ERR_tolerance) {
  double ERR_tot = 0;
  for (const auto &feature : features) {
    ERR_tot += feature.ERR;
  }
  return ((1 - ERR_tot) < ERR_tolerance);
}

namespace Regression {
// Computes feature coefficients for feature batch X with respect to a single
// response variable y

void write_csv(const std::vector<std::vector<Feature>>&rds, const std::string& filename)
{
    std::ofstream file(filename);
    file << "Feature,ERR,g,theta" << std::endl;
    for (const auto &rd : rds)
    {
        for (const auto &feature : rd)
        {
            file << feature.index << "," << feature.ERR << "," << feature.g << ","
                 << feature.coeff << "\n";
        }
    }
}

std::vector<Feature> single_response_regression(Eigen::Ref<const Mat> X,
                                           Eigen::Ref<const Vec> y,
                                           double ERR_tolerance) {
  using namespace FROLS;
  size_t N_features = X.cols();
  Mat Q_global = Mat::Zero(X.rows(), N_features);
  Mat Q_current = Q_global;
  iVec feature_indices = iVec::Constant(N_features, -1);
  Mat A = Mat::Identity(N_features, N_features);
  Vec g = Vec::Zero(N_features);

  std::vector<Feature> rd;
  rd.resize(N_features);
  size_t end_idx = N_features;
  // Perform one feature selection iteration for each feature
  for (int j = 0; j < N_features; j++) {
    // Compute remaining variance by orthogonalizing the current feature
    Q_current = used_feature_orthogonalize(X, Q_global, feature_indices, j);
    // Determine the best feature to add to the feature set
    rd[j] = Features::feature_select(Q_current, y, feature_indices);
    feature_indices[j] = rd[j].index;
    g[j] = rd[j].g;

    Q_global.col(j) = Q_current.col(feature_indices[j]);
    for (int m = 0; m < j; m++) {
      A(m, j) = cov_normalize(Q_global.col(m), X.col(feature_indices[j]));
    }

    // If ERR-tolerance is met, return non-orthogonalized parameters
    if (feature_tolerance_check(rd, ERR_tolerance)) {
      end_idx = j + 1;
      rd.resize(end_idx);
      break;
    }
    Q_current.setZero();
  }
  Vec coefficients =
      A.topLeftCorner(end_idx, end_idx).transpose().inverse() * g.head(end_idx);
  // assign coefficients to features
  for (int i = 0; i < end_idx; i++) {
    rd[i].coeff = coefficients[i];
  }
  return rd;
}

std::vector<std::vector<Feature>>
multiple_response_regression(Eigen::Ref<const Mat> Y, Eigen::Ref<const Mat> X,
                             double ERR_tolerance) {
  size_t N_response = Y.cols();
  std::vector<std::vector<Feature>> result(N_response);
  {
    for (int i = 0; i < N_response; i++) {
      result[i] = single_response_regression(X, Y.col(i), ERR_tolerance);
    }
  }
  return result;
}

const std::string regression_data_summary(const std::vector<Feature> &rd) {
  std::string summary;
  summary += "Best Features:\n";
  for (const auto &feature : rd) {
    summary += "Index: " + std::to_string(feature.index) +
               "\tERR: " + std::to_string(feature.ERR) +
               "\tg: " + std::to_string(feature.g) + "\n";
  }
  summary += "Coefficients:\n";
  for (const auto &feature : rd) {
    summary += std::to_string(feature.coeff) + "\n";
  }
  return summary;
}

const std::string
regression_data_summary(const std::vector<std::vector<Feature>> &rds) {
  std::string summary;
  size_t response_idx = 0;
  for (const auto &rd : rds) {
    summary += "Response variable: " + std::to_string(response_idx) + "\n";
    summary += regression_data_summary(rd);
    response_idx++;
  }
  return summary;
}
} // namespace Regression
} // namespace FROLS

#endif // FROLS_HPP