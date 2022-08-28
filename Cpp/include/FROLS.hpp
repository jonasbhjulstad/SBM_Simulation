#include "FROLS_Typedefs.hpp"
#include "FROLS_feature.hpp"
#include <algorithm>
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
Regression_Data single_response_batch(const Mat &X, const Vec &y,
                                      double ERR_tolerance) {
  size_t N_features = X.cols();
  Mat Q_global = Mat::Zero(X.rows(), N_features);
  Mat Q_current = Q_global;
  iVec feature_indices = iVec::Constant(N_features, -1);
  Mat A = Mat::Identity(N_features, N_features);
  Vec g = Vec::Zero(N_features);

  Regression_Data rd;
  rd.best_features.resize(N_features);
  // Perform one feature selection iteration for each feature
  for (int j = 0; j < N_features; j++) {
    // Compute remaining variance by orthogonalizing the current feature
    Q_current = used_feature_orthogonalize(X, Q_global, feature_indices, j);
    // Determine the best feature to add to the feature set
    rd.best_features[j] = feature_select(X, y, feature_indices);
    feature_indices[j] = rd.best_features[j].index;

    for (int m = 0; m < j; m++) {
      A(m, j) = cov_normalize(Q_current.col(m), X.col(feature_indices[j]));
    }
    Q_global.col(j) = Q_current.col(feature_indices[j]);

    // If ERR-tolerance is met, return non-orthogonalized parameters
    if (feature_tolerance_check(rd.best_features, ERR_tolerance)) {
      rd.coefficients = A.topLeftCorner(j, j).transpose().inverse() * g.head(j);
      return rd;
    }
    Q_current.setZero();
  }
  return rd;
}

std::vector<Regression_Data>
multiple_response_batch(const Mat &Y, const Mat& U,
                        double ERR_tolerance = 1e-1) {
  size_t N_response = Y.cols();
  std::vector<Regression_Data> result(N_response);
  Mat X(U.rows(), U.cols() + Y.cols());
  X << U, Y;
  {
    for (int i = 0; i < N_response; i++) {
      result[i] = single_response_batch(X, Y.col(i), ERR_tolerance);
    }
  }
  return result;
}
} // namespace Regression
} // namespace FROLS