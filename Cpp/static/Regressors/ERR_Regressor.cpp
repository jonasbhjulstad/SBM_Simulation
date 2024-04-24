#include "ERR_Regressor.hpp"

#include <vector>

namespace FROLS::Regression {
std::vector<std::vector<Feature>> ERR_Regressor::candidate_regression(
    const std::vector<Mat> &X_list, const std::vector<Mat> &Q_list_global,
    const std::vector<Vec> &y_list, const std::vector<Feature> &used_features) const {
  uint32_t N_timeseries = X_list.size();
  uint32_t N_features = X_list[0].cols();
  // Get indices for unused features
  std::vector<int> used_indices;
  used_indices.reserve(used_features.size());
  std::transform(used_features.begin(), used_features.end(),
                 std::back_inserter(used_indices),
                 [](const Feature &f) { return f.index; });
  std::vector<uint32_t> candidate_idx;
  if (used_indices.size() > 0)
  {
   candidate_idx = unused_feature_indices(used_features, N_features);
  }
  else
  {
    candidate_idx = FROLS::range(0, N_features);
  }
  std::vector<std::vector<Feature>> candidates(N_timeseries);
  for (int i = 0; i < N_timeseries; i++) {
    candidates[i].resize(N_features - used_features.size());

    std::transform(candidate_idx.begin(), candidate_idx.end(),
        candidates[i].begin(), [=](const uint32_t &idx) {
          Feature f = single_feature_regression(X_list[i].col(idx), y_list[i]);
          f.index = idx;
          return f;
        });
  }
  return candidates;
}

Feature ERR_Regressor::single_feature_regression(const Vec &x,
                                                 const Vec &y) const {
  Feature f;
  f.g = cov_normalize(x, y);
  f.f_ERR = f.g * f.g * ((x.transpose() * x) / (y.transpose() * y)).value();
  f.tag = FEATURE_REGRESSION;
  return f;
}

bool ERR_Regressor::tolerance_check(const std::vector<Mat> &Q_list,
                                    const std::vector<Vec> &y_list,
                                    const std::vector<Feature> &best_features,
                                    uint32_t cutoff_idx) const {
  float ERR_tot = 0;
  uint32_t N_timeseries = Q_list.size();
  for (int i = 0; i < N_timeseries; i++) {
    float ERR_timeseries_tot = 0;
    for (const auto &feature : best_features) {
      ERR_timeseries_tot += feature.f_ERR;
    }
    ERR_tot += ERR_timeseries_tot;
  }
  ERR_tot /= N_timeseries;
  return (1 - ERR_tot) < tol;
}
bool ERR_Regressor::best_feature_measure(const Feature &f0, const Feature &f1) {
  return (1 - f0.f_ERR) < (1 - f1.f_ERR);
}

void ERR_Regressor::theta_solve(const Mat &A, const Vec &g,
                                std::vector<Feature> &features) const {
  Vec coefficients = A.inverse() * g;
  for (int i = 0; i < coefficients.rows(); i++) {
    features[i].theta = coefficients[i];
  }
}

bool ERR_Regressor::objective_condition(float f0, float f1) const {
  return f0 > f1;
}

} // namespace FROLS::Regression


