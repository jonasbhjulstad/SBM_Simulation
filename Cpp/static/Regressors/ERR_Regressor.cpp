#include "ERR_Regressor.hpp"
#include <Math.hpp>
#include <vector>
namespace FROLS::Regression {

Feature ERR_Regressor::feature_select(const Mat &X, const Vec &y,
                                      const iVec &used_features) const {
  size_t N_features = X.cols();
  double ERR, g;
  Feature best_feature;
  for (int i = 0; i < N_features; i++) {
    // If the feature is already used, skip it
    if (!used_features.cwiseEqual(i).any()) {
      Vec xi = X.col(i);
      g = cov_normalize(xi, y);
      ERR = g * g * ((xi.transpose() * xi) / (y.transpose() * y)).value();
      if (ERR > best_feature.f_ERR) {
        best_feature.f_ERR = ERR;
        best_feature.g = g;
        best_feature.index = i;
      }
    }
  }
  return best_feature;
}

bool ERR_Regressor::tolerance_check(
    const Mat &Q, const Vec &y,
    const std::vector<Feature> &best_features) const {
  double ERR_tot = 0;
  for (const auto &feature : best_features) {
    ERR_tot += feature.f_ERR;
  }
  return ERR_tot > tol;
}

} // namespace FROLS::Regression
