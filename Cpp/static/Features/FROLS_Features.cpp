#include "FROLS_Features.hpp"
#include "combinations_with_replacement.hpp"
#include <FROLS_Algorithm.hpp>
#include <FROLS_Math.hpp>
#include <itertools.hpp>
#include <vector>
namespace FROLS::Features {
Feature feature_select(const Mat &X, const Vec &y, const iVec &used_features) {
  size_t N_features = X.cols();
  double ERR, g;
  Feature best_feature;
  for (int i = 0; i < N_features; i++) {
    // If the feature is already used, skip it
    if (!used_features.cwiseEqual(i).any()) {
      Vec xi = X.col(i);
      g = cov_normalize(xi, y);
      ERR = g * g * ((xi.transpose() * xi) / (y.transpose() * y)).value();
      if (ERR > best_feature.ERR) {
        best_feature.ERR = ERR;
        best_feature.g = g;
        best_feature.index = i;
      }
    }
  }
  return best_feature;
}
} // namespace FROLS::Features
