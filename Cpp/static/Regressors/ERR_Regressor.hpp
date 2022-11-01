#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP

#include "Regressor.hpp"

namespace FROLS::Regression {
struct ERR_Regressor : public Regressor {
  ERR_Regressor(const Regressor_Param &p) : Regressor(p) {}

private:
  std::vector<std::vector<Feature>>
  candidate_regression(const std::vector<Mat> &X_list,
                       const std::vector<Mat> &Q_list_global,
                       const std::vector<Vec> &y_list,
                       const std::vector<Feature> &used_features) const;

  bool tolerance_check(const std::vector<Mat> &Q_list, const std::vector<Vec> &y_list,
                       const std::vector<Feature> &best_features,
                       uint32_t cutoff_idx) const;
  Feature single_feature_regression(const Vec &x, const Vec &y) const;

  static bool best_feature_measure(const Feature &, const Feature &);
  bool objective_condition(float, float) const;
  void theta_solve(const Mat &A, const Vec &g,
                   std::vector<Feature> &features) const;
};
} // namespace FROLS::Regression

#endif