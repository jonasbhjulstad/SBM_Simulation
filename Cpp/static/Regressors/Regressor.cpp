#include "Regressor.hpp"
#include <FROLS_Eigen.hpp>
#include <FROLS_Execution.hpp>
#include <fmt/format.h>

namespace FROLS::Regression {

Regressor::Regressor(const Regressor_Param &p)
    : tol(p.tol), theta_tol(p.theta_tol), N_terms_max(p.N_terms_max) {}

std::vector<Feature>
Regressor::single_fit(const std::vector<Mat> &X_list,
                      const std::vector<Vec> &y_list) const {
  uint32_t N_features = X_list[0].cols();
  uint32_t N_rows = X_list[0].rows();
  uint32_t N_timeseries = X_list.size();
  std::vector<Mat> Q_list_global(N_timeseries);
  std::vector<Mat> Q_list_current(N_timeseries);
  for (int i = 0; i < N_timeseries; i++) {
    Q_list_global[i].resize(N_rows, N_features);
    Q_list_global[i].setZero();
    Q_list_current[i].resize(N_rows, N_features);
  }

  Mat A = Mat::Zero(N_features, N_features);
  Vec g = Vec::Zero(N_features);
  std::vector<Feature> best_features;
  best_features.reserve(N_terms_max);
  uint32_t end_idx = 0;

  // fmt::print("Max features: {}\n", N_terms_max);
  // Perform one feature selection iteration for each feature
  for (int j = 0; j < N_features; j++) {
    // fmt::print("Feature {}\n", j+1);
    // Compute remaining variance by orthogonalizing the current feature
    // std::transform(X_list.begin(), X_list.end(), Q_list_current.begin(),
    // [&Q_list_global](const Mat &X) { return used_feature_orthogonalize(X,
    // Q_global, best_features);});

    for (int i = 0; i < N_timeseries; i++) {
      Q_list_current[i] = used_feature_orthogonalize(
          X_list[i], Q_list_global[i], best_features);
    }
    // used_feature_orthogonalize(X, Q_global, best_features);
    // Determine the best feature to add to the feature set
    Feature f = best_feature_select(Q_list_current, Q_list_global, y_list,
                                    best_features);
    if (f.tag == FEATURE_INVALID) {
      end_idx = j+1;
      break;
    }

    best_features.push_back(f);
    for (int i = 0; i < N_timeseries; i++) {
      Q_list_global[i].col(j) = Q_list_current[i].col(best_features[j].index);
    }
    // Q_global.col(j) = Q_current.col(best_features[j].index);

    g[j] = best_features[j].g;

    for (int m = 0; m < j; m++) {
      int a_mj_avg = 0;
      for (int i = 0; i < N_timeseries; i++) {
        a_mj_avg += cov_normalize(Q_list_global[i].col(m),
                                  X_list[i].col(best_features[j].index));
      }
      A(m, j) = a_mj_avg / N_timeseries;
    }
    A(j, j) = 1;

    // If ERR-tolerance is met, return non-orthogonalized parameters
    if (tolerance_check(Q_list_global, y_list, best_features, j + 1) || (j == (N_terms_max - 1))) {
      end_idx = j + 1;
      break;
    }

    std::for_each(Q_list_current.begin(), Q_list_current.end(),
                  [](Mat &Q) { Q.setZero(); });
  }
  theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx),
              best_features);
  for (int i = 0; i < y_list.size(); i++){
  // std::cout << y_list[i].head(5).transpose() <<  std::endl;
                      }
  return best_features;
}

Feature Regressor::best_feature_select(
    const std::vector<Mat> &X_list, const std::vector<Mat> &Q_list_global,
    const std::vector<Vec> &y_list,
    const std::vector<Feature> &used_features) const {
  const std::vector<std::vector<Feature>> candidates =
      candidate_regression(X_list, Q_list_global, y_list, used_features);
  // std::vector<Feature> thresholded_candidates;
  // std::copy_if(candidates.begin(), candidates.end(),
  // std::back_inserter(thresholded_candidates),
  //              [&](const auto &f) {
  //                  return abs(f.g) > theta_tol;
  //              });
  static bool warn_msg = true;

  Feature res;
  switch (candidates[0].size()) {
  case 0:
    if (warn_msg)
      std::cout << "[Regressor] Warning: threshold is too high for candidates"
                << std::endl;
    warn_msg = false;
    break;
  default:
    res = feature_selection_criteria(candidates);
    break;
  }

  return res;
}

Vec Regressor::predict(const Mat &Q,
                       const std::vector<Feature> &features) const {
  Vec y_pred(Q.rows());
  y_pred.setZero();
  uint32_t i = 0;
  for (const auto &feature : features) {
    if (feature.f_ERR == -1) {
      break;
    }
    y_pred += Q.col(i) * feature.g;
    i++;
  }
  return y_pred;
}


std::vector<std::vector<Feature>> Regressor::transform_fit(
    const std::vector<Mat>& X_raw, const std::vector<Mat>& U_raw,
    const std::vector<Mat>& Y_list, Features::Feature_Model &model) {
  uint32_t N_timeseries = X_raw.size();
  uint32_t N_response = Y_list[0].cols();
  std::vector<Mat> XU_list(N_timeseries);
  for (int i = 0; i < N_timeseries; i++) {
    XU_list[i] = Mat::Zero(X_raw[i].rows(), X_raw[i].cols() + U_raw[i].cols());
    XU_list[i] << X_raw[i], U_raw[i];
  }
  std::vector<Mat> X_list(N_timeseries);
  std::transform(XU_list.begin(), XU_list.end(), X_list.begin(),
                 [&model](const Mat &XU) { return model.transform(XU); });

  std::vector<std::vector<Feature>> feature_list(N_response);
  for (int i = 0; i < N_response; i++) {
    feature_list[i].reserve(N_terms_max);
  }
  for (int j = 0; j < N_response; j++) {
    std::vector<Vec> y_list(N_timeseries);
    for (int i = 0; i < N_timeseries; i++) {
      y_list[i] = Y_list[i].col(j);
    }
    feature_list[j] = single_fit(X_list, y_list);
  }
  return feature_list;
}

std::vector<std::vector<Feature>>
Regressor::transform_fit(const Regression_Data &rd,
                         Features::Feature_Model &model){
  return transform_fit(rd.X, rd.U, rd.Y, model);
}

Feature Regressor::feature_selection_criteria(
    const std::vector<std::vector<Feature>> &features) const {
      uint32_t N_timeseries = features.size();
      uint32_t N_features = features[0].size();
      std::vector<float> ERRs(N_features,0);

      for (int i = 0; i < N_timeseries; i++)
      {
        for (int j = 0; j < N_features; j++)
        {
          ERRs[j] += features[i][j].f_ERR;
        }
      }
      uint32_t best_feature_idx = 0;
      for (int j = 0; j < N_features; j++)
      {
        if (objective_condition(ERRs[j], ERRs[best_feature_idx]))
        {
          best_feature_idx = j;
        }
        {
          best_feature_idx = j;
        }
      }
      Feature best_avg_feature{};
      best_avg_feature.f_ERR = 0;
      for (int i = 0; i < N_timeseries; i++)
      {
        best_avg_feature.g += features[i][best_feature_idx].g;
        best_avg_feature.f_ERR += features[i][best_feature_idx].f_ERR;
      }

      best_avg_feature.g /= N_timeseries;
      best_avg_feature.f_ERR /= N_timeseries;
      best_avg_feature.index = features[0][best_feature_idx].index;
      best_avg_feature.tag = FEATURE_REGRESSION;

  return best_avg_feature;
}

std::vector<uint32_t>
Regressor::unused_feature_indices(const std::vector<Feature> &features,
                                  uint32_t N_features) const {
  std::vector<uint32_t> used_idx(features.size());
  std::transform(features.begin(), features.end(), used_idx.begin(),
                 [&](auto &f) { return f.index; });
  return filtered_range(used_idx, 0, N_features);
}

int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
