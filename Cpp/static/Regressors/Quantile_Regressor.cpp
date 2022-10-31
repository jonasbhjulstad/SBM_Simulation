#include "Quantile_Regressor.hpp"
// #include "ortools/base/commandlineflags.h"
// #include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"
// #include "ortools/linear_solver/linear_solver.pb.h"
// #include "ortools/sat/cp_model.h"
#include <FROLS_Execution.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
namespace FROLS::Regression {
Quantile_Regressor::Quantile_Regressor(const Quantile_Param &p)
    : tau(p.tau), Regressor(p), solver_type(p.solver_type),
      problem_type(p.problem_type) {}

void Quantile_Regressor::theta_solve(const Mat &A, const Vec &g,
                                     std::vector<Feature> &features) const {
  std::vector<Feature> feature_tmp;
  feature_tmp.reserve(features.size());
  features[0].theta = features[0].g;

  Vec coefficients = A.inverse() * g;
  for (int i = 0; i < coefficients.rows(); i++) {
    features[i].theta = coefficients[i];
  }
}

Feature Quantile_Regressor::single_feature_regression(const Vec &x,
                                                      const Vec &y) const
  {

    using namespace operations_research;
    uint32_t N_rows = x.rows();

    using namespace operations_research;
    static int count = 0;
    std::string solver_name = "Quantile_Solver_" + std::to_string(count);
    std::unique_ptr<MPSolver> solver = std::make_unique<MPSolver>(solver_name, problem_type);
    const float infinity = solver->infinity();
    // theta_neg = solver->MakeNumVar(0.0, infinity, "theta_neg");
    // theta_pos = solver->MakeNumVar(0.0, infinity, "theta_pos");
    operations_research::MPVariable *theta =
        solver->MakeNumVar(-infinity, infinity, "theta");
    std::vector<operations_research::MPVariable *> u_pos;
    std::vector<operations_research::MPVariable *> u_neg;
    std::vector<operations_research::MPConstraint *> g(N_rows);

    operations_research::MPObjective *objective;
    count++;
    objective = solver->MutableObjective();
    objective->SetMinimization();
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);

    std::for_each(u_pos.begin(), u_pos.end(),
                  [=](auto& u) { objective->SetCoefficient(u, tau); });
    std::for_each(u_neg.begin(), u_neg.end(),
                  [=](auto& u) { objective->SetCoefficient(u, (1 - tau)); });
    // std::generate(g.begin(), g.end(),
    //               [&]() { return solver->MakeRowConstraint(); });

    float eps = 4.f;
    static int counter = 0;
    counter++;
    for (int i = 0; i < N_rows; i++) {
      // g[i]->SetCoefficient(theta_pos, x(i));
      // g[i]->SetCoefficient(theta_neg, -x(i));
      g[i] = solver->MakeRowConstraint(y(i), y(i));
      g[i]->SetCoefficient(theta, x(i));
      g[i]->SetCoefficient(u_pos[i], 1);
      g[i]->SetCoefficient(u_neg[i], -1);
      // g[i]->SetBounds(y[i], y[i]);
    }
    //            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
    const bool solver_status = solver->Solve() == MPSolver::OPTIMAL;

    if (solver_status) {
      float f = objective->Value();

      std::vector<float> u_neg_sol(N_rows);
      std::vector<float> u_pos_sol(N_rows);
      for (int i = 0; i < N_rows; i++) {
        u_neg_sol[i] = u_neg[i]->solution_value();
        u_pos_sol[i] = u_pos[i]->solution_value();
      }
      // float theta_sol = (theta_pos->solution_value() -
      // theta_neg->solution_value()); float theta_pos_sol =
      // theta_pos->solution_value(); float theta_neg_sol =
      // theta_neg->solution_value();
      float theta_sol = theta->solution_value();
      // solver->Clear();
      // fmt::print("Solve status, {}, {}, {}\n", solver_status, f, theta_sol);
      // if(theta_sol == 0)
      // std::cout << y.head(10).transpose() << std::endl;
      // std::for_each(g.begin(), g.end(), [&](auto &gi) { gi->Clear(); });
      return Feature{f, theta_sol, 0, 0., FEATURE_REGRESSION};
    } else {
      std::cout << "[Quantile_Regressor] Warning: Quantile regression failed"
                << std::endl;
      std::for_each(g.begin(), g.end(), [](auto &gi) { gi->Clear(); });
      return Feature{std::numeric_limits<float>::infinity(), 0, 0, 0., FEATURE_INVALID};
    }
  // }
}

std::vector<std::vector<Feature>> Quantile_Regressor::candidate_regression(
    const std::vector<Mat> &X_list, const std::vector<Mat> &Q_list_global,
    const std::vector<Vec> &y_list,
    const std::vector<Feature> &used_features) const
    {
  uint32_t N_timeseries = X_list.size();
  uint32_t N_features = X_list[0].cols();

  // Get indices for unused features
  std::vector<int> used_indices;
  used_indices.reserve(used_features.size());
  std::transform(used_features.begin(), used_features.end(),
                 std::back_inserter(used_indices),
                 [](const Feature &f) { return f.index; });
  std::vector<uint32_t> candidate_idx =
      unused_feature_indices(used_features, N_features);

  std::vector<std::vector<Feature>> candidates(N_timeseries);
  std::generate(candidates.begin(), candidates.end(), [&]() {
    return std::vector<Feature>(candidate_idx.size());
  });
  for (int i = 0; i < N_timeseries; i++) {
    candidates[i].resize(N_features - used_features.size());
    Vec y_diff = y_list[i] - predict(Q_list_global[i], used_features);
    std::transform(candidate_idx.begin(), candidate_idx.end(),
                   candidates[i].begin(), [=](const uint32_t &idx) {
                     Feature f =
                         single_feature_regression(X_list[i].col(idx), y_diff);
                     f.index = idx;
                     return f;
                   });
  }
  return candidates;
}

bool Quantile_Regressor::tolerance_check(
    const std::vector<Mat> &X_list, const std::vector<Vec> &y_list,
    const std::vector<Feature> &best_features, uint32_t cutoff_idx) const {
  uint32_t N_timeseries = X_list.size();
  uint32_t N_rows = X_list[0].rows();

  float aMAE = 0;
  for (int i = 0; i < N_timeseries; i++) {
    Vec y_pred = predict(X_list[i], best_features);
    Vec diff = y_list[i] - y_pred;
    aMAE +=
        (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum() / N_rows;
  }
  aMAE /= N_timeseries;
  bool no_improvement =
      (best_features.size() > 1) && (best_features.back().f_ERR < aMAE);
  if (no_improvement) {
    // std::cout << "[Quantile_Regressor] Warning: Termination due to lack of
    // improvement" << std::endl;
    return true;
  } else if (aMAE < tol) {
    // std::cout << "[Quantile_Regressor] Status: Successful tolerance
    // termination" << std::endl;
    return true;
  }
  // std::cout << "[Quantile_Regressor] Error: " << err << std::endl;
  return false;
}

Feature Quantile_Regressor::feature_selection_criteria(
    const std::vector<std::vector<Feature>> &features) const {
  uint32_t N_timeseries = features.size();
  uint32_t N_features = features[0].size();
  std::vector<float> MAEs(N_features);

  for (int i = 0; i < N_timeseries; i++) {
    for (int j = 0; j < N_features; j++) {
      MAEs[j] += features[i][j].f_ERR;
    }
  }
  uint32_t best_feature_idx = 0;
  for (int j = 0; j < N_features; j++) {
    if ((MAEs[j] > 0) && (MAEs[j] < MAEs[best_feature_idx])) {
      best_feature_idx = j;
    }
  }
  Feature best_avg_feature;
  best_avg_feature.f_ERR = 0;
  for (int j = 0; j < N_features; j++) {
    best_avg_feature.g += features[j][best_feature_idx].g;
    best_avg_feature.f_ERR += features[j][best_feature_idx].f_ERR;
  }

  best_avg_feature.g /= N_timeseries;
  best_avg_feature.f_ERR /= N_timeseries;
  best_avg_feature.index = features[0][best_feature_idx].index;
  best_avg_feature.tag = FEATURE_REGRESSION;


  return best_avg_feature;
}

} // namespace FROLS::Regression
