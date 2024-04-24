#include "Quantile_Regressor.hpp"
// #include "ortools/base/commandlineflags.h"
// #include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"
// #include "ortools/linear_solver/linear_solver.pb.h"
// #include "ortools/sat/cp_model.h"

#include <filesystem>
#include <fmt/format.h>
#include <limits>
namespace FROLS::Regression {
Quantile_Regressor::Quantile_Regressor(const Quantile_Param &p)
    : tau(p.tau), Regressor(p), solver_type(p.solver_type),
      problem_type(p.problem_type),y_tol(p.y_tol), x_tol(p.x_tol) {}

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
    if ((y.lpNorm<Eigen::Infinity>() < y_tol) || (x.lpNorm<Eigen::Infinity>() < x_tol))
    {
      float ynorm = y.lpNorm<Eigen::Infinity>();
      float xnorm = x.lpNorm<Eigen::Infinity>();
      return Feature{std::numeric_limits<float>::infinity(), 0, 0, 0., FEATURE_INVALID};
    }


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

      float theta_sol = theta->solution_value();
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

  bool Quantile_Regressor::objective_condition(float f0, float f1) const
  {
    return f0 < f1;
  }


} // namespace FROLS::Regression
