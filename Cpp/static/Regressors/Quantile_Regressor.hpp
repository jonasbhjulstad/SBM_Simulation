#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
#include "ortools/base/commandlineflags.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#include <memory>
#include <numeric>
#include <thread>

namespace FROLS::Regression {
struct Quantile_Param : public Regressor_Param {
  float tau = .95;
  uint32_t N_rows;
  uint32_t N_threads = 4;
  const std::string solver_type = "CLP";
  operations_research::MPSolver::OptimizationProblemType problem_type =
      operations_research::MPSolver::OptimizationProblemType::GLOP_LINEAR_PROGRAMMING;
};



struct Quantile_Regressor : public Regressor {
  const float tau;

  Quantile_Regressor(const Quantile_Param &p);
  Feature
  feature_selection_criteria(const std::vector<Feature> &features) const;

  void theta_solve(crMat &A, crVec &g, crMat &X, crVec &y,
                   std::vector<Feature> &features) const;

private:
  Feature single_feature_regression(const Vec &x, const Vec &y) const;

  std::vector<Feature>
  candidate_regression(crMat &X, crMat &Q_global, crVec &y,
                       const std::vector<Feature> &used_features) const;

  bool tolerance_check(crMat &Q, crVec &y,
                       const std::vector<Feature> &best_features) const;

  uint32_t feature_selection_idx = 0;

  operations_research::MPSolver::OptimizationProblemType problem_type;
  const std::string solver_type;

  // Quantile_LP construct_solver(uint32_t N_rows) const;
};
} // namespace FROLS::Regression

#endif