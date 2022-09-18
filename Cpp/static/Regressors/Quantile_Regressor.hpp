#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
#include <ortools/base/commandlineflags.h>
//#include <ortools/base/init_google.h>
#include <ortools/base/logging.h>
#include <ortools/linear_solver/linear_solver.h>
#include <ortools/linear_solver/linear_solver.pb.h>
#include <numeric>
#include <memory>

namespace FROLS::Regression {
    struct Quantile_Regressor : public Regressor {
        const double tau;
        const std::string solver_type;

        Quantile_Regressor(double tau, double tol, double theta_tol, const std::string solver_type = "CLP");

    private:
        Feature single_feature_quantile_regression(crVec &x, crVec &y, size_t feature_index);

        Feature feature_select(crMat &X, crVec &y, const std::vector<Feature> &used_features);

        bool tolerance_check(crMat &Q, crVec &y, const std::vector<Feature> &best_features) const;

        std::shared_ptr<spdlog::logger> qr_logger;
        std::shared_ptr<spdlog::logger> subproblem_logger;
        size_t feature_selection_idx = 0;

        void construct_solver(size_t N_rows);

        operations_research::MPObjective * objective;
        std::unique_ptr<operations_research::MPSolver> solver;
        operations_research::MPVariable *theta_neg;
        operations_research::MPVariable *theta_pos;
        std::vector<operations_research::MPVariable *> u_pos;
        std::vector<operations_research::MPVariable *> u_neg;
        std::vector<operations_research::MPConstraint *> g;

    };
}


#endif