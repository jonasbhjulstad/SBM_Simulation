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
    struct Quantile_LP
    {
        const double tau;
        const std::string solver_type;
        Quantile_LP(double tau, const std::string& solver_type) : tau(tau), solver_type(solver_type){}
        void construct(size_t N_rows)
        {
            using namespace operations_research;
            MPSolver::OptimizationProblemType problem_type;
            if (!MPSolver::ParseSolverType(solver_type, &problem_type)) {
                throw std::runtime_error("Solver id " + solver_type + " not recognized");
            }

            if (!MPSolver::SupportsProblemType(problem_type)) {
                throw std::runtime_error("Supports for solver " + solver_type + " not linked in.");
            }
            solver = std::make_unique<MPSolver>("Quantile_Solver", problem_type);
            const double infinity = solver->infinity();

            theta_neg = solver->MakeNumVar(0.0, infinity, "theta_neg");
            theta_pos = solver->MakeNumVar(0.0, infinity, "theta_pos");

            solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
            solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);
            objective = solver->MutableObjective();
            objective->SetMinimization();
            std::for_each(u_pos.begin(), u_pos.end(),
                          [=](auto u) { objective->SetCoefficient(u, tau / N_rows); });
            std::for_each(u_neg.begin(), u_neg.end(),
                          [=](auto u) { objective->SetCoefficient(u, (1 - tau) / N_rows); });
            g.resize(N_rows);
            std::for_each(g.begin(), g.end(), [&](auto &gi) { gi = solver->MakeRowConstraint(); });
        }
        ~Quantile_LP()
        {
            solver->Clear();
        }

        operations_research::MPObjective * objective;
        std::unique_ptr<operations_research::MPSolver> solver;
        operations_research::MPVariable *theta_neg;
        operations_research::MPVariable *theta_pos;
        std::vector<operations_research::MPVariable *> u_pos;
        std::vector<operations_research::MPVariable *> u_neg;
        std::vector<operations_research::MPConstraint *> g;
    };

    struct Quantile_Regressor : public Regressor {
        const double tau;
        const std::string solver_type;


        Quantile_Regressor(double tau, double tol, double theta_tol, size_t N_terms_max = std::numeric_limits<size_t>::infinity(), const std::string solver_type = "GLOP");

    private:
        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        std::vector<Feature> candidate_regression(crMat &X, crVec &y, const std::vector<Feature> &used_features)const;

        bool tolerance_check(crMat &Q, crVec &y, const std::vector<Feature> &best_features) const;

        size_t feature_selection_idx = 0;

        Quantile_LP construct_solver(size_t N_rows) const;


    };
}


#endif