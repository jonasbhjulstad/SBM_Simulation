#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
#include <ortools/base/commandlineflags.h>
#include <ortools/base/logging.h>
#include <ortools/linear_solver/linear_solver.h>
#include <ortools/linear_solver/linear_solver.pb.h>
#include <ortools/glop/parameters.pb.h>
#include <numeric>
#include <memory>

namespace FROLS::Regression
{
    struct Quantile_Param : public Regressor_Param
    {
        float tau = .95;
        const std::string solver_type = "GLOP";
    };
    struct Quantile_LP
    {
        const float tau;
        const std::string solver_type;
        Quantile_LP(float tau, const std::string &solver_type) : tau(tau), solver_type(solver_type) {}
        void construct(uint32_t N_rows)
        {
            using namespace operations_research;
            MPSolver::OptimizationProblemType problem_type;
            if (!MPSolver::ParseSolverType(solver_type, &problem_type))
            {
                throw std::runtime_error("Solver id " + solver_type + " not recognized");
            }

            if (!MPSolver::SupportsProblemType(problem_type))
            {
                throw std::runtime_error("Supports for solver " + solver_type + " not linked in.");
            }
            solver = std::make_unique<MPSolver>("Quantile_Solver", problem_type);
            const float infinity = solver->infinity();

            // theta_neg = solver->MakeNumVar(0.0, infinity, "theta_neg");
            // theta_pos = solver->MakeNumVar(0.0, infinity, "theta_pos");
            theta = solver->MakeNumVar(-infinity, infinity, "theta");

            solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
            solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);
            objective = solver->MutableObjective();
            objective->SetMinimization();
            std::for_each(u_pos.begin(), u_pos.end(),
                          [=](auto u)
                          { objective->SetCoefficient(u, tau); });
            std::for_each(u_neg.begin(), u_neg.end(),
                          [=](auto u)
                          { objective->SetCoefficient(u, (1 - tau) ); });
            g.resize(N_rows);
            std::for_each(g.begin(), g.end(), [&](auto &gi)
                          { gi = solver->MakeRowConstraint(); });
            // solver->SetSolverSpecificParametersAsString("primal_feasibility_tolerance: 1e-6");
            

        }
        ~Quantile_LP()
        {
            solver->Clear();
        }

        operations_research::MPObjective *objective;
        std::unique_ptr<operations_research::MPSolver> solver;
        // operations_research::MPVariable *theta_neg;
        // operations_research::MPVariable *theta_pos;
        operations_research::MPVariable *theta;
        std::vector<operations_research::MPVariable *> u_pos;
        std::vector<operations_research::MPVariable *> u_neg;
        std::vector<operations_research::MPConstraint *> g;
    };

    struct Quantile_Regressor : public Regressor
    {
        const float tau;
        const std::string solver_type;

        Quantile_Regressor(const Quantile_Param &p);
        Feature feature_selection_criteria(const std::vector<Feature> &features) const;

        void theta_solve(crMat &A, crVec &g, crMat& X, crVec& y, std::vector<Feature> &features) const;

    private:
        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        std::vector<Feature> candidate_regression(crMat &X, crMat& Q_global, crVec &y, const std::vector<Feature> &used_features) const;

        bool tolerance_check(crMat &Q, crVec &y, const std::vector<Feature> &best_features) const;

        uint32_t feature_selection_idx = 0;

        Quantile_LP construct_solver(uint32_t N_rows) const;

    };
}

#endif