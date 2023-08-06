#ifndef SYCL_GRAPH_QUANTILE_REGRESSION_HPP
#define SYCL_GRAPH_QUANTILE_REGRESSION_HPP

#include <ortools/linear_solver/linear_solver.h>
#include <Eigen/Dense>
#include <iostream>
#include <string>

float quantile_regression(const Eigen::VectorXf& x, const Eigen::VectorXf& y, float tau, float y_tol = 0.0F, float x_tol = 0.0F)
{
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    using namespace operations_research;
    operations_research::MPSolver::OptimizationProblemType problem_type = operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING;
    // if all y is 0, return 0
    if (y.template lpNorm<Eigen::Infinity>() == 0 || x.template lpNorm<Eigen::Infinity>() == 0)
    {
        return 0.0F;
    }

    // std::cout << "x: " << x.transpose() << std::endl;
    // std::cout << "y: " << y.transpose() << std::endl;

    if ((y.template lpNorm<Eigen::Infinity>() < y_tol) || (x.template lpNorm<Eigen::Infinity>() < x_tol))
    {
        float const ynorm = y.lpNorm<Eigen::Infinity>();
        float const xnorm = x.lpNorm<Eigen::Infinity>();
        return std::numeric_limits<float>::infinity();
    }

    uint32_t const N_rows = x.rows();
    static uint32_t count = 0;
    std::string const solver_name = "Quantile_Solver_" + std::to_string(count++);
    std::unique_ptr<MPSolver> solver = std::make_unique<MPSolver>(solver_name, problem_type);
    const float infinity = solver->infinity();
    // theta_neg = solver->MakeNumVar(0.0, infinity, "theta_neg");
    // theta_pos = solver->MakeNumVar(0.0, infinity, "theta_pos");
    operations_research::MPVariable *theta =
        solver->MakeNumVar(-infinity, infinity, "theta");
    std::vector<operations_research::MPVariable *> u_pos;
    std::vector<operations_research::MPVariable *> u_neg;
    std::vector<operations_research::MPConstraint *> g(N_rows);

    operations_research::MPObjective *objective = nullptr;
    objective = solver->MutableObjective();
    objective->SetMinimization();
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
    solver->MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);

    std::for_each(u_pos.begin(), u_pos.end(),
                  [=](auto &u)
                  { objective->SetCoefficient(u, tau); });
    std::for_each(u_neg.begin(), u_neg.end(),
                  [=](auto &u)
                  { objective->SetCoefficient(u, (1 - tau)); });
    // std::generate(g.begin(), g.end(),
    //               [&]() { return solver->MakeRowConstraint(); });

    for (int i = 0; i < N_rows; i++)
    {
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

    float theta_sol = std::numeric_limits<float>::infinity();

    if (solver_status)
    {
        float const f = objective->Value();

        std::vector<float> u_neg_sol(N_rows);
        std::vector<float> u_pos_sol(N_rows);
        for (int i = 0; i < N_rows; i++)
        {
            u_neg_sol[i] = u_neg[i]->solution_value();
            u_pos_sol[i] = u_pos[i]->solution_value();
        }

        theta_sol = theta->solution_value();
    }
    else
    {

        std::cout << "[Quantile_Regressor] Warning: Quantile regression failed"
                  << std::endl;
        std::for_each(g.begin(), g.end(), [](auto &gi)
                      { gi->Clear(); });
    }

    return theta_sol;
}
#endif
