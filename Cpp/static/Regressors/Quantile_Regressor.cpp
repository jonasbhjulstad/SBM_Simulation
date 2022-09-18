#include "Quantile_Regressor.hpp"

#include <filesystem>
#include <limits>

namespace FROLS::Regression {
    Quantile_Regressor::Quantile_Regressor(double tau, double tol, double theta_tol, const std::string solver_type)
            : tau(tau), Regressor(tol, theta_tol), solver_type(solver_type),
              qr_logger(spdlog::basic_logger_mt(("Quantile_Regressor_" + std::to_string(regressor_id)).c_str(),
                                                (std::string(FROLS_LOG_DIR) + "/quantile_regressor_" +
                                                 std::to_string(regressor_id) + ".txt").c_str(), true)),
              subproblem_logger(spdlog::basic_logger_mt("Subproblem_Logger", (std::string(FROLS_LOG_DIR) +
                                                                              "/quantile_subproblem_variables.txt").c_str(),
                                                        true)) {
        qr_logger->set_level(spdlog::level::debug);
        std::filesystem::create_directory(std::string(FROLS_LOG_DIR) + "/Quantile_Regression/");

        google::InitGoogleLogging("Quantile_Log");

    }

    Feature Quantile_Regressor::single_feature_quantile_regression(crVec &x, crVec &y,
                                                                size_t feature_idx) {
        {


            using namespace operations_research;
            static size_t N_rows;
            if (N_rows != x.rows()) {
                N_rows = x.rows();
                construct_solver(N_rows);
            }


            for (int i = 0; i < N_rows; i++) {
                g[i]->SetCoefficient(theta_pos, x(i));
                g[i]->SetCoefficient(theta_neg, -x(i));
                g[i]->SetCoefficient(u_pos[i], 1);
                g[i]->SetCoefficient(u_neg[i], -1);
                g[i]->SetBounds(y[i], y[i]);
            }

//            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;


            static size_t subproblem_iter = 0;
            static size_t feature_idx_prev = 0;
            if (feature_idx_prev > feature_idx) {
                subproblem_iter++;
            }
            const bool solver_status = solver->Solve() == MPSolver::OPTIMAL;
            Feature candidate;
            candidate.f_ERR = std::numeric_limits<double>::infinity();

            if (solver_status) {
                double f = objective->Value();

                std::vector<double> u_neg_sol(N_rows);
                std::vector<double> u_pos_sol(N_rows);
                for (int i = 0; i < N_rows; i++) {
                    u_neg_sol[i] = u_neg[i]->solution_value();
                    u_pos_sol[i] = u_pos[i]->solution_value();
                }
                double theta_sol = theta_pos->solution_value() - theta_neg->solution_value();
                qr_logger->info("{:^15}{:^15}{:^15.3f}{:^15.3f}", feature_idx, (int) solver_status, theta_sol, f);
                subproblem_logger->info("Theta+:{:^15.3f}\tTheta-:{:^15.3f}", theta_pos->solution_value(),
                                        theta_neg->solution_value());
                subproblem_logger->info("u+:\t{:.3f}", fmt::join(u_pos_sol, ","));
                subproblem_logger->info("u-:\t{:.3f}", fmt::join(u_neg_sol, ","));
                subproblem_logger->info("Regression problem " + std::to_string(subproblem_iter) + ", feature " +
                                        std::to_string(feature_idx));
                std::for_each(g.begin(), g.end(), [&](auto &gi) { gi->Clear(); });
                return Feature{f, theta_sol, feature_idx, 0.};
            } else {
                std::cout << "[Quantile_Regressor] Warning: Quantile regression failed" << std::endl;
                std::for_each(g.begin(), g.end(), [&](auto &gi) { gi->Clear(); });
                return Feature{std::numeric_limits<double>::infinity(), 0.,0,0.};
            }



        }
    }

    Feature Quantile_Regressor::feature_select(crMat &X, crVec &y,
                                            const std::vector<Feature> &used_features) {
        std::vector<size_t> used_indices(used_features.size());
        size_t N_rows = X.rows();
        std::transform(used_features.begin(), used_features.end(), used_indices.begin(),
                       [](auto feature) { return feature.index; });
        qr_logger->info("Feature selection with used indices:\t{}", fmt::join(used_indices, ","));
        qr_logger->info("{:^15}{:^15}{:^15}{:^15}", "Subproblem", "success", "theta", "f");


        size_t N_features = X.cols();

        Vec y_diff = y - predict(X, used_features);
        qr_logger->info("Response y:\t{:.3f}", fmt::join(std::vector(y.data(), y.data() + N_rows), ","));
        qr_logger->info("Orthogonalized y:\t{:.3f}",
                        fmt::join(std::vector(y_diff.data(), y_diff.data() + N_rows), ","));
        Feature best_feature;
        best_feature.f_ERR = std::numeric_limits<double>::infinity();
        for (int i = 0; i < N_features; i++) {
            // If the feature is already used, skip it
            if (std::none_of(used_features.begin(), used_features.end(), [&i](auto f) { return f.index == i; })) {
                Feature candidate = single_feature_quantile_regression(X.col(i), y_diff, i);
                best_feature = (candidate.f_ERR < best_feature.f_ERR) ? candidate : best_feature;
            }
        }

        qr_logger->info("Best feature:{:^15}{:^15.3f}{:^15.3f}", best_feature.index, best_feature.g,
                        best_feature.f_ERR);
        return best_feature;
    }

    bool Quantile_Regressor::tolerance_check(
            crMat &X, crVec &y, const std::vector<Feature> &best_features) const {
        Vec y_pred = predict(X, best_features);
        Vec diff = y - y_pred;
        size_t N_samples = y.rows();
        double err = (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum() / N_samples;
        qr_logger->info("Tolerance-check Average MAE:{:^15.3f}", err);
        return err < tol;
    }

    void Quantile_Regressor::construct_solver(size_t N_rows) {
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
                      [&](const auto &u) { objective->SetCoefficient(u, tau / N_rows); });
        std::for_each(u_neg.begin(), u_neg.end(),
                      [&](const auto &u) { objective->SetCoefficient(u, (1 - tau) / N_rows); });
        g.resize(N_rows);
        std::for_each(g.begin(), g.end(), [&](auto &gi) { gi = solver->MakeRowConstraint(); });


    }

} // namespace FROLS::Features
