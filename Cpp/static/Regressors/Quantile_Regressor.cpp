#include "Quantile_Regressor.hpp"

#include <filesystem>
#include <limits>
#include <execution>
namespace FROLS::Regression {
    Quantile_Regressor::Quantile_Regressor(double tau, double tol, double theta_tol, size_t N_terms_max, const std::string solver_type)
            : tau(tau), Regressor(tol, theta_tol, N_terms_max), solver_type(solver_type){}


    Feature Quantile_Regressor::single_feature_regression(const Vec &x, const Vec &y) const {
        {


            using namespace operations_research;
            size_t N_rows = x.rows();
            Quantile_LP LP(tau, solver_type);
            LP.construct(N_rows);

            for (int i = 0; i < N_rows; i++) {
                LP.g[i]->SetCoefficient(LP.theta_pos, x(i));
                LP.g[i]->SetCoefficient(LP.theta_neg, -x(i));
                LP.g[i]->SetCoefficient(LP.u_pos[i], 1);
                LP.g[i]->SetCoefficient(LP.u_neg[i], -1);
                LP.g[i]->SetBounds(y[i], y[i]);
            }

//            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;

            const bool solver_status = LP.solver->Solve() == MPSolver::OPTIMAL;
            Feature candidate;
            candidate.f_ERR = std::numeric_limits<double>::infinity();

            if (solver_status) {
                double f = LP.objective->Value();

                std::vector<double> u_neg_sol(N_rows);
                std::vector<double> u_pos_sol(N_rows);
                for (int i = 0; i < N_rows; i++) {
                    u_neg_sol[i] = LP.u_neg[i]->solution_value();
                    u_pos_sol[i] = LP.u_pos[i]->solution_value();
                }
                double theta_sol = LP.theta_pos->solution_value() - LP.theta_neg->solution_value();
                std::for_each(LP.g.begin(), LP.g.end(), [&](auto &gi) { gi->Clear(); });
                return Feature{f, theta_sol, 0, 0.};
            } else {
                std::cout << "[Quantile_Regressor] Warning: Quantile regression failed" << std::endl;
                std::for_each(LP.g.begin(), LP.g.end(), [&](auto &gi) { gi->Clear(); });
                return Feature{std::numeric_limits<double>::infinity(), 0., 0, 0.};
            }


        }
    }

    std::vector<Feature> Quantile_Regressor::candidate_regression(crMat &X, crVec &y, const std::vector<Feature>& used_features) const {
        const Vec y_diff = y - predict(X, used_features);

        std::vector<size_t> candidate_idx = unused_feature_indices(used_features, X.cols());
        std::vector<Feature> candidates(candidate_idx.size());
        std::transform(candidate_idx.begin(), candidate_idx.end(), candidates.begin(),
                       [=](const size_t &idx) {
                           Feature f = single_feature_regression(X.col(idx), y);
                           f.index = idx;
                           return f;
                       });
//        for (int i = 0; i < candidate_idx.size(); i++)
//        {
//            candidates[i] = single_feature_regression(X.col(i), y);
//            candidates[i].index = candidate_idx[i];
//        }
        return candidates;
    }

    bool Quantile_Regressor::tolerance_check(
            crMat &X, crVec &y, const std::vector<Feature> &best_features) const {
        Vec y_pred = predict(X, best_features);
        Vec diff = y - y_pred;
        size_t N_samples = y.rows();
        double err = (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum() / N_samples;
        return err < tol;
    }

} // namespace FROLS::Features
