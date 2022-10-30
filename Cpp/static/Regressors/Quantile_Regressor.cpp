#include "Quantile_Regressor.hpp"

#include <FROLS_Execution.hpp>
#include <filesystem>
#include <limits>
#include <fmt/format.h>
namespace FROLS::Regression {
    Quantile_Regressor::Quantile_Regressor(const Quantile_Param &p)
            : tau(p.tau), Regressor(p), solver_type(p.solver_type), problem_type(p.problem_type){}

    void Quantile_Regressor::theta_solve(crMat &A, crVec &g, crMat& Q, crVec& y, std::vector<Feature> &features) const {
        std::vector<Feature> feature_tmp;
        feature_tmp.reserve(features.size());
        Mat y_diffs(Q.rows(), features.size()+1);
        y_diffs.col(0) = y - Q.col(0)*features[0].g;
        features[0].theta = features[0].g;

        Vec coefficients =
                A.inverse() * g;
        for (
                int i = 0;
                i < coefficients.rows();
                i++) {
            features[i].
                    theta = coefficients[i];
        }

    }


    Feature Quantile_Regressor::single_feature_regression(const Vec &x, const Vec &y) const {
        {


            using namespace operations_research;
            uint32_t N_rows = x.rows();

            using namespace operations_research;
            MPSolver solver("Quantile_Solver", problem_type);
            const float infinity = solver.infinity();
            // theta_neg = solver.MakeNumVar(0.0, infinity, "theta_neg");
            // theta_pos = solver.MakeNumVar(0.0, infinity, "theta_pos");
            operations_research::MPVariable *theta = solver.MakeNumVar(-infinity, infinity, "theta");
            std::vector<operations_research::MPVariable *> u_pos;
            std::vector<operations_research::MPVariable *> u_neg;
            std::vector<operations_research::MPConstraint *> g;
            operations_research::MPObjective *objective;

            solver.MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
            solver.MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);
            objective = solver.MutableObjective();
            objective->SetMinimization();
            std::for_each(u_pos.begin(), u_pos.end(),
                        [=](auto u) { objective->SetCoefficient(u, tau); });
            std::for_each(u_neg.begin(), u_neg.end(),
                        [=](auto u) { objective->SetCoefficient(u, (1-tau)); });
            g.resize(N_rows);
            std::for_each(g.begin(), g.end(),
                        [&](auto &gi) { gi = solver.MakeRowConstraint(); });


            float eps = 4.f;
            static int counter = 0;
            counter++;
            for (int i = 0; i < N_rows; i++) {
                // g[i]->SetCoefficient(theta_pos, x(i));
                // g[i]->SetCoefficient(theta_neg, -x(i));
                g[i]->SetCoefficient(theta, x(i));
                g[i]->SetCoefficient(u_pos[i], 1);
                g[i]->SetCoefficient(u_neg[i], -1);
                g[i]->SetBounds(y[i], y[i]);
            }
//            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
            const bool solver_status = solver.Solve() == MPSolver::OPTIMAL;
            Feature candidate;
            candidate.f_ERR = std::numeric_limits<float>::infinity();

            if (solver_status) {
                float f = objective->Value();

                std::vector<float> u_neg_sol(N_rows);
                std::vector<float> u_pos_sol(N_rows);
                for (int i = 0; i < N_rows; i++) {
                    u_neg_sol[i] = u_neg[i]->solution_value();
                    u_pos_sol[i] = u_pos[i]->solution_value();
                }
                // float theta_sol = (theta_pos->solution_value() - theta_neg->solution_value());
                // float theta_pos_sol = theta_pos->solution_value();
                // float theta_neg_sol = theta_neg->solution_value();
                float theta_sol = theta->solution_value();
                // fmt::print("Solve status, {}, {}, {}\n", solver_status, f, theta_sol);
                // if(theta_sol == 0)
                    // std::cout << y.head(10).transpose() << std::endl;
                

                // std::for_each(g.begin(), g.end(), [&](auto &gi) { gi->Clear(); });
                return Feature{f, theta_sol, 0, 0., FEATURE_REGRESSION};
            } else {
                std::cout << "[Quantile_Regressor] Warning: Quantile regression failed" << std::endl;
                // std::for_each(g.begin(), g.end(), [](auto &gi) { gi->Clear(); });
                return Feature{};
            }
        }
    }

    std::vector<Feature>
    Quantile_Regressor::candidate_regression(crMat &X, crMat& Q_global, crVec &y, const std::vector<Feature> &used_features) const {
        //get used indices of used_features
        std::vector<int> used_indices;
        used_indices.reserve(used_features.size());
        std::transform(used_features.begin(), used_features.end(), std::back_inserter(used_indices),
                       [](const Feature &f) { return f.index; });
        
        Vec y_diff = y - predict(Q_global, used_features);
        std::vector<uint32_t> candidate_idx = unused_feature_indices(used_features, X.cols());
        std::vector<Feature> candidates(candidate_idx.size());
        std::transform(FROLS::execution::par_unseq, candidate_idx.begin(), candidate_idx.end(), candidates.begin(),
                       [=](const uint32_t &idx) {
                           Feature f = single_feature_regression(X.col(idx), y_diff);
                           f.index = idx;
                           return f;
                       });
        
        return candidates;
    }

    bool Quantile_Regressor::tolerance_check(
            crMat &X, crVec &y, const std::vector<Feature> &best_features) const {
        Vec y_pred = predict(X, best_features);
        Vec diff = y - y_pred;
        uint32_t N_samples = y.rows();
        float err = (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum() / N_samples;
        bool no_improvement = (best_features.size() > 1) && (best_features.back().f_ERR < err);
        if (no_improvement)
        {
            // std::cout << "[Quantile_Regressor] Warning: Termination due to lack of improvement" << std::endl;
            return true;
        }
        else if(err < tol)
        {
            // std::cout << "[Quantile_Regressor] Status: Successful tolerance termination" << std::endl;
            return true;
        }
        // std::cout << "[Quantile_Regressor] Error: " << err << std::endl;
        return false;
    }

    Feature Quantile_Regressor::feature_selection_criteria(const std::vector<Feature> &features) const
    {
        return *std::min_element(features.begin(), features.end(), [](const Feature &f1, const Feature &f2) {
            return f1.f_ERR < f2.f_ERR;
        });
    }


} // namespace FROLS::Features
