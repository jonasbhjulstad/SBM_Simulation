#include "Quantile_Regressor.hpp"
#include <casadi/casadi.hpp>
#include <filesystem>
#include <limits>

namespace FROLS::Regression {

    bool Quantile_Regressor::single_feature_quantile_regression(crVec &x, crVec &y, Feature& best_feature, size_t feature_idx) const{
        using namespace casadi;
        // tau - Quantile
        size_t N_rows = x.rows();
        using namespace casadi;
        MX theta_pos = MX::sym("theta_pos");
        MX theta_neg = MX::sym("theta_neg");
        MX u_pos = MX::sym("u_pos", N_rows);
        MX u_neg = MX::sym("u_neg", N_rows);


        std::map<std::string, DM> res;
        std::map<std::string, DM> arg;
        arg["lbx"] = 0;
        arg["ubx"] = inf;
        arg["lbg"] = 0;
        arg["ubg"] = 0;
        arg["x0"] = 1e-6;
        Dict opts;
        opts["ipopt.print_level"] = 0;
        opts["ipopt.linear_solver"] = "ma27";
        opts["ipopt.file_print_level"] = 0;
        opts["print_time"] = 0;
        opts["ipopt.sb"] = "yes";
        static size_t subproblem_iter = 0;
        static size_t feature_idx_prev = 0;
        if (feature_idx_prev > feature_idx)
        {
            subproblem_iter++;
        }
        std::filesystem::create_directory(std::string(FROLS_LOG_DIR) + "/Quantile_Regression/");
        opts["ipopt.output_file"] = (std::string(FROLS_LOG_DIR) +
                                     "/Quantile_Regression/Quantile_Regression_Optimization_" +
                                     std::to_string(subproblem_iter) + "_Feature_" + std::to_string(feature_idx) +
                                     ".txt");
        opts["ipopt.file_print_level"] = 6;
        DM dm_y = DM(std::vector<double>(y.data(), y.data() + N_rows));
        DM xi =
                DM(std::vector<double>(x.data(), x.data() + N_rows));
        MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
        MX f_obj_vec = (tau * (u_pos) + (1 - tau) * (u_neg)) / N_rows;
        double W_initial = 100000;
        f_obj_vec(0) *= W_initial;
        MX f_obj = sum1(f_obj_vec);
        MXDict nlp = {{"x", vertcat(theta_pos, theta_neg, u_pos, u_neg)},
                      {"f", f_obj},
                      {"g", g}};

        Function solver = nlpsol("solver", "ipopt", nlp, opts);
        res = solver(arg);

        std::vector<double> res_x = res["x"].get_elements();
        bool solver_status = solver.stats()["success"];
        double f = res["f"](0).scalar();
        double theta_val = res_x[0] - res_x[1];
        qr_logger->info("{:^15}{:^15}{:^15.3f}{:^15.3f}", feature_idx, (int) solver_status, theta_val, f);

        if (solver_status) {
            if (f < best_feature.f_ERR) {
                best_feature.f_ERR = f;
                best_feature.g = theta_val;
                best_feature.index = feature_idx;
            }
        } else {
            std::cout << "[Quantile_Regressor] Warning: Quantile regression failed" << std::endl;
        }

        double theta_pos_opt = res_x[0];
        double theta_neg_opt = res_x[1];
        std::vector<double> u_pos_opt = std::vector(res_x.begin()+2, res_x.begin() + N_rows + 2);
        std::vector<double> u_neg_opt = std::vector(res_x.begin()+2 + N_rows, res_x.end());
        subproblem_logger->info("Regression problem " + std::to_string(subproblem_iter) + ", feature " + std::to_string(feature_idx));
        subproblem_logger->info("Theta+:{:^15.3f}\tTheta-:{:^15.3f}", theta_pos_opt, theta_neg_opt);
        subproblem_logger->info("u+:\t{:.3f}", fmt::join(u_pos_opt, ","));
        subproblem_logger->info("u-:\t{:.3f}", fmt::join(u_neg_opt, ","));
        return solver_status;

    }

    void Quantile_Regressor::feature_select(crMat &X, crVec &y,
                                            std::vector<Feature> &used_features) const {
        std::vector<size_t> used_indices(used_features.size());
        size_t N_rows = X.rows();
        std::transform(used_features.begin(), used_features.end(), used_indices.begin(),
                       [](auto feature) { return feature.index; });
        qr_logger->info("Feature selection with used indices:\t{}", fmt::join(used_indices, ","));
        qr_logger->info("{:^15}{:^15}{:^15}{:^15}", "Subproblem", "success", "theta", "f");


        size_t N_features = X.cols();
        Feature best_feature;
        best_feature.f_ERR = std::numeric_limits<double>::max();

        Vec y_orth = vec_orthogonalize(y, predict(X, used_features));
        qr_logger->info("Response y:\t{:.3f}", fmt::join(std::vector(y.data(), y.data() + N_rows), ","));
        qr_logger->info("Orthogonalized y:\t{:.3f}",
                        fmt::join(std::vector(y_orth.data(), y_orth.data() + N_rows), ","));

        for (int i = 0; i < N_features; i++) {
            // If the feature is already used, skip it


            if (std::none_of(used_features.begin(), used_features.end(), [&i](auto f) { return f.index == i; })) {
                bool solve_status = single_feature_quantile_regression(X.col(i), y_orth, best_feature, i);
            }
        }

        qr_logger->info("Best feature:{:^15}{:^15.3f}{:^15.3f}", best_feature.index, best_feature.g,
                        best_feature.f_ERR);
        used_features.push_back(best_feature);
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

} // namespace FROLS::Features
