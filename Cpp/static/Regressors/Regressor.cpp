#include "Regressor.hpp"
#include <FROLS_Execution.hpp>
#include <fmt/format.h>
#include <FROLS_Eigen.hpp>

namespace FROLS::Regression {

    Regressor::Regressor(const Regressor_Param &p)
            : tol(p.tol), theta_tol(p.theta_tol), N_terms_max(p.N_terms_max) {}


    std::vector<Feature> Regressor::single_fit(const Mat &X, const Vec &y) const {
        uint32_t N_features = X.cols();
        Mat Q_global = Mat::Zero(X.rows(), N_features);
        Mat Q_current = Q_global;
        Mat A = Mat::Zero(N_features, N_features);
        Vec g = Vec::Zero(N_features);
        std::vector<Feature> best_features;
        best_features.reserve(N_terms_max);
        uint32_t end_idx = N_features;

        // fmt::print("Max features: {}\n", N_terms_max);
        // Perform one feature selection iteration for each feature
        for (int j = 0; j < N_features; j++) {
            // fmt::print("Feature {}\n", j+1);
            // Compute remaining variance by orthogonalizing the current feature
            Q_current =
                    used_feature_orthogonalize(X, Q_global, best_features);
            // Determine the best feature to add to the feature set
            Feature f = best_feature_select(Q_current, Q_global, y, best_features);

            if ((f.tag == FEATURE_INVALID) || (j >= (N_terms_max))) {
                end_idx = j;
                break;
            }
            best_features.push_back(f);
            Q_global.col(j) = Q_current.col(best_features[j].index);


            g[j] = best_features[j].g;

            for (int m = 0; m < j; m++) {
                A(m, j) = cov_normalize(Q_global.col(m), X.col(best_features[j].index));
            }
            A(j, j) = 1;

            // If ERR-tolerance is met, return non-orthogonalized parameters
            if (tolerance_check(Q_global.leftCols(j +1), y, best_features)) {
                end_idx = j+1;
                break;
            }


            Q_current.setZero();
        }
        theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx), X, y, best_features);

        return best_features;
    }

    std::vector<Feature>
    Regressor::single_fit(const Mat &X, const Vec &y, std::vector<Feature> preselect_features) const {

        std::for_each(preselect_features.begin(), preselect_features.end(), [](auto& f){f.tag = FEATURE_PRESELECTED;});
        Vec y_diff = y;
        std::for_each(preselect_features.begin(), preselect_features.end(), [&](const auto& f){
            y_diff -= X.col(f.index)*f.theta;
        });
        Mat X_unused = X;

        std::vector<Feature> result;
        result.insert(result.begin(), preselect_features.begin(), preselect_features.end());
        std::vector<Feature> identified_features = single_fit(X_unused, y_diff);
        result.insert(result.end(), identified_features.begin(), identified_features.end());
        return result;

    }

    Feature Regressor::best_feature_select(crMat &X, crMat& Q_global, crVec &y, const std::vector<Feature> &used_features) const {
        const std::vector<Feature> candidates = candidate_regression(X, Q_global, y, used_features);
        std::vector<Feature> thresholded_candidates;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(thresholded_candidates),
                     [&](const auto &f) {
                         return abs(f.g) > theta_tol;
                     });
        static bool warn_msg = true;

        Feature res;
        switch (thresholded_candidates.size()) {
            case 0:
                if (warn_msg)
                    std::cout << "[Regressor] Warning: threshold is too high for candidates" << std::endl;
                warn_msg = false;
                break;
            case 1:
                res = thresholded_candidates[0];
                break;
            default:
                res = feature_selection_criteria(thresholded_candidates);
                break;
        }

        return res;
    }

    std::vector<Feature> Regressor::fit(crMat &X, crVec &y) {
        if ((X.rows() != y.rows())) {
            throw std::invalid_argument("X, U and y must have same number of rows");
        }
        std::vector<Feature> result;
        result = this->single_fit(X, y);
        return result;
    }

    Vec Regressor::predict(crMat &Q, const std::vector<Feature> &features) const {
        Vec y_pred(Q.rows());
        y_pred.setZero();
        uint32_t i = 0;
        for (const auto &feature: features) {
            if (feature.f_ERR == -1) {
                break;
            }
            y_pred += Q.col(i) * feature.g;
            i++;
        }
        return y_pred;
    }

    struct Fit_Data {
        std::vector<Feature> preselect_features;
        Vec y;
    };


    std::vector<Feature> Regressor::transform_fit(crMat &X_raw, crMat &U_raw, crVec &y,
                                  Features::Feature_Model &model) {
        Mat XU(X_raw.rows(), X_raw.cols() + U_raw.cols());
        XU << X_raw, U_raw;
        Mat X = model.transform(XU);
        return single_fit(X, y);
    }


    std::vector<Feature> Regressor::transform_fit(const std::vector<std::string>& filenames, const std::vector<std::string>& colnames_x, const std::vector<std::string>& colnames_u, const std::string& colname_y, Features::Feature_Model& model)
    {
        uint32_t Nx = colnames_x.size();
        using namespace FROLS;
        DataFrameStack dfs(filenames);
        Mat X, U;
        Vec y;

        X = dataframe_to_matrix(dfs, colnames_x,
                                0, -2);
        y = dataframe_to_vector(dfs, colname_y, 1, -1);
        U = dataframe_to_matrix(dfs, colnames_u, 0, -2);
        std::vector<uint32_t> N_rows = dfs.get_N_rows();

        return transform_fit(X, U, y, model);
    }


    std::vector<uint32_t>
    Regressor::unused_feature_indices(const std::vector<Feature> &features, uint32_t N_features) const {
        std::vector<uint32_t> used_idx(features.size());
        std::transform(features.begin(), features.end(), used_idx.begin(), [&](auto &f) { return f.index; });
        return filtered_range(used_idx, 0, N_features);
    }

    int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
