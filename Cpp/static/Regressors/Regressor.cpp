#include "Regressor.hpp"
#include <omp.h>
#include <algorithm>
#include <execution>

namespace FROLS::Regression {

    Regressor::Regressor(double tol, double theta_tol, size_t N_terms_max)
            : tol(tol), theta_tol(theta_tol), N_terms_max(N_terms_max), regressor_id(regressor_count) {}

    Mat Regressor::used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                              const std::vector<Feature> &used_features) const {
        size_t N_features = X.cols();
        size_t N_samples = X.rows();
        Mat Q_current = Mat::Zero(X.rows(), X.cols());
        for (int k = 0; k < N_features; k++) {
            if (std::none_of(used_features.begin(), used_features.end(),
                             [&](const auto &feature) { return feature.index == k; })) {
                Q_current.col(k) = vec_orthogonalize(X.col(k), Q.leftCols(used_features.size()));
            }
        }
        return Q_current;
    }

    std::vector<Feature> Regressor::single_fit(const Mat &X, const Vec &y) const {
        size_t N_features = X.cols();
        Mat Q_global = Mat::Zero(X.rows(), N_features);
        Mat Q_current = Q_global;
        Mat A = Mat::Zero(N_features, N_features);
        Vec g = Vec::Zero(N_features);

        std::vector<Feature> best_features;
        size_t end_idx = N_features;
        // Perform one feature selection iteration for each feature
        for (int j = 0; j < N_features; j++) {
            // Compute remaining variance by orthogonalizing the current feature
            Q_current =
                    used_feature_orthogonalize(X, Q_global, best_features);
            // Determine the best feature to add to the feature set
            Feature f = best_feature_select(Q_current, y, best_features);

            if ((f.f_ERR == -1) || (j == N_terms_max)) {
                end_idx = j;
                break;
            }
            best_features.push_back(f);

            //std::cout << "Feature " << best_features.back().index << std::endl;
            g[j] = best_features[j].g;

            Q_global.col(j) = Q_current.col(best_features[j].index);
            for (int m = 0; m < j; m++) {
                A(m, j) = cov_normalize(Q_global.col(m), X.col(best_features[j].index));
            }
            A(j, j) = 1;

            // If ERR-tolerance is met, return non-orthogonalized parameters
            if (tolerance_check(Q_global.leftCols(j + 1), y, best_features)) {
                end_idx = j+1;
                break;
            }


            Q_current.setZero();
        }
        theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx), best_features);

        return best_features;
    }

    void Regressor::theta_solve(crMat &A, crVec &g, std::vector<Feature> &features) const {
        Vec coefficients =
                A.inverse() * g;
        //Lectures on network systems
        // assign coefficients to features
        for (
                int i = 0;
                i < coefficients.rows();
                i++) {
            features[i].
                    theta = coefficients[i];
        }
    }

    Feature Regressor::best_feature_select(crMat &X, crVec &y, const std::vector<Feature> &used_features) const {
        const std::vector<Feature> candidates = candidate_regression(X, y, used_features);
        std::vector<Feature> thresholded_candidates;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(thresholded_candidates),
                     [&](const auto &f) { return abs(f.g) > theta_tol; });
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
                res = *std::max_element(thresholded_candidates.begin(), thresholded_candidates.end(),
                                        [](const Feature &f0, const Feature &f1) { return f0.f_ERR < f1.f_ERR; });
                break;
        }

        return res;
    }


    std::vector<std::vector<Feature>> Regressor::fit(crMat &X, crMat &Y) {
        if ((X.rows() != Y.rows())) {
            throw std::invalid_argument("X, U and Y must have same number of rows");
        }
        size_t N_response = Y.cols();
        std::vector<std::vector<Feature>> result(N_response);
        auto cols = Y.colwise();
        std::transform(std::execution::par_unseq,cols.begin(), cols.end(), result.begin(),
                       [=](const auto &yi) { return this->single_fit(X, yi); });
//        for (int i = 0; i < N_response; i++) {
//            result[i] = single_fit(X, Y.col(i));
//        }
        return result;
    }

    Vec Regressor::predict(crMat &Q, const std::vector<Feature> &features) const {
        Vec y_pred(Q.rows());
        y_pred.setZero();
        size_t i = 0;
        for (const auto &feature: features) {
            if (feature.f_ERR == -1) {
                break;
            }
            y_pred += Q.col(i) * feature.g;
            i++;
        }
        return y_pred;
    }

    void Regressor::transform_fit(crMat &X_raw, crMat &U_raw, crMat &Y,
                                  Features::Feature_Model &model) {
        Mat XU(X_raw.rows(), X_raw.cols() + U_raw.cols());
        XU << X_raw, U_raw;
        Mat X = model.transform(XU);
        model.features = fit(X, Y);
    }

    std::vector<size_t>
    Regressor::unused_feature_indices(const std::vector<Feature> &features, size_t N_features) const {
        std::vector<size_t> used_idx(features.size());
        std::transform(features.begin(), features.end(), used_idx.begin(), [&](auto &f) { return f.index; });
        return filtered_range(used_idx, 0, N_features);
    }

    int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
