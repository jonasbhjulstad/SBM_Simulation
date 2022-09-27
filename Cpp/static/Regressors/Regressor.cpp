#include "Regressor.hpp"
#include <omp.h>
#include <algorithm>
#include <execution>

namespace FROLS::Regression {

    Regressor::Regressor(const Regressor_Param &p)
            : tol(p.tol), theta_tol(p.theta_tol), N_terms_max(p.N_terms_max) {}


    std::vector<Feature> Regressor::single_fit(const Mat &X, const Vec &y) const {
        size_t N_features = X.cols();
        Mat Q_global = Mat::Zero(X.rows(), N_features);
        Mat Q_current = Q_global;
        Mat A = Mat::Zero(N_features, N_features);
        Vec g = Vec::Zero(N_features);
        std::vector<Feature> best_features;
        best_features.reserve(N_terms_max);
        size_t end_idx = N_features;

        // Perform one feature selection iteration for each feature
        for (int j = 0; j < N_features; j++) {

            // Compute remaining variance by orthogonalizing the current feature
            Q_current =
                    used_feature_orthogonalize(X, Q_global, best_features);
            // Determine the best feature to add to the feature set
            Feature f = best_feature_select(Q_current, y, best_features);

            if ((f.f_ERR == -std::numeric_limits<double>::infinity()) || (j >= (N_terms_max))) {
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
            if (tolerance_check(Q_global.leftCols(j + +1), y, best_features)) {
                end_idx = j+1;
                break;
            }


            Q_current.setZero();
        }
        theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx), best_features);

        return best_features;
    }

    std::vector<Feature>
    Regressor::single_fit(const Mat &X, const Vec &y, const std::vector<Feature> preselect_features) const {
//        Mat X_preselect_orth = X;
//        for (int i = 0; i < preselect_features.size(); i++) {
//            for (int j = 0; j < X.cols(); j++) {
//                X_preselect_orth.col(j) = vec_orthogonalize(X_preselect_orth.col(j),
//                                                            X.col(preselect_features[i].index) *
//                                                            preselect_features[i].theta);
//            }
//        }

        Vec y_diff = y;
        std::for_each(preselect_features.begin(), preselect_features.end(), [&](const auto& f){
            y_diff -= X.col(f.index)*f.theta;
        });
        Mat X_unused = X;
        std::for_each(preselect_features.begin(), preselect_features.end(), [&](const auto& f){X_unused.col(f.index).setZero();});

        std::vector<Feature> result;
        result.insert(result.begin(), preselect_features.begin(), preselect_features.end());
        std::vector<Feature> identified_features = single_fit(X_unused, y_diff);
        result.insert(result.end(), identified_features.begin(), identified_features.end());
        return result;

    }

    void Regressor::theta_solve(crMat &A, crVec &g, std::vector<Feature> &features) const {
        Vec coefficients =
                A.inverse() * g;
        //Lectures on network systems
        // assign coefficients to features
        std::cout << A << std::endl;
        std::cout << g << std::endl;
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
                     [&](const auto &f) {
                         return ((abs(f.g) > theta_tol) && (f.f_ERR != -std::numeric_limits<double>::infinity()));
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
                res = *std::max_element(thresholded_candidates.begin(), thresholded_candidates.end(),
                                        best_feature_measure);
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
        std::transform(std::execution::par_unseq, cols.begin(), cols.end(), result.begin(),
                       [=](const auto &yi) { return this->single_fit(X, yi); });
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

    struct Fit_Data {
        std::vector<Feature> preselect_features;
        Vec y;
    };

    void Regressor::transform_fit(crMat &X_raw, crMat &U_raw, crMat &Y,
                                  Features::Feature_Model &model) {
        Mat XU(X_raw.rows(), X_raw.cols() + U_raw.cols());
        XU << X_raw, U_raw;
        std::vector<Fit_Data> data(Y.cols());
        model.preselected_features.resize(Y.cols());
        for (int i = 0; i < Y.cols(); i++) {
            data[i] = Fit_Data{model.preselected_features[i], Y.col(i)};
        }
        model.features.resize(Y.cols());
        std::transform(std::execution::par_unseq, data.begin(), data.end(), model.features.begin(), [&](const auto &d) {
            Mat X = model.transform(XU, d.preselect_features);
            return single_fit(X, d.y, d.preselect_features);
        });

    }


    std::vector<size_t>
    Regressor::unused_feature_indices(const std::vector<Feature> &features, size_t N_features) const {
        std::vector<size_t> used_idx(features.size());
        std::transform(features.begin(), features.end(), used_idx.begin(), [&](auto &f) { return f.index; });
        return filtered_range(used_idx, 0, N_features);
    }

    bool Regressor::best_feature_measure(const Feature &f0, const Feature &f1) {
        return f0.f_ERR < f1.f_ERR;
    }

    int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
