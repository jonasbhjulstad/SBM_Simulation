#include "Feature_Model.hpp"
#include "Regressor.hpp"

namespace FROLS::Features {

    Feature_Model::Feature_Model(const size_t N_output_features, const std::vector<size_t> _ignore_idx,
                                 const std::vector<std::vector<Feature>> preselected_features) : N_output_features(
            N_output_features), preselected_features(preselected_features) {
        std::copy(_ignore_idx.begin(), _ignore_idx.end(), std::back_inserter(ignore_idx));
    }

    Vec Feature_Model::transform(crMat &X_raw, size_t target_index, bool &index_failure) {
        return _transform(X_raw, target_index, index_failure);
    }

    Mat Feature_Model::transform(crMat &X_raw, const std::vector<Feature> preselected_features) {
        size_t N_input_features = X_raw.cols();
        size_t N_rows = X_raw.rows();
        Mat X_poly(N_rows, N_output_features + preselected_features.size());
        size_t feature_idx = 0;
        size_t col_iter = 0;
        bool index_failure = 0;
        candidate_feature_idx.reserve(N_output_features);
        candidate_feature_idx.clear();
        preselect_feature_idx.reserve(preselected_features.size());
        preselect_feature_idx.clear();
        do {
            bool not_ignored = std::none_of(ignore_idx.begin(), ignore_idx.end(),
                                           [&](const auto &ig_idx) { return feature_idx == ig_idx; });
            bool is_preselected = std::any_of(preselected_features.begin(), preselected_features.end(),
                                              [&](const auto &ps_feature) { return feature_idx == ps_feature.index; });
            if (not_ignored) {

                X_poly.col(col_iter) = transform(X_raw, feature_idx, index_failure);
                if (!index_failure) {
                    col_iter++;
                    candidate_feature_idx.push_back(feature_idx);
                }
                if (is_preselected)
                {
                    preselect_feature_idx.push_back(feature_idx);
                }
            }
            feature_idx++;
        } while ((col_iter < (N_output_features + preselected_features.size())) && !index_failure);

        X_poly.conservativeResize(N_rows, col_iter);
        return X_poly;
    }


    Vec Feature_Model::step(crVec &x, crVec &u) {
        Vec x_next(x.rows());
        Mat X(1, x.rows() + u.rows());
        X << x.transpose(), u.transpose();
        bool index_failure = false;
        x_next.setZero();
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < features[i].size(); j++) {
                x_next(i) +=
                        features[i][j].theta *
                        _transform(X, candidate_feature_idx[features[i][j].index], index_failure).value();
                if (index_failure) {
                    break;
                }
            }
        }

        return x_next;
    }

    Mat Feature_Model::simulate(crVec &x0, crMat &U, size_t Nt) {
        Mat X(Nt + 1, x0.rows());
        X.setZero();
        X.row(0) = x0;
        for (int i = 0; i < Nt; i++) {
            X.row(i + 1) = step(X.row(i), U.row(i));
        }
        return X;
    }


} // namespace FROLS::Features