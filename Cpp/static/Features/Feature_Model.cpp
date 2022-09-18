#include "Feature_Model.hpp"

namespace FROLS::Features {

    Feature_Model::Feature_Model(const size_t N_output_features, const std::vector<size_t> ignore_idx) : N_output_features(N_output_features), ignore_idx(ignore_idx),
                                                                         feature_logger(spdlog::basic_logger_mt(
                                                                                 "feature_logger",
                                                                                 (std::string(FROLS_LOG_DIR) +
                                                                                  "/feature_log.txt").c_str(), true)) {
        feature_logger->
                set_level(spdlog::level::debug);
        feature_logger->info("{:^15}{:^15}", "Feature_Name", "Index");
    }

    Vec Feature_Model::transform(crMat &X_raw, size_t target_index, bool& index_failure) {
        feature_logger->info("{:^15}{:^15}", feature_name(target_index, false), target_index);
        return _transform(X_raw, target_index, index_failure);
    }

    Mat Feature_Model::transform(crMat &X_raw) {
        size_t N_input_features = X_raw.cols();
        size_t N_rows = X_raw.rows();
        Mat X_poly(N_rows, N_output_features);
        size_t feature_idx = 0;
        size_t col_iter = 0;
        bool index_failure = 0;
        candidate_feature_idx.reserve(N_output_features);
        candidate_feature_idx.clear();
        do
        {
            if (std::none_of(ignore_idx.begin(), ignore_idx.end(), [&](const auto& ig_idx){return feature_idx == ig_idx;}))
            {
                X_poly.col(col_iter) = transform(X_raw, feature_idx, index_failure);
                if (index_failure)
                {
                    feature_logger->info("Feature Index exceeded at index {:^15}", feature_idx);
                }
                else{
                    col_iter++;
                    candidate_feature_idx.push_back(feature_idx);
                }
            }
            feature_idx++;
        } while ((col_iter < N_output_features) && !index_failure);
        X_poly.conservativeResize(N_rows, col_iter);
        return X_poly;
    }


    Vec Feature_Model::step(crVec &x, crVec &u) {
        Vec x_next(x.rows());
        x_next.setZero();
        Mat X(1, x.rows() + u.rows());
        X << x.transpose(), u.transpose();
        bool index_failure = false;
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < features[i].size(); j++) {
                x_next(i) +=
                        features[i][j].theta * _transform(X, candidate_feature_idx[features[i][j].index], index_failure).value();
                if (index_failure)
                {
                    break;
                }
            }
        }
        return x_next;
    }

    Mat Feature_Model::simulate(crVec &x0, crMat &U, size_t Nt) {
        Mat X(Nt + 1, x0.rows());
        X.row(0) = x0;
        for (int i = 0; i < Nt; i++) {
            X.row(i + 1) = step(X.row(i), U.row(i));
        }
        return X;
    }


} // namespace FROLS::Features