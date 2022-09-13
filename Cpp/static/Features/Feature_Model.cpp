#include "Feature_Model.hpp"

namespace FROLS::Features {

    Feature_Model::Feature_Model() :
            feature_logger(spdlog::basic_logger_mt(
                    "feature_logger",
                    (std::string(FROLS_LOG_DIR) + "/feature_log.txt").c_str(), true)) {
        feature_logger->
                set_level(spdlog::level::debug);
        feature_logger->info("{:^15}{:^15}", "Feature_Name", "Index");
    }

    Vec Feature_Model::transform(crMat &X_raw, size_t target_index) const {
        feature_logger->info("{:^15}{:^15}", feature_name(target_index, false), target_index);
        return _transform(X_raw, target_index);
    }

    Mat Feature_Model::transform(crMat &X_raw) const {
        return
                _transform(X_raw);
    }


    Vec Feature_Model::step(crVec &x, crVec &u) const {
        Vec x_next(x.rows());
        x_next.setZero();
        Mat X(1, x.rows() + u.rows());
        X << x.transpose(), u.transpose();
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < features[i].size(); j++) {
                x_next(i) +=
                        features[i][j].theta * _transform(X, features[i][j].index).value();
            }
        }
        return x_next;
    }

    Mat Feature_Model::simulate(crVec &x0, crMat &U, size_t Nt) const {
        Mat X(Nt + 1, x0.rows());
        X.row(0) = x0;
        for (int i = 0; i < Nt; i++) {
            X.row(i + 1) = step(X.row(i), U.row(i));
        }
        return X;
    }

} // namespace FROLS::Features