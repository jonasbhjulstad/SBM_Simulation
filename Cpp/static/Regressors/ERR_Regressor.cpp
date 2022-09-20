#include "ERR_Regressor.hpp"
#include <vector>
#include <execution>

namespace FROLS::Regression {

    std::vector<Feature> ERR_Regressor::candidate_regression(crMat &X, crVec &y,
                                                             const std::vector<Feature> &used_features) const {
        std::vector<size_t> candidate_idx = unused_feature_indices(used_features, X.cols());
        std::vector<Feature> candidates(X.cols() - used_features.size());
        std::transform(std::execution::par_unseq,candidate_idx.begin(), candidate_idx.end(), candidates.begin(),
                       [&](const size_t &idx) {
                           Feature f = single_feature_regression(X.col(idx), y);
                           f.index = idx;
                           return f;
                       });
        return candidates;
    }

    Feature ERR_Regressor::single_feature_regression(crVec &x, crVec &y) const {
        Feature f;
        f.g = cov_normalize(x, y);
        f.f_ERR = f.g * f.g * ((x.transpose() * x) / (y.transpose() * y)).value();
        return f;
    }

    bool ERR_Regressor::tolerance_check(
            crMat &Q, crVec &y,
            const std::vector<Feature> &best_features) const {
        double ERR_tot = 0;
        for (const auto &feature: best_features) {
            ERR_tot += feature.f_ERR;
        }
        return (1 - ERR_tot) < tol;
    }

} // namespace FROLS::Regression
