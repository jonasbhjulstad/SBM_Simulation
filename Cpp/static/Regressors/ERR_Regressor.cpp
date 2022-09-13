#include "ERR_Regressor.hpp"
#include <vector>

namespace FROLS::Regression {

    void ERR_Regressor::feature_select(crMat &X, crVec &y,
                                       std::vector<Feature> &used_features) const {
        std::vector<size_t> used_indices(used_features.size());
        std::transform(used_features.begin(), used_features.end(), used_indices.begin(),
                       [](auto feature) { return feature.index; });
        ERR_logger->info("Feature selection with used indices:\t{}", fmt::join(used_indices, ","));
        ERR_logger->info("{:^15}{:^15}{:^15}", "Subproblem", "theta", "ERR");

        size_t N_features = X.cols();
        double ERR, g;
        Feature best_feature;
        for (int i = 0; i < N_features; i++) {


            // If the feature is already used, skip it
            if (std::none_of(used_features.begin(), used_features.end(), [&i](auto f) { return f.index == i; })) {
                Vec xi = X.col(i);
                g = cov_normalize(xi, y);
                ERR = g * g * ((xi.transpose() * xi) / (y.transpose() * y)).value();
                if (ERR > best_feature.f_ERR) {
                    best_feature.f_ERR = ERR;
                    best_feature.g = g;
                    best_feature.index = i;
                }
                ERR_logger->info("{:^15}{:^15}{:^15.3f}", i, g, ERR);
            }

        }
        ERR_logger->info("Best feature:{:^15}{:^15.3f}{:^15.3f}", best_feature.index, best_feature.g,
                         best_feature.f_ERR);
        used_features.push_back(best_feature);
    }

    bool ERR_Regressor::tolerance_check(
            crMat &Q, crVec &y,
            const std::vector<Feature> &best_features) const {
        double ERR_tot = 0;
        for (const auto &feature: best_features) {
            ERR_tot += feature.f_ERR;
        }
        return ERR_tot > tol;
    }

} // namespace FROLS::Regression
