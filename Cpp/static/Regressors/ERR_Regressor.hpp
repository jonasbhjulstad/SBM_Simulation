#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP

#include "Regressor.hpp"

namespace FROLS::Regression {
    struct ERR_Regressor : public Regressor {
        ERR_Regressor(double tol, double theta_tol, size_t N_terms_max = std::numeric_limits<size_t>::infinity()) : Regressor(tol, theta_tol, N_terms_max){}

    private:
        std::vector<Feature> candidate_regression(crMat &X, crVec &y,
                                            const std::vector<Feature> &used_features) const;

        bool tolerance_check(crMat &Q, crVec &y,
                             const std::vector<Feature> &best_features) const;

        Feature single_feature_regression(crVec &x, crVec &y) const;


    };
} // namespace FROLS::Regression

#endif