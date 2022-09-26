#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP

#include "Regressor.hpp"

namespace FROLS::Regression {
    struct ERR_Regressor : public Regressor {
        ERR_Regressor(const Regressor_Param& p) : Regressor(p){}

    private:
        std::vector<Feature> candidate_regression(crMat &X, crVec &y,
                                            const std::vector<Feature> &used_features) const;

        bool tolerance_check(crMat &Q, crVec &y,
                             const std::vector<Feature> &best_features) const;

        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        static bool best_feature_measure(const Feature&, const Feature&);
    };
} // namespace FROLS::Regression

#endif