#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP

#include "Regressor.hpp"

namespace FROLS::Regression {
    struct ERR_Regressor : public Regressor {
        ERR_Regressor(const Regressor_Param& p) : Regressor(p){}

    private:
        std::vector<Feature> candidate_regression(crMat &X,  crMat& Q_global, crVec &y,
                                            const std::vector<Feature> &used_features) const;

        bool tolerance_check(crMat &Q, crVec &y,
                             const std::vector<Feature> &best_features) const;

        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        static bool best_feature_measure(const Feature&, const Feature&);
        Feature feature_selection_criteria(const std::vector<Feature> &features) const;

        void theta_solve(crMat &A, crVec &g, crMat& X, crVec& y, std::vector<Feature> &features) const;
    };
} // namespace FROLS::Regression

#endif