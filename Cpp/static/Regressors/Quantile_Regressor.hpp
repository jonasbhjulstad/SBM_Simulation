#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP
#include <Typedefs.hpp>
#include "Regressor.hpp"

namespace FROLS::Regression
{   
    struct Quantile_Regressor: public Regressor
    {
        const double tau;
        Quantile_Regressor(double tau, double tol): tau(tau), Regressor(tol) {}
        private:
        Feature feature_select(const Mat &X, const Vec &y, const iVec &used_features) const;
        bool tolerance_check(const Mat& Q, const Vec& y, const std::vector<Feature>& best_features) const;
    };
}


#endif