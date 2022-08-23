#ifndef FROLS_FEATURE_HPP
#define FROLS_FEATURE_HPP
#include "FROLS_Typedefs.hpp"
#include "FROLS_math.hpp"
#include "FROLS_Data.hpp"
namespace FROLS
{
    Feature feature_select(const Mat& X, const Vec& y, const iVec& used_features)
    {
        size_t N_features = X.cols();
        double ERR, g;
        Feature best_feature;
        for (int i = 0; i < N_features; i++)
        {
            //If the feature is already used, skip it
            if (!used_features.cwiseEqual(i).any())
            {
                Vec xi = X.col(i);
                g = cov_normalize(xi, y);
                ERR = g*g*((xi.transpose() * xi)/(y.transpose()*y)).value();
                if (ERR > best_feature.ERR)
                {
                    best_feature.ERR = ERR;
                    best_feature.g = g;
                    best_feature.index = i;
                }
            }
        }
        return best_feature;
    }

}
#endif