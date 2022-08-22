#ifndef FROLS_FEATURE_HPP
#define FROLS_FEATURE_HPP
#include "FROLS_Typedefs.hpp"
#include "FROLS_math.hpp"
namespace FROLS
{
    struct Feature
    {
        double ERR = 0; //Error Reduction Ratio
        Vec g; //Feature (Orthogonalized Linear-in-the-parameters form)
        size_t index; //Index of the feature in the original feature set
    }

    Feature feature_select(const Mat& X, const Vec& y, const iVec& used_features)
    {
        size_t N_features = X.cols();
        Vec g = Vec::Zero(N_features);
        Vec ERR = Vec::Zero(N_features);
        for (int i = 0; i < N_features; i++)
        {
            if (!used_features.cwiseEqual(i).any())
            {
                Vec xi = X.col(i);
                g[i] = cov_normalize(xi, y);
                ERR[m] = g[m].square()*xi.T * xi/(y.T*y);
            }
        }
        Feature best_feature;
        best_feature.ERR = ERR.maxCoeff();
        best_feature.g = g.cwiseProduct(ERR.cwiseEqual(best_feature.ERR));
        best_feature.index = ERR.cwiseEqual(best_feature.ERR).index_max();
        return best_feature;
    }

}

#endif