#ifndef FROLS_DATA_HPP
#define FROLS_DATA_HPP
#include <vector>
namespace FROLS
{
    struct Feature
    {
        double ERR = 0; //Error Reduction Ratio
        double g; //Feature (Orthogonalized Linear-in-the-parameters form)
        size_t index; //Index of the feature in the original feature set
    };

    struct Regression_Data
    {
        std::vector<Feature> best_features; //Best regression-features in ERR-decreasing order
        Vec coefficients; //[N_features x N_features] rowwise coefficient matrix
    };

}

#endif