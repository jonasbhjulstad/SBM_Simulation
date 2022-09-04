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

    const std::string regression_data_summary(const Regression_Data& rd)
    {
        std::string summary = "Regression_Data Summary:\n";
        summary += "Best Features:\n";
        for (const auto& feature : rd.best_features)
        {
            summary += "Index: " + std::to_string(feature.index) + "\tERR: " + std::to_string(feature.ERR) + "\tg: " + std::to_string(feature.g) + "\n";
        }
        summary += "Coefficients:\n";
        for (const auto& coeff : rd.coefficients)
        {
            summary += std::to_string(coeff) + "\n";
        }
        return summary;
    }

}

#endif