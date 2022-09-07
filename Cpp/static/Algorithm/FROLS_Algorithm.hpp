#ifndef FROLS_HPP
#define FROLS_HPP
#include <FROLS_Polynomial.hpp>
#include <FROLS_Features.hpp>
#include <FROLS_Math.hpp>
#include <FROLS_Typedefs.hpp>
#include <algorithm>
#include <fstream>
namespace FROLS {
// Orthogonalizes x with respect to previously selected features in Q
Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                               const iVec &used_indices,
                               size_t current_feature_idx);
bool feature_tolerance_check(const std::vector<Feature> &features,
                             double ERR_tolerance);

namespace Regression {
// Computes feature coefficients for feature batch X with respect to a single
// response variable y

std::vector<Feature> single_response_regression(Eigen::Ref<const Mat> X,
                                                Eigen::Ref<const Vec> y,
                                                double ERR_tolerance);

std::vector<std::vector<Feature>>
multiple_response_regression(Eigen::Ref<const Mat> X, Eigen::Ref<const Mat> Y, double ERR_tolerance);



const std::string regression_data_summary(const std::vector<Feature> &rd);

const std::string
regression_data_summary(const std::vector<std::vector<Feature>> &rds);
} // namespace Regression
} // namespace FROLS

#endif // FROLS_HPP