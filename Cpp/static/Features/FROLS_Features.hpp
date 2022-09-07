#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP
#include "FROLS_Polynomial.hpp"
#include <FROLS_Typedefs.hpp>
namespace FROLS::Features {
Feature feature_select(const Mat &X, const Vec &y, const iVec &used_features);

} // namespace FROLS::Features

#endif