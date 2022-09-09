#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP
#include "Regressor.hpp"
#include <Typedefs.hpp>
namespace FROLS::Regression {
struct ERR_Regressor : public Regressor {
  ERR_Regressor(double tol) : Regressor(tol) {}

private:
  Feature feature_select(const Mat &X, const Vec &y,
                         const iVec &used_features) const;
  bool tolerance_check(const Mat &Q, const Vec &y,
                       const std::vector<Feature> &best_features) const;
};
} // namespace FROLS::Regression

#endif