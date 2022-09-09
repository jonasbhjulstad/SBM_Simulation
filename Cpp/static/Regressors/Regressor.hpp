#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP
#include <Feature_Model.hpp>
#include <Typedefs.hpp>
#include <vector>

namespace FROLS::Regression {
struct Regressor {
  const double tol;
  Regressor(double tol) : tol(tol) {}
  std::vector<std::vector<Feature>> fit(crMat &X, crMat &Y) const;
  void
  transform_fit(crMat& X, crMat& U, crMat& Y,
                Features::Feature_Model &model)const;

private:
  std::vector<Feature> single_fit(crMat X, crVec y) const;
  Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                 const iVec &used_indices) const;

  virtual Feature feature_select(const Mat &X, const Vec &y,
                                 const iVec &used_features) const = 0;
  virtual bool
  tolerance_check(const Mat &Q, const Vec &y,
                  const std::vector<Feature> &best_features) const = 0;
};
} // namespace FROLS::Regression

#endif