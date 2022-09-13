#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP
#include <FROLS_Path_Config.hpp>
#include <Feature_Model.hpp>
#include <Math.hpp>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>


namespace FROLS::Regression {
struct Regressor {
  const double tol;
  Regressor(double tol);
  std::vector<std::vector<Feature>> fit(crMat &X, crMat &Y) const;
  void
  transform_fit(crMat& X, crMat& U, crMat& Y,
                Features::Feature_Model &model)const;
  protected:
  Vec predict(crMat& X, const std::vector<Feature>& features) const;
private:
  std::vector<Feature> single_fit(crMat X, crVec y) const;
  Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                 const iVec &used_indices) const;
  virtual void feature_select(crMat &X, crVec &y,
                                 std::vector<Feature>& best_features) const = 0;
  virtual bool
  tolerance_check(crMat &Q, crVec &y,
                  const std::vector<Feature> &best_features) const = 0;
  std::shared_ptr<spdlog::logger> regressor_logger;

};
} // namespace FROLS::Regression

#endif