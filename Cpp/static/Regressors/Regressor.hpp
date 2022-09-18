#ifndef FROLS_REGRESSOR_HPP
#define FROLS_REGRESSOR_HPP
#include <FROLS_Path_Config.hpp>
#include <Feature_Model.hpp>
#include <FROLS_Math.hpp>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>


namespace FROLS::Regression {
struct Regressor {
  const double tol, theta_tol;
  const size_t regressor_id;
  Regressor(double tol, double theta_tol);
  std::vector<std::vector<Feature>> fit(crMat &X, crMat &Y);
  void
  transform_fit(crMat& X, crMat& U, crMat& Y,
                Features::Feature_Model &model);
  protected:
  Vec predict(crMat& X, const std::vector<Feature>& features) const;
private:
  std::vector<Feature> single_fit(crMat X, crVec y);
  Mat used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                 const std::vector<Feature>& used_features) const;
  virtual Feature feature_select(crMat &X, crVec &y,
                                 const std::vector<Feature>& best_features) = 0;
  virtual bool
  tolerance_check(crMat &Q, crVec &y,
                  const std::vector<Feature> &best_features) const = 0;
  std::shared_ptr<spdlog::logger> regressor_logger;
  static int regressor_count;
};
} // namespace FROLS::Regression

#endif