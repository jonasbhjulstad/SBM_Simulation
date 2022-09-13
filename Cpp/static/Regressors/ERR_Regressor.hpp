#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP
#include "Regressor.hpp"
namespace FROLS::Regression {
struct ERR_Regressor : public Regressor {
  ERR_Regressor(double tol) : Regressor(tol),
                              ERR_logger(spdlog::basic_logger_mt("ERR_Regressor", (std::string(FROLS_LOG_DIR) + "/ERR_regressor.txt").c_str(), true)){}

private:
  void feature_select(crMat &X, crVec &y,
                         std::vector<Feature> &used_features) const;
  bool tolerance_check(crMat &Q, crVec &y,
                       const std::vector<Feature> &best_features) const;
  std::shared_ptr<spdlog::logger> ERR_logger;
};
} // namespace FROLS::Regression

#endif