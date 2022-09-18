#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP
#include "Regressor.hpp"
namespace FROLS::Regression {
struct ERR_Regressor : public Regressor {
  ERR_Regressor(double tol, double theta_tol) : Regressor(tol, theta_tol),
                              ERR_logger(spdlog::basic_logger_mt(("ERR_Regressor_" + std::to_string(regressor_id)).c_str(), (std::string(FROLS_LOG_DIR) + "/ERR_regressor_" + std::to_string(regressor_id) + ".txt").c_str(), true)){}
private:
  Feature feature_select(crMat &X, crVec &y,
                         const std::vector<Feature> &used_features);
  bool tolerance_check(crMat &Q, crVec &y,
                       const std::vector<Feature> &best_features) const;
  std::shared_ptr<spdlog::logger> ERR_logger;
};
} // namespace FROLS::Regression

#endif