#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP
#include "Regressor.hpp"
#include <numeric>
namespace FROLS::Regression
{   
    struct Quantile_Regressor: public Regressor
    {
        const double tau;
        Quantile_Regressor(double tau, double tol): tau(tau), Regressor(tol),
        qr_logger(spdlog::basic_logger_mt("Quantile_Regressor", (std::string(FROLS_LOG_DIR) + "/quantile_regressor.txt").c_str(), true)),
                                                    subproblem_logger(spdlog::basic_logger_mt("Subproblem_Logger", (std::string(FROLS_LOG_DIR) + "/quantile_subproblem_variables.txt").c_str(), true))
        {
            qr_logger->set_level(spdlog::level::debug);
        }
        private:
        bool single_feature_quantile_regression(crVec &x, crVec &y, Feature& best_feature, size_t feature_index) const;
        void feature_select(crMat &X, crVec &y, std::vector<Feature> &used_features) const;
        bool tolerance_check(crMat& Q, crVec& y, const std::vector<Feature>& best_features) const;
        std::shared_ptr<spdlog::logger> qr_logger;
        std::shared_ptr<spdlog::logger> subproblem_logger;
        size_t feature_selection_idx = 0;
    };
}


#endif