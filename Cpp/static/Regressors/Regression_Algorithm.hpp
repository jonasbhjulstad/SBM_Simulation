//
// Created by arch on 9/22/22.
//

#ifndef FROLS_REGRESSION_ALGORITHM_HPP
#define FROLS_REGRESSION_ALGORITHM_HPP
#include <functional>
#include <string>
#include <Regressor.hpp>
#include <Feature_Model.hpp>
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
namespace FROLS::Regression
{
    void from_file_regression(std::function<std::string(uint32_t)> MC_fname_f, const std::vector<std::string> &colnames_x,
                              const std::vector<std::string> &colnames_u, uint32_t N_sims,
                              FROLS::Regression::Regressor &regressor, FROLS::Features::Feature_Model  &feature_model,
                              std::function<std::string(uint32_t)> outsim_f, bool differentiate = false);
}
#endif //FROLS_REGRESSION_ALGORITHM_HPP
