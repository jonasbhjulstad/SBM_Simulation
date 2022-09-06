#include "FROLS_Regression_JIT.hpp"
#include <FROLS_Features.hpp>

namespace FROLS::JIT
{
    //generate casadi function from regression data
    casadi::Function polynomial_function(const Polynomial_Model &rd)
    {
        using namespace FROLS;
        using namespace casadi;
        MX x0 = MX::sym("x", rd.N_states);
        MX u = MX::sym("u", rd.N_control_inputs);
        //iterate over response variables in rd
        for (const auto& response_features: rd.features)
        {
            for (const auto& feature: response_features)
            {
                auto xki = Features::Polynomial::single_feature_transform(x0, rd.d_max);

            }
        }

    }
    //JIT-compile regression function
    casadi::Function compile_regression_function(const Regression_Model &rd);
}

