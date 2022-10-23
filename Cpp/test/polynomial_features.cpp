#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>
#include <Polynomial_Discrete.hpp>


int main()
{
    using namespace FROLS;
    using namespace FROLS::Regression;
    uint32_t d_max = 2;
    uint32_t N_output_features = 100;
    uint32_t Nu = 1;
    uint32_t Nx = 3;
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);

    for (int i = 0; i < N_output_features; i++)
    {
        std::cout << i << ":\t" <<  model.feature_name(i) << std::endl;
    }

}