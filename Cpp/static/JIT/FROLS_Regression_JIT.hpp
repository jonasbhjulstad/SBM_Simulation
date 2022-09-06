#ifndef FROLS_REGRESSION_JIT_HPP
#define FROLS_REGRESSION_JIT_HPP
#include <casadi/casadi.hpp>
#include <FROLS_Typedefs.hpp>
#include <FROLS_Polynomial.hpp>
namespace FROLS::JIT
{
    //generate casadi function from regression data
    casadi::Function generate_regression_function(const Polynomial_Model &rd);
    //JIT-compile regression function
    casadi::Function compile_regression_function(const Polynomial_Model &rd);
}


#endif // FROLS_REGRESSION_JIT_HPP