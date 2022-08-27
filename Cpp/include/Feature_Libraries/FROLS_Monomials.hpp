#ifndef FROLS_MONOMIALS_HPP
#define FROLS_MONOMIALS_HPP
#include <FROLS_Typedefs.hpp>
#include <array>
#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include <functional>
#include <nvector/nvector_serial.h>
#include <stddef.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
namespace FROLS::Features {

typedef int (*RhsFn)(double, N_Vector, N_Vector, void *);
template <size_t Nx>
using MonomialFn =
    std::function<std::array<double, Nx>(const std::array<double, Nx>)>;
template <size_t Nx>
using jacMonomialFn =
    std::function<std::array<double, Nx * Nx>(const std::array<double, Nx>)>;

template <size_t Nx, size_t N_control> struct MonomialRegressionData {
  double theta_x[Nx];
  double theta_u[N_control];
  std::array<std::array<size_t, Nx>> monomial_powers;
};

template <size_t Nx>
int polynomial_CVRhsFn(realtype t, N_Vector y, N_Vector ydot, void *p_param) {
  std::array<double, Nx> y_data = NV_DATA_S(y);
  double *ydot_data = NV_DATA_S(ydot);
  std::array<double, Nx> x;
  MonomialRegressionData md = *(MonomialRegressionData)p_param;

  for (int k = 0; k < N_features; k++) {
    for (int i = 0; i < Nx; i++) {
      auto powers_i = md.monomial_powers[i] double monomial = 1;
      for (int j = 0; j < Nx; j++) {
        monomial *= pow(y_data[j], j);
      }
      ydot_data[i] += monomial;
    }
  }
  return 0;
}

template <size_t Nx>
jacMonomialFn<Nx> generate_jac_monomial(size_t orders[Nx]) {
  jacMonomialFn<Nx> jacMonomial = [orders](const std::array<double, Nx> x) {
    std::array<double, Nx * Nx> result;
    for (size_t i = 0; i < Nx; i++) {
      for (size_t j = 0; j < Nx; j++) {
        if (i == j) {
          result[i * Nx + j] = orders[i] * pow(x[i], orders[i] - 1);
        } else {
          result[i * Nx + j] = 0;
        }
      }
    }
    return result;
  };
  return jacMonomial;
}

template <size_t Nx>
int generate_polynomial_CVLsJacTimesVecFn(N_Vector v, N_Vector Jv, realtype t,
                                          N_Vector y, void *param) {
  std::array<double, Nx> x;
  for (size_t i = 0; i < Nx; i++) {
    x[i] = NV_Ith_S(y, i);
  }
  std::array<double, Nx * Nx> result;
  std::array<MonomialFn<Nx>, Nx> jac_monomial_fns = (MonomialFn<Nx> *)param;
  double *theta = (double *)(jac_monomial_fns.back() + 1);
  for (size_t i = 0; i < Nx; i++) {
    for (size_t j = 0; j < Nx; j++) {
      result[i * Nx + j] = 0;
      for (size_t k = 0; k < jac_monomial_fns.size(); k++) {
        result[i * Nx + j] += theta[k] * jac_monomial_fns[k](x)[i * Nx + j];
      }
    }
  }
  for (size_t i = 0; i < Nx * Nx; i++) {
    NV_Ith_S(Jv, i) = result[i];
  }
  return 0;
}

} // namespace FROLS::Features

#endif