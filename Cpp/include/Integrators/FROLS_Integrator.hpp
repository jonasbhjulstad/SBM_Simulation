#ifndef FROLS_INTEGRATOR_HPP
#define FROLS_INTEGRATOR_HPP

#include "Integrator_Models.hpp"

namespace FROLS {
template <size_t Nx>
class FROLS_Integrator : public CVODE_Integrator<Nx, FROLS_Integrator<Nx>> {
  FROLS_Integrator(const std::array<realtype, Nx> &x0, realtype dt,
                   realtype abs_tol = 1e-5, realtype reltol = 1e-5,
                   realtype t0 = 0)
      : CVODE_Integrator<Nx, FROLS_Integrator<Nx>>(x0, dt, abs_tol, reltol,
                                                   t0) {}
    reassign_jacobian()
    {
        
    }
};
} // namespace FROLS

#endif