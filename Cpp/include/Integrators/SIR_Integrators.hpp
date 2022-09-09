#ifndef FROLS_SIR_INTEGRATORS_HPP
#define FROLS_SIR_INTEGRATORS_HPP
#include <Integrators/Integrator_Models.hpp>
#include <algorithm>
#include <random>
namespace FROLS::Integrators {

struct SIR_Stochastic {
public:
  static constexpr size_t Nx = 3;
  std::random_device rd;
  std::mt19937 generator;
  double alpha, beta, N_pop, dt;

  SIR_Stochastic(const std::array<double, Nx> &x0, double alpha, double beta,
                 double N_pop, double dt)
      : alpha(alpha), beta(beta), N_pop(N_pop), dt(dt) {}

  std::array<double, 3> step(const std::array<double, 3> &x) {
    double p_I = 1 - std::exp(-beta * x[1] / N_pop * dt);
    double p_R = 1 - std::exp(-alpha * dt);

    double K_SI;
    double K_IR;
    std::uint64_t seed = 777;
    std::binomial_distribution<> SI_dist(x[0], p_I);
    std::binomial_distribution<> IR_dist(x[1], p_R);

    K_SI = SI_dist(generator);
    K_IR = IR_dist(generator);

    std::array<double, 3> delta_x = {-K_SI, K_SI - K_IR, K_IR};

    std::array<double, 3> x_next;
    for (int i = 0; i < Nx; i++) {
      x_next[i] = std::max({x[i] + delta_x[i], 0.});
    }
    return x_next;
  }
};

// Simple function that calculates the differential equation.
int SIR_eval_f(double t, N_Vector x, N_Vector x_dot, void *param) {
  double *x_data = N_VGetArrayPointer(x);
  double *x_dot_data = N_VGetArrayPointer(x_dot);
  double S = x_data[0];
  double I = x_data[1];
  double R = x_data[2];
  double alpha = ((double *)param)[0];
  double beta = ((double *)param)[1];
  double N_pop = ((double *)param)[2];

  x_dot_data[0] = -beta * S * I / N_pop;
  x_dot_data[1] = beta * S * I / N_pop - alpha * I;
  x_dot_data[2] = alpha * I;

  return 0;
}

// Jacobian function vector routine.
int SIR_eval_jac(N_Vector v, N_Vector Jv, double t, N_Vector x, N_Vector fx,
                 void *param, N_Vector tmp) {
  double *x_data = N_VGetArrayPointer(x);
  double *Jv_data = N_VGetArrayPointer(Jv);
  double *v_data = N_VGetArrayPointer(v);
  double S = x_data[0];
  double I = x_data[1];
  double R = x_data[2];
  double alpha = ((double *)param)[0];
  double beta = ((double *)param)[1];
  double N_pop = ((double *)param)[2];

  Jv_data[0] = -beta * I / N_pop * v_data[0] - beta * S / N_pop * v_data[1];
  Jv_data[1] = beta * I / N_pop * v_data[0] + beta * S / N_pop * v_data[1] -
               alpha * v_data[1];
  Jv_data[2] = alpha * v_data[1];

  return 0;
}

struct SIR_Deterministic : public CVODE_Integrator<3, SIR_Deterministic> {
  double param[3];
  SIR_Deterministic(const std::array<double, 3> &x0, double alpha, double beta,
                    double N_pop, double dt)
      : CVODE_Integrator<3, SIR_Deterministic>(x0, dt) {
    param[0] = alpha;
    param[1] = beta;
    param[2] = N_pop;
    assert(this->initialize_solver(SIR_eval_f, SIR_eval_jac, (void *)param) ==
           EXIT_SUCCESS);
  }
};
} // namespace FROLS::Integrators
#endif // SIR_INTEGRATORS_HPP