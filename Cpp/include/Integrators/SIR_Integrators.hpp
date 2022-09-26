#ifndef FROLS_SIR_INTEGRATORS_HPP
#define FROLS_SIR_INTEGRATORS_HPP

#include <Integrators/Integrator_Models.hpp>
#include <algorithm>
#include <cassert>
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

    struct SIR_Param {
        double alpha;
        double beta;
        double N_pop;
    };

    // Simple function that calculates the differential equation.
    template<SIR_Param(param_gen)(void *p, double t)>
    int SIR_eval_f(double t, N_Vector x, N_Vector x_dot, void *param) {
        double *x_data = N_VGetArrayPointer(x);
        double *x_dot_data = N_VGetArrayPointer(x_dot);
        double S = x_data[0];
        double I = x_data[1];
        double R = x_data[2];
        SIR_Param p = param_gen(param, t);

        x_dot_data[0] = -p.beta * S * I / p.N_pop;
        x_dot_data[1] = p.beta * S * I / p.N_pop - p.alpha * I;
        x_dot_data[2] = p.alpha * I;

        return 0;
    }

    // Jacobian function vector routine.
    template<SIR_Param(param_gen)(void *p, double t)>
    int SIR_eval_jac(N_Vector v, N_Vector Jv, double t, N_Vector x, N_Vector fx,
                     void *param, N_Vector tmp) {
        double *x_data = N_VGetArrayPointer(x);
        double *Jv_data = N_VGetArrayPointer(Jv);
        double *v_data = N_VGetArrayPointer(v);
        double S = x_data[0];
        double I = x_data[1];
        double R = x_data[2];
        SIR_Param p = param_gen(param, t);

        Jv_data[0] = -p.beta * I / p.N_pop * v_data[0] - p.beta * S / p.N_pop * v_data[1];
        Jv_data[1] = p.beta * I / p.N_pop * v_data[0] + p.beta * S / p.N_pop * v_data[1] -
                     p.alpha * v_data[1];
        Jv_data[2] = p.alpha * v_data[1];
        return 0;
    }

    SIR_Param SIR_param_gen(void *p, double t) {
        return *static_cast<SIR_Param *>(p);
    }

    struct SIR_Sine_Param {
        double alpha;
        double beta;
        double N_pop;
        double omega;
        double amplitude;
        double offset;
    };


    SIR_Param SIR_sine_param_gen(void *param, double t) {
        SIR_Sine_Param ps = *static_cast<SIR_Sine_Param *>(param);
        SIR_Param p;
        p.alpha = ps.alpha;
        p.beta = std::max({ps.beta + ps.amplitude * sin(t * ps.omega + ps.offset), 0.});
        p.N_pop = ps.N_pop;
        return p;
    }

    template<size_t Nt, typename Param, SIR_Param(param_gen)(void *p, double t)>
    struct SIR_Models : public CVODE_Integrator<3, Nt, SIR_Models<Nt, Param, param_gen>, SIR_Param> {
        Param param;

        SIR_Models(const std::array<double, 3> &x0, const Param &p, double dt, realtype abs_tol = 1e-5,
                   realtype reltol = 1e-5)
                : CVODE_Integrator<3, Nt, SIR_Models<Nt, Param, param_gen>, SIR_Param>(x0, dt, abs_tol, reltol, 0),
                  param(p) {
            int flag = this->initialize_solver(SIR_eval_f<param_gen>, SIR_eval_jac<param_gen>, (void *) &param);
            assert(flag ==
                   EXIT_SUCCESS);
        }
    };

    template<size_t Nt>
    using SIR_Continous = SIR_Models<Nt, SIR_Param, SIR_param_gen>;
    template<size_t Nt>
    using SIR_Sine = SIR_Models<Nt, SIR_Sine_Param, SIR_sine_param_gen>;

    template<size_t Nt>
    struct SIR_Discrete : public Model_Integrator<3, Nt, SIR_Discrete<Nt>, SIR_Sine_Param> {
        using Base = Model_Integrator<3, Nt, SIR_Discrete<Nt>, SIR_Sine_Param>;
        static constexpr size_t Nx = 3;
        const SIR_Sine_Param p;
        size_t iter = 0;
        using Base::t_current;
        SIR_Discrete(const SIR_Sine_Param& p) : p(p){}

        std::array<double, Nx> step(const std::array<double, Nx>& x)
        {
            std::array<double, Nx> res;
            double beta = std::max({0., p.beta + p.amplitude*sin(p.omega*t_current + p.offset)});
            res[0] = x[0] -beta*x[0]*x[1];
            res[1] = x[1] + beta*x[0]*x[1] - p.alpha*x[1];
            res[2] = x[2] + p.alpha*x[1];
            t_current++;
            return res;
        }
    };

} // namespace FROLS::Integrators
#endif // SIR_INTEGRATORS_HPP