#ifndef FROLS_SIR_INTEGRATORS_HPP
#define FROLS_SIR_INTEGRATORS_HPP

#include <Integrators/Integrator_Models.hpp>
#include <algorithm>
#include <cassert>
#include <random>

namespace FROLS::Integrators
{

    struct SIR_Stochastic
    {
    public:
        static constexpr uint16_t Nx = 3;
        std::random_device rd;
        std::mt19937 generator;
        float alpha, beta, N_pop, dt;

        SIR_Stochastic(const std::array<float, Nx> &x0, float alpha, float beta,
                       float N_pop, float dt)
            : alpha(alpha), beta(beta), N_pop(N_pop), dt(dt) {}

        std::array<float, 3> step(const std::array<float, 3> &x)
        {
            float p_I = 1 - std::exp(-beta * x[1] / N_pop * dt);
            float p_R = 1 - std::exp(-alpha * dt);

            float K_SI;
            float K_IR;
            std::uint64_t seed = 777;
            std::binomial_distribution<> SI_dist(x[0], p_I);
            std::binomial_distribution<> IR_dist(x[1], p_R);

            K_SI = SI_dist(generator);
            K_IR = IR_dist(generator);

            std::array<float, 3> delta_x = {-K_SI, K_SI - K_IR, K_IR};

            std::array<float, 3> x_next;
            for (int i = 0; i < Nx; i++)
            {
                x_next[i] = std::max({x[i] + delta_x[i], .0f});
            }
            return x_next;
        }
    };

    template <typename dType = float>
    struct SIR_Param
    {
        float alpha;
        float beta;
        float N_pop;
    };

    // Simple function that calculates the differential equation.
    template <SIR_Param<>(param_gen)(void *p, float t)>
    int SIR_eval_f(float t, N_Vector x, N_Vector x_dot, void *param)
    {
        float *x_data = N_VGetArrayPointer(x);
        float *x_dot_data = N_VGetArrayPointer(x_dot);
        float S = x_data[0];
        float I = x_data[1];
        float R = x_data[2];
        SIR_Param<> p = param_gen(param, t);

        x_dot_data[0] = -p.beta * S * I / p.N_pop;
        x_dot_data[1] = p.beta * S * I / p.N_pop - p.alpha * I;
        x_dot_data[2] = p.alpha * I;

        return 0;
    }

    // Jacobian function vector routine.
    template <SIR_Param<>(param_gen)(void *p, float t)>
    int SIR_eval_jac(N_Vector v, N_Vector Jv, float t, N_Vector x, N_Vector fx,
                     void *param, N_Vector tmp)
    {
        float *x_data = N_VGetArrayPointer(x);
        float *Jv_data = N_VGetArrayPointer(Jv);
        float *v_data = N_VGetArrayPointer(v);
        float S = x_data[0];
        float I = x_data[1];
        float R = x_data[2];
        SIR_Param<> p = param_gen(param, t);

        Jv_data[0] = -p.beta * I / p.N_pop * v_data[0] - p.beta * S / p.N_pop * v_data[1];
        Jv_data[1] = p.beta * I / p.N_pop * v_data[0] + p.beta * S / p.N_pop * v_data[1] -
                     p.alpha * v_data[1];
        Jv_data[2] = p.alpha * v_data[1];
        return 0;
    }

    SIR_Param<> SIR_param_gen(void *p, float t)
    {
        return *static_cast<SIR_Param<> *>(p);
    }

    struct SIR_Sine_Param
    {
        float alpha;
        float beta;
        float N_pop;
        float omega;
        float amplitude;
        float offset;
    };

    SIR_Param<> SIR_sine_param_gen(void *param, float t)
    {
        SIR_Sine_Param ps = *static_cast<SIR_Sine_Param *>(param);
        SIR_Param<> p;
        p.alpha = ps.alpha;
        p.beta = std::max({ps.beta + ps.amplitude * sin(t * ps.omega + ps.offset), .0f});
        p.N_pop = ps.N_pop;
        return p;
    }

    template <uint16_t Nt, typename Param, SIR_Param<>(param_gen)(void *p, float t)>
    struct SIR_Models : public CVODE_Integrator<3, Nt, SIR_Models<Nt, Param, param_gen>, SIR_Param<>>
    {
        Param param;

        SIR_Models(const std::array<float, 3> &x0, const Param &p, float dt, float abs_tol = 1e-5,
                   float reltol = 1e-5)
            : CVODE_Integrator<3, Nt, SIR_Models<Nt, Param, param_gen>, SIR_Param<>>(x0, dt, abs_tol, reltol, 0),
              param(p)
        {
            int flag = this->initialize_solver(SIR_eval_f<param_gen>, SIR_eval_jac<param_gen>, (void *)&param);
            assert(flag ==
                   EXIT_SUCCESS);
        }
    };

    template <uint16_t Nt>
    using SIR_Continous = SIR_Models<Nt, SIR_Param<>, SIR_param_gen>;
    template <uint16_t Nt>
    using SIR_Sine = SIR_Models<Nt, SIR_Sine_Param, SIR_sine_param_gen>;

    template <uint16_t Nt>
    struct SIR_Discrete : public Model_Integrator<3, Nt, SIR_Discrete<Nt>, SIR_Sine_Param>
    {
        using Base = Model_Integrator<3, Nt, SIR_Discrete<Nt>, SIR_Sine_Param>;
        static constexpr uint16_t Nx = 3;
        const SIR_Sine_Param p;
        uint16_t iter = 0;
        using Base::t_current;
        SIR_Discrete(const SIR_Sine_Param &p) : p(p) {}

        std::array<float, Nx> step(const std::array<float, Nx> &x)
        {
            std::array<float, Nx> res;
            float beta = std::max({.0f, p.beta + p.amplitude * sin(p.omega * t_current + p.offset)});
            res[0] = x[0] - beta * x[0] * x[1];
            res[1] = x[1] + beta * x[0] * x[1] - p.alpha * x[1];
            res[2] = x[2] + p.alpha * x[1];
            t_current++;
            return res;
        }
    };

} // namespace FROLS::Integrators
#endif // SIR_INTEGRATORS_HPP