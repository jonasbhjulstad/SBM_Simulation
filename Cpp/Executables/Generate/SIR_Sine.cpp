//
// Created by arch on 9/17/22.
//
#include <Integrators/SIR_Integrators.hpp>
#include <DataFrame.hpp>
constexpr size_t Nt = 100;

std::string SIR_Sine_filename(size_t sim_iter)
{
    return "SIR_Sine_Trajectory_" + std::to_string(sim_iter) + ".csv";
}
int main()
{
    using namespace FROLS;
    double R0 = 1.2;
    double alpha = 1./9;
    double beta_mean = R0*alpha;
    double beta_omega = .5*(2*M_PI);
    double beta_std = beta_mean/2;
    double beta_offset = 0.;
    double N_pop = 1000;
    double I0 = N_pop/10;
    std::array<double, 3> x0 = {N_pop - I0, I0, 0.};
    double beta_p[] = {beta_std, beta_omega, beta_offset};
    double dt = 0.5;
    Integrators::SIR_Sine<Nt> model(x0, {alpha, beta_mean, N_pop, beta_omega, beta_std, beta_offset}, dt);

    auto traj = model.simulate(x0);


}