#include <gtest/gtest.h>
#include <Integrators/SIR_Integrators.hpp>
#include <array>

TEST(SIR_Integrator, integrator_test)
{
    std::array<float, 3> x0 = {900, 100, 0};
    float R0 = 1.2;
    float alpha = 1./9;
    float beta = R0*alpha;
    float dt = 5;
    float N_pop = 1000;

    FROLS::Integrators::SIR_Deterministic model(x0, alpha, beta, N_pop, dt);
    // auto result = model.run_trajectory(100);

    //Expect population to remain the same
    for (auto x: result)
    {
        EXPECT_NEAR(x[0] + x[1] + x[2], N_pop, 1e-4);
    }

}