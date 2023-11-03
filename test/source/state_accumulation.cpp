#include <doctest/doctest.h>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <Sycl_Buffer_Routines/Buffer_Utils.hpp>
void state_accumulation_test()
{
    sycl::queue q(sycl::gpu_selector_v);

    auto  N_vertices = 100;
    auto N_sims = 10;
    auto Nt = 100;
    auto N_communities = 2;
// sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, std::shared_ptr<sycl::buffer<SIR_State, 3>> &v_buf, std::shared_ptr<sycl::buffer<uint32_t, 2>> &vcm_buf, std::shared_ptr<sycl::buffer<State_t, 3>>& community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims)
    auto v_buf = std::make_shared<sycl::buffer<SIR_State, 3>>(sycl::buffer<SIR_State, 3>(sycl::range<3>(Nt, N_sims, N_vertices)));
    std::vector<SIR_State> v_data(Nt * N_sims * N_vertices);
    for (int i = 0; i < v_data.size(); i++)
    {
        v_data[i] = SIR_INDIVIDUAL_S;
    }

    // auto vcm_buf = std::make_shared<sycl::buffer<uint32_t, 2>>(sycl::buffer<uint32_t, 2>(sycl::range<2>(N_sims, N_vertices)));
    std::vector<uint32_t> vcm_data(N_sims * N_vertices, 0);
    sycl::event event;
    auto vcm_buf = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, vcm_data, sycl::range<1>(N_vertices),event);
    event.wait();

    auto state_buf = std::make_shared<sycl::buffer<State_t, 3>>(sycl::buffer<State_t, 3>(sycl::range<3>(Nt, N_sims, N_communities)));
// sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, std::shared_ptr<sycl::buffer<SIR_State, 3>> &v_buf, std::shared_ptr<sycl::buffer<uint32_t, 2>> &vcm_buf, std::shared_ptr<sycl::buffer<State_t, 3>>& community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims)
    std::vector<sycl::event> events;
    accumulate_community_state(q, events, v_buf, vcm_buf, state_buf, sycl::range<1>(N_sims), sycl::range<1>(N_sims), N_sims);
}

#ifdef SBM_SIMULATION_TEST_ENABLE_CUDA
TEST("State Accumulation")
{
    state_accumulation_test();
}
#endif
