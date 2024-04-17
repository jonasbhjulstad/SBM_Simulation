#include <SIR_SBM/simulation.hpp>
#include <SIR_SBM/ticktock.hpp>
#include <SIR_SBM/population_count.hpp>
using namespace SIR_SBM;

int main()
{
    int N_pop = 100;
    int N_communities = 2;
    int seed = 10;
    float p_in = 0.5;
    float p_out = 1.0;
    TickTock t;
    t.tick();
    auto graph = generate_planted_SBM<oneapi::dpl::ranlux48>(N_pop, N_communities, p_in, p_out, seed);
    t.tock_print();

    sycl::queue q{sycl::gpu_selector_v}; // Create a queue on the default device
    Sim_Param p;
    p.Nt = 100;
    p.Nt_alloc = 100;
    p.N_I_terminate = 1;
    p.N_sims = 100;
    p.seed = 10;
    auto SB = Sim_Buffers<oneapi::dpl::ranlux48>(q, graph, p);
    SB.wait();


    initialize(q, SB.state, SB.rngs, 0.1).wait();
    
    std::vector<Population_Count> count(N_communities*p.N_sims*p.Nt, {0, 0, 0});
    {
        auto count_buf = sycl::buffer<Population_Count, 3>{count.data(), sycl::range<3>(N_communities, p.N_sims, p.Nt)};
        partition_population_count(q, SB.state, count_buf, SB.vpm, 0).wait();
    }

    auto rec_evt = recover(q, SB.state, SB.rngs, 0.1, 0);

    rec_evt.wait();

    {
        auto count_buf = sycl::buffer<Population_Count, 3>{count.data(), sycl::range<3>(N_communities, p.N_sims, p.Nt)};
        partition_population_count(q, SB.state, count_buf, SB.vpm, 0).wait();
    }



    return 0;

}