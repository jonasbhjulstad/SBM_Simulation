#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/ticktock.hpp>
#include <SIR_SBM/queue_select.hpp>

using namespace SIR_SBM;

int main()
{

    int N_pop = 100;
    int N_communities = 10;
    int seed = 10;
    float p_in = 0.5;
    float p_out = 1.0;
    TickTock t;
    t.tick();
    auto graph = generate_planted_SBM<oneapi::dpl::ranlux48>(N_pop, N_communities, p_in, p_out, seed);
    t.tock_print();

    auto N_in = bipartite_max_edges(N_pop, N_pop)*N_communities;
    auto N_out = bipartite_max_edges(N_pop, N_pop)*n_choose_k(N_communities, 2);

    t.tick();
    auto graph_out = generate_planted_SBM<oneapi::dpl::ranlux48>(N_pop, N_communities, 0.0, 1.0, seed);
    throw_if(graph_out.N_edges() != N_out, "Wrong number of planted out edges");
    t.tock_print();
    t.tick();
    auto graph_in = generate_planted_SBM<oneapi::dpl::ranlux48>(N_pop, N_communities, 1.0, 0.0, seed);
    int N_edges = graph_in.N_edges();
    throw_if(graph_in.N_edges() != N_in, "Wrong number of planted in edges");
    t.tock_print();
    return 0;
}