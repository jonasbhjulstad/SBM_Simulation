
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Utils/path_config.hpp>
#include <itertools.hpp>
#include <random>
#include <iostream>
int main(int argc, char **argv)
{
    auto seed = 239;
    std::mt19937 gen(seed);

    auto N_tot = 5e4;
    auto N_communities = 3;
    auto N_pop = N_tot / N_communities;
    auto avg_degree = 3;

    auto p_out = 0.5f;
    auto lbd_out = (N_tot - N_pop) * p_out;
    auto lbd_in = (N_pop)*avg_degree - lbd_out;
    auto p_in = lbd_in / N_pop;


    auto [edgelist, nodelist] = generate_planted_SBM(N_pop, N_communities, p_in, p_out, gen());

    return 0;
}
