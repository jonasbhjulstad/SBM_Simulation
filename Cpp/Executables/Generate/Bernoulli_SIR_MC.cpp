
#include "Bernoulli_SIR_MC_Dynamic.hpp"
#include <quantiles.hpp>
#include <Sycl_Graph_Path_Config.hpp>
#include <Sycl_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
#include <utility>
#include <chrono>



constexpr float p_I0 = 1.0;
constexpr uint32_t N_pop =200;
constexpr float p_ER = 1.00;
constexpr uint32_t Nt = 50;
constexpr uint32_t NV = N_pop;
size_t nk = FROLS::n_choose_k(NV, 2);
uint32_t NE = 1.5*nk;
int main() {
    using namespace SYCL::Graph;
    using namespace Network_Models;
    std::cout << NE << std::endl;
    MC_SIR_Params<>p;
    p.N_pop = N_pop;
    p.p_ER = p_ER;

    //mark start timestamp
    auto start = std::chrono::high_resolution_clock::now();

    std::random_device rd{};
    std::vector<uint32_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&](){return rd();});
    auto enum_seeds = enumerate(seeds);

    std::mt19937_64 rng(rd());

    std::vector<MC_SIR_VectorData> simdatas(p.N_sim);
    auto G = generate_SIR_ER_graph(N_pop, p_ER, rd());
    std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto& es) {
        uint32_t iter = es.first;
        uint32_t seed = es.second;
        auto MC_params = generate_interaction_probabilities(p, rng, Nt);
        std::vector<float> p_Is(Nt);
        std::transform(MC_params.begin(), MC_params.end(), p_Is.begin(), [](auto& p) {return p.p_I;});
        if ((iter % (p.N_sim / 10)) == 0)
        {
            std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
        }
        return MC_SIR_simulation(G, p, seed, p_Is);
    });

    

    std::for_each(simdatas.begin(), simdatas.end(), [&, n= 0](const auto& simdata)mutable
    {
        traj_to_file(p, simdata, n++, Nt);
    });

    using namespace std::placeholders;
    const std::vector<std::string> colnames = {"S", "I", "R", "p_I"};
    auto MC_fname_f = std::bind(MC_filename, p.N_pop, p.p_ER, _1, "SIR");
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames, MC_fname_f, q_fname_f);

    //mark end timestamp
    auto end = std::chrono::high_resolution_clock::now();
    //print time
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    return 0;
}