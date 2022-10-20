
#include "Bernoulli_SIR_MC.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
#include <utility>
#include <chrono>
template <uint32_t Nt, typename dType=float>
void traj_to_file(const FROLS::MC_SIR_Params<>& p, const FROLS::MC_SIR_SimData<Nt>& d, uint32_t iter)
{
    //print iter
    FROLS::DataFrame df;
    std::array<dType, Nt+1> p_Is;
    std::transform(d.p_vec.begin(), d.p_vec.end(), p_Is.begin(), [](auto& p) { return p.p_I; });
    std::array<dType, Nt+1> p_Rs;
    std::fill(p_Rs.begin(), p_Rs.end(), p.p_R);
    p_Is.back() = 0.;
    df.assign("S", d.traj[0]);
    df.assign("I", d.traj[1]);
    df.assign("R", d.traj[2]);
    df.assign("p_I", p_Is);
    df.assign("p_R", p_Rs);
    auto t = FROLS::range(0, Nt+1);
    df.assign("t", t);
    df.resize(Nt+1);
    df.write_csv(FROLS::MC_filename(p.N_pop, p.p_ER, iter, "SIR"),
                 ",", p.csv_termination_tol);
}
constexpr size_t n_choose_k(size_t n, size_t k)
{
    return (k == 0) ? 1 : (n * n_choose_k(n - 1, k - 1)) / k;
}

constexpr float p_I0 = 1.0;
constexpr uint32_t N_pop =200;
constexpr float p_ER = 1.00;
constexpr uint32_t Nt = 50;
constexpr uint32_t NV = N_pop;
constexpr size_t nk = n_choose_k(NV, 2);
constexpr uint32_t NE = 1.5*nk;
int main() {
    using namespace FROLS;
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
    typedef Network_Models::SIR_Bernoulli_Network<SIR_VectorGraph, decltype(rng), Nt> SIR_Bernoulli_Network;
    std::vector<std::mutex*> v_mx(NV+1);
    //create mutexes
    for (auto& mx : v_mx) {
        mx = new std::mutex();
    }
    std::vector<std::mutex*> e_mx(NE+1);
    //create mutexes
    for (auto& mx : e_mx) {
        mx = new std::mutex();
    }

    SIR_VectorGraph G(v_mx, e_mx);
    generate_erdos_renyi<SIR_VectorGraph, decltype(rng)>(G, p.N_pop, p.p_ER, SIR_S, rng);
    std::vector<MC_SIR_SimData<Nt>> simdatas(p.N_sim);
    std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto& es) {
        uint32_t iter = es.first;
        uint32_t seed = es.second;
        if ((iter % (p.N_sim / 10)) == 0)
        {
            std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
        }
        return MC_SIR_simulation<decltype(G), Nt>(G, p, seed);
    });

    

    std::for_each(simdatas.begin(), simdatas.end(), [&, n= 0](const auto& simdata)mutable
    {
        traj_to_file<Nt>(p, simdata, n++);
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